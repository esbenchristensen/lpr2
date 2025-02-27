import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from dataset import PlateDataset
from crnn import CRNN
from torch.nn import CTCLoss
from tqdm import tqdm

# ---------------------------------------------------
# 1) Hyperparameters
# ---------------------------------------------------
imgH = 32                  # image height after resizing
nc = 1                     # number of channels (grayscale)
nclass = 37                # 36 characters (A-Z and 0-9) + 1 for CTC blank
nh = 256                   # LSTM hidden size
batch_size = 16            # using batch size 16 for efficient training
total_epochs = 100         # total training epochs
# Two-phase learning rate schedule:
# Phase 1: Epochs 0–79 use LR = 0.001
# Phase 2: Epochs 80–100 use LR = 1e-4
lr_phase1 = 0.001
lr_phase2 = 1e-4
lr_switch_epoch = 80       # switch learning rate at epoch 80

patience = 10              # early stopping patience (epochs without improvement)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the alphabet (index 0 is reserved for CTC blank)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ---------------------------------------------------
# 2) Collate Function (with padding)
# ---------------------------------------------------
def collate_fn(batch):
    # Sort batch by image width in descending order
    batch.sort(key=lambda x: x["image"].shape[-1], reverse=True)
    images = [sample["image"] for sample in batch]
    labels = [sample["label"] for sample in batch]
    lengths = [sample["length"] for sample in batch]
    
    # Pad images so they have the same width
    max_width = max(img.shape[-1] for img in images)
    padded_images = []
    for img in images:
        pad_width = max_width - img.shape[-1]
        if pad_width > 0:
            img = torch.nn.functional.pad(img, (0, pad_width), "constant", 0)
        padded_images.append(img)
    images = torch.stack(padded_images)
    labels_concat = torch.cat(labels)
    return images, labels_concat, lengths

# ---------------------------------------------------
# 3) Create Dataset and Split into Train/Validation
# ---------------------------------------------------
# Ensure your synthetic dataset folder "dk_synthetic_dataset" contains 10,000 images
# and "dk_synthetic_dataset/labels.txt" contains 10,000 corresponding lines.
dataset = PlateDataset(root_dir="dk_synthetic_dataset", labels_file="dk_synthetic_dataset/labels.txt")
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Total samples: {total_samples} | Training: {train_size} | Validation: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---------------------------------------------------
# 4) Create Model, Loss, Optimizer, and Scheduler
# ---------------------------------------------------
model = CRNN(imgH, nc, nclass, nh).to(device)
criterion = CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=lr_phase1)

# We'll manually adjust the learning rate at epoch lr_switch_epoch
def adjust_learning_rate(epoch):
    if epoch == lr_switch_epoch:
        for g in optimizer.param_groups:
            g['lr'] = lr_phase2
        print(f"*** Switching to Phase 2 LR: {lr_phase2} at epoch {epoch+1} ***")

# Optionally, use a StepLR scheduler if desired:
# scheduler = StepLR(optimizer, step_size=lr_switch_epoch, gamma=lr_phase2 / lr_phase1)

# ---------------------------------------------------
# 5) Beam Search Decoder for CTC (Beam width = 10)
# ---------------------------------------------------
def beam_search_decoder(output, beam_width=10):
    """
    A simple beam search decoder for CTC.
    Assumes output shape is (T, 1, nclass) with log probabilities.
    Returns the decoded string.
    """
    T = output.shape[0]
    beam = [("", 0.0)]
    for t in range(T):
        new_beam = {}
        log_probs = output[t, 0, :]  # shape: (nclass,)
        for seq, score in beam:
            for c in range(nclass):
                new_seq = seq
                new_score = score + log_probs[c].item()
                if c != 0:  # if not blank, append character
                    new_seq = seq + alphabet[c-1]
                if new_seq in new_beam:
                    prev_score = new_beam[new_seq]
                    new_beam[new_seq] = math.log(math.exp(prev_score) + math.exp(new_score))
                else:
                    new_beam[new_seq] = new_score
        beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
    best_seq, _ = beam[0]
    # Collapse repeated characters (CTC collapse)
    final_seq = []
    prev_char = ""
    for char in best_seq:
        if char != prev_char:
            final_seq.append(char)
        prev_char = char
    return "".join(final_seq)

# ---------------------------------------------------
# 6) Inference Function using Beam Search Decoder
# ---------------------------------------------------
def infer(sample):
    model.eval()
    image = sample["image"].unsqueeze(0).to(device)  # shape: (1, 1, H, W)
    with torch.no_grad():
        preds = model(image)  # shape: (T, 1, nclass)
    decoded_text = beam_search_decoder(preds, beam_width=10)
    return decoded_text

# ---------------------------------------------------
# 7) Training and Validation Loop with Early Stopping
# ---------------------------------------------------
def train_and_validate():
    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_start = time.time()
    
    for epoch in range(total_epochs):
        adjust_learning_rate(epoch)
        model.train()
        epoch_start = time.time()
        train_epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} (Train)")
        for images, labels, lengths in pbar:
            images = images.to(device)
            preds = model(images)  # shape: (T, batch, nclass)
            T, batch_sz, _ = preds.size()
            preds_log_softmax = preds.log_softmax(2)
            
            input_lengths = torch.full((batch_sz,), T, dtype=torch.long).to(device)
            labels = labels.to(device)
            
            loss = criterion(preds_log_softmax, labels, input_lengths, torch.tensor(lengths).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        
        # Validation step
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for images, labels, lengths in val_loader:
                images = images.to(device)
                preds = model(images)
                T, batch_sz, _ = preds.size()
                preds_log_softmax = preds.log_softmax(2)
                input_lengths = torch.full((batch_sz,), T, dtype=torch.long).to(device)
                labels = labels.to(device)
                loss = criterion(preds_log_softmax, labels, input_lengths, torch.tensor(lengths).to(device))
                val_epoch_loss += loss.item()
        avg_val_loss = val_epoch_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{total_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epoch Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), f"weights/crnn_epoch_{epoch+1}.pth")
        
        # Early stopping based on validation loss improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"*** Early stopping at epoch {epoch+1}: no improvement for {patience} epochs ***")
            break
        
        # Optionally, print predictions for a few validation samples for debugging
        model.eval()
        sample_predictions = []
        for i, sample in enumerate(val_dataset):
            if i >= 5:  # print only for first 5 validation samples
                break
            true_label_indices = sample["label"].tolist()
            true_text = "".join([alphabet[idx-1] for idx in true_label_indices])
            predicted = infer(sample)
            sample_predictions.append(f"Val Sample {i}: True={true_text} | Predicted={predicted}")
        for line in sample_predictions:
            print(line)
            
    total_end = time.time()
    print(f"Training complete! Total time: {total_end - total_start:.2f}s")

# ---------------------------------------------------
# 8) Final Test on Validation Set
# ---------------------------------------------------
def test_final():
    model.eval()
    for i, sample in enumerate(val_dataset):
        true_label_indices = sample["label"].tolist()
        true_text = "".join([alphabet[idx-1] for idx in true_label_indices])
        predicted = infer(sample)
        print(f"Validation Sample {i}: True={true_text} | Predicted={predicted}")

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    print("Starting final training on 10,000 samples (80% train, 20% val).")
    train_and_validate()
    print("Testing final validation results:")
    test_final()