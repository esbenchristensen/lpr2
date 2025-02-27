import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PlateDataset
from crnn import CRNN
from torch.nn import CTCLoss
from tqdm import tqdm

# -----------------------
# Training Hyperparameters
# -----------------------
imgH = 32           # image height after resizing
nc = 1              # number of channels (grayscale)
nclass = 37         # 36 characters (A-Z, 0-9) + 1 for CTC blank
nh = 256            # LSTM hidden size
batch_size = 16
epochs = 200
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    # Sort batch by image width in descending order (helps with CTC)
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

# -----------------------
# Create Dataset and DataLoader
# -----------------------
dataset = PlateDataset(
    root_dir="dk_synthetic_dataset",       # or "synthetic_dataset" if that's your folder name
    labels_file="dk_synthetic_dataset/labels.txt"
)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


print("Dataset length:", len(dataset))
# -----------------------
# Create CRNN Model, Loss, and Optimizer
# -----------------------
model = CRNN(imgH, nc, nclass, nh).to(device)
criterion = CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------
# Training Loop
# -----------------------
def train():
    model.train()
    total_start = time.time()  # record overall start time

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, lengths in pbar:
            images = images.to(device)  # shape: (batch, 1, H, W)
            preds = model(images)       # shape: (T, batch, nclass)
            T, batch_sz, _ = preds.size()
            preds_log_softmax = preds.log_softmax(2)
            
            # For CTC loss, input_lengths is T for every batch element
            input_lengths = torch.full(size=(batch_sz,), fill_value=T, dtype=torch.long).to(device)
            labels = labels.to(device)
            
            loss = criterion(preds_log_softmax, labels, input_lengths, torch.tensor(lengths).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # Estimate remaining time: how many epochs left * time of this epoch
        epochs_left = epochs - (epoch + 1)
        est_remaining = epoch_time * epochs_left

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, "
              f"Epoch Time: {epoch_time:.2f}s, Est. Remaining: {est_remaining:.2f}s")

        # Save a checkpoint after each epoch
        torch.save(model.state_dict(), f"weights/crnn_epoch_{epoch+1}.pth")

    total_end = time.time()
    total_time = total_end - total_start
    print(f"Training complete! Total time: {total_time:.2f}s.")

if __name__ == "__main__":
    train()