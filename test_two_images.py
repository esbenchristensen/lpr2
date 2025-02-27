import os
import cv2
import torch
import numpy as np
from crnn import CRNN

# Define your alphabet (0 is reserved for CTC blank)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decoder(output):
    # output shape: (T, batch, nclass)
    output = output.cpu().detach().numpy()
    pred_texts = []
    for b in range(output.shape[1]):
        best_path = np.argmax(output[:, b, :], axis=1)
        decoded = []
        prev = -1
        for idx in best_path:
            if idx != prev and idx != 0:  # 0 is blank
                decoded.append(alphabet[idx - 1])
            prev = idx
        pred_texts.append("".join(decoded))
    return pred_texts

def load_crnn_model(weights_path="crnn_epoch_200.pth"):
    # Ensure the number of classes is len(alphabet)+1 for CTC blank
    model = CRNN(imgH=32, nc=1, nclass=len(alphabet)+1, nh=256)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """
    Loads an image, converts to grayscale, resizes to height=32 (preserving aspect ratio),
    and normalizes to [0,1]. Returns a torch tensor with shape (1, 1, H, W).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    H, W = img.shape
    new_H = 32
    new_W = int(W * new_H / H)
    img = cv2.resize(img, (new_W, new_H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # shape: (1, 1, H, W)
    return img_tensor.to(device)

def predict_image(model, image_path):
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        preds = model(img_tensor)  # shape: (T, batch, nclass)
    text = greedy_decoder(preds)[0]
    return text

def main():
    # Update these paths to your test images
    # For example, use one normal and one angled image from your synthetic dataset:
    test_images = [
        "dk_synthetic_dataset/plate_angled_000_TO76194.png",
        "dk_synthetic_dataset/plate_normal_000_TO76194.png"
    ]
    
    # Ground truth labels from your labels.txt for these images (if known)
    ground_truths = {
        "plate_normal_000_TO76194.png": "TO76194",
        "plate_angled_000_TO76194.png": "TO76194"
    }
    
    model = load_crnn_model("crnn_epoch_200.pth")
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Test image not found: {img_path}")
            continue
        predicted = predict_image(model, img_path)
        filename = os.path.basename(img_path)
        truth = ground_truths.get(filename, "Unknown")
        print(f"Image: {filename}")
        print(f"  Ground Truth: {truth}")
        print(f"  Predicted   : {predicted}")
        print("-----------------------------------------------------")

if __name__ == "__main__":
    main()