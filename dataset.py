import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PlateDataset(Dataset):
    def __init__(self, root_dir, labels_file):
        """
        root_dir: Directory containing the synthetic plate images.
        labels_file: A text file where each line is formatted as:
                     <filename> <plate_text>
        """
        self.root_dir = root_dir
        self.samples = []
        with open(labels_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename, text = parts[0], parts[1]
                    self.samples.append((filename, text))
        print(f"[Dataset] Loaded {len(self.samples)} samples from {labels_file}")
        
        # Define the alphabet: A-Z and 0-9.
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        # Create a mapping from character to index; reserve 0 for the CTC blank.
        self.char_to_idx = {char: i+1 for i, char in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        img_path = os.path.join(self.root_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found.")
        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Resize image to fixed height of 32 pixels, preserving aspect ratio
        H, W = img.shape
        new_H = 32
        new_W = int(W * new_H / H)
        img = cv2.resize(img, (new_W, new_H))
        # Expand dimensions so the image becomes (1, H, W)
        img = np.expand_dims(img, axis=0)
        # Convert the plate text to a list of label indices
        label = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        sample = {"image": torch.from_numpy(img), "label": torch.tensor(label, dtype=torch.long), "length": len(label)}
        return sample