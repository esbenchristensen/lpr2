from dataset import PlateDataset

def main():
    # Update these paths to match your dataset folder and labels file.
    root_dir = "dk_synthetic_dataset"  # Folder where your images are stored.
    labels_file = "dk_synthetic_dataset/labels.txt"
    
    dataset = PlateDataset(root_dir=root_dir, labels_file=labels_file)
    
    # Print details for the first 10 samples in the dataset.
    for idx in range(10):
        sample = dataset[idx]
        print(f"Sample {idx}:")
        print("  Image shape:", sample["image"].shape)
        print("  Label tensor:", sample["label"])
        print("  Label length:", sample["length"])
    
if __name__ == "__main__":
    main()