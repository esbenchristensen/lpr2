import torch
import cv2
import numpy as np
from crnn import CRNN

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # 36 chars
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

def load_model(weights_path="crnn_epoch_200.pth"):
    model = CRNN(imgH=32, nc=1, nclass=len(alphabet)+1, nh=256)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not read image:", image_path)
        return None
    # Resize to height=32, preserve aspect ratio
    H, W = img.shape
    new_H = 32
    new_W = int(W * new_H / H)
    img = cv2.resize(img, (new_W, new_H))
    img = img.astype(np.float32) / 255.0
    # shape: (1, H, W)
    img = np.expand_dims(img, axis=0)
    # shape: (batch=1, channels=1, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_tensor)  # (T, batch, nclass)
    texts = greedy_decoder(preds)
    return texts[0]

if __name__ == "__main__":
    # Example usage
    my_model = load_model("crnn_epoch_200.pth")
    test_img = "plate_angled_000_TO76194.png"  # replace with your own test image
    result_text = predict(my_model, test_img)
    print("Recognized text:", result_text)