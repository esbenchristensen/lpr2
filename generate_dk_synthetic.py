import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# -----------------------
# 1) Global Config
# -----------------------
OUTPUT_DIR = "dk_synthetic_dataset"
IMG_WIDTH, IMG_HEIGHT = 520, 110   # base plate dimensions
BLUE_ICON_WIDTH = int(IMG_WIDTH * 0.15)  # ~15% for the blue DK icon
FONT_PATH = "/Users/esbenchristensen/Github/lpr2/licensefont.ttf"  # Replace with your font file (.ttf or .otf)
NUM_IMAGES = 2000                  # Generate 1000 base images for demonstration
RANDOM_ANGLE_MAX = 15             # up to ±25 degrees rotation
PERSPECTIVE_MAX = 0.15            # up to 20% perspective distortion
SCALE_MIN = 0.7                   # Minimum scale factor
SCALE_MAX = 1.2                   # Maximum scale factor

os.makedirs(OUTPUT_DIR, exist_ok=True)

# We'll store labels in a text file in the same directory
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.txt")

# -----------------------
# 2) Functions
# -----------------------

def generate_plate_text():
    """
    Danish plates often have 2 letters + 5 digits, e.g. 'BT29579'.
    We'll generate exactly that format.
    """
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    digits = ''.join(random.choices("0123456789", k=5))
    return letters + digits

def create_plate_image(plate_text):
    """
    Create a base plate image with:
      - White background
      - Blue DK icon on the left
      - Plate text using a custom font
    Returns a Pillow Image (RGB).
    """
    # Create blank white image
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw the blue DK icon on the left
    for x in range(BLUE_ICON_WIDTH):
        for y in range(IMG_HEIGHT):
            draw.point((x, y), fill=(0, 0, 255))  # pure blue in RGB

    # Load custom font
    try:
        font = ImageFont.truetype(FONT_PATH, size=70)  # Adjust size as needed
    except IOError:
        print("WARNING: Could not load custom font, using default.")
        font = ImageFont.load_default()

    # Position the text: just to the right of the blue area
    text_x = BLUE_ICON_WIDTH + 10
    text_y = 20  # trial offset
    draw.text((text_x, text_y), plate_text, font=font, fill=(0, 0, 0))
    return img

def rotate_image_pil(img, angle):
    """
    Rotate the PIL image around its center by the given angle.
    Expand=True so we don't crop corners.
    """
    return img.rotate(angle, expand=True, fillcolor=(255,255,255))

def perspective_transform_cv2(pil_img):
    """
    Convert PIL -> OpenCV, apply random perspective transform, then back to PIL.
    """
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    shift_x = int(PERSPECTIVE_MAX * w)
    shift_y = int(PERSPECTIVE_MAX * h)

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [random.randint(-shift_x, shift_x), random.randint(-shift_y, shift_y)],
        [w + random.randint(-shift_x, shift_x), random.randint(-shift_y, shift_y)],
        [w + random.randint(-shift_x, shift_x), h + random.randint(-shift_y, shift_y)],
        [random.randint(-shift_x, shift_x), h + random.randint(-shift_y, shift_y)]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cv_warped = cv2.warpPerspective(cv_img, M, (w, h), borderValue=(255,255,255))
    warped_pil = Image.fromarray(cv2.cvtColor(cv_warped, cv2.COLOR_BGR2RGB))
    return warped_pil

# -----------------------
# 3) Main
# -----------------------
if __name__ == "__main__":
    # Open the labels file in write mode
    with open(LABELS_FILE, "w") as label_f:
        for i in range(NUM_IMAGES):
            plate_text = generate_plate_text()
            # Create the base plate image
            base_img = create_plate_image(plate_text)

            # Randomly scale the plate image to simulate different distances
            scale_factor = random.uniform(SCALE_MIN, SCALE_MAX)
            new_width = int(IMG_WIDTH * scale_factor)
            new_height = int(IMG_HEIGHT * scale_factor)
            scaled_img = base_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save the "normal" (scaled) plate
            normal_filename = f"plate_normal_{i:03d}_{plate_text}.png"
            normal_path = os.path.join(OUTPUT_DIR, normal_filename)
            scaled_img.save(normal_path)
            
            # Write label for normal image
            # Format: filename <space> text
            label_f.write(f"{normal_filename} {plate_text}\n")

            # Apply random rotation
            angle = random.uniform(-RANDOM_ANGLE_MAX, RANDOM_ANGLE_MAX)
            rotated_img = rotate_image_pil(scaled_img, angle)

            # Then apply perspective transform
            perspective_img = perspective_transform_cv2(rotated_img)

            # Save the final "angled" plate
            angled_filename = f"plate_angled_{i:03d}_{plate_text}.png"
            angled_path = os.path.join(OUTPUT_DIR, angled_filename)
            perspective_img.save(angled_path)

            # Write label for angled image
            label_f.write(f"{angled_filename} {plate_text}\n")

    print(f"Done! Generated {NUM_IMAGES} normal plates + {NUM_IMAGES} angled plates.")
    print(f"Labels saved to: {LABELS_FILE}")