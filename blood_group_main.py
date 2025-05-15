from PIL import Image
import numpy as np
import tensorflow as tf

# Load trained clot detection model
model = tf.keras.models.load_model('best_model.h5')

# Constants
IMG_SIZE = (128, 128)

# Preprocess a PIL image for prediction
def preprocess_image(pil_img):
    pil_img = pil_img.resize(IMG_SIZE)
    pil_img = pil_img.convert('RGB')
    img_array = np.array(pil_img) / 255.0
    img_array = img_array.reshape((1, 128, 128, 3))
    return img_array

# Classify a region image (returns clot status and confidence)
def classify_region(region_img):
    img_input = preprocess_image(region_img)
    prediction = model.predict(img_input)[0][0]
    is_clotted = prediction > 0.5
    return is_clotted, prediction

# Full image processing and blood group determination
def identify_blood_group(image_path):
    image = Image.open(image_path)
    width, height = image.size

    # Split into 3 equal vertical regions: Anti-A, Anti-B, Anti-D
    region_a = image.crop((0, 0, width // 3, height))
    region_b = image.crop((width // 3, 0, 2 * width // 3, height))
    region_d = image.crop((2 * width // 3, 0, width, height))

    # Classify each region
    clot_a, conf_a = classify_region(region_a)
    clot_b, conf_b = classify_region(region_b)
    clot_d, conf_d = classify_region(region_d)

    # Apply blood group logic
    if clot_a and clot_b:
        group = "AB"
    elif clot_a:
        group = "A"
    elif clot_b:
        group = "B"
    else:
        group = "O"

    rh = "+" if clot_d else "-"
    blood_group = group + rh

    # Output
    print("=== Blood Group Identification ===")
    print(f"Anti-A Region: {'Clotted' if clot_a else 'Non-Clotted'} ({conf_a*100:.2f}%)")
    print(f"Anti-B Region: {'Clotted' if clot_b else 'Non-Clotted'} ({conf_b*100:.2f}%)")
    print(f"Anti-D Region: {'Clotted' if clot_d else 'Non-Clotted'} ({conf_d*100:.2f}%)")
    print(f"\n🩸 Predicted Blood Group: {blood_group}")

# Run this block
if __name__ == "__main__":
    image_path = r"C:\Users\hp\Documents\split_dataset\test\A+ve\7_A_+ve.jpg"  # Path to your input image
    identify_blood_group(image_path)
