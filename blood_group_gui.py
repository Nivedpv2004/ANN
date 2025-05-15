import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Constants
IMG_SIZE = (128, 128)

# GUI setup
root = tk.Tk()
root.title("Blood Group Identifier")
root.geometry("500x600")
root.config(bg="white")

# Widgets
img_label = tk.Label(root, bg="white")
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="white", justify="left")
result_label.pack(pady=10)


# Preprocess a PIL image
def preprocess_image(pil_img):
    pil_img = pil_img.resize(IMG_SIZE)
    pil_img = pil_img.convert('RGB')
    img_array = np.array(pil_img) / 255.0
    img_array = img_array.reshape((1, 128, 128, 3))
    return img_array


# Predict clotting in a region
def classify_region(region_img):
    processed = preprocess_image(region_img)
    prediction = model.predict(processed)[0][0]
    return prediction > 0.5, prediction


# Full image classification (blood group)
def classify_combined_image(img_path):
    image = Image.open(img_path)
    width, height = image.size

    # Split into 3 equal vertical regions: Anti-A, Anti-B, Anti-D
    region_a = image.crop((0, 0, width // 3, height))
    region_b = image.crop((width // 3, 0, 2 * width // 3, height))
    region_d = image.crop((2 * width // 3, 0, width, height))

    # Classify each
    clot_a, conf_a = classify_region(region_a)
    clot_b, conf_b = classify_region(region_b)
    clot_d, conf_d = classify_region(region_d)

    # Apply logic
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

    # Display result
    result_text = f"🩸 Predicted Blood Group: {blood_group}\n\n"
    result_text += f"Anti-A: {'Clotted' if clot_a else 'Non-Clotted'} ({conf_a * 100:.2f}%)\n"
    result_text += f"Anti-B: {'Clotted' if clot_b else 'Non-Clotted'} ({conf_b * 100:.2f}%)\n"
    result_text += f"Anti-D: {'Clotted' if clot_d else 'Non-Clotted'} ({conf_d * 100:.2f}%)"
    result_label.config(text=result_text)


# Upload button function
def upload_and_classify():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((350, 150))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        classify_combined_image(file_path)


# Upload button
upload_btn = tk.Button(root, text="Upload Blood Test Image", command=upload_and_classify,
                       font=("Arial", 14), bg="#4CAF50", fg="white")
upload_btn.pack(pady=20)

# Run the GUI
root.mainloop()
