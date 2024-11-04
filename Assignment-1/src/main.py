import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL, LabelFrame
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.input_img = None

    def load_image(self, path):
        image = np.fromfile(path, dtype=np.uint16).reshape((self.height, self.width))
        self.input_img = np.clip(image, 0, 4095)
        return self.input_img

    def demosaic(self, image):
        return cv2.demosaicing(image, cv2.COLOR_BayerGB2BGR)

    def white_balance(self, image):
        b, g, r = cv2.split(image)
        avg = (np.mean(b) + np.mean(g) + np.mean(r)) / 3
        b, g, r = cv2.multiply(b, avg / np.mean(b)), cv2.multiply(g, avg / np.mean(g)), cv2.multiply(r, avg / np.mean(r))
        return cv2.merge([b, g, r])

    def denoise(self, image, strength):
        return cv2.GaussianBlur(image, (strength, strength), 0)

    def gamma_correction(self, image, gamma=2.2):
        image = (image / 4095.0) ** (1 / gamma)
        return (image * 255).astype(np.uint8)

    def sharpen(self, image, amount=1.5):
        blurred = cv2.GaussianBlur(image, (0, 0), amount)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def normalize_and_contrast(self, image):
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    def process_image(self, image, wb=True, denoise_strength=3, gamma=2.2, sharpen_amount=1.5):
        image = self.demosaic(image)
        if wb:
            image = self.white_balance(image)
        image = self.denoise(image, denoise_strength)
        image = self.gamma_correction(image, gamma)
        image = self.sharpen(image, sharpen_amount)
        image = self.normalize_and_contrast(image)
        return image

def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, path="processed_image.jpg"):
    cv2.imwrite(path, image)
    print(f"Image saved as {path}")

class ImageApp:
    def __init__(self, root):
        self.processor = ImageProcessor(1920, 1280)
        self.image = None
        self.root = root
        self.create_ui()

    def create_ui(self):
        self.root.title("Image Processing Tool")
        self.root.geometry("500x600")
        
        # Title Label
        title_label = tk.Label(self.root, text="Image Processing Tool", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # Frame for file loading
        file_frame = LabelFrame(self.root, text="Load Image", font=("Helvetica", 12, "bold"), padx=10, pady=10)
        file_frame.pack(pady=10, fill="x")

        load_button = tk.Button(file_frame, text="Select Image", font=("Helvetica", 12), command=self.load_image, width=15)
        load_button.pack(pady=5)

        # Frame for processing options
        options_frame = LabelFrame(self.root, text="Processing Options", font=("Helvetica", 12, "bold"), padx=10, pady=10)
        options_frame.pack(pady=10, fill="both", expand=True)

        # White Balance
        self.wb_var = tk.IntVar(value=1)
        wb_check = tk.Checkbutton(options_frame, text="Apply White Balance", variable=self.wb_var, font=("Helvetica", 12))
        wb_check.pack(anchor="w")

        # Denoise Strength
        self.denoise_slider = Scale(options_frame, from_=1, to=10, orient=HORIZONTAL, label="Denoise Strength",
                                    font=("Helvetica", 10), length=400)
        self.denoise_slider.set(3)
        self.denoise_slider.pack(pady=5)

        # Gamma Correction
        self.gamma_slider = Scale(options_frame, from_=1.0, to=3.0, resolution=0.1, orient=HORIZONTAL, label="Gamma Correction",
                                  font=("Helvetica", 10), length=400)
        self.gamma_slider.set(2.2)
        self.gamma_slider.pack(pady=5)

        # Sharpen Amount
        self.sharpen_slider = Scale(options_frame, from_=0.5, to=3.0, resolution=0.1, orient=HORIZONTAL, label="Sharpen Amount",
                                    font=("Helvetica", 10), length=400)
        self.sharpen_slider.set(1.5)
        self.sharpen_slider.pack(pady=5)

        # Process and Save Button
        button_frame = LabelFrame(self.root, text="Actions", font=("Helvetica", 12, "bold"), padx=10, pady=10)
        button_frame.pack(pady=10, fill="x")

        process_button = tk.Button(button_frame, text="Process Image", font=("Helvetica", 12), command=self.process_image, width=15)
        process_button.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Raw Image Files", "*.raw")])
        if file_path:
            self.image = self.processor.load_image(file_path)
            img_display = Image.fromarray((self.image / 4095 * 255).astype(np.uint8))
            img_display = ImageTk.PhotoImage(img_display)
            img_label = tk.Label(self.root, image=img_display)
            img_label.image = img_display
            img_label.pack(pady=10)

    def process_image(self):
        if self.image is None:
            return

        processed_image = self.processor.process_image(
            image=self.image,
            wb=self.wb_var.get(),
            denoise_strength=self.denoise_slider.get(),
            gamma=self.gamma_slider.get(),
            sharpen_amount=self.sharpen_slider.get()
        )
        
        show_image(processed_image)
        save_image(processed_image)

root = tk.Tk()
app = ImageApp(root)
root.mainloop()
