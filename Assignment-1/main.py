import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from isp_functions import *

class ImageProcessingUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ISP Tuning Tool")

        # Upload Button
        self.upload_button = tk.Button(master, text="Upload RAW Image", command=self.upload_image)
        self.upload_button.pack()

        # Image Panel
        self.image_panel = tk.Label(master)
        self.image_panel.pack()

    def upload_image(self):
        raw_file = filedialog.askopenfilename(filetypes=[("RAW Files", "*.raw")])
        if not raw_file:
            return

        # Process the uploaded image through the pipeline
        processed_image = self.process_image_pipeline(raw_file)

        # Convert to Image and Display
        img = Image.fromarray(processed_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_panel.imgtk = imgtk
        self.image_panel.config(image=imgtk)

    def process_image_pipeline(self, raw_file):
        """Complete pipeline to process Bayer RAW image to RGB."""
        raw_image = load_raw_image(raw_file)
        rgb_image = demosaic_edge_based(raw_image)
        wb_image = gray_world_white_balance(rgb_image)
        denoise_image = gaussian_denoise(wb_image)
        gamma_image = gamma_correction(denoise_image)
        sharpen_image = unsharp_mask(gamma_image)
        return sharpen_image

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingUI(root)
    root.mainloop()
