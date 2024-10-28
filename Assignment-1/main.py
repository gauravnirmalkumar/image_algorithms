import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
from isp_functions import *

class ImageProcessingUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ISP Tuning Tool")
        
        # Create main containers
        self.left_frame = ttk.Frame(master, padding="10")
        self.right_frame = ttk.Frame(master, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialize image variables
        self.original_image = None
        self.current_image = None
        
        self.setup_image_display()
        self.setup_controls()
        self.setup_processing_buttons()

    def setup_image_display(self):
        """Set up the image display area"""
        # Create canvas for image display
        self.canvas = tk.Canvas(self.left_frame, bg='gray20', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Upload button
        upload_frame = ttk.Frame(self.left_frame)
        upload_frame.pack(fill=tk.X, pady=5)
        self.upload_button = ttk.Button(upload_frame, text="Upload RAW Image", 
                                      command=self.upload_image)
        self.upload_button.pack(pady=5)

    def setup_controls(self):
        """Enhanced control panel with additional parameters"""
        # Color Enhancement Controls
        color_frame = ttk.LabelFrame(self.right_frame, text="Color Enhancement", 
                                   padding="5")
        color_frame.pack(fill=tk.X, pady=5)
        
        # Saturation control
        self.saturation = tk.DoubleVar(value=1.5)  # Default to higher saturation
        ttk.Label(color_frame, text="Saturation:").pack()
        saturation_slider = ttk.Scale(color_frame, from_=0.0, to=2.0,
                                    variable=self.saturation,
                                    orient=tk.HORIZONTAL,
                                    command=self.on_slider_change)
        saturation_slider.pack(fill=tk.X)
        
        # Contrast control
        self.contrast = tk.DoubleVar(value=1.2)  # Default to slightly higher contrast
        ttk.Label(color_frame, text="Contrast:").pack()
        contrast_slider = ttk.Scale(color_frame, from_=0.5, to=1.5,
                                  variable=self.contrast,
                                  orient=tk.HORIZONTAL,
                                  command=self.on_slider_change)
        contrast_slider.pack(fill=tk.X)
        
        # Color temperature control
        self.temperature = tk.IntVar(value=0)
        ttk.Label(color_frame, text="Color Temperature:").pack()
        temp_slider = ttk.Scale(color_frame, from_=-100, to=100,
                              variable=self.temperature,
                              orient=tk.HORIZONTAL,
                              command=self.on_slider_change)
        temp_slider.pack(fill=tk.X)

        # White Balance Controls with strength
        wb_frame = ttk.LabelFrame(self.right_frame, text="White Balance", 
                                padding="5")
        wb_frame.pack(fill=tk.X, pady=5)
        
        self.wb_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(wb_frame, text="Enable", 
                       variable=self.wb_enabled,
                       command=self.process_and_display).pack()
        
        self.wb_strength = tk.DoubleVar(value=1.0)
        ttk.Label(wb_frame, text="Strength:").pack()
        wb_slider = ttk.Scale(wb_frame, from_=0.0, to=2.0,
                            variable=self.wb_strength,
                            orient=tk.HORIZONTAL,
                            command=self.on_slider_change)
        wb_slider.pack(fill=tk.X)

        # Denoise Controls
        denoise_frame = ttk.LabelFrame(self.right_frame, text="Denoising", padding="5")
        denoise_frame.pack(fill=tk.X, pady=5)
        
        self.denoise_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(denoise_frame, text="Enable", 
                       variable=self.denoise_enabled,
                       command=self.process_and_display).pack()
        
        self.kernel_size = tk.IntVar(value=5)
        ttk.Label(denoise_frame, text="Kernel Size:").pack()
        kernel_slider = ttk.Scale(denoise_frame, from_=3, to=15, 
                                variable=self.kernel_size,
                                orient=tk.HORIZONTAL, 
                                command=self.on_slider_change)
        kernel_slider.pack(fill=tk.X)

        # Gamma Controls
        gamma_frame = ttk.LabelFrame(self.right_frame, text="Gamma Correction", 
                                   padding="5")
        gamma_frame.pack(fill=tk.X, pady=5)
        
        self.gamma_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(gamma_frame, text="Enable", 
                       variable=self.gamma_enabled,
                       command=self.process_and_display).pack()
        
        self.gamma_value = tk.DoubleVar(value=2.2)
        ttk.Label(gamma_frame, text="Gamma:").pack()
        gamma_slider = ttk.Scale(gamma_frame, from_=1.0, to=3.0, 
                               variable=self.gamma_value,
                               orient=tk.HORIZONTAL, 
                               command=self.on_slider_change)
        gamma_slider.pack(fill=tk.X)

        # Sharpening Controls
        sharp_frame = ttk.LabelFrame(self.right_frame, text="Sharpening", padding="5")
        sharp_frame.pack(fill=tk.X, pady=5)
        
        self.sharp_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(sharp_frame, text="Enable", 
                       variable=self.sharp_enabled,
                       command=self.process_and_display).pack()
        
        self.sharp_strength = tk.DoubleVar(value=1.0)
        ttk.Label(sharp_frame, text="Strength:").pack()
        sharp_slider = ttk.Scale(sharp_frame, from_=0.0, to=2.0, 
                               variable=self.sharp_strength,
                               orient=tk.HORIZONTAL, 
                               command=self.on_slider_change)
        sharp_slider.pack(fill=tk.X)

    def setup_processing_buttons(self):
        """Set up the processing and save buttons"""
        button_frame = ttk.Frame(self.right_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="Process Image",
                                       command=self.process_and_display)
        self.process_button.pack(fill=tk.X, pady=2)
        
        self.save_button = ttk.Button(button_frame, text="Save Image",
                                    command=self.save_image)
        self.save_button.pack(fill=tk.X, pady=2)

    def upload_image(self):
        """Handle image upload and initial processing"""
        raw_file = filedialog.askopenfilename(filetypes=[("RAW Files", "*.raw")])
        if not raw_file:
            return

        self.original_image = load_raw_image(raw_file)
        self.process_and_display()

    def process_and_display(self):
        """Process the image with current parameters and display it"""
        if self.original_image is None:
            return

        # Gather current parameters
        params = {
            'wb_enabled': self.wb_enabled.get(),
            'wb_strength': self.wb_strength.get(),
            'temperature': self.temperature.get(),
            'contrast': self.contrast.get(),
            'saturation': self.saturation.get(),
            'denoise_enabled': self.denoise_enabled.get(),
            'kernel_size': self.kernel_size.get(),
            'gamma_enabled': self.gamma_enabled.get(),
            'gamma_value': self.gamma_value.get(),
            'sharp_enabled': self.sharp_enabled.get(),
            'sharp_strength': self.sharp_strength.get()
        }

        # Process image through pipeline
        self.current_image = process_pipeline(self.original_image, params)
        
        # Convert to PhotoImage and display
        image = Image.fromarray(self.current_image)
        # Resize if needed
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=photo, anchor=tk.CENTER)
        self.canvas.image = photo  # Keep a reference!

    def on_slider_change(self, _):
        """Handle slider value changes"""
        self.process_and_display()

    def save_image(self):
        """Save the processed image"""
        if self.current_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), 
                      ("JPEG files", "*.jpg"), 
                      ("All files", "*.*")]
        )
        if file_path:
            Image.fromarray(self.current_image).save(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingUI(root)
    root.mainloop()