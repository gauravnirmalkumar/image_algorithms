import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from PIL.ExifTags import TAGS
import piexif
import threading

class ModernHDRProcessor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("HDR Image Processor")
        self.window.configure(bg='#f0f0f0')
        self.window.geometry("1280x900")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('Modern.TFrame', background='#f0f0f0')
        self.style.configure('Modern.TButton', padding=10, font=('Helvetica', 10))
        self.style.configure('Modern.TLabel', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('Title.TLabel', background='#f0f0f0', font=('Helvetica', 14, 'bold'))
        
        # State variables
        self.input_images = []
        self.exposure_times = []
        self.current_hdr = None
        self.current_ldr = None
        self.image_previews = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.window, style='Modern.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="HDR Image Processor", 
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))
        
        # Top frame for image selection and parameters
        top_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        top_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Image selection frame
        image_frame = ttk.LabelFrame(
            top_frame, 
            text="Image Selection", 
            padding="10",
            style='Modern.TFrame'
        )
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image preview frame
        self.preview_frame = ttk.Frame(image_frame, style='Modern.TFrame')
        self.preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create 4 image preview placeholders
        for i in range(4):
            preview = ttk.Label(self.preview_frame, text=f"Image {i+1}")
            preview.grid(row=0, column=i, padx=5)
            self.image_previews.append(preview)
        
        # Load button and status
        ttk.Button(
            image_frame,
            text="Load 4 Images",
            command=self.load_images,
            style='Modern.TButton'
        ).pack(pady=(0, 5))
        
        self.status_label = ttk.Label(
            image_frame,
            text="No images loaded",
            style='Modern.TLabel'
        )
        self.status_label.pack()
        
        # Exposure times display
        self.exposure_label = ttk.Label(
            image_frame,
            text="",
            style='Modern.TLabel',
            wraplength=400
        )
        self.exposure_label.pack(pady=5)
        
        # Parameters frame
        param_frame = ttk.LabelFrame(
            top_frame,
            text="Processing Parameters",
            padding="10",
            style='Modern.TFrame'
        )
        param_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # Merge method
        ttk.Label(
            param_frame,
            text="Merge Algorithm:",
            style='Modern.TLabel'
        ).pack(anchor=tk.W)
        
        self.merge_method = ttk.Combobox(
            param_frame,
            values=["Debevec", "Robertson", "Mertens"],
            state="readonly",
            width=30
        )
        self.merge_method.set("Debevec")
        self.merge_method.pack(pady=(0, 10))
        
        # Tone mapping method
        ttk.Label(
            param_frame,
            text="Tone Mapping:",
            style='Modern.TLabel'
        ).pack(anchor=tk.W)
        
        self.tonemap_method = ttk.Combobox(
            param_frame,
            values=["Drago", "Reinhard", "Mantiuk"],
            state="readonly",
            width=30
        )
        self.tonemap_method.set("Drago")
        self.tonemap_method.pack(pady=(0, 10))
        
        # Processing buttons
        button_frame = ttk.Frame(param_frame, style='Modern.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame,
            text="Process HDR",
            command=self.process_hdr_threaded,
            style='Modern.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Save Result",
            command=self.save_result,
            style='Modern.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            param_frame,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(pady=10)
        
        # Main preview frame
        preview_label_frame = ttk.LabelFrame(
            main_frame,
            text="HDR Preview",
            padding="10",
            style='Modern.TFrame'
        )
        preview_label_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_label = ttk.Label(preview_label_frame)
        self.preview_label.pack(expand=True)

    def get_exposure_time(self, image_path):
        try:
            exif_dict = piexif.load(image_path)
            if exif_dict['Exif']:
                exposure_data = exif_dict['Exif'].get(33434)
                if exposure_data:
                    num, den = exposure_data
                    return num / den
            
            img = Image.open(image_path)
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'ExposureTime':
                        if isinstance(value, tuple):
                            return value[0] / value[1]
                        return float(value)
            
            return None
        except Exception as e:
            print(f"Error reading EXIF data: {str(e)}")
            return None

    def create_thumbnail(self, image, size=(150, 150)):
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 1:
            new_width = size[0]
            new_height = int(size[0] / aspect_ratio)
        else:
            new_height = size[1]
            new_width = int(size[1] * aspect_ratio)
            
        resized = cv2.resize(image, (new_width, new_height))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
        return ImageTk.PhotoImage(image)

    def load_images(self):
        filepaths = filedialog.askopenfilenames(
            title="Select 4 Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if len(filepaths) != 4:
            messagebox.showerror("Error", "Please select exactly 4 images")
            return
            
        self.input_images = []
        self.exposure_times = []
        exposure_info = "Exposure Times:\n"
        
        # Clear existing previews
        for preview in self.image_previews:
            preview.configure(image='', text='')
        
        for i, filepath in enumerate(filepaths):
            # Load image
            img = cv2.imread(filepath)
            if img is None:
                messagebox.showerror("Error", f"Failed to load {filepath}")
                return
            self.input_images.append(img)
            
            # Create and update thumbnail
            thumbnail = self.create_thumbnail(img)
            self.image_previews[i].configure(image=thumbnail)
            self.image_previews[i].image = thumbnail
            
            # Get exposure time
            exposure_time = self.get_exposure_time(filepath)
            if exposure_time:
                self.exposure_times.append(exposure_time)
                exposure_info += f"Image {i+1}: {exposure_time:.4f}s\n"
            else:
                messagebox.showwarning("Warning", f"Could not detect exposure time for {os.path.basename(filepath)}")
                
        # If we couldn't detect all exposure times, use defaults
        if len(self.exposure_times) != 4:
            self.exposure_times = np.array([1/320.0, 1/160.0, 1/60.0, 1/15.0], dtype=np.float32)
            exposure_info += "\nUsing default exposure times: 1/320s, 1/160s, 1/60s, 1/15s"
        else:
            self.exposure_times = np.array(self.exposure_times, dtype=np.float32)
        
        self.status_label.config(text=f"{len(self.input_images)} images loaded")
        self.exposure_label.config(text=exposure_info)
        self.resize_images()

    def resize_images(self):
        if not self.input_images:
            return
            
        heights = [img.shape[0] for img in self.input_images]
        widths = [img.shape[1] for img in self.input_images]
        
        min_height = min(heights)
        min_width = min(widths)
        
        for i in range(len(self.input_images)):
            if heights[i] != min_height or widths[i] != min_width:
                self.input_images[i] = cv2.resize(self.input_images[i], (min_width, min_height))

    def create_merge_algorithm(self):
        method = self.merge_method.get()
        if method == "Debevec":
            return cv2.createMergeDebevec()
        elif method == "Robertson":
            return cv2.createMergeRobertson()
        else:  # Mertens
            return cv2.createMergeMertens()

    def create_tonemap_algorithm(self):
        method = self.tonemap_method.get()
        if method == "Drago":
            return cv2.createTonemapDrago()
        elif method == "Reinhard":
            return cv2.createTonemapReinhard()
        else:  # Mantiuk
            return cv2.createTonemapMantiuk()

    def process_hdr_threaded(self):
        self.progress.start()
        threading.Thread(target=self.process_hdr, daemon=True).start()

    def process_hdr(self):
        if not self.input_images:
            messagebox.showerror("Error", "Please load images first")
            self.progress.stop()
            return
            
        try:
            # Create merge algorithm
            merge_algorithm = self.create_merge_algorithm()
            
            # Merge images
            if self.merge_method.get() == "Mertens":
                self.current_hdr = merge_algorithm.process(self.input_images)
            else:
                self.current_hdr = merge_algorithm.process(self.input_images, times=self.exposure_times)
            
            # Apply tone mapping
            tonemap_algorithm = self.create_tonemap_algorithm()
            self.current_ldr = tonemap_algorithm.process(self.current_hdr)
            
            # Normalize and convert to 8-bit
            self.current_ldr = np.clip(self.current_ldr * 255, 0, 255).astype('uint8')
            
            # Update preview
            self.window.after(0, self.update_preview)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.progress.stop()

    def update_preview(self):
        if self.current_ldr is None:
            return
            
        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(self.current_ldr, cv2.COLOR_BGR2RGB)
        
        # Resize for preview if needed
        max_preview_size = 800
        height, width = rgb_image.shape[:2]
        
        if height > max_preview_size or width > max_preview_size:
            scale = max_preview_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
            
        # Convert to PhotoImage
        image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image)
        
        # Update preview label
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def save_result(self):
        if self.current_ldr is None:
            messagebox.showerror("Error", "No processed image to save")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filepath:
            cv2.imwrite(filepath, self.current_ldr)
            messagebox.showinfo("Success", "Image saved successfully")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ModernHDRProcessor()
    app.run()