import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from PIL.ExifTags import TAGS
import piexif
import threading
from datetime import datetime

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
        
        # HDR Processing parameters
        self.hdr_params = {
            'gamma': 0.77,
            'alpha': 0.35,
            'gamma_local': 1.5
        }
        
        # Available algorithms
        self.merge_algorithms = {
            'Simple Weighted': self.merge_exposures,
            'Debevec': self.merge_debevec,
            'Mertens': self.merge_mertens
        }
        
        self.tone_mapping_algorithms = {
            'Local Tone Mapping': self.tone_map,
            'Reinhard': self.tone_map_reinhard,
            'Drago': self.tone_map_drago,
            'Mantiuk': self.tone_map_mantiuk
        }
        
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.window, style='Modern.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="HDR Image Processor", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Top frame for controls
        top_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        top_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Image selection frame
        image_frame = ttk.LabelFrame(top_frame, text="Image Selection", padding="10", style='Modern.TFrame')
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.preview_frame = ttk.Frame(image_frame, style='Modern.TFrame')
        self.preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        for i in range(3):
            preview = ttk.Label(self.preview_frame, text=f"Image {i+1}")
            preview.grid(row=0, column=i, padx=5)
            self.image_previews.append(preview)
        
        ttk.Button(image_frame, text="Load Images", command=self.load_images, style='Modern.TButton').pack(pady=(0, 5))
        
        # Algorithm selection frame
        algo_frame = ttk.LabelFrame(top_frame, text="Algorithm Selection", padding="10", style='Modern.TFrame')
        algo_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Merge algorithm dropdown
        ttk.Label(algo_frame, text="Merge Algorithm:", style='Modern.TLabel').pack(anchor=tk.W)
        self.merge_var = tk.StringVar(value='Mertens')
        merge_dropdown = ttk.Combobox(algo_frame, textvariable=self.merge_var, values=list(self.merge_algorithms.keys()))
        merge_dropdown.pack(fill=tk.X, pady=(0, 10))
        
        # Tone mapping algorithm dropdown
        ttk.Label(algo_frame, text="Tone Mapping:", style='Modern.TLabel').pack(anchor=tk.W)
        self.tone_var = tk.StringVar(value='Local Tone Mapping')
        tone_dropdown = ttk.Combobox(algo_frame, textvariable=self.tone_var, values=list(self.tone_mapping_algorithms.keys()))
        tone_dropdown.pack(fill=tk.X, pady=(0, 10))
        tone_dropdown.bind('<<ComboboxSelected>>', self.update_parameter_visibility)
        
        # Parameters frame
        self.param_frame = ttk.LabelFrame(top_frame, text="Processing Parameters", padding="10", style='Modern.TFrame')
        self.param_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # Local tone mapping parameters (initially hidden)
        self.local_params_frame = ttk.Frame(self.param_frame, style='Modern.TFrame')
        
        # Gamma parameter with value display
        gamma_frame = ttk.Frame(self.local_params_frame, style='Modern.TFrame')
        gamma_frame.pack(fill=tk.X, pady=5)
        ttk.Label(gamma_frame, text="Gamma:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.gamma_value = tk.StringVar(value="3.0")
        ttk.Label(gamma_frame, textvariable=self.gamma_value, style='Modern.TLabel').pack(side=tk.RIGHT)
        self.gamma_scale = ttk.Scale(self.local_params_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL,
                                   command=lambda v: self.gamma_value.set(f"{float(v):.2f}"))
        self.gamma_scale.set(0.77)
        self.gamma_scale.pack(fill=tk.X)
        
        # Processing buttons
        button_frame = ttk.Frame(self.param_frame, style='Modern.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Process HDR", command=self.process_hdr_threaded, style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Result", command=self.save_result, style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Full Preview", command=self.show_full_preview, style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.param_frame, mode='indeterminate', length=200)
        self.progress.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(self.param_frame, text="", style='Modern.TLabel')
        self.status_label.pack(pady=5)
        
        # Preview frame
        preview_label_frame = ttk.LabelFrame(main_frame, text="HDR Preview", padding="10", style='Modern.TFrame')
        preview_label_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_label = ttk.Label(preview_label_frame)
        self.preview_label.pack(expand=True)
        
        # Initialize parameter visibility
        self.update_parameter_visibility()

    def update_parameter_visibility(self, event=None):
        # Show/hide local tone mapping parameters based on selected algorithm
        if self.tone_var.get() == 'Local Tone Mapping':
            self.local_params_frame.pack(fill=tk.X, pady=5)
        else:
            self.local_params_frame.pack_forget()

    def align_images(self, images):
        """
        Enhanced image alignment with better error handling and robustness
        """
        try:
            # Convert images to 8-bit format first
            imgs_8bit = []
            for img in images:
                if img.dtype != np.uint8:
                    normalized = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    normalized = img
                imgs_8bit.append(normalized)

            # Convert images to grayscale for alignment
            gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs_8bit]

            # Initialize alignment object
            alignMTB = cv2.createAlignMTB()
            alignMTB.setMaxBits(2)

            # Perform MTB alignment
            aligned = []
            alignMTB.process(imgs_8bit, aligned)

            # If MTB alignment fails, try ECC alignment
            if not self._check_alignment_quality(aligned):
                aligned = self._align_ecc(imgs_8bit, gray_images)

                # If ECC fails, try feature-based alignment
                if not self._check_alignment_quality(aligned):
                    aligned = self._align_features(imgs_8bit, gray_images)

                    # If all alignments fail, return original images with warning
                    if not self._check_alignment_quality(aligned):
                        self.status_label.configure(text="Warning: Using original images (alignment uncertain)")
                        return imgs_8bit

            return aligned

        except Exception as e:
            self.status_label.configure(text=f"Alignment failed: {str(e)}")
            return imgs_8bit  # Return original images if alignment fails

    def _align_mtb(self, images):
        """
        Median Threshold Bitmap alignment
        """
        try:
            alignMTB = cv2.createAlignMTB()
            alignMTB.setMaxBits(2)  # Increase robustness
            aligned = []
            imgs_8bit = [img.astype(np.uint8) if img.dtype != np.uint8 else img for img in images]
            alignMTB.process(imgs_8bit, aligned)
            return aligned
        except:
            return images

    def _align_ecc(self, images, gray_images):
        """
        Enhanced Correlation Coefficient alignment
        """
        try:
            aligned = [images[0]]  # Reference image
            warp_mode = cv2.MOTION_HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

            # Define region of interest for alignment (center portion of image)
            height, width = gray_images[0].shape
            roi_margin = 0.1  # 10% margin
            roi = (
                int(width * roi_margin), int(height * roi_margin),
                int(width * (1 - 2 * roi_margin)), int(height * (1 - 2 * roi_margin))
            )

            for i in range(1, len(images)):
                try:
                    # Calculate ECC transform
                    warp_matrix = np.eye(3, 3, dtype=np.float32)
                    _, warp_matrix = cv2.findTransformECC(
                        gray_images[0][roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]],
                        gray_images[i][roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]],
                        warp_matrix, warp_mode, criteria
                    )

                    # Apply transform
                    aligned_img = cv2.warpPerspective(
                        images[i], warp_matrix,
                        (width, height),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    )
                    aligned.append(aligned_img)
                except:
                    aligned.append(images[i])  # Keep original if alignment fails

            return aligned
        except:
            return images

    def _align_features(self, images, gray_images):
        """
        Feature-based alignment using SIFT/ORB and homography
        """
        try:
            # Try SIFT first, fall back to ORB if not available
            try:
                detector = cv2.SIFT_create()
            except:
                detector = cv2.ORB_create(nfeatures=2000)

            aligned = [images[0]]  # Reference image
            height, width = gray_images[0].shape

            for i in range(1, len(images)):
                try:
                    # Detect keypoints and compute descriptors
                    kp1, des1 = detector.detectAndCompute(gray_images[0], None)
                    kp2, des2 = detector.detectAndCompute(gray_images[i], None)

                    # Match features
                    if isinstance(detector, cv2.SIFT):
                        matcher = cv2.BFMatcher(cv2.NORM_L2)
                    else:
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

                    matches = matcher.knnMatch(des1, des2, k=2)

                    # Apply ratio test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                    if len(good_matches) > 10:
                        # Get matched keypoints
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Calculate homography
                        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                        if H is not None:
                            # Apply transform
                            aligned_img = cv2.warpPerspective(images[i], H, (width, height))
                            aligned.append(aligned_img)
                        else:
                            aligned.append(images[i])
                    else:
                        aligned.append(images[i])
                except:
                    aligned.append(images[i])

            return aligned
        except:
            return images

    def _check_alignment_quality(self, aligned_images):
        """
        Check if alignment was successful by comparing image similarities
        """
        try:
            if len(aligned_images) < 2:
                return False

            reference = cv2.cvtColor(aligned_images[0], cv2.COLOR_BGR2GRAY)
            min_similarity = float('inf')

            for img in aligned_images[1:]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Calculate normalized cross-correlation
                result = cv2.matchTemplate(reference, gray, cv2.TM_CCORR_NORMED)
                similarity = np.max(result)
                min_similarity = min(min_similarity, similarity)

            # Threshold for acceptable alignment
            return min_similarity > 0.8
        except:
            return False

    def align_and_merge(self, images, exposure_times):
        """
        Combines alignment and merging with quality checks
        """
        try:
            # Step 1: Initial alignment
            aligned_images = self.align_images(images)

            # Step 2: Check if we need to crop borders
            aligned_images = self._crop_borders(aligned_images)

            # Step 3: Merge aligned images
            merge_algo = self.merge_algorithms[self.merge_var.get()]
            hdr_result = merge_algo(aligned_images, exposure_times)

            return hdr_result
        except Exception as e:
            raise Exception(f"Alignment and merge failed: {str(e)}")

    def _crop_borders(self, images, crop_percent=0.02):
        """
        Crop image borders to remove alignment artifacts
        """
        if not images:
            return images

        height, width = images[0].shape[:2]
        crop_x = int(width * crop_percent)
        crop_y = int(height * crop_percent)

        cropped = []
        for img in images:
            cropped.append(img[crop_y:-crop_y, crop_x:-crop_x])

        return cropped

    def calibrate_exposures(self):
        """
        Calibrate exposure values using image content
        """
        if not self.input_images:
            messagebox.showerror("Error", "No images loaded")
            return
        
        try:
            # Create CalibrateDebevec object
            calibrate = cv2.createCalibrateDebevec()
            
            # Convert images to 8-bit if necessary
            imgs_8bit = [img.astype(np.uint8) if img.dtype != np.uint8 else img for img in self.input_images]
            
            # Calibrate response function
            response = calibrate.process(imgs_8bit, np.array(self.exposure_times, dtype=np.float32))
            
            # Estimate new exposure times
            middle_exposure = np.median(self.exposure_times)
            new_exposures = []
            
            for img in imgs_8bit:
                # Calculate average brightness
                avg_brightness = np.mean(img)
                # Estimate relative exposure
                relative_exposure = avg_brightness / 128.0  # Assuming middle gray is 128
                new_exposure = middle_exposure / relative_exposure
                new_exposures.append(new_exposure)
            
            # Update exposure times
            self.exposure_times = new_exposures
            
            # Update UI
            self.status_label.configure(text="Exposure times calibrated")
            self.display_exposure_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed: {str(e)}")


    def get_exposure_time(self, image_path):
        """
        Extract exposure time from image EXIF metadata.
        Returns exposure time in seconds as a float.
        """
        try:
            # Try using PIL first
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif is not None:
                    for tag_id in exif:
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == 'ExposureTime':
                            # ExposureTime is typically stored as a ratio (e.g., 1/60)
                            value = exif[tag_id]
                            if isinstance(value, tuple):
                                return float(value[0]) / float(value[1])
                            return float(value)
            
            # If PIL doesn't work, try piexif as backup
            try:
                exif_dict = piexif.load(image_path)
                if exif_dict['Exif']:
                    # ExposureTime tag ID is 33434
                    exposure = exif_dict['Exif'].get(33434)
                    if exposure:
                        return float(exposure[0]) / float(exposure[1])
            except:
                pass
            
            # If no EXIF data found, try to estimate from image properties
            img = cv2.imread(image_path)
            if img is not None:
                # Estimate exposure time based on image brightness
                average_brightness = np.mean(img)
                # Map brightness to a reasonable exposure time range (1/4000 to 1 second)
                estimated_exposure = (average_brightness / 255.0) * (1.0 - 1/4000) + 1/4000
                print(f"Warning: No EXIF data found for {image_path}. Estimated exposure time: {estimated_exposure:.4f}s")
                return estimated_exposure
            
            raise ValueError("Could not determine exposure time")
            
        except Exception as e:
            print(f"Error reading exposure time from {image_path}: {str(e)}")
            # Return a default exposure time as fallback
            return 1/60.0

    def validate_exposure_times(self):
        """
        Validate that we have valid exposure times for all loaded images.
        Returns True if all exposure times are valid, False otherwise.
        """
        if not self.exposure_times:
            messagebox.showerror("Error", "No exposure times available")
            return False
            
        if len(self.exposure_times) != len(self.input_images):
            messagebox.showerror("Error", "Missing exposure times for some images")
            return False
            
        # Check for invalid values
        for exp_time in self.exposure_times:
            if exp_time <= 0 or exp_time > 30:  # 30 seconds is a reasonable upper limit
                messagebox.showerror("Error", f"Invalid exposure time detected: {exp_time}")
                return False
                
        return True

    def display_exposure_info(self):
        """
        Display exposure information for loaded images.
        """
        if not self.exposure_times:
            return
            
        info_window = tk.Toplevel(self.window)
        info_window.title("Exposure Information")
        info_window.geometry("400x300")
        
        # Create a text widget to display information
        text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(info_window, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Display exposure information
        text_widget.insert(tk.END, "Exposure Times:\n\n")
        for i, exp_time in enumerate(self.exposure_times):
            text_widget.insert(tk.END, f"Image {i+1}: {exp_time:.4f} seconds\n")
            # Calculate EV difference from middle exposure
            if i > 0:
                ev_diff = np.log2(self.exposure_times[i] / self.exposure_times[i-1])
                text_widget.insert(tk.END, f"EV difference from previous: {ev_diff:.1f} stops\n")
            text_widget.insert(tk.END, "\n")
        
        text_widget.configure(state=tk.DISABLED)  # Make text read-only
    def merge_exposures(self, images, exposure_times):
        """
        Your original HDR merging algorithm
        """
        aligned_images = []
        for img in images:
            if img is not None:
                aligned_images.append(img.astype(np.float32) / 255.0)
        
        weights = np.array([1/et for et in exposure_times])
        weights = weights / np.sum(weights)
        
        hdr_image = np.zeros_like(aligned_images[0])
        for img, w in zip(aligned_images, weights):
            hdr_image += img * w
            
        return hdr_image
        
    def create_thumbnail(self, img, max_size=200):
        """Create a thumbnail for preview"""
        height, width = img.shape[:2]
        scale = max_size / max(height, width)
        
        if scale < 1:
            new_size = (int(width * scale), int(height * scale))
            img = cv2.resize(img, new_size)
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_img)
        return ImageTk.PhotoImage(image)

    def load_images(self):
        filepaths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not filepaths:
            return
            
        self.input_images = []
        self.exposure_times = []
        
        # Create temporary directory to save images
        temp_dir = "temp_hdr_images/"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, filepath in enumerate(filepaths[:3]):  # Limit to 3 images
            # Get exposure time first
            exposure_time = self.get_exposure_time(filepath)
            if exposure_time:
                self.exposure_times.append(exposure_time)
            else:
                messagebox.showerror("Error", f"Could not get exposure time for {filepath}")
                continue
            
            # Load and save image
            img = cv2.imread(filepath)
            if img is None:
                messagebox.showerror("Error", f"Failed to load {filepath}")
                continue
                
            cv2.imwrite(os.path.join(temp_dir, f"image_{i}.jpg"), img)
            self.input_images.append(img)
            
            # Update preview
            thumbnail = self.create_thumbnail(img)
            self.image_previews[i].configure(image=thumbnail)
            self.image_previews[i].image = thumbnail
        
        # Validate exposure times
        if self.validate_exposure_times():
            self.image_dir = temp_dir
            # Sort images by exposure time
            sorted_indices = np.argsort(self.exposure_times)
            self.input_images = [self.input_images[i] for i in sorted_indices]
            self.exposure_times = [self.exposure_times[i] for i in sorted_indices]
            
            # Add exposure info button
            if not hasattr(self, 'exposure_info_button'):
                self.exposure_info_button = ttk.Button(
                    self.preview_frame,
                    text="Show Exposure Info",
                    command=self.display_exposure_info,
                    style='Modern.TButton'
                )
                self.exposure_info_button.grid(row=1, column=1, pady=5)
                
            self.status_label.configure(text="Images loaded successfully")
        else:
            # Clean up if validation failed
            self.input_images = []
            self.exposure_times = []

    def process_hdr_threaded(self):
        self.progress.start()
        threading.Thread(target=self.process_hdr, daemon=True).start()

    def process_hdr(self):
        try:
            if not self.input_images or not self.exposure_times:
                raise Exception("No images loaded")
                
            # Convert input images to proper format
            processed_images = []
            for img in self.input_images:
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                processed_images.append(img)
                
            # Get selected algorithms
            merge_algo = self.merge_algorithms[self.merge_var.get()]
            tone_map_algo = self.tone_mapping_algorithms[self.tone_var.get()]
            
            # Update parameters from UI
            self.hdr_params.update({
                'lambda_': 50,
                'gamma': self.gamma_scale.get(),
                'saturation_local': 2
            })
            
            # Align and merge images
            aligned_images = self.align_images(processed_images)
            self.current_hdr = merge_algo(aligned_images, np.array(self.exposure_times))
            
            # Apply tone mapping
            self.current_ldr = tone_map_algo(self.current_hdr)
            
            # ew
            self.window.after(0, self.update_preview)
            self.status_label.configure(text="HDR processing completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.progress.stop()

    def merge_exposures(self, images, exposure_times):
        """
        Your original HDR merging algorithm
        """
        aligned_images = []
        for img in images:
            if img is not None:
                aligned_images.append(img.astype(np.float32) / 255.0)
        
        weights = np.array([1/et for et in exposure_times])
        weights = weights / np.sum(weights)
        
        hdr_image = np.zeros_like(aligned_images[0])
        for img, w in zip(aligned_images, weights):
            hdr_image += img * w
            
        return hdr_image

    def tone_map(self, hdr_image):
        """
        Implement robust tone mapping to convert HDR to LDR
        """
        gamma = self.hdr_params['gamma']
        
        # Handle NaN and Inf values
        hdr_image = np.nan_to_num(hdr_image, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure positive values
        hdr_image = np.maximum(hdr_image, 1e-8)
        
        # Normalize to [0, 1] with robust min/max
        min_val = np.percentile(hdr_image, 1)  # Use 1st percentile instead of min
        max_val = np.percentile(hdr_image, 99)  # Use 99th percentile instead of max
        
        # Avoid division by zero
        if max_val - min_val < 1e-8:
            max_val = min_val + 1e-8
            
        img_norm = np.clip((hdr_image - min_val) / (max_val - min_val), 0, 1)
        
        # Apply gamma correction
        return np.power(img_norm, gamma)
    
    def merge_debevec(self, images, exposure_times):
        """
        Fixed Debevec merge with proper image conversion and error handling
        """
        try:
            # Convert images to 8-bit format required by createMergeDebevec
            imgs_8bit = []
            for img in images:
                # Ensure proper normalization before conversion
                if img.dtype != np.uint8:
                    normalized = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    normalized = img
                imgs_8bit.append(normalized)

            # Create merger
            merger = cv2.createMergeDebevec()

            # Process with proper exposure times format
            exposure_times = np.array(exposure_times, dtype=np.float32)
            hdr = merger.process(imgs_8bit, times=exposure_times.copy())

            # Handle potential invalid values
            hdr = np.nan_to_num(hdr, nan=0.0, posinf=1.0, neginf=0.0)

            # Normalize if necessary
            if np.max(hdr) > 0:
                hdr = hdr / np.max(hdr)

            return hdr

        except Exception as e:
            raise Exception(f"Debevec merge failed: {str(e)}")
    
    def merge_mertens(self, images, exposure_times):
        """
        Fixed Mertens fusion with proper image conversion and error handling
        """
        try:
            # Convert images to 8-bit format required by createMergeMertens
            imgs_8bit = []
            for img in images:
                # Ensure proper normalization before conversion
                if img.dtype != np.uint8:
                    normalized = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    normalized = img
                imgs_8bit.append(normalized)
            
            # Create merger with optimized weights
            merger = cv2.createMergeMertens(
                contrast_weight=1.0,
                saturation_weight=1.0,
                exposure_weight=0.0
            )
            
            # Process images
            result = merger.process(imgs_8bit)
            
            # Handle potential invalid values
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure result is in 0-1 range
            result = np.clip(result, 0, 1)
            
            # Normalize if necessary
            if np.max(result) > 0:
                result = result / np.max(result)
            
            return result
            
        except Exception as e:
            self.status_label.configure(text=f"Mertens fusion failed: {str(e)}")
            raise e
        
    
    # Additional tone mapping algorithms
    def tone_map_reinhard(self, hdr_image):
        tonemap = cv2.createTonemapReinhard(
            gamma=self.gamma_scale.get(),
            intensity=0.5,
            light_adapt=0.8,
            color_adapt=0.0
        )
        return tonemap.process(hdr_image)

    def tone_map_drago(self, hdr_image):
        """
        Improved Drago tone mapping with proper value handling
        """
        try:
            # Normalize and handle invalid values
            hdr_image = np.nan_to_num(hdr_image, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure positive values
            hdr_image = np.maximum(hdr_image, 1e-8)
            
            # Normalize to a reasonable range
            hdr_image = hdr_image / np.max(hdr_image)
            
            tonemap = cv2.createTonemapDrago(
                gamma=self.gamma_scale.get(),
                saturation=0.7,
                bias=0.85
            )
            
            result = tonemap.process(hdr_image.astype(np.float32))
            return np.clip(result, 0, 1)
            
        except Exception as e:
            raise Exception(f"Drago tone mapping failed: {str(e)}")
    
    def tone_map_mantiuk(self, hdr_image):
        """
        Improved Mantiuk tone mapping with proper value handling
        """
        try:
            # Normalize and handle invalid values
            hdr_image = np.nan_to_num(hdr_image, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure positive values
            hdr_image = np.maximum(hdr_image, 1e-8)
            
            # Normalize to a reasonable range
            hdr_image = hdr_image / np.max(hdr_image)
            
            tonemap = cv2.createTonemapMantiuk(
                gamma=self.gamma_scale.get(),
                scale=0.7,
                saturation=0.85
            )
            
            result = tonemap.process(hdr_image.astype(np.float32))
            return np.clip(result, 0, 1)
            
        except Exception as e:
            raise Exception(f"Mantiuk tone mapping failed: {str(e)}")
            return tonemap.process(hdr_image)
    
    def show_full_preview(self):
        if self.current_ldr is None:
            messagebox.showerror("Error", "No HDR image processed yet")
            return

        preview_window = tk.Toplevel(self.window)
        preview_window.title("HDR Result")

        # Make window fullscreen
        preview_window.attributes('-fullscreen', True)

        # Convert the HDR image for display
        display_img = (self.current_ldr * 255).astype(np.uint8)
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        # Get screen dimensions
        screen_width = preview_window.winfo_screenwidth()
        screen_height = preview_window.winfo_screenheight()

        # Calculate scaling factor to fit image to screen while maintaining aspect ratio
        image_height, image_width = rgb_image.shape[:2]
        width_ratio = screen_width / image_width
        height_ratio = screen_height / image_height
        scale_factor = min(width_ratio, height_ratio)

        # Resize image to fit screen
        new_width = int(image_width * scale_factor)
        new_height = int(image_height * scale_factor)
        rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Convert to PhotoImage and display
        image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image)

        # Create label to display image
        label = ttk.Label(preview_window, image=photo)
        label.image = photo  # Keep reference

        # Center the image
        label.place(relx=0.5, rely=0.5, anchor='center')

        # Add escape key binding to close window
        preview_window.bind('<Escape>', lambda e: preview_window.destroy())

        # Add close button
        close_button = ttk.Button(
            preview_window,
            text="Close (Esc)",
            command=preview_window.destroy,
            style='Modern.TButton'
        )
        close_button.place(relx=0.95, rely=0.05, anchor='ne')


    def update_preview(self):
        if self.current_ldr is None:
            return
            
        try:
            # Ensure proper normalization before conversion
            if self.current_ldr.dtype == np.float32 or self.current_ldr.dtype == np.float64:
                # Handle invalid values
                preview_img = np.nan_to_num(self.current_ldr, nan=0.0, posinf=1.0, neginf=0.0)
                # Ensure proper range [0, 1]
                preview_img = np.clip(preview_img, 0, 1)
                # Convert to uint8
                preview_img = (preview_img * 255).astype(np.uint8)
            else:
                preview_img = self.current_ldr
                
            rgb_image = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            
            # Resize for preview if needed
            max_preview_size = 800
            height, width = rgb_image.shape[:2]
            if height > max_preview_size or width > max_preview_size:
                scale = max_preview_size / max(height, width)
                rgb_image = cv2.resize(rgb_image, (int(width * scale), int(height * scale)))
            
            image = Image.fromarray(rgb_image)
            self.current_preview = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.current_preview)
            self.preview_label.image = self.current_preview
            
        except Exception as e:
            messagebox.showerror("Error", f"Preview update failed: {str(e)}")

    def save_result(self):
        if self.current_ldr is None:
            messagebox.showerror("Error", "No image to save")
            return
        
        # Get current timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile=f"hdr_result_{timestamp}.jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("OpenEXR files", "*.exr"),
                ("TIFF files", "*.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            # Save based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.jpg', '.jpeg']:
                # Convert to uint8 for JPEG
                save_img = (self.current_ldr * 255).astype(np.uint8)
                cv2.imwrite(file_path, save_img)
                
            elif ext == '.png':
                # Save as 16-bit PNG
                save_img = (self.current_ldr * 65535).astype(np.uint16)
                cv2.imwrite(file_path, save_img)
                
            elif ext == '.exr':
                # Save HDR data in OpenEXR format
                try:
                    import OpenEXR
                    import Imath
                    # Implementation for OpenEXR saving
                    pass
                except ImportError:
                    messagebox.showerror("Error", "OpenEXR support not available")
                    return
                    
            elif ext == '.tiff':
                # Save as 32-bit TIFF
                save_img = self.current_ldr.astype(np.float32)
                cv2.imwrite(file_path, save_img)
            
            # Save exposure metadata if possible
            try:
                with open(file_path + ".txt", "w") as f:
                    f.write("HDR Processing Parameters:\n")
                    f.write(f"Exposure Times: {self.exposure_times}\n")
                    f.write(f"Lambda: {self.hdr_params['lambda_']}\n")
                    f.write(f"Gamma: {self.hdr_params['gamma']}\n")
                    f.write(f"Local Saturation: {self.hdr_params['saturation_local']}\n")
            except:
                pass
                
            messagebox.showinfo("Success", "Image saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'image_dir') and os.path.exists(self.image_dir):
            try:
                for file in os.listdir(self.image_dir):
                    os.remove(os.path.join(self.image_dir, file))
                os.rmdir(self.image_dir)
            except Exception as e:
                print(f"Cleanup error: {str(e)}")

    def run(self):
        try:
            self.window.mainloop()
        finally:
            self.cleanup()

def main():
    app = ModernHDRProcessor()
    app.run()

if __name__ == "__main__":
    main()