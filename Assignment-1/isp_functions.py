import numpy as np
import cv2

def load_raw_image(filepath, width=1920, height=1280):
    """Load and reshape a 12-bit Bayer RAW image into a normalized 8-bit image."""
    raw_data = np.fromfile(filepath, dtype=np.uint16).reshape((height, width))
    
    # Extract the 12-bit pixel values and normalize them to 8-bit (0-255)
    raw_data = np.clip(raw_data, 0, 4095)  # Ensure values stay within 12-bit range
    normalized = (raw_data / 4095.0 * 255).astype(np.uint8)
    return normalized

def demosaic_edge_based(image):
    """Convert GRBG Bayer pattern to RGB using OpenCV."""
    return cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)

def gray_world_white_balance(image):
    """Apply the Gray World white balance algorithm to remove color cast."""
    avg_rgb = np.mean(image, axis=(0, 1))
    scale = avg_rgb.mean() / avg_rgb
    return np.clip(image * scale, 0, 255).astype(np.uint8)

def gaussian_denoise(image, kernel_size=5):
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def gamma_correction(image, gamma=2.2):
    """Apply sRGB gamma correction."""
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, lookup_table)

def unsharp_mask(image, strength=1.5, blur_size=5):
    """Apply unsharp masking to sharpen the image."""
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
