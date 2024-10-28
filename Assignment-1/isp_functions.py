# isp_functions.py

import numpy as np
import cv2

def load_raw_image(filepath, width=1920, height=1280):
    """
    Load and reshape a 12-bit Bayer RAW image into a normalized 8-bit image.
    
    Args:
        filepath (str): Path to the RAW image file
        width (int): Width of the image in pixels
        height (int): Height of the image in pixels
        
    Returns:
        numpy.ndarray: Normalized 8-bit image array
    """
    raw_data = np.fromfile(filepath, dtype=np.uint16).reshape((height, width))
    raw_data = np.clip(raw_data, 0, 4095)  # Ensure values stay within 12-bit range
    normalized = (raw_data / 4095.0 * 255).astype(np.uint8)
    return normalized

def demosaic_edge_based(image):
    """
    Convert GRBG Bayer pattern to RGB using OpenCV.
    
    Args:
        image (numpy.ndarray): Input Bayer pattern image
        
    Returns:
        numpy.ndarray: Demosaiced RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)

def gray_world_white_balance(image, enabled=True):
    """
    Apply the Gray World white balance algorithm to remove color cast.
    
    Args:
        image (numpy.ndarray): Input RGB image
        enabled (bool): Whether to apply the white balance
        
    Returns:
        numpy.ndarray: White balanced image if enabled, otherwise original image
    """
    if not enabled:
        return image
        
    avg_rgb = np.mean(image, axis=(0, 1))
    scale = avg_rgb.mean() / avg_rgb
    return np.clip(image * scale, 0, 255).astype(np.uint8)

def gaussian_denoise(image, enabled=True, kernel_size=5):
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        image (numpy.ndarray): Input image
        enabled (bool): Whether to apply denoising
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        
    Returns:
        numpy.ndarray: Denoised image if enabled, otherwise original image
    """
    if not enabled:
        return image
        
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def gamma_correction(image, enabled=True, gamma=2.2):
    """
    Apply sRGB gamma correction.
    
    Args:
        image (numpy.ndarray): Input image
        enabled (bool): Whether to apply gamma correction
        gamma (float): Gamma value
        
    Returns:
        numpy.ndarray: Gamma corrected image if enabled, otherwise original image
    """
    if not enabled:
        return image
        
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 
                           for i in range(256)]).astype("uint8")
    return cv2.LUT(image, lookup_table)

def unsharp_mask(image, enabled=True, strength=1.0, blur_size=5):
    """
    Apply unsharp masking to sharpen the image.
    
    Args:
        image (numpy.ndarray): Input image
        enabled (bool): Whether to apply sharpening
        strength (float): Strength of the sharpening effect
        blur_size (int): Size of the Gaussian blur kernel
        
    Returns:
        numpy.ndarray: Sharpened image if enabled, otherwise original image
    """
    if not enabled:
        return image
        
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

def process_pipeline(image, params):
    """
    Process an image through the entire ISP pipeline with given parameters.
    
    Args:
        image (numpy.ndarray): Input RAW image
        params (dict): Dictionary containing processing parameters
        
    Returns:
        numpy.ndarray: Processed image
    """
    # Demosaic
    rgb_image = demosaic_edge_based(image)
    
    # Apply each processing step with parameters
    wb_image = gray_world_white_balance(rgb_image, params['wb_enabled'])
    denoise_image = gaussian_denoise(wb_image, params['denoise_enabled'], 
                                   params['kernel_size'])
    gamma_image = gamma_correction(denoise_image, params['gamma_enabled'], 
                                 params['gamma_value'])
    final_image = unsharp_mask(gamma_image, params['sharp_enabled'], 
                              params['sharp_strength'])
    
    return final_image