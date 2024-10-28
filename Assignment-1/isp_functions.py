# isp_functions.py

import numpy as np
import cv2

def load_raw_image(filepath, width=1920, height=1280):
    """Load and reshape a 12-bit Bayer RAW image into a normalized 8-bit image."""
    try:
        raw_data = np.fromfile(filepath, dtype=np.uint16).reshape((height, width))
        raw_data = np.clip(raw_data, 0, 4095)  # Ensure values stay within 12-bit range
        normalized = (raw_data / 4095.0 * 255).astype(np.uint8)
        return normalized
    except Exception as e:
        print(f"Error loading RAW image: {e}")
        return None

def demosaic_edge_based(image):
    """Convert GRBG Bayer pattern to RGB using OpenCV."""
    try:
        return cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)
    except Exception as e:
        print(f"Error in demosaicing: {e}")
        return image

def adjust_saturation(image, saturation_factor=1.5):
    """
    Adjust color saturation of the image.
    Args:
        image: RGB image
        saturation_factor: Value > 1 increases saturation, < 1 decreases it
    """
    try:
        if saturation_factor == 1.0:
            return image
            
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_factor
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
    except Exception as e:
        print(f"Error adjusting saturation: {e}")
        return image

def adjust_contrast(image, contrast_factor=1.2):
    """
    Adjust image contrast using histogram stretching.
    Args:
        image: Input image
        contrast_factor: Value > 1 increases contrast, < 1 decreases it
    """
    try:
        mean = np.mean(image)
        adjusted = (image - mean) * contrast_factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error adjusting contrast: {e}")
        return image

def gray_world_white_balance(image, enabled=True, strength=1.0):
    """
    Enhanced white balance with adjustable strength.
    Args:
        image: Input RGB image
        enabled: Whether to apply white balance
        strength: Strength of the white balance effect
    """
    try:
        if not enabled:
            return image
            
        avg_rgb = np.mean(image, axis=(0, 1))
        scale = (avg_rgb.mean() / avg_rgb) ** strength
        balanced = np.clip(image * scale, 0, 255).astype(np.uint8)
        return balanced
    except Exception as e:
        print(f"Error in white balance: {e}")
        return image

def gaussian_denoise(image, enabled=True, kernel_size=5):
    """
    Apply Gaussian blur to reduce noise.
    Args:
        image: Input image
        enabled: Whether to apply denoising
        kernel_size: Size of the Gaussian kernel
    """
    try:
        if not enabled:
            return image
            
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    except Exception as e:
        print(f"Error in denoising: {e}")
        return image

def gamma_correction(image, enabled=True, gamma=2.2):
    """
    Apply sRGB gamma correction.
    Args:
        image: Input image
        enabled: Whether to apply gamma correction
        gamma: Gamma value
    """
    try:
        if not enabled:
            return image
            
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 
                               for i in range(256)]).astype("uint8")
        return cv2.LUT(image, lookup_table)
    except Exception as e:
        print(f"Error in gamma correction: {e}")
        return image

def unsharp_mask(image, enabled=True, strength=1.0, blur_size=5):
    """
    Apply unsharp masking to sharpen the image.
    Args:
        image: Input image
        enabled: Whether to apply sharpening
        strength: Strength of the sharpening effect
        blur_size: Size of the Gaussian blur kernel
    """
    try:
        if not enabled:
            return image
            
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    except Exception as e:
        print(f"Error in sharpening: {e}")
        return image

def color_temperature_adjust(image, temperature=0):
    """
    Adjust color temperature (warm/cool).
    Args:
        image: Input image
        temperature: Range -100 (cooler) to 100 (warmer)
    """
    try:
        if temperature == 0:
            return image
            
        img_float = image.astype(float)
        
        # Adjust RGB channels based on temperature
        if temperature > 0:  # Warmer
            factor = temperature / 100
            img_float[:,:,2] *= (1 + factor * 0.2)  # More red
            img_float[:,:,0] *= (1 - factor * 0.2)  # Less blue
        else:  # Cooler
            factor = -temperature / 100
            img_float[:,:,2] *= (1 - factor * 0.2)  # Less red
            img_float[:,:,0] *= (1 + factor * 0.2)  # More blue
            
        return np.clip(img_float, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error adjusting color temperature: {e}")
        return image

def process_pipeline(image, params):
    """
    Process an image through the entire ISP pipeline with given parameters.
    Args:
        image: Input RAW image
        params: Dictionary containing processing parameters
    """
    try:
        # Demosaic
        rgb_image = demosaic_edge_based(image)
        
        # White balance with strength control
        wb_image = gray_world_white_balance(rgb_image, 
                                          params['wb_enabled'],
                                          params['wb_strength'])
        
        # Color temperature adjustment
        temp_image = color_temperature_adjust(wb_image, 
                                            params['temperature'])
        
        # Contrast adjustment
        contrast_image = adjust_contrast(temp_image, 
                                       params['contrast'])
        
        # Color saturation
        saturated_image = adjust_saturation(contrast_image, 
                                          params['saturation'])
        
        # Denoise if enabled
        denoised_image = gaussian_denoise(saturated_image,
                                        params['denoise_enabled'],
                                        params['kernel_size'])
        
        # Gamma correction
        gamma_image = gamma_correction(denoised_image,
                                     params['gamma_enabled'],
                                     params['gamma_value'])
        
        # Final sharpening
        final_image = unsharp_mask(gamma_image,
                                  params['sharp_enabled'],
                                  params['sharp_strength'])
        
        return final_image
    except Exception as e:
        print(f"Error in processing pipeline: {e}")
        return image