# Basic Implementation of Image Signal Processor (ISP)
##INTRODUCTION
This project is a foundational implementation of an Image Signal Processor (ISP), designed to process raw image data and apply essential image processing functions. 
The program demonstrates basic ISP operations such as Debayering, noise reduction, sharpening, and other signal adjustments.
##FEATURES
Emmetra’s ISP implementation includes the following five key algorithms for a streamlined, effective image processing pipeline:
  ###1. Demosaic Layer with Edge-Based Interpolation
      Reconstructs full-color images from sensor data using edge-based interpolation, enhancing detail by minimizing color artifacts and preserving natural edges.
  ###2. White Balance
      Adjusts color temperature to ensure accurate color reproduction, correcting for different lighting conditions so whites appear neutral and colors are balanced.
  ###3. Denoise Using a Gaussian Filter
      Reduces noise in low-light or high-ISO images by smoothing unwanted artifacts while preserving key details, resulting in a cleaner overall image.
  ###4. Gamma Correction
      Applies a gamma curve to adjust brightness and contrast, ensuring a natural tonal range and accurate midtones for a visually balanced output.
  ###5. Sharpening Filter
      Enhances image clarity by accentuating edges and fine details, creating a sharper, more defined look without introducing additional noise.

###USER INTERFACE
This ISP implementation includes a user-friendly interface designed to simplify interaction. It allows users to visualize each algorithm’s effect on the image.

