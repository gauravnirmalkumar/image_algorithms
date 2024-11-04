# Basic Implementation of Image Signal Processor (ISP)

## Introduction
This project is a foundational implementation of an Image Signal Processor (ISP), designed to process raw image data and apply essential image processing functions. The program demonstrates basic ISP operations such as Debayering, noise reduction, sharpening, and other signal adjustments.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
  - [Demosaicing](#demosaicing)
  - [White Balance](#white-balance)
  - [Denoising](#denoising)
  - [Gamma Correction](#gamma-correction)
  - [Sharpening Filters](#sharpening-filters)
- [Results](#results)

## Features
Emmetra's ISP implementation includes the following five key algorithms for a streamlined, effective image processing pipeline:

| Algorithm | Description |
|-----------|-------------|
| Demosaic Layer | Edge-based interpolation for full-color image reconstruction |
| White Balance | Color temperature adjustment for accurate reproduction |
| Denoise | Gaussian filter for noise reduction in low-light conditions |
| Gamma Correction | Brightness and contrast adjustment for natural tonal range |
| Sharpening | Edge enhancement for improved clarity |

### User Interface
| Feature | Description |
|---------|-------------|
| Image Upload | Simple click-to-upload interface for raw files |
| Adjustment Sliders | Interactive controls for denoising, gamma, and sharpening |
| Preview | Real-time visualization of processing effects |
| Auto-save | Processed images automatically saved to current directory |

## Installation

### Requirements
```bash
pip install -r Requirements.txt
```
### Usage
- Load Raw Image: Click on upload image and choose the intended raw file to be processed
- Use the denoising, gamma correction and sharpening sliders as needed
- Click on process image to view the processed image, while the image will also be saved into the current directory

### Running the Application
```bash
python main.py
```

## Technical Details

### Demosaicing

| Technique | Description | Use Cases | Limitations |
|-----------|-------------|-----------|-------------|
| Nearest Neighbor | Simple method using closest pixel duplication | Low-cost applications | Color fringing, low quality |
| Bilinear | Averages surrounding pixel colors | Basic processing | Blurry images, edge artifacts |
| Edge-Aware | Adjusts interpolation based on detected edges | High-quality imaging | Computationally intensive |
| Frequency-Based | Analyzes spatial frequencies for interpolation | Professional devices | High processing requirements |
| Machine Learning | Uses trained models for color prediction | Smartphone cameras | Requires significant resources |

### White Balance

| Method | Description | Best For | Limitations |
|--------|-------------|----------|-------------|
| Auto WB | Automatic scene detection | Everyday shooting | Struggles with mixed lighting |
| Preset WB | Predefined lighting settings | Controlled environments | Limited adaptability |
| Custom WB | Manual white reference setting | Studio photography | Requires reference card |
| Manual Adjustment | Fine-tuning of RGB channels | Creative control | Time-consuming |

### Denoising

| Technique | Description | Applications | Limitations |
|-----------|-------------|--------------|-------------|
| Spatial Filtering | Local area averaging | General noise reduction | May blur details |
| Frequency Domain | Transform-based filtering | Periodic noise removal | Computationally heavy |
| Wavelet | Multi-scale decomposition | Professional imaging | Complex implementation |
| Machine Learning | Neural network-based | Modern cameras | Resource intensive |
| Temporal | Multi-frame processing | Video applications | Motion sensitivity |

### Gamma Correction

| Type | Description | Use Cases | Limitations |
|------|-------------|-----------|-------------|
| Linear | Simple brightness mapping | Basic workflows | Unnatural results |
| Piecewise | Segmented adjustment | HDR imaging | Implementation complexity |
| Parametric | Custom curve definition | Professional editing | Requires expertise |
| Inverse | Display preparation | Print preparation | Careful management needed |

### Sharpening Filters

| Method | Description | Best For | Limitations |
|--------|-------------|----------|-------------|
| Unsharp Mask | Edge enhancement via blur subtraction | General use | Potential halos |
| High-Pass | Frequency-based edge retention | Professional editing | Time-consuming |
| Laplace | Second derivative edge detection | Strong enhancement | Noise sensitive |
| Deconvolution | Mathematical blur reversal | Scientific imaging | Complex setup |
| ML-Based | Adaptive enhancement | Modern software | Resource heavy |

## Results

### Image 1: Demosaicing and Gamma Correction
| Image Type | Image |
|------------|-------|
| With Demosaicing and Gamma Correction | <img width="956" alt="Demosaic_gamma1" src="https://github.com/user-attachments/assets/97499940-d15e-4e7e-9a5a-0b2085e5cbfa"> |

### Image 2: Adding White Balance
| Image Type | Image |
|------------|-------|
| With Demosaicing, White Balance and Gamma Correction | <img width="955" alt="Demosaic_WhiteBalance_gamma" src="https://github.com/user-attachments/assets/065edabc-53ba-4c5b-9f7b-c6645726af5b"> |

### Image 3: Adding Denoising
| Image Type | Image |
|------------|-------|
| With Demosaicing, White Balance, Denoising and Gamma | ![Demosaic_WhiteBalance_denoise_gamma](https://github.com/user-attachments/assets/9aa14d17-b13f-4a2a-aab9-0755f813c8e0) |

### Image 4: Complete Pipeline
| Image Type | Image |
|------------|-------|
| With Demosaicing, White Balance, Denoising, Gamma Correction, and Sharpening | ![Demosaic_WhiteBalance_denoise_gamma_sharp](https://github.com/user-attachments/assets/ee3e653a-17d8-4501-b107-4dd5e3c46253) |

### Final Result
| Image Type | Image |
|------------|-------|
| Final Image After Basic ISP Implementation | <img width="958" alt="Final_image_after_ISP" src="https://github.com/user-attachments/assets/463d2ec4-9ba0-4350-9015-4fad645fd29e"> |
