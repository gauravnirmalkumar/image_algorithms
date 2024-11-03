# HDR Image Processing Application

This repository provides an application for HDR (High Dynamic Range) image processing with multiple techniques. The application is written in Python and utilizes OpenCV for Mertens merge, tone mapping, and exposure detection.

## Table of contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [HDR Techniques](#hdr-merging-algorithms)
  - [Simple Weighted Merge](#simple-weighted-merge)
  - [Merten](#merten)
  - [Debevec](#debevec)
- [Tone Mapping Techniques](#tone-mapping-techniques)
  - [Local Tone Mapping](#local-tone-mapping)
  - [Global Tone Mapping](#global-tone-mapping)
    - [Reinhard Tone Mapping](#reinhard-tone-mapping)
    - [Drago Tone Mapping](#drago-tone-mapping)
    - [Mantiuk  Tone Mapping](#mantiuk-tone-mapping)
- [Example Outputs](#example-outputs)

---

## Introduction

HDR imaging allows capturing images with a higher range of luminosity levels than standard digital imaging. This application implements several HDR techniques to create realistic images by merging multiple exposures and applying different tone-mapping algorithms.

## Features

- **Multi-Exposure Support**: Load multiple images with different exposures (underexposed, normal, overexposed) to create a single HDR image.
- **Automatic Exposure Detection**: Identifies and categorizes input images based on their exposure levels.
- **HDR Merging Algorithms**: Supports multiple merging algorithms, including Merten’s and Debevec's methods, for blending exposures.
- **Tone Mapping Techniques**: Offers different tone mapping methods, such as Reinhard, Drago, Mantiuk, and Local Tone Mapping.
- **Real-Time Preview**: Enables users to see the effect of tone mapping adjustments on the HDR output.
- **Image Export**: Allows saving the processed HDR images in common formats.

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd src
pip install -r requirements.txt
```

## Usage
- Load Images: Upload up to three images with different exposures (e.g., underexposed, properly exposed, overexposed) for merging.
- Select HDR Merging Method: Choose from available merging algorithms, including Merten's merge, Debevec and simple weighted.
- Apply Tone Mapping: Customize tone mapping settings to enhance the final image’s detail, brightness, and contrast.
- Save Output: Export your HDR image to your desired format.

## Running the Application
```bash
python app.py
```
The GUI will open, where you can load your images, apply tone mapping adjustments, and preview the HDR output.

## HDR Merging Algorithms

### Simple Weighted Merge

#### Description
The Simple Weighted Merge algorithm combines images by assigning weights to pixel values based on brightness. Images with different exposures contribute to the final HDR based on how well-exposed each pixel is, resulting in a balanced image that preserves details across the range of exposures.

#### How It Works
- **Exposure-Based Weighting**: Each image is weighted by the brightness of its pixels. Bright, well-exposed areas have more influence in the final image, while underexposed or overexposed regions are given lower weights.
- **Merging**: The algorithm combines all the weighted images to produce an HDR output where each pixel represents a blend of the input images, emphasizing well-exposed areas and reducing the effect of under- or over-exposed regions.

#### When to Use
- **Efficiency**: This algorithm is computationally inexpensive and easy to implement, making it suitable for quick HDR generation.
- **Balanced Scenes**: It works well for scenes with moderate lighting differences, such as indoor spaces with natural light but without extreme brightness contrasts.

#### Limitations
- **High-Contrast Scenarios**: In scenes with significant contrast, like direct sunlight and deep shadows, the algorithm may struggle to capture full detail without introducing artifacts.

---

### Merten

#### Description
Merten's Merge enhances HDR merging by assigning weights based on contrast and saturation rather than just brightness. This approach enables the algorithm to preserve more fine-grained details, making it particularly useful for scenes with complex textures and varied lighting.

#### How It Works
- **Contrast and Saturation Weighting**: Pixels with higher contrast and richer colors are given higher weights. This prioritizes well-defined details and vibrant areas over flat or monochromatic regions, enhancing the final image’s depth and realism.
- **Blending Process**: By combining images according to these weights, Merten’s Merge produces an HDR output that emphasizes detail-rich areas, creating a visually compelling result without over-brightening or flattening the image.

#### When to Use
- **Detail Preservation**: This algorithm is ideal for scenes with fine textures or where visual detail is important, such as landscapes with varied foliage or architectural photos.
- **Complex Lighting**: Works well in environments with dynamic lighting, as it maintains depth without overemphasizing brightness.

#### Limitations
- **Increased Computational Cost**: Because this algorithm requires additional processing to calculate contrast and saturation, it is slower than simpler methods.
- **Artifacts in Low-Contrast Scenes**: If the scene lacks sufficient contrast, the algorithm may not perform as effectively, potentially resulting in a flat or washed-out image.

---

### Debevec

#### Description
The Debevec algorithm is a classic approach in HDR imaging that uses a camera response function to account for the non-linear relationship between pixel brightness and real-world light intensities. By adjusting for exposure and calibrating each image, this algorithm produces a true HDR image that retains detail across a wide range of luminance values.

#### How It Works
- **Response Function Calculation**: The algorithm uses a set of images taken at different exposures to calculate a response function for each pixel, enabling a more accurate representation of the scene’s true brightness.
- **Weighting Based on Pixel Brightness**: While combining the images, pixels that are too bright or too dark are given lower weights to avoid issues like blown-out highlights or crushed shadows.
- **HDR Creation**: After calculating the camera response function and adjusting for exposure levels, the algorithm merges the images into a final HDR output that accurately represents the full dynamic range of the scene.

#### When to Use
- **High Contrast Scenes**: Particularly useful for scenes with extreme lighting variations, such as sunsets, or photos taken in challenging lighting conditions where detail is crucial.
- **True HDR Generation**: Ideal when the goal is to create a realistic HDR output that closely represents the original lighting.

#### Limitations
- **Calibration Requirement**: This algorithm needs a carefully calibrated response function, which may require more time and fine-tuning.
- **High Computational Load**: The response function calculation is resource-intensive, making it slower compared to other methods. It may be less suitable for quick HDR processing or real-time applications.

## Tone Mapping Techniques

### Overview
After creating an HDR image, the brightness levels often exceed the range of a standard display. **Tone Mapping** is the process of converting an HDR image’s high dynamic range into a lower dynamic range that can be viewed on typical screens while preserving visual details. This section explores two main types of tone mapping techniques: **Local Tone Mapping** and **Global Tone Mapping**.

---

### Local Tone Mapping

#### Description
Local Tone Mapping adjusts the brightness and contrast of small regions within the image independently. This technique enhances local contrast and brings out details in specific parts of the image, making it effective for creating visually rich images where textures and details are important.

#### How It Works
- **Adaptive Adjustments**: Local Tone Mapping evaluates brightness and contrast on a pixel-by-pixel basis, taking into account the surrounding pixel values. Each pixel’s tone is adjusted relative to its neighboring pixels to maintain local contrast.
- **Enhanced Textures and Details**: By applying adjustments only to specific areas, this technique can highlight textures and fine details without affecting the entire image.
  
#### When to Use
- **High-Detail Scenes**: Ideal for images where textures and minute details are critical, such as close-up shots of objects or natural landscapes.
- **Complex Lighting Conditions**: Works well in scenes with varied lighting, such as forests or architectural interiors, as it brings out subtle contrasts.

#### Limitations
- **Higher Computational Cost**: Local adjustments are resource-intensive and may take longer to process.
- **Halo Artifacts**: Excessive adjustments may introduce halo artifacts around objects, where contrast transitions become unnatural or overly emphasized.

---

### Global Tone Mapping

#### Description
Global Tone Mapping applies a uniform adjustment across the entire image, transforming the HDR image into a viewable format without considering local variations in brightness or contrast. While simpler, it provides a more consistent look, making it suitable for balanced scenes where local detail adjustments are less crucial.

#### Techniques in Global Tone Mapping

##### Reinhard Tone Mapping

- **Description**: Reinhard Tone Mapping is a widely-used technique that aims to produce realistic results by approximating the brightness perception of the human eye.
- **How It Works**: This technique applies a logarithmic compression to brightness levels, effectively balancing the brightest and darkest regions of the image. It adjusts the overall intensity to prevent overly bright or dark spots.
- **Advantages**: Produces natural-looking images that retain overall contrast and brightness, making it well-suited for realistic rendering.
- **Limitations**: May not highlight fine details as effectively as other techniques since it focuses on a balanced approach rather than enhancing local contrast.

##### Drago Tone Mapping

- **Description**: Drago Tone Mapping is designed to enhance the visibility of details in dark regions without compromising the brighter parts of the image, making it suitable for high-contrast scenes.
- **How It Works**: This technique compresses the dynamic range with an emphasis on preserving the luminance of darker areas, using a logarithmic function tailored to human visual perception.
- **Advantages**: Retains detail in shadowed areas while maintaining brightness in highlights, making it ideal for HDR scenes with significant contrast.
- **Limitations**: Can lead to desaturation in very bright areas, which may make certain colors appear washed out.

##### Mantiuk Tone Mapping

- **Description**: Mantiuk Tone Mapping is an advanced technique that enhances both local contrast and visual detail by applying a model of human visual sensitivity to contrast changes.
- **How It Works**: This method uses complex algorithms to adjust both global and local contrast, aiming to maximize perceived detail based on the eye's sensitivity to different contrasts and luminance levels.
- **Advantages**: Produces highly detailed images with enhanced textures and sharp contrasts, making it well-suited for artistic or highly detailed HDR applications.
- **Limitations**: May introduce artifacts or unnatural effects if over-applied, and requires significant computational power.

## Example Outputs

In this section, we provide an example of HDR merging and tone mapping using three differently exposed images. The final results are shown after applying Merten's Merge for HDR creation and Local Tone Mapping to enhance details.

### Input Images
The three input images below were captured at different exposure levels:
- **Image 1**: Underexposed to capture bright highlights.
- **Image 2**: Balanced exposure for midtones.
- **Image 3**: Overexposed to capture shadow details.


#### Input Images
Here are the three input images captured at different exposures:

##### Underexposed
![image1](https://github.com/user-attachments/assets/b25a715d-a8ac-4376-a0c1-c1c132cb7314)
##### Midtones 
![image2](https://github.com/user-attachments/assets/0e803a12-c3c7-47be-bd86-32e2eb8a032e)
##### Overexposed
![image3](https://github.com/user-attachments/assets/c0b9bce6-2b14-4fa7-b817-0a5f28cbea9f)

## Final HDR Output Using Merten’s Merge and Local Tone Mapping

![hdr_result_20241103_220757](https://github.com/user-attachments/assets/fa5a2065-3e34-4b3f-bcbb-b39f22f33728)

#### Key Highlights
- **Dynamic Range**: Merten’s Merge preserves detail across various exposure levels, ensuring that highlights and shadows are well-balanced.
- **Enhanced Texture and Local Contrast**: Local Tone Mapping adds depth by adjusting brightness and contrast on a localized level, bringing out finer details in regions such as shadows and textured surfaces.
- **Natural Colors**: The image maintains a natural, visually appealing color balance with saturated tones that enhance the realism.




