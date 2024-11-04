# Basic Implementation of Image Signal Processor (ISP)
## Introduction
This project is a foundational implementation of an Image Signal Processor (ISP), designed to process raw image data and apply essential image processing functions. 
The program demonstrates basic ISP operations such as Debayering, noise reduction, sharpening, and other signal adjustments.
## Features
Emmetra’s ISP implementation includes the following five key algorithms for a streamlined, effective image processing pipeline:
  #### 1. Demosaic Layer with Edge-Based Interpolation :
  Reconstructs full-color images from sensor data using edge-based interpolation, enhancing detail by minimizing color artifacts and preserving natural edges. 
  #### 2. White Balance
  Adjusts color temperature to ensure accurate color reproduction, correcting for different lighting conditions so whites appear neutral and colors are balanced.
  #### 3. Denoise Using a Gaussian Filter
  Reduces noise in low-light or high-ISO images by smoothing unwanted artifacts while preserving key details, resulting in a cleaner overall image.
  #### 4. Gamma Correction
  Applies a gamma curve to adjust brightness and contrast, ensuring a natural tonal range and accurate midtones for a visually balanced output.
  #### 5. Sharpening Filter
  Enhances image clarity by accentuating edges and fine details, creating a sharper, more defined look without introducing additional noise.

#### User Interface
This ISP implementation includes a user-friendly interface designed to simplify interaction. It allows users to visualize each algorithm’s effect on the image.
## Installation
#### Installing all required dependencies:
      pip install -r Requirements.txt
##Usage
- Load Raw Image: Click on upload image and choose the intended raw file to be processed
- Use the denoising, gamma correction and sharpening sliders as needed
- Click on process image to view the processed image, while the image will also be saved into the current directory

## Running The Application
      python main.py
This will opne the GUI, following the above mentioned steps to process your raw images 

# Demosaicing

Demosaicing is a digital image processing technique used to reconstruct full-color images from raw data captured by image sensors, which typically employ a Bayer filter array. Each pixel in this array captures only one color (red, green, or blue), meaning the other two color values must be interpolated. Demosaicing algorithms are essential for filling in these missing color values and producing a complete color image.

## Demosaicing Techniques

### 1. **Nearest Neighbor Interpolation**
   - **Description**: This is the simplest method, where each missing color value is filled by duplicating the nearest neighbor's color.
   - **Use Cases**: Primarily used in low-cost or low-power applications where processing resources are limited.
   - **Limitations**: It often leads to significant color artifacts, such as “color fringing” and low image quality, especially around edges.

### 2. **Bilinear Interpolation**
   - **Description**: In this technique, the missing color values are calculated by averaging the colors of the surrounding pixels.
   - **Use Cases**: Commonly used in basic image processing pipelines due to its simplicity and efficiency.
   - **Limitations**: Bilinear interpolation can produce blurry images, with noticeable artifacts around edges due to its inability to distinguish edges from flat areas.

### 3. **Edge-Aware Demosaicing**
   - **Description**: Edge-aware algorithms detect edges in the image and adjust the interpolation process to avoid blending colors across these edges. This helps maintain image sharpness and reduces color artifacts.
   - **Use Cases**: Suitable for high-quality imaging where edge preservation is critical, such as in photography or videography.
   - **Limitations**: Computationally more intensive, making it less ideal for real-time or low-power applications.

### 4. **Frequency-Based Demosaicing**
   - **Description**: Frequency-based algorithms analyze the spatial frequencies in the image and apply different interpolation methods based on these frequencies, effectively separating high-frequency (edge) and low-frequency (flat) regions.
   - **Use Cases**: Common in professional imaging devices where high accuracy is essential.
   - **Limitations**: These algorithms require significant processing power and can introduce artifacts if frequencies are misinterpreted.

### 5. **Machine Learning-Based Demosaicing**
   - **Description**: This technique uses trained models to predict missing color values, allowing for highly accurate color reconstruction, even in complex textures and edges.
   - **Use Cases**: Increasingly used in smartphone cameras and other devices where high-quality image processing is required.
   - **Limitations**: Requires a large amount of training data and processing power, making it challenging for devices with limited resources.

## Use Cases

- **Consumer Cameras**: Demosaicing is essential in consumer cameras (smartphones, DSLRs) to deliver high-quality images from compact sensors.
- **Medical Imaging**: Edge-aware and machine learning-based techniques are often used to ensure clarity and detail in diagnostic imaging.
- **Surveillance and Security**: Lower-cost methods like bilinear interpolation may be used for cost-effectiveness, though edge-aware methods are used when detail is necessary.

## Limitations of Demosaicing

While demosaicing is vital for color reconstruction, it has inherent limitations:

- **Artifacts**: Demosaicing can introduce color artifacts, such as moiré patterns, color fringing, and false colors around edges.
- **Processing Requirements**: More advanced techniques like edge-aware and machine learning-based demosaicing require significant processing power and memory, limiting their usage in real-time or low-power devices.
- **Edge and Detail Preservation**: No demosaicing algorithm perfectly preserves edges and fine details; trade-offs between sharpness, color accuracy, and noise are often required.

Demosaicing continues to evolve, with ongoing research focused on minimizing these limitations to produce even more natural and accurate color images

# White Balance in Cameras

White balance (WB) is a critical process in photography and videography that ensures colors are rendered accurately under different lighting conditions. It adjusts the colors in an image so that the whites appear neutral, allowing other colors to be reproduced correctly. Without proper white balance, images can take on an unnatural color cast, such as yellowish or bluish tones.

## Understanding Color Temperature

Color temperature is measured in Kelvin (K) and refers to the hue of a specific type of light source. Different light sources emit varying colors:

- **Tungsten Light (Incandescent)**: ~2800K (warm, yellowish)
- **Fluorescent Light**: ~4000K (cool, greenish)
- **Daylight**: ~5500K to 6500K (neutral, white)
- **Shade**: ~7000K to 8000K (cool, bluish)

### How White Balance Works

The camera's white balance setting compensates for the color temperature of the light source. It adjusts the intensities of the red, green, and blue (RGB) channels in the image to achieve a neutral white, thus ensuring that other colors are rendered accurately.

## White Balance Techniques

### 1. **Auto White Balance (AWB)**
   - **Description**: The camera automatically detects the scene's lighting conditions and adjusts the white balance accordingly.
   - **Use Cases**: Convenient for everyday shooting where lighting conditions change frequently.
   - **Limitations**: May struggle with mixed lighting conditions or when the scene contains dominant colors, leading to inaccurate color representation.

### 2. **Preset White Balance**
   - **Description**: Photographers can select from a range of predefined settings (e.g., daylight, cloudy, tungsten, fluorescent) that correspond to common lighting situations.
   - **Use Cases**: Useful for consistent results in controlled lighting environments.
   - **Limitations**: Requires understanding of lighting conditions; not adaptable to unique situations.

### 3. **Custom White Balance**
   - **Description**: Photographers can manually set the white balance by taking a reference shot of a neutral gray or white object under the lighting conditions they are using.
   - **Use Cases**: Ideal for studio photography or any scenario with controlled lighting.
   - **Limitations**: Requires additional steps and equipment (a gray card), which may not always be feasible.

### 4. **Manual White Balance Adjustment**
   - **Description**: Advanced users can manually adjust the RGB channels to fine-tune the white balance according to their preferences.
   - **Use Cases**: Useful for creative control over color rendition in artistic photography.
   - **Limitations**: Requires a good understanding of color theory and can be time-consuming.

## Use Cases

- **Portrait Photography**: Proper white balance is essential to ensure skin tones appear natural and flattering under various lighting conditions.
- **Product Photography**: Accurate color representation is critical when showcasing products, particularly in e-commerce.
- **Landscape Photography**: White balance helps capture the true colors of natural scenes, enhancing the visual appeal of the image.

## Limitations of White Balance

While white balance is vital for achieving accurate colors, it has some limitations:

- **Inconsistencies Under Mixed Lighting**: Different light sources in a single scene can create challenges for accurate white balance, leading to color casts.
- **Post-Processing Adjustments**: While white balance can be adjusted in post-processing, it’s always better to capture the best possible balance in-camera to minimize artifacts and maintain image quality.
- **Subjectivity**: Artistic choices can lead to differing opinions on what constitutes "correct" white balance, making it sometimes a subjective process.

Overall, effective white balance is crucial for producing high-quality images that accurately reflect the scene's colors and lighting conditions.

# Denoising in Cameras

Denoising is a crucial image processing technique used to reduce unwanted noise in photographs, particularly those taken in low-light conditions or with high ISO settings. Noise can manifest as graininess, color distortion, or random variations in brightness, detracting from image quality. Effective denoising enhances the overall clarity and detail of images, making them more visually appealing.

## Understanding Image Noise

Image noise typically arises from several sources, including:

- **Sensor Noise**: Variability in the sensor's response to light, often exacerbated in low-light situations or when using high ISO settings.
- **Shot Noise**: Random variations in the number of photons hitting the sensor, which is more pronounced in low-light conditions.
- **Quantization Noise**: Results from the process of converting an analog signal to a digital format, introducing rounding errors in pixel values.

### Types of Noise

1. **Luminance Noise**: Variations in brightness that can appear as grainy textures in the image. This type of noise is often more acceptable than color noise in terms of perceived image quality.

2. **Chrominance Noise**: Variations in color that manifest as random color speckles or splotches. This type of noise is typically more noticeable and undesirable, as it can significantly alter the color accuracy of the image.

## Denoising Techniques

### 1. **Spatial Filtering**
   - **Description**: Techniques like Gaussian blur, median filtering, or bilateral filtering smooth out noise by averaging pixel values in a localized area.
   - **Use Cases**: Useful for reducing luminance noise while preserving edge detail.
   - **Limitations**: May introduce blurring, particularly in fine details, and may not effectively remove chrominance noise.

### 2. **Frequency Domain Filtering**
   - **Description**: This method transforms the image into the frequency domain using techniques like the Fourier Transform, allowing separation of noise from useful image data. Noise typically occupies higher frequencies, which can be reduced or eliminated.
   - **Use Cases**: Effective for removing periodic noise patterns and achieving a cleaner image.
   - **Limitations**: Computationally intensive and can distort image details if not carefully managed.

### 3. **Wavelet Denoising**
   - **Description**: Wavelet transforms decompose the image into different frequency components, enabling targeted noise reduction at various scales while preserving image detail.
   - **Use Cases**: Common in professional photography and scientific imaging where preserving detail is critical.
   - **Limitations**: Complexity of implementation and potential artifacts if parameters are not optimally set.

### 4. **Machine Learning-Based Denoising**
   - **Description**: These techniques leverage neural networks trained on large datasets to learn how to effectively separate noise from signal in images. The models can intelligently adapt to various types of noise and image content.
   - **Use Cases**: Increasingly popular in consumer cameras and software applications due to their high effectiveness and adaptability.
   - **Limitations**: Requires substantial computational resources and training data; may struggle with unusual noise patterns not present in the training set.

### 5. **Temporal Denoising**
   - **Description**: In video applications, this technique uses information from multiple frames to reduce noise, leveraging the temporal coherence of video data to improve overall quality.
   - **Use Cases**: Particularly useful in low-light video recordings where frame-to-frame consistency can help in noise reduction.
   - **Limitations**: Sensitive to motion; fast-moving subjects can lead to ghosting or blurring effects.

## Use Cases

- **Low-Light Photography**: Denoising is essential for capturing clear and usable images in low-light conditions without excessive grain.
- **Astrophotography**: High ISO settings often lead to significant noise, making effective denoising crucial for capturing celestial details.
- **Medical Imaging**: In fields like MRI or X-ray imaging, noise reduction is vital for ensuring diagnostic clarity and accuracy.

## Limitations of Denoising

Despite its importance, denoising comes with several limitations:

- **Detail Loss**: Overzealous denoising can result in loss of fine details, leading to an overly smooth or unnatural appearance.
- **Artifact Introduction**: Some denoising algorithms can introduce artifacts, such as halo effects or blurring around edges, if not applied judiciously.
- **Computational Demands**: Advanced denoising techniques, especially those based on machine learning, can require significant processing power and may not be suitable for real-time applications.

Overall, effective denoising is crucial for producing high-quality images that retain detail and accuracy, especially in challenging shooting conditions. The choice of denoising technique depends on the specific application, the type of noise present, and the desired balance between noise reduction and detail preservation.
# Gamma Correction in Cameras

Gamma correction is a vital image processing technique used to adjust the brightness and contrast of images. It compensates for the nonlinear relationship between the encoded luminance values in digital images and human perception of brightness, ensuring that images appear natural and well-balanced across different display devices.

## Understanding Gamma

Gamma is a numerical value that defines the nonlinear curve used to map the input signal (brightness values) to the output signal (display brightness). The relationship can be expressed as:

${Output} = {Input}^{gamma}$

Where:
- **Input**: The original brightness value.
- **Output**: The adjusted brightness value.
- **γ (gamma)**: A constant that determines the shape of the curve.

### Gamma Values

- **γ < 1**: Brightens the image. Useful for enhancing dark areas and improving visibility in shadows.
- **γ = 1**: No change; the input and output are equal.
- **γ > 1**: Darkens the image, which can help control highlights and improve detail in bright areas.

## How Gamma Correction Works

Gamma correction adjusts the pixel values of an image based on the gamma curve. This curve allows for a more perceptually uniform distribution of brightness, enhancing detail in shadows and highlights. By correcting for the display's nonlinear response, gamma correction ensures that the image's brightness levels align more closely with human vision.

## Gamma Correction Techniques

### 1. **Linear Gamma Correction**
   - **Description**: This method applies a linear mapping to the pixel values, resulting in a straightforward adjustment of brightness.
   - **Use Cases**: Simple applications where minimal manipulation is required, typically in linear color space workflows.
   - **Limitations**: Does not account for the nonlinear response of display devices, which may lead to unnatural images.

### 2. **Piecewise Linear Gamma Correction**
   - **Description**: This technique uses a piecewise linear function to adjust different segments of the brightness spectrum independently, allowing for greater control.
   - **Use Cases**: Useful in scenarios where specific tonal ranges need adjustment, such as in high dynamic range (HDR) imaging.
   - **Limitations**: More complex to implement than simple linear gamma correction.

### 3. **Parametric Gamma Correction**
   - **Description**: This method allows users to define custom gamma curves based on parameters, providing fine-tuning capabilities.
   - **Use Cases**: Common in professional image editing software for creative control over image aesthetics.
   - **Limitations**: Requires a deeper understanding of gamma values and their effects on image quality.

### 4. **Inverse Gamma Correction**
   - **Description**: This technique applies the inverse of the gamma function to the image, often used in preparation for display or printing.
   - **Use Cases**: Necessary when converting images to formats that require linear data for processing, such as during printing.
   - **Limitations**: Can lead to incorrect results if not carefully managed, particularly when combined with other adjustments.

## Use Cases

- **Photography**: Enhances the quality of images by ensuring that details are preserved in both shadows and highlights, making photographs more visually appealing.
- **Video Production**: Ensures consistency in brightness and contrast across frames, which is crucial for maintaining a cohesive look in films and videos.
- **Game Development**: Gamma correction is used to achieve realistic lighting effects, improving the overall visual quality of gaming environments.

## Limitations of Gamma Correction

While gamma correction is essential for image quality, it has some limitations:

- **Potential for Clipping**: Incorrect gamma settings can lead to clipping in highlights or shadows, resulting in loss of detail in bright or dark areas.
- **Display Dependency**: Different displays may have varying gamma characteristics, requiring careful calibration to achieve consistent results across devices.
- **Subjectivity**: The choice of gamma value can be subjective, as different applications and artistic intentions may call for different corrections.

Overall, gamma correction is a crucial technique for ensuring that images appear natural and balanced, accommodating the nonlinear response of human vision and display devices. Proper application of gamma correction enhances the visual quality of images across various domains, from photography to video production.
# Sharpening Filters in Cameras

Sharpening filters are essential image processing techniques used to enhance the perceived clarity and detail of images. By increasing contrast along the edges and fine details, sharpening helps to create a more defined and visually appealing image. However, it is crucial to apply sharpening judiciously to avoid introducing artifacts.

## Understanding Image Sharpness

Sharpness refers to the clarity and detail visible in an image, which is influenced by several factors:

- **Edge Definition**: The perceived sharpness of an image is largely determined by the clarity of edges and transitions between different colors and brightness levels.
- **Contrast**: Higher contrast along edges enhances the perception of sharpness.
- **Detail**: Fine details, such as textures, contribute to overall sharpness.

### Types of Sharpening

1. **Luminance Sharpening**: Targets the brightness information in the image, enhancing edges without altering color information. This method is typically preferred as it maintains color fidelity while improving detail.

2. **Chrominance Sharpening**: Enhances color information in the image, but can lead to unwanted color artifacts if over-applied. It is generally less common than luminance sharpening.

## Sharpening Techniques

### 1. **Unsharp Masking (USM)**
   - **Description**: A widely used technique that enhances edges by subtracting a blurred version of the image from the original. The process involves adjusting three parameters: amount, radius, and threshold.
     - **Amount**: Determines the strength of the sharpening effect.
     - **Radius**: Controls how much area around an edge is affected.
     - **Threshold**: Defines how different the sharpened pixel must be from the surrounding pixels to be affected.
   - **Use Cases**: Common in both photography and print media for improving detail.
   - **Limitations**: Can produce halo effects and increase noise if applied excessively.

### 2. **High-Pass Filtering**
   - **Description**: This technique involves applying a high-pass filter to retain only the high-frequency components (edges) of the image. The high-pass layer is then blended with the original image to enhance sharpness.
   - **Use Cases**: Often used in professional photo editing and retouching.
   - **Limitations**: Requires careful blending to avoid unnatural effects and can be time-consuming.

### 3. **Laplace Sharpening**
   - **Description**: This method uses the Laplacian operator to detect edges based on second derivatives. The edges are then enhanced, resulting in a sharper image.
   - **Use Cases**: Suitable for applications requiring strong edge enhancement.
   - **Limitations**: Can introduce noise and artifacts if not used judiciously.

### 4. **Deconvolution Sharpening**
   - **Description**: This advanced technique attempts to reverse the effects of blurring by using mathematical algorithms to restore image details. It requires a known point spread function (PSF) of the blur.
   - **Use Cases**: Common in scientific imaging and specialized photography where precision is critical.
   - **Limitations**: Computationally intensive and can lead to artifacts if the PSF is not accurately defined.

### 5. **Machine Learning-Based Sharpening**
   - **Description**: These techniques utilize deep learning models trained on large datasets to intelligently enhance image sharpness, adapting to various image characteristics.
   - **Use Cases**: Increasingly used in modern camera software and mobile applications for automatic sharpening.
   - **Limitations**: Requires significant computational resources and training data; may not generalize well to images outside the training set.

## Use Cases

- **Portrait Photography**: Sharpening helps emphasize facial features and textures while maintaining a natural appearance.
- **Landscape Photography**: Enhances fine details in nature scenes, making textures and patterns more pronounced.
- **Product Photography**: Critical for showcasing product details and ensuring that textures are visible and appealing in advertising.

## Limitations of Sharpening Filters

While sharpening filters are powerful tools, they come with some limitations:

- **Artifact Introduction**: Over-sharpening can lead to halo effects, noise amplification, and unnatural-looking images, detracting from overall quality.
- **Increased Noise**: Sharpening can accentuate existing noise, particularly in low-light images or areas with fine detail.
- **Subjective Interpretation**: The ideal level of sharpening can be subjective, varying based on personal preferences and intended use.

Overall, sharpening filters are invaluable for enhancing image clarity and detail. When applied appropriately, they can significantly improve the quality of images, making them more visually compelling across various applications. However, careful consideration of the sharpening technique and parameters is essential to achieve the desired results without introducing artifacts.

### image 1

| Image Type         | Image                                                                                   
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| With Demosaicing And Gamma Correction  |    <img width="956" alt="Demosaic_ _gamma1" src="https://github.com/user-attachments/assets/97499940-d15e-4e7e-9a5a-0b2085e5cbfa">|
                                                                                                                               | 

### image 2

| Image Type         | Image                                                                                   
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| With Demosaicing, White Balance And Gamma Correction  |  <img width="955" alt="Demosaic_ _WhiteBalance_ gamma" src="https://github.com/user-attachments/assets/065edabc-53ba-4c5b-9f7b-c6645726af5b">|                                                                                                                                                                             
                                                                                                                                                
                                                                                                                                                             

### image 3

| Image Type                      | Image                                                                                                                      |         
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| With Demosaicing and Denoising,White Balance, and Gamma | ![Demosaic_ _WhiteBalance_ _denoise_gamma](https://github.com/user-attachments/assets/9aa14d17-b13f-4a2a-aab9-0755f813c8e0)|
                                                                                                                           

### image 4

| Image Type                             | Image                                                                                                       |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------|
| With Demosaicing, White Balance, Denoising, Gamma Correction, and Sharpening    |![Demosaic_ _WhiteBalance_ _denoise_gamma](https://github.com/user-attachments/assets/ee3e653a-17d8-4501-b107-4dd5e3c46253)|

#Final Image After Basic ISP Implementation
<img width="958" alt="Final_image_after_ISP" src="https://github.com/user-attachments/assets/463d2ec4-9ba0-4350-9015-4fad645fd29e">



                                                                                                                       
                                                                                                                                       
