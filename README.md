# Image Algorithms

This repository contains implementations of various image processing algorithms focused on ISP pipeline implementation, denoising techniques, and HDR imaging.

## Overview

- **Assignment 1**: Basic ISP Pipeline Implementation
- **Assignment 2**: Advanced Denoising and Sharpness Techniques
- **Assignment 3**: HDR Imaging Implementation

## Getting Started

### Prerequisites

- Python 3.10.11
- OpenCV
- NumPy
- PyTorch (for AI-based denoising in Assignment 2)
- Raw image viewer (PixelViewer or IrfanView with RAW plugin)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gauravnirmalkumar/image_algorithms.git
cd image_algorithms
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Assignment Details

### Assignment 1: Basic ISP Pipeline
```bash
cd Assignment-1
```
Implements basic image signal processing routines including:
- Demosaicing (5x5 edge-based interpolation)
- White Balance (Gray World algorithm)
- Denoising (5x5 Gaussian filter)
- Gamma Correction (sRGB)
- Sharpening (Unsharp mask)

### Assignment 2: Denoising and Sharpness
```bash
cd Assignment-2
```
Implements and compares various denoising and sharpening techniques:
- Traditional filters (Median, Bilateral, Gaussian)
- AI-based denoising
- Edge enhancement using Laplacian filter
- SNR analysis for different gray tones

### Assignment 3: HDR Imaging
```bash
cd Assignment-3
```
HDR implementation including:
- Multi-exposure image merging
- Tone mapping
- HDR to LDR conversion

## Directory Structure
```
image_algorithms/
├── Assignment-1/
│   ├── src/
│   ├── data/
│   └── docs/
├── Assignment-2/
│   ├── src/
│   ├── data/
│   └── docs/
├── Assignment-3/
│   ├── src/
│   ├── data/
│   └── docs/
└── requirements.txt
```

## Usage

Each assignment directory contains its own README with specific instructions. Generally:

1. Navigate to the assignment directory
2. Run the main script:
```bash
python src/main.py
```
3. Check the output in the respective output directory

## Input Data Format

- Raw images: 12-bit Bayer pattern (GRBG)
- Resolution: 1920x1280
- View raw images using PixelViewer or IrfanView with RAW plugin

## Configuration

For viewing RAW files in PixelViewer:
- Bit Depth: 12 bits
- Bayer Pattern: GRBG
- Resolution: 1920x1280

## Results

Each assignment generates output in its respective directory:
- Assignment 1: Processed images with different pipeline combinations
- Assignment 2: Comparison results of different denoising/sharpening techniques
- Assignment 3: Final HDR images and tone-mapped results

## Documentation

Detailed documentation for each assignment can be found in their respective `/docs` directories:
- Implementation approach
- Algorithm details
- Results analysis
- Performance metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- Course instructors and TAs
- Open source community for various image processing libraries
- Contributors and collaborators

## Contact

Project Link: [https://github.com/gauravnirmalkumar/AIDenoisingAndEdgeEnhancement](https://github.com/gauravnirmalkumar/AIDenoisingAndEdgeEnhancement)
