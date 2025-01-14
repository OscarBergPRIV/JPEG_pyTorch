# JPEG_pyTorch
This repo contains the implementation of a single-channel JPEG in pyTorch. The quantization happens gradient-aware, such that the full Encoder-Decoder remains trainable.

# Trainable JPEG Compression for Single-Channel Images in PyTorch

This repository implements a trainable JPEG compression algorithm for single-channel images using PyTorch. The approach allows training the model with two distinct configurations:
- **Fully Straight-Through Estimator (STE)** for gradients.
- **STE only for the Quantization Module** for more control.

Useful for [Split-computing for DNNs](https://arxiv.org/abs/1902.01000)


## Key Features
- **Trainable Compression**: The JPEG compression pipeline, including the Discrete Cosine Transform (DCT) and quantization, is differentiable.
- **Customizable Training Options**:
  - Fully STE: Gradients pass through all parts of the pipeline.
  - STE for quantization only: Offers a middle ground for training with quantization-aware constraints.
- **Modular Design**: While the pipeline omits Huffman and Run-Length Encoding, these lossless techniques can be added in post-processing for complete JPEG functionality.
  
## Image 1
![Cat Quality 1](./imgs/cat_quality_1.PNG)

## Image 2
![Cat Quality 10](./imgs/cat_quality_10.PNG)

## Image 3
![Cat Quality 100](./imgs/cat_quality_100.PNG)


## How It Works
The implementation adheres to the following steps:
1. **Block-wise DCT Transformation**: Applies an 8x8 DCT transform to image blocks.
2. **Quantization**: Compresses the DCT coefficients using trainable quantization tables.
3. **Decompression**: Inverse quantization and DCT for reconstructing the image.
4. **Differentiability**: Gradients are approximated using the STE for effective training.

## Limitations
- **Lossless Encoding**: Huffman and Run-Length Encoding are not included in the pipeline but can be integrated separately.
- **Single-Channel Input**: This implementation focuses on grayscale images (1 channel). Extending to RGB is possible but not implemented here.

## Installation
Clone this repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd JPEG_pyTorch
