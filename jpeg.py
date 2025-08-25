import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
from PIL import Image
import io

# -----------------------------
# Block splitting/merging
# -----------------------------
class BlockSplitting(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size

    def forward(self, image):
        B, C, H, W = image.shape
        if H % self.block_size != 0 or W % self.block_size != 0:
            raise ValueError(f"Image dimensions ({H}x{W}) are not divisible by block size {self.block_size}.")

        image_reshaped = image.view(B, C, H // self.block_size, self.block_size, W // self.block_size, self.block_size)
        image_transposed = image_reshaped.permute(0, 1, 2, 4, 3, 5)
        num_blocks = (H // self.block_size) * (W // self.block_size)
        blocks = image_transposed.contiguous().view(B, C * num_blocks, self.block_size, self.block_size)
        return blocks


class BlockMerging(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size

    def forward(self, blocks, original_size, num_channels):
        H, W = original_size
        B = blocks.shape[0]
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        image_reshaped = blocks.view(B, num_channels, num_blocks_h, num_blocks_w, self.block_size, self.block_size)
        image_transposed = image_reshaped.permute(0, 1, 2, 4, 3, 5)
        image = image_transposed.contiguous().view(B, num_channels, H, W)
        return image

# -----------------------------
# JPEG Quantization table
# -----------------------------
QT = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=torch.float32)

# -----------------------------
# Proper libjpeg-style quality scaling
# -----------------------------
def quality_to_scaled_QT(QT, quality):
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    QT_scaled = torch.floor((QT * scale + 50) / 100)
    QT_scaled = torch.clamp(QT_scaled, 1, 255)
    return QT_scaled

# -----------------------------
# Quantization / Dequantization
# -----------------------------
class QuantizationCoef(nn.Module):
    def __init__(self, QT, ste=False):
        super().__init__()
        self.register_buffer('QT', QT)
        self.round = StraightThroughRound.apply if ste else torch.round

    def forward(self, inputs):
        return self.round(inputs.float() / self.QT)


class DequantizationCoef(nn.Module):
    def __init__(self, QT):
        super().__init__()
        self.register_buffer('QT', QT)

    def forward(self, inputs):
        return inputs * self.QT


class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# -----------------------------
# JPEG Module
# -----------------------------
class QuantizationModule(nn.Module):
    def __init__(self, img_size=(672, 672), quality=10, device="cpu", ste=False):
        super().__init__()
        self.h, self.w = img_size
        self.jpeg_quality = quality  # save for display

        QT_scaled = quality_to_scaled_QT(QT, quality).to(device)
        self.quantCF = QuantizationCoef(QT_scaled, ste=ste)
        self.dequantCF = DequantizationCoef(QT_scaled)
        self.bsplit = BlockSplitting()
        self.bmerge = BlockMerging()

        self.size = 8
        self.norm_factors = torch.zeros(self.size).to(device)
        self.cos_terms = torch.zeros(self.size, self.size, self.size, self.size).to(device)

        for x in range(self.size):
            self.norm_factors[x] = 1 / math.sqrt(2) if x == 0 else 1
        for x in range(self.size):
            for y in range(self.size):
                for u in range(self.size):
                    for v in range(self.size):
                        self.cos_terms[x, y, u, v] = (
                            math.cos((2 * x + 1) * u * math.pi / (2 * self.size)) *
                            math.cos((2 * y + 1) * v * math.pi / (2 * self.size))
                        )

    def quantize(self, input_tensor, min_value=None, max_value=None):
        if min_value is None:
            min_value = input_tensor.min()
        if max_value is None:
            max_value = input_tensor.max()
        scale = (max_value - min_value) / 255.0
        quantized_tensor = torch.round(((input_tensor - min_value) / scale)).clamp(0, 255)
        return quantized_tensor, min_value, max_value

    def dequantize(self, quantized_tensor, min_value, max_value):
        scale = (max_value - min_value) / 255.0
        return quantized_tensor.float() * scale + min_value

    def dct_2d_explicit(self, input_tensor):
        bs, c, h, w = input_tensor.shape
        input_tensor = input_tensor - 128
        dct_coefficients = torch.zeros_like(input_tensor)
        for u in range(h):
            for v in range(w):
                dct_coefficients[:, :, u, v] = (
                    0.25 * self.norm_factors[u] * self.norm_factors[v] *
                    torch.sum(
                        input_tensor *
                        self.cos_terms[:, :, u, v].unsqueeze(0).unsqueeze(0),
                        dim=(2, 3)
                    )
                )
        return dct_coefficients

    def idct_2d_explicit(self, dct_coefficients):
        bs, c, h, w = dct_coefficients.shape
        reconstructed = torch.zeros_like(dct_coefficients)
        for x in range(h):
            for y in range(w):
                reconstructed[:, :, x, y] = (
                    0.25 * torch.sum(
                        self.norm_factors.unsqueeze(1) *
                        self.norm_factors.unsqueeze(0) *
                        dct_coefficients *
                        self.cos_terms[x, y, :, :].unsqueeze(0).unsqueeze(0),
                        dim=(2, 3)
                    )
                )
        reconstructed += 128.0
        return reconstructed

    def forward(self, input_tensor):
        num_channels = input_tensor.shape[1]
        quantized_tensor, min_value, max_value = self.quantize(input_tensor)
        quantized_tensor = self.bsplit(quantized_tensor)
        dct_coeffs = self.dct_2d_explicit(quantized_tensor)
        dct_coeffs = self.quantCF(dct_coeffs)
        dct_coeffs = self.dequantCF(dct_coeffs)
        reconstructed_tensor = self.idct_2d_explicit(dct_coeffs)
        output_tensor = self.dequantize(reconstructed_tensor, min_value, max_value)
        output_tensor = self.bmerge(output_tensor, (self.h, self.w), num_channels=num_channels)
        return output_tensor

def read_jpeg_to_torch_8bit(image_path):
    return read_image(image_path)

def show_tensor(tensor, title=""):
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    plt.imshow(arr, cmap="gray")
    plt.title(title)
    plt.axis("off")

# -----------------------------
# Example main (with PIL comparison)
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quality = 20  # Match your example
    quant_module = QuantizationModule(img_size=(672, 672), quality=quality, device=device, ste=False)
    quant_module.to(device)

    image_path = './cat_superres_with_ort.jpg'
    img_tensor = read_jpeg_to_torch_8bit(image_path).to(device)

    channel_index = 0
    single_channel_img = img_tensor[channel_index, :, :]
    input_tensor = single_channel_img.view(1, 1, 672, 672).float()  # Float for safety

    # Your custom JPEG reconstruction
    output_tensor_custom = quant_module(input_tensor)

    # PIL (libjpeg) comparison: Compress/decompress the same single channel
    input_arr = single_channel_img.cpu().numpy().astype(np.uint8)
    img_pil = Image.fromarray(input_arr, mode='L')
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    img_reconstructed_pil = Image.open(buffer)
    output_arr_pil = np.array(img_reconstructed_pil)
    output_tensor_pil = torch.from_numpy(output_arr_pil).view(1, 1, 672, 672).float().to(device)

    # Compute MAEs
    mae_custom = torch.mean(torch.abs(output_tensor_custom - input_tensor))
    mae_pil = torch.mean(torch.abs(output_tensor_pil - input_tensor))
    print(f"MAE (Custom vs. Original): {mae_custom.item():.4f}")
    print(f"MAE (PIL/libjpeg vs. Original): {mae_pil.item():.4f}")
    print(f"Difference tensor (Custom - Original):\n{output_tensor_custom - input_tensor}")
    print(f"Difference tensor (PIL/libjpeg - Original):\n{output_tensor_pil - input_tensor}")

    # Visualize
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    show_tensor(input_tensor[0, 0], title="Original Image")
    plt.subplot(1, 3, 2)
    show_tensor(output_tensor_custom[0, 0], title=f"Custom JPEG (Quality={quality})")
    plt.subplot(1, 3, 3)
    show_tensor(output_tensor_pil[0, 0], title=f"PIL/libjpeg JPEG (Quality={quality})")
    plt.show()
