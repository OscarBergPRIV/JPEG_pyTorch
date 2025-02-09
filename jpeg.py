import torch
import torch.nn as nn
import math
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt

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
    """
    Module to merge non-overlapping blocks back into the original image.

    This reverses the BlockSplitting process, reconstructing the full image from its blocks.
    """

    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size

    def forward(self, blocks, original_size, num_channels):
        H, W = original_size  # Original height and width
        B = blocks.shape[0]   # Batch size

        # Compute number of blocks along height and width
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size

        # Reshape blocks to separate (C, num_blocks_h, num_blocks_w, block_size, block_size)
        image_reshaped = blocks.view(B, num_channels, num_blocks_h, num_blocks_w, self.block_size, self.block_size)

        # Permute dimensions to arrange blocks correctly
        image_transposed = image_reshaped.permute(0, 1, 2, 4, 3, 5)

        # Merge the block dimensions to reconstruct the full image
        image = image_transposed.contiguous().view(B, num_channels, H, W)

        return image

# Original JPEG Q-Table
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


def quality2factor(quality):
    """
    quality (1-100) to quantization factor

    Args:
        quality (float): quality value (1, 100)

    Returns:
        float: quantization factor
    """

    if 1 > quality or quality > 100:
        raise ValueError("Only quality vals in range (1, 100) acceptable")

    if quality >= 50:
        factor = 200.0 - (quality * 2.0)
    else:
        factor = 5000.0 / quality

    return factor / 100.0

class QuantizationCoef(nn.Module):

    def __init__(self, QT, factor=0.5, ste=False):
        """
        Initializes the Quantization module.

        Args:
            q_table (torch.Tensor): Quantization table used to scale the DCT coefficients.
            factor (float, optional): Scaling factor to adjust the quantization strength.
                                      A higher factor results in more aggressive quantization.
                                      Default is 0.5.
        """
        super().__init__()

        if ste:
            self.round = StraightThroughRound.apply
        else:
            self.round = torch.round

        self.register_buffer('QT', QT)

        self.factor = factor + 1e-5

    def forward(self, inputs):
        """
        Quantization based on original JPEG

        Args:
            inputs (torch.Tensor): Output of DCT

        Returns:
            torch.Tensor: Quantized output of DCT
        """

        pre_quant_input = inputs.float() / (self.QT * self.factor)

        quant_input = self.round(pre_quant_input)

        return quant_input


class DequantizationCoef(nn.Module):

    def __init__(self, q_table, factor=0.5):
        """
        Args:
            QT (torch.Tensor): Quantization table

            factor (float, optional): Factor based on given quality
        """
        super().__init__()

        self.register_buffer('QT', QT)

        self.factor = factor + 1e-5

    def forward(self, inputs):
        """
        Dequantization of inputs

        Args:
            quantized_blocks (torch.Tensor): Quantized DCT coefficients with shape (B, C, H, W).

        Returns:
            torch.Tensor: Dequantized blocks with the same shape as input, restoring the original scale.
        """

        # Multiply the quantized coefficients by the quantization table and scaling factor
        output = inputs * (self.QT * self.factor)
        return output


class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuantizationModule(nn.Module):
    def __init__(self, img_size = (672, 672), quality = 10, device="cpu", ste=False):
        """
        Initializes the QuantizationModule.
        """
        super(QuantizationModule, self).__init__()
        self.h = img_size[0]
        self.w = img_size[1]

        scaling_factor = quality2factor(quality)
        print(scaling_factor)
        self.quantCF = QuantizationCoef(QT, scaling_factor, ste=ste)
        self.dequantCF = DequantizationCoef(QT, scaling_factor)
        self.bsplit = BlockSplitting()
        self.bmerge = BlockMerging()

        self.size = 8  # Patch size
        self.norm_factors = torch.zeros(self.size).to(device)
        self.cos_terms = torch.zeros(self.size, self.size, self.size, self.size).to(device)


        self.round = None
        if ste:
            self.round = StraightThroughRound.apply
        else:
            self.round = torch.round

        # Compute normalization factors
        for x in range(self.size):
            self.norm_factors[x] = 1 / math.sqrt(2) if x == 0 else 1

        # Precompute cosine terms
        for x in range(self.size):
            for y in range(self.size):
                for u in range(self.size):
                    for v in range(self.size):
                        self.cos_terms[x, y, u, v] = (
                            math.cos((2 * x + 1) * u * math.pi / (2 * self.size)) *
                            math.cos((2 * y + 1) * v * math.pi / (2 * self.size))
                        )


    def quantize(self, input_tensor, min_value=None, max_value=None):
        """
        Quantize the input tensor into 8-bit representation.

        Args:
            input_tensor (torch.Tensor): Input tensor to be quantized.
            min_value (float): Minimum value for the range (optional).
            max_value (float): Maximum value for the range (optional).

        Returns:
            torch.Tensor: Quantized tensor (integer values in range [0, 255]).
            float, float: The min and max values used for quantization.
        """
        if min_value is None:
            min_value = input_tensor.min()
        if max_value is None:
            max_value = input_tensor.max()

        scale = (max_value - min_value) / 255.0
        quantized_tensor = self.round(((input_tensor - min_value) / scale)).clamp(0, 255) # .to(torch.uint8)

        return quantized_tensor, min_value, max_value

    def dequantize(self, quantized_tensor, min_value, max_value):
        """
        Dequantize an 8-bit tensor back to its original range.

        Args:
            quantized_tensor (torch.Tensor): Quantized tensor (integer values in range [0, 255]).
            min_value (float): Minimum value of the original range.
            max_value (float): Maximum value of the original range.

        Returns:
            torch.Tensor: Dequantized tensor in the original range.
        """
        scale = (max_value - min_value) / 255.0
        dequantized_tensor = quantized_tensor.float() * scale + min_value

        return dequantized_tensor

    def dct_2d_explicit(self, input_tensor):
        """
        Compute the 2D Discrete Cosine Transform (DCT-II) for a batch of patches.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (bs, c, 8, 8).

        Returns:
            torch.Tensor: DCT coefficients of shape (bs, c, 8, 8).
        """
        bs, c, h, w = input_tensor.shape
        assert h == 8 and w == 8, "This implementation assumes 8x8 patches for height and width."

        input_tensor = input_tensor - 128

        # Compute DCT coefficients
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
        """
        Compute the 2D Inverse Discrete Cosine Transform (IDCT-II) for a batch of patches.

        Args:
            dct_coefficients (torch.Tensor): DCT coefficients of shape (bs, c, 8, 8).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (bs, c, 8, 8).
        """
        bs, c, h, w = dct_coefficients.shape
        assert h == 8 and w == 8, "This implementation assumes 8x8 patches for height and width."

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



class JPEGED(torch.nn.Module):

    def __init__(self, quality, device, img_size):
        super().__init__()
        self.quality = quality
        self.jpeg = QuantizationModule(img_size=img_size, quality=quality, device=device)


    class _JPEGEDFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, jpeg):
            x, _ = jpeg(input_tensor)
            
            ctx.save_for_backward()
            return x

        @staticmethod
        def backward(ctx, grad_output, grad_encoded=None):
            grad_input = grad_output.clone()
            return grad_input, None, None

    def forward(self, input_tensor):
        return self._JPEGEDFunction.apply(input_tensor, self.jpeg)


def read_jpeg_to_torch_8bit(image_path):
  """
  Reads a JPEG image, converts it to a PyTorch tensor, and ensures 8-bit data.

  Args:
    image_path: Path to the JPEG image file.

  Returns:
    A PyTorch tensor representing the image with 8-bit data.
  """

  img = read_image(image_path)

  return img


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_module = QuantizationModule(img_size = (672, 672), quality = 50, device=device, ste=False)
    quant_module.to(device)

    input_tensor = torch.randn(10, 1 , 672, 672).to(device)
    a = torch.tensor(4.0, requires_grad=True)
    input_tensor *= a
    output_tensor = quant_module(input_tensor)
    output_tensor.mean().backward()
    print("GRAD: ", a.grad)

    print("="*50)

    image_path = './cat_superres_with_ort.jpg'
    img_tensor = read_jpeg_to_torch_8bit(image_path)
    img_tensor = img_tensor.to(device)
    channel_index = 0

    single_channel_img = img_tensor[channel_index, :, :]

    input_tensor = single_channel_img.view(1, 1, 672, 672)
    single_channel_img_np = input_tensor.cpu().numpy()
    
    img = Image.fromarray(single_channel_img_np[0, 0])
    img.show()

    # Forward pass
    output_tensor = quant_module(input_tensor)

    img = Image.fromarray(output_tensor[0, 0].cpu().numpy())
    img.show()

    #print("dct_coeffs shape: ", dct_coeffs.shape)
