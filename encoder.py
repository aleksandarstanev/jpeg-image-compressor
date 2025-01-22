"""
Discrete Cosine Transform (DCT) compressor.
Reconstructs an image by using only the first K coefficients of its 8x8 DCT,
OR by quantizing all coefficients of the 8x8 DCT.
"""

import argparse
import itertools
import math
import os

from skimage.metrics import structural_similarity
import numpy as np
import cv2 as cv

import utils

def main():
    # Parse the command line arguments to attributes of 'args'
    parser = argparse.ArgumentParser(description='Discrete Cosine Transform (DCT) compressor. '
                                                 'Reconstructs an image by using only the first K coefficients '
                                                 'of its 8x8 DCT, or by quantizing all coefficients of the 8x8 DCT.')
    
    parser.add_argument('--input', dest='image_path', required=True, type=str,
                        help='Path to the image to be compressed.')
    
    parser.add_argument('--output', dest='output_path', required=True, type=str,
                        help='Path to the output (compressed) image.')
    
    parser.add_argument('--coeffs', dest='num_coeffs', required=False, type=int,
                        help='Number of coefficients that will be used to reconstruct the original image, '
                             'without quantization.')
    
    parser.add_argument('--scale-factor', dest='scale_factor', required=False, type=float, default=1,
                        help='Scale factor for the quantization step (the higher, the more quantization loss).')
    
    args = parser.parse_args()

    # Read image
    orig_img = cv.imread(args.image_path, cv.IMREAD_COLOR)
    img = np.float32(orig_img)

    # Convert to YCrCb color space
    img_ycc = cv.cvtColor(img, code=cv.COLOR_BGR2YCrCb)

    # Perform 4:2:0 chroma subsampling
    img_ycc = chroma_subsampling(img_ycc)

    # Split into channels and compress each channel separately.
    rec_img = np.empty_like(img)
    for channel_num in range(3):
        if channel_num == 0:
            quantization_matrix = utils.luminance_quantization_matrix
        else:
            quantization_matrix = utils.chrominance_quantization_matrix

        single_channel_image = approximate_single_channel_image(img_ycc[:, :, channel_num],
                                            num_coeffs=args.num_coeffs,
                                            scale_factor=args.scale_factor,
                                            quantization_matrix=quantization_matrix)
        
        rec_img[:, :, channel_num] = single_channel_image

    # Convert back to RGB from YCrCb
    rec_img_rgb = cv.cvtColor(rec_img, code=cv.COLOR_YCrCb2BGR)
    rec_img_rgb[rec_img_rgb < 0] = 0
    rec_img_rgb[rec_img_rgb > 255] = 255
    rec_img_rgb = np.uint8(rec_img_rgb)

    # Calculate conversion metrics to evaluate the quality of the approximation
    err_img = abs(np.array(rec_img_rgb, dtype=float) - np.array(orig_img, dtype=float))

    # Mean Squared Error (MSE)
    # Lower MSE is better.
    mse = (err_img ** 2).mean()

    print(f"Mean Squared Error (MSE): {mse}")

    # Peak Signal-to-Noise Ratio (PSNR)
    # Higher PSNR is better. Typically, PSNR > 30 dB is considered good.
    psnr = 10 * math.log10((255 ** 2) / mse)
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr}")

    # Structural Similarity Index Measure (SSIM)
    # Higher SSIM is better. Typically, SSIM > 0.8 is considered good.
    ssim = structural_similarity(
        cv.cvtColor(np.float32(rec_img_rgb), code=cv.COLOR_BGR2GRAY),
        cv.cvtColor(np.float32(orig_img), code=cv.COLOR_BGR2GRAY), 
        data_range=255
    )
    print(f"Structural Similarity Index Measure (SSIM): {ssim}")

    # Save the compressed image
    if args.output_path.endswith('.jpg') or args.output_path.endswith('.jpeg'):
        print(f"Saving the compressed image to {args.output_path}")
        cv.imwrite(args.output_path, rec_img_rgb)
    else:
        raise ValueError("Output file must have a .jpg or .jpeg extension")
    
    # Calculate compression ratio
    orig_img_size = os.path.getsize(args.image_path)
    rec_img_size = os.path.getsize(args.output_path)

    compression_ratio = rec_img_size / orig_img_size

    print(f"Original image size: {orig_img_size} bytes")
    print(f"Compressed image size: {rec_img_size} bytes")
    print(f"Compression ratio: {compression_ratio}")
    
    # Show the compressed image
    cv.imshow('Compressed image', rec_img_rgb)
    cv.waitKey(0)

def chroma_subsampling(img):
    """
    Performs 4:2:0 chroma subsampling on the YCrCb image.
    Subsamples the chrominance channels (Cr and Cb) both horizontally and vertically.
    Returns the image with subsampled chrominance channels.
    """

    img_copy = img.copy()
    
    height, width = img.shape[:2]
    
    # Subsample Cr and Cb channels (channels 1 and 2)
    for channel in [1, 2]:
        # Resize chrominance channel to half width AND half height (4:2:0 subsampling)
        chrominance = cv.resize(img_copy[:, :, channel], 
                              (width//2, height//2),
                              interpolation=cv.INTER_LINEAR)
        
        # Resize back to original size
        img_copy[:, :, channel] = cv.resize(chrominance,
                                          (width, height),
                                          interpolation=cv.INTER_NEAREST)
    
    return img_copy

def approximate_single_channel_image(img, num_coeffs=None, scale_factor=1, quantization_matrix=None):
    """
    Approximates a single channel image by using only the first coefficients of the DCT.
    1) The image is chopped into 8x8 pixels patches and the DCT is applied to each patch.
    2) If num_coeffs is provided, only the first K DCT coefficients are kept.
    3) If not, all the elements are quantized using the JPEG quantization matrix and the scale_factor.
    4) Finally, the resulting coefficients are used to approximate the original patches with the IDCT, and the image is
     reconstructed back again from these patches.
    :param img: Image to be approximated.
    :param num_coeffs: Number of DCT coefficients to use.
    :param scale_factor: Scale factor to use in the quantization step.
    :return: The approximated image.
    """

    if len(img.shape) != 2:
        raise ValueError('Input image must be a single channel 2D array')

    height = img.shape[0]
    width = img.shape[1]

    if height % 8 != 0 or width % 8 != 0:
        # If the image dimensions are not multiple of 8, we can pad the image with zeroes.
        # For the sake of simplicity, this is not implemented here.
        raise ValueError("Image dimensions (%s, %s) must be multiple of 8" % (height, width))

    # Split into 8 x 8 pixels blocks
    img_blocks = [img[j:j + 8, i:i + 8]
                  for (j, i) in itertools.product(range(0, height, 8),
                                                  range(0, width, 8))]

    # DCT transform every 8x8 block
    dct_blocks = [cv.dct(img_block) for img_block in img_blocks]

    if num_coeffs is not None:
        # Keep only the first K DCT coefficients of every block
        reduced_dct_coeffs = [utils.zig_zag(dct_block, num_coeffs) for dct_block in dct_blocks]
    else:
        # Quantize all the DCT coefficients using the quantization matrix and the scaling factor
        reduced_dct_coeffs = [np.round(dct_block / (quantization_matrix * scale_factor))
                              for dct_block in dct_blocks]

        # Scale back the coefficients to the original range
        reduced_dct_coeffs = [reduced_dct_coeff * (quantization_matrix * scale_factor)
                          for reduced_dct_coeff in reduced_dct_coeffs]

    # Inverse DCT of every block
    rec_img_blocks = [cv.idct(coeff_block) for coeff_block in reduced_dct_coeffs]

    # Reshape the reconstructed image blocks
    rec_img = []
    for chunk_row_blocks in utils.chunks(rec_img_blocks, width / 8):
        for row_block_num in range(8):
            for block in chunk_row_blocks:
                rec_img.extend(block[row_block_num])
    rec_img = np.array(rec_img).reshape(height, width)

    return rec_img


if __name__ == '__main__':
    main()
