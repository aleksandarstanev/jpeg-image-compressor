# JPEG Image Compressor

## Description

A Python-based JPEG image compression tool that implements the core principles of JPEG compression. This tool supports two compression methods:

1. Traditional JPEG compression using customizable quantization tables with a scaling factor
2. DCT coefficient truncation, where you can specify the number of coefficients to retain

## Usage

### With Quantization table

```bash
python3 encoder.py --input <path_to_image> --output <output_path> --scale-factor <scale_factor>
```

### Using the first K coefficients of the DCT

```bash
python3 encoder.py --input <path_to_image> --output <output_path> --coeffs <number_of_coefficients>
```
