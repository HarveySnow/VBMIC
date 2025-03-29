# VBMIC: Variable Bitrate Multi-view Image Compression

VBMIC is a novel lightweight model designed for efficient variable bitrate multi-view image compression, tailored for resource-limited environments. It leverages deep learning to optimize compression rates while maintaining high image quality across varying inter-view correlations.

# Key Features

- **Adaptive Bitrate Encoding**: Dynamically adjust encoding bitrates based on the redundancy among images.

- **Feature Scaling Module**: Adjusts feature maps according to variable bit factors for efficient compression.

- **Model Pruning**: Reduces computational complexity and storage overhead via stepwise channel pruning.

- **High Compression Efficiency**: Achieves competitive PSNR, MS-SSIM, and FID scores at low bitrates.

## Performance

- **Datasets**: Tested on Cityscapes, InStereo2K, and WildTrack datasets.
- **Metrics**: Evaluated using PSNR, MS-SSIM, and FID.
- **Results**: Demonstrates superior performance compared to existing models at low bitrates (0.02 to 0.30).

## Usage

1. **Setup Environment**: Ensure Python 3.7 and necessary libraries (PyTorch, CompressAI) are installed.
2. **Model Training**: Train the VBMIC model using the provided training scripts.
3. **Compression**: Apply the trained model to compress multi-view images.
4. **Decompression**: Decode compressed bitstream using the VBMIC decoding process.

## Contributing

Contributions to VBMIC are welcome! Please submit a pull request or open an issue for any improvements, bug fixes, or feature requests.

## Authors

- Junwei Zhou
- Yuxuan Zhao
- Yujie Song
- Tian Xiang
- Yanchao Yang
- Jianwen Xiang
