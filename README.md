# Crop Detection System

A deep learning-based system for classifying three types of crops: jute, rice, and wheat. The system uses transfer learning with MobileNetV2 and is optimized for mobile deployment using TensorFlow Lite.

## Features

- Real-time crop classification
- Support for three crop types: jute, rice, and wheat
- Image-based prediction
- Real-time webcam prediction
- Mobile-optimized model
- Configurable confidence threshold
- Unknown class detection for low-confidence predictions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Webcam (for real-time prediction)
- Sufficient disk space for model and dataset

## Installation

1. **Clone the repository**
```bash
git clone [https://github.com/yourusername/CropDetection.git](https://github.com/almustakim/CropDetection.git)
cd CropDetection
```

2. **Create a virtual environment (recommended)**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## Project Structure

```
CropDetection/
├── dataset/
│   ├── train/
│   │   ├── jute/
│   │   ├── rice/
│   │   └── wheat/
│   └── validation/
│       ├── jute/
│       ├── rice/
│       └── wheat/
├── testImage/
│   └── test.jpg
├── train.py
├── predict.py
├── webcam_predict.py
├── model.h5
├── model.tflite
└── requirements.txt
```

## Dataset Preparation

1. **Organize your images**:
   - Place training images in `dataset/train/<crop_type>/`
   - Place validation images in `dataset/validation/<crop_type>/`
   - Supported crop types: jute, rice, wheat

2. **Image requirements**:
   - Format: JPG or PNG
   - Size: Will be resized to 150x150 pixels
   - Quality: Clear, well-lit images
   - Recommended: 50-100 images per class for training

## Usage

### 1. Training the Model

```bash
python train.py
```

This will:
- Train the model using your dataset
- Save the model as `model.h5`
- Convert and save the TFLite model as `model.tflite`

### 2. Image Prediction

```bash
python predict.py path/to/image.jpg [--confidence_threshold 0.75]
```

Example:
```bash
python predict.py testImage/test.jpg
```

### 3. Real-time Webcam Prediction

```bash
python webcam_predict.py
```

- Press 'q' to quit the webcam feed

## Model Details

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 150x150x3 (RGB images)
- **Output**: 3 classes (jute, rice, wheat)
- **Confidence Threshold**: 75% (configurable)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU Issues**
   - Install CUDA toolkit
   - Install cuDNN
   - Or use CPU version of TensorFlow

3. **Webcam Not Working**
   - Check webcam permissions
   - Verify webcam is not in use by another application

4. **Low Accuracy**
   - Increase training data
   - Improve image quality
   - Adjust confidence threshold

## Performance Optimization

1. **For Better Accuracy**:
   - Use high-quality images
   - Increase training dataset size
   - Adjust confidence threshold

2. **For Faster Inference**:
   - Use GPU if available
   - Reduce input image size
   - Use TFLite model

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the MobileNetV2 model
- OpenCV for image processing capabilities
- All contributors and users of this project

## Contact

For support or questions, please:
- Open an issue in the repository
- Contact the development team
- Check the documentation

## Future Improvements

- [ ] Add support for more crop types
- [ ] Implement batch processing
- [ ] Add API endpoints
- [ ] Improve model accuracy
- [ ] Add mobile app integration

## Version History

- v1.0.0
  - Initial release
  - Basic crop classification
  - Webcam support
  - TFLite conversion 
