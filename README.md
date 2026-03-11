# Food Image Classifier

A deep learning model for classifying food images using PyTorch and transfer learning with ResNet50.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Your dataset should be organized as follows:
```
your_dataset_folder/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── validation/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    ├── class2/
    │   └── ...
    └── ...
```

## Usage

### Basic Training

```python
from food_classifier import FoodClassifier

# Initialize the classifier
classifier = FoodClassifier(
    data_dir="/path/to/your/dataset",
    batch_size=32,
    learning_rate=0.001,
    num_epochs=10
)

# Train the model
classifier.train()

# Plot training history
classifier.plot_training_history('training_history.png')

# Save the model
classifier.save_model('my_model.pth')
```

### Making Predictions

```python
from food_classifier import FoodClassifier

# Initialize classifier
classifier = FoodClassifier(data_dir="/path/to/your/dataset")

# Load trained model
classifier.load_model('best_model.pth')

# Predict on a new image
predicted_class, confidence = classifier.predict('/path/to/image.jpg')
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

### Custom Configuration

```python
classifier = FoodClassifier(
    data_dir="/path/to/your/dataset",
    batch_size=64,          # Larger batch size (if you have enough GPU memory)
    learning_rate=0.0001,   # Lower learning rate for fine-tuning
    num_epochs=20           # More epochs for better results
)
```

## Features

- **Transfer Learning**: Uses pre-trained ResNet50 for better accuracy with less data
- **Data Augmentation**: Applies random transformations to training images
- **Automatic Best Model Saving**: Saves the model with best validation accuracy
- **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
- **Training Visualization**: Plots loss and accuracy curves
- **Easy Prediction**: Simple interface for classifying new images

## Model Architecture

- Base: ResNet50 (pre-trained on ImageNet)
- Custom classifier head with dropout for regularization
- Supports any number of food classes

## Output Files

After training, you'll get:
- `best_model.pth` - Model with best validation accuracy
- `food_classifier_final.pth` - Final model after all epochs
- `training_history.png` - Plots showing loss and accuracy over time

## Tips for Better Results

1. **More data**: Aim for at least 100+ images per class
2. **Balanced classes**: Try to have similar numbers of images for each class
3. **Image quality**: Use clear, well-lit images
4. **Longer training**: Increase `num_epochs` for better results (15-25 epochs)
5. **GPU usage**: The model will automatically use GPU if available

## Troubleshooting

- **Out of memory**: Reduce `batch_size` to 16 or 8
- **Slow training**: Set `num_workers=0` in the DataLoader if you have issues
- **Poor accuracy**: Try training for more epochs or adjust learning rate

# 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