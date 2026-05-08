# Food Image Classifier

A deep learning model for classifying food images using PyTorch and transfer learning with ResNet50.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
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

## Model Architecture

- Base: ResNet50 (pre-trained on ImageNet)
- Custom classifier head with dropout for regularization
- Supports any number of food classes

## Output Files
- `best_model.pth` - Model with best validation accuracy
- `food_classifier_final.pth` - Final model after all epochs
- `training_history.png` - Plots showing loss and accuracy over time
