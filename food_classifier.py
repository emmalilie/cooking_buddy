import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class FoodClassifier:
    def __init__(self, data_dir, batch_size=32, learning_rate=0.0001, num_epochs=10, load_model_path=None):
        """
        Initialize the food classifier
        
        Args:
            data_dir: Path to directory containing 'train' and 'validation' folders
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            load_model_path: Path to existing model to continue training (optional)
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model_path = load_model_path
        
        print(f"Using device: {self.device}")
        
        # Data transformations
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        self.train_dataset = datasets.ImageFolder(
            root=self.data_dir / 'train',
            transform=self.train_transforms
        )
        
        self.val_dataset = datasets.ImageFolder(
            root=self.data_dir / 'val',
            transform=self.val_transforms
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Get class names
        self.class_names = self.train_dataset.classes
        self.num_classes = len(self.class_names)
        
        print(f"\nDataset Information:")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Load existing model or build new one
        if self.load_model_path and os.path.exists(self.load_model_path):
            print(f"\n{'='*50}")
            print(f"Loading existing model from: {self.load_model_path}")
            print(f"{'='*50}")
            self._load_existing_model(self.load_model_path)
        else:
            if self.load_model_path:
                print(f"\nWarning: Model path '{self.load_model_path}' not found. Building new model.")
            # Initialize new model
            self.model = self._build_model()
            
            # Loss function and optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3
            )
    
    def _build_model(self):
        """Build the model using transfer learning with ResNet50"""
        # Load pre-trained ResNet50
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        # for param in model.parameters():
            # param.requires_grad = False
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        
        # Replace the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        model = model.to(self.device)
        return model
    
    def _load_existing_model(self, filepath):
        """Load existing model and continue from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            print(f"Loaded training history: {len(self.history['train_loss'])} epochs completed")
            if self.history['val_acc']:
                print(f"Previous best validation accuracy: {max(self.history['val_acc']):.2f}%")
        
        # Verify class names match
        if 'class_names' in checkpoint:
            saved_classes = checkpoint['class_names']
            if saved_classes != self.class_names:
                raise ValueError(
                    f"Class mismatch! Saved model has {len(saved_classes)} classes: {saved_classes}\n"
                    f"Current dataset has {len(self.class_names)} classes: {self.class_names}"
                )
        
        # Build model architecture
        self.model = self._build_model()
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded successfully")
        
        # Initialize optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Optimizer state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("  Continuing with fresh optimizer...")
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Scheduler state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        print(f"\nReady to continue training from previous checkpoint!")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_loss = running_loss / len(self.train_dataset)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_dataset)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Train the model"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        # Get best validation accuracy from history
        best_val_acc = max(self.history['val_acc']) if self.history['val_acc'] else 0.0
        starting_epoch = len(self.history['train_loss'])
        
        if starting_epoch > 0:
            print(f"Continuing from epoch {starting_epoch + 1}")
            print(f"Current best validation accuracy: {best_val_acc:.2f}%")
        
        for epoch in range(self.num_epochs):
            actual_epoch = starting_epoch + epoch + 1
            print(f"\nEpoch {actual_epoch}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"✓ Saved new best model (Val Acc: {val_acc:.2f}%)")
        
        print("\n" + "="*50)
        print(f"Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"Total epochs trained: {len(self.history['train_loss'])}")
        print("="*50)
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(self.history['val_loss'], label='Validation Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()
    
    def save_model(self, filepath='food_classifier.pth'):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'class_names': self.class_names,
            'history': self.history
        }, filepath)
    
    def load_model(self, filepath='food_classifier.pth'):
        """Load a saved model (for inference only - use load_model_path in __init__ for continuing training)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Model loaded from {filepath}")
    
    def predict(self, image_path):
        """Predict the class of a single image"""
        from PIL import Image
        
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transforms(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score

    def evaluate_model(self, dataset_type='val'):
        """
        Evaluate model accuracy and generate confusion matrix
        
        Args:
            dataset_type: 'val' or 'train'
        """
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        loader = self.val_loader if dataset_type == 'val' else self.train_loader
        dataset = self.val_dataset if dataset_type == 'val' else self.train_dataset
        
        print(f"\nEvaluating on {dataset_type} set...")
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Overall accuracy
        accuracy = accuracy_score(all_labels, all_preds) * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        
        # Per-class report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=axes[0])
        axes[0].set_title(f'Confusion Matrix (Counts)\nAccuracy: {accuracy:.2f}%')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # Normalized (percentages per true class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=axes[1])
        axes[1].set_title('Confusion Matrix (% per True Class)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        save_path = f'confusion_matrix_{dataset_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
        
        # Print most confused pairs
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)
        top_confused = np.dstack(np.unravel_index(np.argsort(cm_no_diag.ravel())[::-1], cm.shape))[0]
        
        print("\nTop 5 Most Confused Class Pairs:")
        for i, (true_idx, pred_idx) in enumerate(top_confused[:5]):
            count = cm[true_idx, pred_idx]
            if count > 0:
                print(f"  {i+1}. True: '{self.class_names[true_idx]}' → Predicted: '{self.class_names[pred_idx]}' ({count} times)")
        
        return accuracy, cm


# Example usage
if __name__ == "__main__":
    # Set your data directory path here
    DATA_DIR = r"C:\Users\emmal\OneDrive\Desktop\cooking_buddy\food_images\ingredients"  # Change this to your folder path
    
    # ========================================
    # Option 1: Start fresh training
    # ========================================
    # classifier = FoodClassifier(
    #     data_dir=DATA_DIR,
    #     batch_size=32,
    #     learning_rate=0.001,
    #     num_epochs=10
    # )
    
    # ========================================
    # Option 2: Continue training from existing model
    # ========================================
    #classifier = FoodClassifier(
        #data_dir=DATA_DIR,
        #batch_size=32,
        ##learning_rate=0.000001,
        #num_epochs=2,
        #load_model_path="food_classifier_final.pth"  # Model to continue from
    #)

    classifier = FoodClassifier(
        data_dir=DATA_DIR,
        batch_size=32,
        learning_rate=0.000001,
        num_epochs=0,  # 0 epochs = no training, just evaluate
        load_model_path="food_classifier_final.pth"
    )   

