import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


class NutritionPredictor:
    def __init__(self, model_path, nutrition_csv='food_nutrition.csv'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Build food -> (serving, calories) lookup
        df = pd.read_csv(nutrition_csv)
        df = df.drop_duplicates(subset='Food')
        df['calories_clean'] = df['Calories'].str.replace(' cal', '', case=False).str.strip().astype(int)
        self.nutrition = {
            row['Food'].lower(): {'serving': row['Serving'], 'calories': row['calories_clean']}
            for _, row in df.iterrows()
        }

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.class_names))
        )
        return model.to(self.device)

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        tensor = self.transforms(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, idx = torch.max(probs, 1)

        food = self.class_names[idx.item()]
        info = self.nutrition.get(food.lower(), None)

        serving = info['serving'] if info else 'N/A'
        calories = info['calories'] if info else 'N/A'

        return food, confidence.item() * 100, serving, calories


if __name__ == "__main__":
    predictor = NutritionPredictor(
        model_path='best_model.pth',
        nutrition_csv='cleaned_nutrition.csv'
    )

    food, confidence, serving, calories = predictor.predict('test_image.jpg')
    print(f"{food} ({confidence:.1f}%) — {calories} cal per {serving}")