# Running Cooking Buddy Web App

## Quick Start

1. **Start the backend server:**
   - Double-click `start_server.bat` OR
   - Run: `python api_server.py`
   - Server will start at http://localhost:5000

2. **Open the frontend:**
   - Open `frontend/website.html` in your browser
   - Upload a food image
   - Click "Find My Recipes!"

## Requirements

Make sure you have installed:
```bash
pip install flask flask-cors groq torch torchvision pandas pillow
```

## Files Overview

- `api_server.py` - Flask backend API
- `agent.py` - Main food analysis agent
- `nutrition_predictor.py` - Food classification model
- `frontend/website.html` - Web interface
- `best_model.pth` - Trained model weights
- `cleaned_nutrition.csv` - Nutrition database
- `filtered_recipes.csv` - Recipe database
