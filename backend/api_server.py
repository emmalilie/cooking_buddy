from flask import Flask, request, jsonify
from flask_cors import CORS
from agent import FoodAgent
import base64
import os
import traceback

app = Flask(__name__)
CORS(app)

agent = FoodAgent(
    model_path='best_model.pth',
    nutrition_csv='cleaned_nutrition.csv',
    recipes_csv='filtered_recipes.csv',
    max_recipes=2
)

@app.route('/analyze', methods=['POST'])
def analyze():
    temp_path = 'temp_upload.jpg'
    try:
        print("[API] Received request")
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            print("[API] No image data")
            return jsonify({'error': 'No image provided'}), 400
        
        print("[API] Decoding image...")
        img_bytes = base64.b64decode(image_data)
        
        print("[API] Saving temp file...")
        with open(temp_path, 'wb') as f:
            f.write(img_bytes)
        
        print("[API] Running analysis...")
        result = agent.analyse(temp_path)
        
        print(f"[API] Success! Food: {result['food']}")
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        print(f"[API] ERROR: {str(e)}")
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Cooking Buddy API Server...")
    app.run(debug=True, port=5000)
