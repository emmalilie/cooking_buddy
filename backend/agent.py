from groq import Groq
import pandas as pd
from nutrition_predictor import NutritionPredictor

GROQ_API_KEY = ""


class FoodAgent:
    def __init__(self, model_path, nutrition_csv='cleaned_nutrition.csv', recipes_csv='filtered_recipes.csv', max_recipes=5):
        self.predictor = NutritionPredictor(model_path, nutrition_csv)
        self.client = Groq(api_key=GROQ_API_KEY)
        self.max_recipes = max_recipes

        df = pd.read_csv(recipes_csv)
        df['ingredients_list'] = df['ingredients'].apply(
            lambda x: [i.strip().lower() for i in x.split(',')]
        )
        self.recipes = df

    def find_recipes(self, food, max_recipes=None):
        max_recipes = max_recipes or self.max_recipes
        food_lower = food.lower()

        matched = self.recipes[
            self.recipes['ingredients_list'].apply(
                lambda ings: any(food_lower in ing for ing in ings)
            )
        ].head(max_recipes)

        return matched

    def analyse(self, image_path, max_recipes=None):
        max_recipes = max_recipes or self.max_recipes

        # Step 1: Detect food + calories
        food, confidence, serving, calories = self.predictor.predict(image_path)

        # Step 2: Find matching recipes
        matches = self.find_recipes(food, max_recipes=max_recipes)

        if matches.empty:
            recipes_text = "No recipes found for this ingredient."
            recipes_list = []
        else:
            recipes_text = matches[['id', 'name', 'ingredients', 'steps', 'servings', 'serving_size', 'tags']].to_string(index=False)
            recipes_list = matches[['id', 'name', 'servings', 'serving_size', 'tags']].to_dict(orient='records')

        # Step 3: Ask Groq to summarise
        prompt = f"""I detected the following ingredient from a food image:

- Food: {food}
- Confidence: {confidence:.1f}%
- Serving size: {serving}
- Calories per serving: {calories} cal

Here are {len(matches)} recipes from my database that include this ingredient:

{recipes_text}

Please give a short, friendly summary of:
1. The nutritional value of {food}
2. A brief highlight of each recipe and which would suit different occasions (quick weeknight meal, special dinner, healthy option, etc.)
Keep it concise and practical."""

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
            stop=None
        )

        summary = completion.choices[0].message.content

        return {
            "food": food,
            "confidence": float(confidence),
            "serving": str(serving),
            "calories": int(calories) if isinstance(calories, (int, float)) else str(calories),
            "recipes": recipes_list,
            "summary": summary
        }


if __name__ == "__main__":
    agent = FoodAgent(
        model_path='best_model.pth',
        nutrition_csv='cleaned_nutrition.csv',
        recipes_csv='filtered_recipes.csv',
        max_recipes=2
    )

    result = agent.analyse('test_image.jpg')

    print("=" * 50)
    print(f"Food: {result['food']} | {result['calories']} cal per {result['serving']}")
    print(f"\nTop {len(result['recipes'])} matching recipes:")
    for r in result['recipes']:
        print(f"  [{r['id']}] {r['name']} — {r['servings']} servings, {r['serving_size']}")
    print("\nSummary:")
    print(result['summary'])
