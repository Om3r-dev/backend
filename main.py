from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os
from PIL import Image, ImageEnhance
import io
import json
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys - Set these as environment variables
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY", "your_clarifai_key_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_key_here")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY != "your_openai_key_here" else None

USER_ID = "clarifai"
APP_ID = "main"

# Multiple models for better accuracy
MODELS = [
    {"id": "food-item-recognition", "weight": 1.0},
    {"id": "general", "weight": 0.8},
]

# Food-related keywords to prioritize
FOOD_KEYWORDS = {
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'lemon', 'lime', 'pear', 'peach', 'cherry', 'berries'],
    'vegetables': ['carrot', 'broccoli', 'lettuce', 'tomato', 'onion', 'potato', 'pepper', 'cucumber', 'spinach', 'celery', 'corn', 'beans'],
    'dairy': ['milk', 'cheese', 'butter', 'yogurt', 'cream', 'egg', 'eggs', 'sour cream'],
    'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'salmon', 'bacon', 'ham', 'sausage'],
    'pantry': ['bread', 'pasta', 'rice', 'flour', 'sugar', 'salt', 'oil', 'vinegar', 'sauce', 'condiment'],
    'beverages': ['juice', 'soda', 'water', 'beer', 'wine', 'coffee', 'tea', 'milk']
}

def preprocess_image(image_bytes):
    """Enhance image for better recognition"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Enhance brightness and contrast
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG', quality=95)
        enhanced_bytes = output_buffer.getvalue()
        
        return enhanced_bytes
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return image_bytes

def is_food_related(concept_name):
    """Check if a concept is food-related"""
    concept_lower = concept_name.lower()
    
    for category, keywords in FOOD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in concept_lower or concept_lower in keyword:
                return True, category
    
    food_terms = ['food', 'ingredient', 'meal', 'dish', 'recipe', 'cooking', 'fresh', 'organic']
    for term in food_terms:
        if term in concept_lower:
            return True, 'general_food'
    
    return False, None

def query_clarifai_model(image_base64, model_id):
    """Query a specific Clarifai model"""
    headers = {
        "Authorization": f"Key {CLARIFAI_API_KEY}",
        "Content-Type": "application/json"
    }

    url = f"https://api.clarifai.com/v2/users/{USER_ID}/apps/{APP_ID}/models/{model_id}/outputs"
    
    payload = {
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": image_base64
                    }
                }
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error querying model {model_id}: {e}")
        return None

def analyze_with_openai_vision(image_base64, clarifai_results=None):
    """Use OpenAI Vision to analyze the image and validate/enhance results"""
    if not openai_client:
        return None
    
    try:
        # Prepare the prompt
        base_prompt = """
        Analyze this refrigerator/kitchen image and identify food ingredients that could be used for cooking meals. 
        Focus on:
        1. Fresh produce (fruits, vegetables)
        2. Dairy products (milk, cheese, eggs, yogurt)
        3. Meat/protein items
        4. Pantry staples visible
        5. Condiments and sauces
        
        Please be specific and practical - only list items that are clearly visible and would be useful for meal planning.
        Return your response as a JSON object with this format:
        {
            "food_items": [
                {"name": "item_name", "confidence": 95, "category": "vegetables"},
                ...
            ],
            "analysis": "Brief description of what you see"
        }
        """
        
        # Add Clarifai comparison if available
        if clarifai_results:
            clarifai_items = [item['name'] for item in clarifai_results if item.get('is_food')]
            base_prompt += f"\n\nFor comparison, another AI detected these items: {clarifai_items}. Please validate and provide your own analysis."

        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": base_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content
        print(f"OpenAI raw response: {content}")
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (might be wrapped in markdown)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
        except:
            pass
        
        # Fallback: parse as plain text
        return {
            "food_items": [],
            "analysis": content,
            "raw_response": True
        }
        
    except Exception as e:
        print(f"OpenAI Vision API error: {e}")
        return None

def combine_ai_results(clarifai_results, openai_result):
    """Intelligently combine results from both AI systems"""
    if not openai_result:
        return clarifai_results
    
    # Start with Clarifai results
    combined = {}
    
    # Add Clarifai results
    for item in clarifai_results:
        name = item['name'].lower()
        combined[name] = {
            'name': item['name'],
            'confidence': item['confidence'],
            'source': 'clarifai',
            'is_food': item['is_food'],
            'category': item.get('category', 'unknown')
        }
    
    # Add/enhance with OpenAI results
    if 'food_items' in openai_result:
        for item in openai_result['food_items']:
            name = item['name'].lower()
            confidence = item.get('confidence', 80)
            
            if name in combined:
                # Boost confidence if both AIs agree
                combined[name]['confidence'] = min(95, combined[name]['confidence'] * 1.3)
                combined[name]['source'] = 'both_ais'
            else:
                # Add new item from OpenAI
                combined[name] = {
                    'name': item['name'],
                    'confidence': confidence,
                    'source': 'openai',
                    'is_food': True,
                    'category': item.get('category', 'general_food')
                }
    
    # Convert back to list and sort by confidence
    final_results = list(combined.values())
    final_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return final_results

@app.post("/detect_foods")
async def detect_foods(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        enhanced_contents = preprocess_image(contents)
        image_bytes_base64 = base64.b64encode(enhanced_contents).decode("utf-8")

        print(f"Processing image: {file.filename}")
        
        # Step 1: Query Clarifai models
        clarifai_results = []
        model_results = []
        
        for model_info in MODELS:
            model_id = model_info["id"]
            weight = model_info["weight"]
            
            print(f"Querying Clarifai model: {model_id}")
            result = query_clarifai_model(image_bytes_base64, model_id)
            
            if result:
                model_results.append((result, weight))

        # Process Clarifai results
        if model_results:
            combined_concepts = {}
            
            for model_data, weight in model_results:
                outputs = model_data.get('outputs', [])
                if not outputs:
                    continue
                    
                concepts = outputs[0].get('data', {}).get('concepts', [])
                
                for concept in concepts:
                    name = concept['name']
                    confidence = concept['value'] * weight
                    is_food, category = is_food_related(name)
                    
                    if is_food and confidence > 0.15:
                        if name in combined_concepts:
                            combined_concepts[name]['confidence'] = max(
                                combined_concepts[name]['confidence'], 
                                confidence
                            )
                        else:
                            combined_concepts[name] = {
                                'name': name,
                                'confidence': round(confidence * 100, 1),
                                'is_food': is_food,
                                'category': category
                            }
            
            clarifai_results = list(combined_concepts.values())
            clarifai_results.sort(key=lambda x: x['confidence'], reverse=True)

        # Step 2: Determine if we need AI backup
        max_confidence = max([r['confidence'] for r in clarifai_results], default=0)
        should_use_ai = (
            max_confidence < 60 or  # Low confidence
            len(clarifai_results) < 3 or  # Few items detected
            openai_client is not None  # AI is available
        )

        openai_result = None
        if should_use_ai:
            print("Using OpenAI Vision for enhanced analysis...")
            openai_result = analyze_with_openai_vision(image_bytes_base64, clarifai_results)

        # Step 3: Combine results intelligently
        if openai_result:
            final_results = combine_ai_results(clarifai_results, openai_result)
        else:
            final_results = clarifai_results

        # Step 4: Categorize final results
        high_confidence = [item for item in final_results if item["confidence"] > 70]
        medium_confidence = [item for item in final_results if 40 <= item["confidence"] <= 70]
        low_confidence = [item for item in final_results if 20 <= item["confidence"] < 40]

        response_data = {
            "items": [item["name"] for item in final_results],
            "food_items": final_results,
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "total_food_detected": len(final_results),
            "ai_enhanced": openai_result is not None,
            "max_confidence": max_confidence,
            "analysis_method": "ai_enhanced" if openai_result else "clarifai_only"
        }

        if openai_result and 'analysis' in openai_result:
            response_data["ai_analysis"] = openai_result['analysis']

        return response_data

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest_meals")
async def suggest_meals(ingredients: list):
    """Placeholder for Spoonacular integration"""
    return {
        "message": "Meal suggestions coming soon!",
        "ingredients_received": ingredients,
        "suggested_recipes": []
    }

@app.get("/")
async def root():
    ai_status = "enabled" if openai_client else "disabled"
    return {
        "message": "MealMapper AI-Enhanced API is running",
        "version": "3.0",
        "ai_backup": ai_status
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "clarifai_models": len(MODELS),
        "ai_backup": openai_client is not None
    }