from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os
from PIL import Image, ImageEnhance
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use environment variable for API key
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY", "7a918c1528ad460694d606386c472a63")
USER_ID = "clarifai"
APP_ID = "main"

# Multiple models for better accuracy
MODELS = [
    {"id": "food-item-recognition", "weight": 1.0},
    {"id": "general", "weight": 0.8},
    {"id": "food-item-v1", "weight": 0.9}  # Try this model too
]

# Food-related keywords to prioritize
FOOD_KEYWORDS = {
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'lemon', 'lime', 'pear', 'peach', 'cherry'],
    'vegetables': ['carrot', 'broccoli', 'lettuce', 'tomato', 'onion', 'potato', 'pepper', 'cucumber', 'spinach', 'celery'],
    'dairy': ['milk', 'cheese', 'butter', 'yogurt', 'cream', 'egg', 'eggs'],
    'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'salmon', 'bacon', 'ham'],
    'pantry': ['bread', 'pasta', 'rice', 'flour', 'sugar', 'salt', 'oil', 'vinegar', 'sauce'],
    'beverages': ['juice', 'soda', 'water', 'beer', 'wine', 'coffee', 'tea']
}

def preprocess_image(image_bytes):
    """Enhance image for better recognition"""
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Enhance brightness and contrast
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)  # Increase brightness by 20%
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # Increase contrast by 10%
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG', quality=95)
        enhanced_bytes = output_buffer.getvalue()
        
        return enhanced_bytes
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return image_bytes  # Return original if preprocessing fails

def is_food_related(concept_name):
    """Check if a concept is food-related"""
    concept_lower = concept_name.lower()
    
    # Check against our food keywords
    for category, keywords in FOOD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in concept_lower or concept_lower in keyword:
                return True, category
    
    # Additional food-related terms
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

def combine_model_results(model_results):
    """Combine results from multiple models with weighting"""
    combined_concepts = {}
    
    for model_data, weight in model_results:
        if not model_data or 'outputs' not in model_data:
            continue
            
        outputs = model_data.get('outputs', [])
        if not outputs:
            continue
            
        concepts = outputs[0].get('data', {}).get('concepts', [])
        
        for concept in concepts:
            name = concept['name']
            confidence = concept['value'] * weight
            
            if name in combined_concepts:
                # Average the confidences, giving more weight to higher confidence
                existing_conf = combined_concepts[name]['confidence']
                combined_concepts[name]['confidence'] = max(existing_conf, confidence)
                combined_concepts[name]['count'] += 1
            else:
                combined_concepts[name] = {
                    'confidence': confidence,
                    'count': 1,
                    'name': name
                }
    
    # Convert back to list and boost food-related items
    final_concepts = []
    for name, data in combined_concepts.items():
        is_food, category = is_food_related(name)
        confidence = data['confidence']
        
        # Boost food-related items
        if is_food:
            confidence *= 1.3  # 30% boost for food items
        
        # Boost items detected by multiple models
        if data['count'] > 1:
            confidence *= 1.2  # 20% boost for multi-model detection
        
        final_concepts.append({
            'name': name,
            'value': min(confidence, 1.0),  # Cap at 1.0
            'is_food': is_food,
            'category': category,
            'model_count': data['count']
        })
    
    return sorted(final_concepts, key=lambda x: x['value'], reverse=True)

@app.post("/detect_foods")
async def detect_foods(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        enhanced_contents = preprocess_image(contents)
        image_bytes_base64 = base64.b64encode(enhanced_contents).decode("utf-8")

        print(f"Processing image: {file.filename}")
        
        # Query multiple models
        model_results = []
        for model_info in MODELS:
            model_id = model_info["id"]
            weight = model_info["weight"]
            
            print(f"Querying model: {model_id}")
            result = query_clarifai_model(image_bytes_base64, model_id)
            
            if result:
                model_results.append((result, weight))
                print(f"✓ {model_id} responded successfully")
            else:
                print(f"✗ {model_id} failed")

        if not model_results:
            raise HTTPException(status_code=500, detail="All models failed to respond")

        # Combine and process results
        combined_concepts = combine_model_results(model_results)
        
        print(f"Combined concepts: {len(combined_concepts)}")
        for concept in combined_concepts[:10]:
            print(f"  {concept['name']}: {concept['value']:.3f} (food: {concept['is_food']})")
        
        # Filter and categorize results
        food_items = []
        non_food_items = []
        
        for concept in combined_concepts:
            confidence = concept['value']
            
            if confidence > 0.15:  # Lower threshold but better filtering
                item = {
                    "name": concept["name"],
                    "confidence": round(confidence * 100, 1),
                    "is_food": concept["is_food"],
                    "category": concept.get("category", "unknown"),
                    "models": concept["model_count"]
                }
                
                if concept["is_food"]:
                    food_items.append(item)
                else:
                    non_food_items.append(item)
        
        # Separate by confidence for food items
        high_confidence = [item for item in food_items if item["confidence"] > 60]
        medium_confidence = [item for item in food_items if 30 <= item["confidence"] <= 60]
        low_confidence = [item for item in food_items if 15 <= item["confidence"] < 30]
        
        # Include some high-confidence non-food items that might be food containers
        container_terms = ['bottle', 'jar', 'container', 'package', 'carton', 'can', 'box']
        relevant_non_food = []
        for item in non_food_items[:10]:  # Top 10 non-food items
            if any(term in item['name'].lower() for term in container_terms):
                relevant_non_food.append(item)
        
        return {
            "items": [item["name"] for item in food_items],
            "food_items": food_items,
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "containers_detected": relevant_non_food,
            "total_food_detected": len(food_items),
            "models_used": len([r for r in model_results if r]),
            "image_enhanced": True
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "MealMapper Enhanced API is running", "version": "2.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models": len(MODELS)}

@app.get("/models")
async def get_models():
    return {"models": MODELS, "food_categories": list(FOOD_KEYWORDS.keys())}