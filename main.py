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

# Multiple Clarifai models
CLARIFAI_MODELS = [
    {"id": "food-item-recognition", "weight": 1.0},
    {"id": "general", "weight": 0.8},
]

def preprocess_image(image_bytes):
    """Enhance image for better recognition"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # More aggressive enhancement for better AI detection
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG', quality=95)
        enhanced_bytes = output_buffer.getvalue()
        
        return enhanced_bytes
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return image_bytes

def query_clarifai_models(image_base64):
    """Query all Clarifai models simultaneously"""
    clarifai_results = []
    
    headers = {
        "Authorization": f"Key {CLARIFAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for model_info in CLARIFAI_MODELS:
        model_id = model_info["id"]
        weight = model_info["weight"]
        
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
            print(f"🔍 Querying Clarifai model: {model_id}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                outputs = result.get('outputs', [])
                if outputs:
                    concepts = outputs[0].get('data', {}).get('concepts', [])
                    for concept in concepts:
                        if concept['value'] > 0.15:  # Filter low confidence
                            clarifai_results.append({
                                'name': concept['name'],
                                'confidence': round(concept['value'] * 100 * weight, 1),
                                'source': 'clarifai',
                                'model': model_id
                            })
                print(f"✅ {model_id} returned {len(concepts)} concepts")
            else:
                print(f"❌ {model_id} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model_id} error: {e}")
    
    return clarifai_results

def query_openai_vision(image_base64):
    """Use OpenAI Vision for comprehensive food detection"""
    if not openai_client:
        print("❌ OpenAI client not available")
        return []
    
    try:
        print("🧠 Querying OpenAI Vision...")
        
        prompt = """
        Analyze this refrigerator/kitchen image and identify ALL food ingredients that could be used for cooking.
        
        Focus on detecting:
        - Fresh produce (fruits, vegetables, herbs)
        - Dairy products (milk, cheese, eggs, yogurt, butter)
        - Meat and proteins (chicken, beef, fish, tofu)
        - Pantry items (bread, pasta, rice, sauces)
        - Beverages (juice, milk, water)
        - Condiments and seasonings
        - Leftovers and prepared foods
        - Canned/packaged goods
        
        Be very thorough and specific. Look carefully at containers, packages, and fresh items.
        Consider items that might be partially visible or in the background.
        
        Return ONLY a JSON array of objects with this exact format:
        [
            {"name": "milk", "confidence": 95},
            {"name": "eggs", "confidence": 90},
            {"name": "carrots", "confidence": 85}
        ]
        
        Include confidence scores from 60-99 based on how clearly visible and certain each item is.
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # ✅ FIXED: Changed from "gpt-4-vision-preview"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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
            max_tokens=1500,
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        print(f"🧠 OpenAI raw response: {content[:200]}...")
        
        # Parse JSON response
        try:
            # Try to find JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                items = json.loads(json_str)
                
                openai_results = []
                for item in items:
                    if isinstance(item, dict) and 'name' in item:
                        openai_results.append({
                            'name': item['name'],
                            'confidence': item.get('confidence', 80),
                            'source': 'openai'
                        })
                
                print(f"✅ OpenAI detected {len(openai_results)} items")
                return openai_results
            else:
                print("❌ Could not parse OpenAI JSON response")
                return []
                
        except json.JSONDecodeError as e:
            print(f"❌ OpenAI JSON parse error: {e}")
            return []
        
    except Exception as e:
        print(f"❌ OpenAI Vision error: {e}")
        return []

def merge_duplicate_items(all_items):
    """Smart merging of duplicate items from different AI sources"""
    merged = {}
    
    for item in all_items:
        name = item['name'].lower().strip()
        
        # Handle common variations and synonyms
        name_variations = {
            'egg': 'eggs',
            'carrot': 'carrots',
            'tomato': 'tomatoes',
            'potato': 'potatoes',
            'onion': 'onions',
            'bell pepper': 'pepper',
            'red pepper': 'pepper',
            'green pepper': 'pepper',
            'whole milk': 'milk',
            '2% milk': 'milk',
            'skim milk': 'milk',
            'cheddar cheese': 'cheese',
            'mozzarella cheese': 'cheese',
            'chicken breast': 'chicken',
            'ground beef': 'beef',
            'white bread': 'bread',
            'wheat bread': 'bread',
        }
        
        # Normalize name
        normalized_name = name_variations.get(name, name)
        
        if normalized_name in merged:
            existing = merged[normalized_name]
            
            # Boost confidence when multiple AIs detect the same item
            if item['source'] != existing['source']:
                # Different sources agree - high confidence boost
                confidence_boost = 1.4
                existing['sources'].append(item['source'])
                existing['source'] = 'both_ais'
            else:
                # Same source, multiple models - moderate boost
                confidence_boost = 1.2
            
            # Take the higher confidence and boost it
            new_confidence = max(existing['confidence'], item['confidence']) * confidence_boost
            existing['confidence'] = min(99, round(new_confidence, 1))
            
            # Use the more descriptive name
            if len(item['name']) > len(existing['name']):
                existing['name'] = item['name']
                
        else:
            merged[normalized_name] = {
                'name': item['name'],
                'confidence': item['confidence'],
                'source': item['source'],
                'sources': [item['source']],
                'normalized_key': normalized_name
            }
    
    # Convert back to list and sort by confidence
    final_results = list(merged.values())
    final_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return final_results

@app.post("/detect_foods")
async def detect_foods(file: UploadFile = File(...)):
    try:
        print(f"\n🚀 Starting AI-driven analysis for: {file.filename}")
        
        # Read and preprocess image
        contents = await file.read()
        enhanced_contents = preprocess_image(contents)
        image_base64 = base64.b64encode(enhanced_contents).decode("utf-8")
        
        # Run both AI systems simultaneously
        all_detected_items = []
        
        # Get Clarifai results
        clarifai_items = query_clarifai_models(image_base64)
        all_detected_items.extend(clarifai_items)
        
        # Get OpenAI results
        openai_items = query_openai_vision(image_base64)
        all_detected_items.extend(openai_items)
        
        print(f"\n📊 Raw Detection Results:")
        print(f"Clarifai detected: {len(clarifai_items)} items")
        print(f"OpenAI detected: {len(openai_items)} items")
        print(f"Total raw items: {len(all_detected_items)}")
        
        # Merge duplicates and boost confidence for agreements
        merged_results = merge_duplicate_items(all_detected_items)
        
        print(f"Final merged items: {len(merged_results)}")
        
        # Categorize by confidence levels
        high_confidence = [item for item in merged_results if item['confidence'] >= 75]
        medium_confidence = [item for item in merged_results if 50 <= item['confidence'] < 75]
        low_confidence = [item for item in merged_results if 30 <= item['confidence'] < 50]
        
        # Count AI agreements
        ai_agreements = len([item for item in merged_results if item['source'] == 'both_ais'])
        
        # Log top results
        print(f"\n🎯 Top 10 Results:")
        for i, item in enumerate(merged_results[:10]):
            sources_indicator = "🤖🤖" if item['source'] == 'both_ais' else ("🧠" if item['source'] == 'openai' else "👁️")
            print(f"  {i+1}. {item['name']} ({item['confidence']}%) {sources_indicator}")
        
        return {
            "items": [item["name"] for item in merged_results],
            "food_items": merged_results,
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "total_food_detected": len(merged_results),
            "ai_agreements": ai_agreements,
            "agreement_percentage": round((ai_agreements / len(merged_results) * 100), 1) if merged_results else 0,
            "clarifai_detected": len(clarifai_items),
            "openai_detected": len(openai_items),
            "raw_total": len(all_detected_items),
            "duplicates_merged": len(all_detected_items) - len(merged_results),
            "ai_enhanced": True,
            "analysis_method": "dual_ai_enhanced"
        }

    except Exception as e:
        print(f"❌ Error in detect_foods: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest_meals")
async def suggest_meals(ingredients: list):
    """Enhanced meal suggestions using detected ingredients"""
    return {
        "message": "Spoonacular integration coming soon!",
        "ingredients_received": len(ingredients),
        "detected_ingredients": ingredients,
        "suggested_recipes": [
            {
                "name": "Recipe suggestions will appear here",
                "missing_ingredients": 0,
                "ready_to_cook": True
            }
        ]
    }

@app.get("/")
async def root():
    ai_status = "enabled" if openai_client else "disabled"
    return {
        "message": "MealMapper Dual-AI Enhanced API",
        "version": "4.0",
        "ai_systems": {
            "clarifai": "enabled",
            "openai_vision": ai_status
        },
        "features": ["smart_duplicate_merging", "confidence_boosting", "ai_agreement_detection"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "clarifai_models": len(CLARIFAI_MODELS),
        "openai_available": openai_client is not None,
        "ai_systems": 2 if openai_client else 1
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check AI system status"""
    return {
        "clarifai_key_set": CLARIFAI_API_KEY != "your_clarifai_key_here",
        "openai_key_set": OPENAI_API_KEY != "your_openai_key_here",
        "openai_client_ready": openai_client is not None,
        "models_configured": len(CLARIFAI_MODELS)
    }