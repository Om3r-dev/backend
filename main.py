from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os
from PIL import Image, ImageEnhance
import io
import json
from openai import OpenAI
import requests
from typing import List, Dict, Optional
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys - Set these as environment variables
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY", "api_key_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "api_key_here")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "api_key_here")

# Initialize OpenAI client - FIXED
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and OPENAI_API_KEY != "api_key_here" else None

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
    """Use OpenAI Vision for comprehensive food detection - ENHANCED VERSION"""
    if not openai_client:
        print("❌ OpenAI client not available")
        print(f"   - API key set: {bool(OPENAI_API_KEY)}")
        print(f"   - API key value: {OPENAI_API_KEY[:20] if OPENAI_API_KEY else 'None'}...")
        return []
    
    try:
        print("🧠 Querying OpenAI Vision...")
        print(f"   - Using model: gpt-4o")
        print(f"   - Image size: {len(image_base64)} characters")
        
        # Improved prompt with more explicit instructions
        prompt = """
        You are an expert food identification AI. Analyze this refrigerator/kitchen image with extreme care and attention to detail.

        IMPORTANT INSTRUCTIONS:
        1. Look at EVERY visible item, container, package, bottle, jar, and food product
        2. Read any visible text/labels on packages and containers
        3. Identify fresh produce, dairy, meats, beverages, condiments, leftovers
        4. Include items that are partially visible or in the background
        5. Be specific with names (e.g., "2% milk" not "dairy", "cheddar cheese" not "cheese")

        WHAT TO LOOK FOR:
        - Milk cartons, juice boxes, water bottles
        - Egg cartons, cheese packages, yogurt containers
        - Fresh fruits and vegetables in drawers/shelves
        - Meat packages, deli containers
        - Condiment bottles (ketchup, mustard, sauces)
        - Leftover containers, takeout boxes
        - Bread, tortillas, baked goods
        - Canned goods, jars, packages

        RESPONSE FORMAT:
        Return ONLY a valid JSON array. No other text or explanations.
        Format: [{"name": "specific_item_name", "confidence": 85}]

        Examples:
        [
            {"name": "whole milk", "confidence": 95},
            {"name": "large eggs", "confidence": 90},
            {"name": "cheddar cheese slices", "confidence": 88},
            {"name": "fresh broccoli", "confidence": 85},
            {"name": "ground beef", "confidence": 80}
        ]

        Be thorough and look carefully at every part of the image.
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Changed to gpt-4o for better vision
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
            max_tokens=2000,  # Increased for more detailed responses
            temperature=0.1,   # Lower temperature for more consistent results
        )
        
        content = response.choices[0].message.content.strip()
        
        # More detailed logging
        print(f"🧠 OpenAI full response length: {len(content)} characters")
        print(f"🧠 OpenAI response preview: {content[:300]}...")
        
        # Better JSON parsing with multiple attempts
        try:
            # Method 1: Try direct JSON parsing
            try:
                items = json.loads(content)
                if isinstance(items, list):
                    print("✅ Direct JSON parsing successful")
                else:
                    raise ValueError("Not a list")
            except:
                # Method 2: Extract JSON array with regex
                json_patterns = [
                    r'\[[\s\S]*\]',  # Standard array
                    r'```json\s*(\[[\s\S]*?\])\s*```',  # Code block
                    r'```\s*(\[[\s\S]*?\])\s*```',  # Simple code block
                ]
                
                items = None
                for pattern in json_patterns:
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        try:
                            items = json.loads(json_str)
                            print(f"✅ Regex JSON parsing successful with pattern: {pattern}")
                            break
                        except:
                            continue
                
                if items is None:
                    print("❌ All JSON parsing methods failed")
                    print(f"Raw response: {content}")
                    return []
            
            # Process the items
            openai_results = []
            for item in items:
                if isinstance(item, dict) and 'name' in item:
                    confidence = item.get('confidence', 75)
                    # Ensure confidence is reasonable
                    if confidence < 60:
                        confidence = 60
                    elif confidence > 99:
                        confidence = 99
                        
                    openai_results.append({
                        'name': item['name'],
                        'confidence': confidence,
                        'source': 'openai'
                    })
                else:
                    print(f"⚠️ Skipping invalid item: {item}")
            
            print(f"✅ OpenAI detected {len(openai_results)} valid items")
            
            # Log the detected items for debugging
            if openai_results:
                print("🧠 OpenAI detected items:")
                for i, item in enumerate(openai_results[:5]):  # Show first 5
                    print(f"   {i+1}. {item['name']} ({item['confidence']}%)")
                if len(openai_results) > 5:
                    print(f"   ... and {len(openai_results) - 5} more items")
            else:
                print("🧠 No items detected by OpenAI")
            
            return openai_results
                
        except json.JSONDecodeError as e:
            print(f"❌ OpenAI JSON parse error: {e}")
            print(f"Raw content: {content}")
            return []
        
    except Exception as e:
        print(f"❌ OpenAI Vision error: {e}")
        print(f"Error type: {type(e).__name__}")
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
            
            # Initialize sources if not exists
            if 'sources' not in existing:
                existing['sources'] = [existing['source']]
            
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

class SpoonacularClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.spoonacular.com"
        self.headers = {"Content-Type": "application/json"}
    
    def find_recipes_by_ingredients(self, ingredients: List[str], number: int = 10) -> List[Dict]:
        """Find recipes that can be made with the given ingredients"""
        if not ingredients:
            return []
        
        # Clean and format ingredients for Spoonacular
        formatted_ingredients = [ingredient.strip().lower() for ingredient in ingredients]
        ingredients_string = ",".join(formatted_ingredients)
        
        url = f"{self.base_url}/recipes/findByIngredients"
        params = {
            "apiKey": self.api_key,
            "ingredients": ingredients_string,
            "number": number,
            "ranking": 2,  # Maximize used ingredients
            "ignorePantry": False,  # Include common pantry items
            "limitLicense": False
        }
        
        try:
            print(f"🍳 Searching Spoonacular with ingredients: {ingredients_string}")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                recipes = response.json()
                print(f"✅ Found {len(recipes)} recipes")
                return recipes
            else:
                print(f"❌ Spoonacular API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Spoonacular request failed: {e}")
            return []
    
    def get_recipe_details(self, recipe_id: int) -> Optional[Dict]:
        """Get detailed information about a specific recipe"""
        url = f"{self.base_url}/recipes/{recipe_id}/information"
        params = {
            "apiKey": self.api_key,
            "includeNutrition": False
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"❌ Failed to get recipe details for {recipe_id}: {e}")
            return None
    
    def get_bulk_recipe_details(self, recipe_ids: List[int]) -> List[Dict]:
        """Get details for multiple recipes in one call"""
        if not recipe_ids:
            return []
        
        ids_string = ",".join(str(id) for id in recipe_ids)
        url = f"{self.base_url}/recipes/informationBulk"
        params = {
            "apiKey": self.api_key,
            "ids": ids_string,
            "includeNutrition": False
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"❌ Failed to get bulk recipe details: {e}")
            return []

# Initialize Spoonacular client - FIXED
spoonacular_client = SpoonacularClient(SPOONACULAR_API_KEY) if SPOONACULAR_API_KEY and SPOONACULAR_API_KEY != "api_key_here" else None

def process_spoonacular_recipes(raw_recipes: List[Dict], available_ingredients: List[str]) -> List[Dict]:
    """Process and enhance Spoonacular recipe results"""
    processed_recipes = []
    
    for recipe in raw_recipes:
        # Calculate recipe metrics
        used_ingredients = recipe.get('usedIngredients', [])
        missed_ingredients = recipe.get('missedIngredients', [])
        
        used_count = len(used_ingredients)
        missed_count = len(missed_ingredients)
        total_ingredients = used_count + missed_count
        
        # Calculate match percentage
        match_percentage = (used_count / total_ingredients * 100) if total_ingredients > 0 else 0
        
        # Determine if recipe is "ready to cook" (missing <= 2 ingredients)
        ready_to_cook = missed_count <= 2
        
        # Extract ingredient names
        used_ingredient_names = [ing['name'] for ing in used_ingredients]
        missed_ingredient_names = [ing['name'] for ing in missed_ingredients]
        
        processed_recipe = {
            "id": recipe.get('id'),
            "name": recipe.get('title'),
            "image": recipe.get('image'),
            "used_ingredients": used_ingredient_names,
            "missed_ingredients": missed_ingredient_names,
            "used_ingredient_count": used_count,
            "missed_ingredient_count": missed_count,
            "match_percentage": round(match_percentage, 1),
            "ready_to_cook": ready_to_cook,
            "difficulty": "Easy" if missed_count <= 1 else "Medium" if missed_count <= 3 else "Hard",
            "spoonacular_score": recipe.get('likes', 0)
        }
        
        processed_recipes.append(processed_recipe)
    
    # Sort by match percentage, then by likes
    processed_recipes.sort(key=lambda x: (x['match_percentage'], x['spoonacular_score']), reverse=True)
    
    return processed_recipes

def get_recipe_suggestions(ingredients: List[str], max_recipes: int = 8) -> Dict:
    """Get recipe suggestions based on available ingredients"""
    if not spoonacular_client:
        return {
            "error": "Spoonacular API not configured",
            "recipes": [],
            "total_found": 0
        }
    
    if not ingredients:
        return {
            "error": "No ingredients provided",
            "recipes": [],
            "total_found": 0
        }
    
    # Filter ingredients to focus on food items (remove very generic terms)
    filtered_ingredients = []
    generic_terms = {'food', 'item', 'object', 'product', 'container', 'package'}
    
    for ingredient in ingredients:
        if ingredient.lower().strip() not in generic_terms and len(ingredient.strip()) > 2:
            filtered_ingredients.append(ingredient.strip())
    
    print(f"🔍 Filtered ingredients for recipe search: {filtered_ingredients}")
    
    # Get recipes from Spoonacular
    raw_recipes = spoonacular_client.find_recipes_by_ingredients(
        filtered_ingredients, 
        number=max_recipes * 2  # Get more to have better filtering options
    )
    
    if not raw_recipes:
        return {
            "message": "No recipes found with current ingredients",
            "recipes": [],
            "total_found": 0,
            "ingredients_used": filtered_ingredients
        }
    
    # Process and rank recipes
    processed_recipes = process_spoonacular_recipes(raw_recipes, filtered_ingredients)
    
    # Limit to requested number
    final_recipes = processed_recipes[:max_recipes]
    
    # Calculate summary statistics
    ready_to_cook_count = len([r for r in final_recipes if r['ready_to_cook']])
    avg_match_percentage = sum(r['match_percentage'] for r in final_recipes) / len(final_recipes) if final_recipes else 0
    
    return {
        "recipes": final_recipes,
        "total_found": len(raw_recipes),
        "recipes_returned": len(final_recipes),
        "ready_to_cook_count": ready_to_cook_count,
        "average_match_percentage": round(avg_match_percentage, 1),
        "ingredients_used": filtered_ingredients,
        "api_source": "spoonacular"
    }

@app.post("/suggest_meals")
async def suggest_meals(request_data: dict):
    """Enhanced meal suggestions using Spoonacular API"""
    try:
        # Extract ingredients from request
        ingredients = request_data.get("ingredients", [])
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        
        max_recipes = request_data.get("max_recipes", 8)
        
        print(f"\n🍽️ Generating meal suggestions for {len(ingredients)} ingredients")
        
        # Get recipe suggestions
        suggestions = get_recipe_suggestions(ingredients, max_recipes)
        
        # Add metadata
        suggestions["timestamp"] = "2025-08-19T" + str(len(ingredients))  # Simple timestamp
        suggestions["service_status"] = "active" if spoonacular_client else "limited"
        
        return suggestions
        
    except Exception as e:
        print(f"❌ Error in suggest_meals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recipe suggestion failed: {str(e)}")

@app.get("/recipe/{recipe_id}")
async def get_recipe_details_endpoint(recipe_id: int):
    """Get detailed information about a specific recipe"""
    if not spoonacular_client:
        raise HTTPException(status_code=503, detail="Spoonacular API not configured")
    
    try:
        recipe_details = spoonacular_client.get_recipe_details(recipe_id)
        
        if not recipe_details:
            raise HTTPException(status_code=404, detail="Recipe not found")
        
        # Extract key information
        processed_details = {
            "id": recipe_details.get("id"),
            "title": recipe_details.get("title"),
            "image": recipe_details.get("image"),
            "ready_in_minutes": recipe_details.get("readyInMinutes"),
            "servings": recipe_details.get("servings"),
            "source_url": recipe_details.get("sourceUrl"),
            "spoonacular_url": recipe_details.get("spoonacularSourceUrl"),
            "instructions": recipe_details.get("instructions", ""),
            "summary": recipe_details.get("summary", ""),
            "ingredients": [
                {
                    "name": ing.get("name"),
                    "amount": ing.get("amount"),
                    "unit": ing.get("unit")
                }
                for ing in recipe_details.get("extendedIngredients", [])
            ]
        }
        
        return processed_details
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting recipe details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch recipe details")

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
    """Debug endpoint to check all AI system status - FIXED"""
    return {
        "clarifai_key_set": CLARIFAI_API_KEY != "api_key_here",
        "openai_key_set": OPENAI_API_KEY != "api_key_here",  # FIXED
        "spoonacular_key_set": SPOONACULAR_API_KEY != "api_key_here",  # FIXED
        "openai_client_ready": openai_client is not None,
        "spoonacular_client_ready": spoonacular_client is not None,
        "models_configured": len(CLARIFAI_MODELS),
        "all_services_ready": all([
            CLARIFAI_API_KEY != "api_key_here",
            openai_client is not None,
            spoonacular_client is not None
        ])
    }