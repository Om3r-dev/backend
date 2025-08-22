from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import requests
import google.generativeai as genai
from PIL import Image
import io
import json
from typing import List, Dict, Set
import re
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure APIs
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
CLARIFAI_API_KEY = os.getenv('CLARIFAI_API_KEY')
SPOONACULAR_API_KEY = os.getenv('SPOONACULAR_API_KEY')

class FoodDetectionService:
    def __init__(self):
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        self.food_keywords = {
            'vegetables': ['carrot', 'broccoli', 'spinach', 'lettuce', 'tomato', 'cucumber', 
                          'bell pepper', 'red pepper', 'green pepper', 'yellow pepper', 'onion', 
                          'red onion', 'white onion', 'garlic', 'potato', 'sweet potato', 'celery',
                          'cabbage', 'cauliflower', 'zucchini', 'eggplant', 'mushroom', 'corn',
                          'peas', 'green beans', 'asparagus', 'radish', 'beets', 'kale', 'arugula',
                          'cilantro', 'parsley', 'green onion', 'scallion', 'jalapeño', 'chili'],
            'fruits': ['apple', 'green apple', 'red apple', 'banana', 'orange', 'lemon', 'lime', 
                      'strawberry', 'blueberry', 'grape', 'pear', 'peach', 'plum', 'cherry', 
                      'pineapple', 'mango', 'avocado', 'kiwi', 'melon', 'watermelon', 'pomegranate'],
            'proteins': ['chicken', 'chicken breast', 'chicken thigh', 'beef', 'ground beef', 
                        'pork', 'fish', 'salmon', 'tuna', 'shrimp', 'eggs', 'tofu', 'black beans',
                        'kidney beans', 'lentils', 'chickpeas', 'turkey', 'lamb', 'bacon', 'ham'],
            'dairy': ['milk', 'whole milk', 'skim milk', 'cheese', 'cheddar cheese', 'mozzarella',
                     'yogurt', 'greek yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
                     'parmesan', 'feta', 'goat cheese', 'cream cheese'],
            'grains': ['rice', 'brown rice', 'white rice', 'pasta', 'bread', 'white bread',
                      'whole wheat bread', 'quinoa', 'oats', 'barley', 'flour', 'noodles'],
            'pantry': ['olive oil', 'vegetable oil', 'vinegar', 'balsamic vinegar', 'salt',
                      'pepper', 'sugar', 'honey', 'garlic powder', 'onion powder', 'paprika']
        }

    def detect_with_gemini(self, image_data: bytes) -> List[Dict]:
        """Use Google gemini-2.0-flash to detect food items"""
        try:
            if not image_data:
                logger.warning("No image data provided to Gemini")
                return []
                
            image = Image.open(io.BytesIO(image_data))
            
            prompt = """
            Analyze this food/fridge image and identify EVERY SINGLE SPECIFIC ingredient you can see.
            
            CRITICAL RULES:
            - List each ingredient individually (tomato, onion, carrot) - NOT categories like "vegetables"
            - Be as specific as possible: "red bell pepper" not just "pepper", "cheddar cheese" not just "cheese"
            - Include ALL visible ingredients: fruits, vegetables, meats, dairy, spices, condiments, pantry items
            - For packaged items, identify the food inside: "ground beef" not "package"
            - List fresh herbs specifically: basil, cilantro, parsley, etc.
            
            Return ONLY a JSON array like this:
            [{"name": "tomato", "confidence": 85}, {"name": "red onion", "confidence": 90}, {"name": "cheddar cheese", "confidence": 75}]
            
            Focus on ingredients that could be used for cooking. Be thorough and specific.
            """
            
            response = self.gemini_model.generate_content([prompt, image])
            
            if response and response.text:
                # Try to extract JSON array from response
                json_match = re.search(r'\[(.*?)\]', response.text.replace('\n', ''), re.DOTALL)
                if json_match:
                    try:
                        items = json.loads('[' + json_match.group(1) + ']')
                        valid_items = []
                        for item in items:
                            if isinstance(item, dict) and item.get('name') and item.get('confidence'):
                                valid_items.append({
                                    'name': str(item['name']).lower().strip(),
                                    'confidence': min(int(item['confidence']), 95),
                                    'source': 'gemini'
                                })
                        return valid_items[:25]
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(f"JSON parsing error: {e}")
                
                # Fallback: extract ingredients from text
                items = []
                text_lines = response.text.lower().split('\n')
                for line in text_lines:
                    # Look for ingredient patterns
                    if any(word in line for word in ['ingredient', 'food', 'item', '-', '•', '1.', '2.']):
                        # Extract ingredient names
                        words = re.findall(r'\b[a-zA-Z\s]{3,20}\b', line)
                        for word in words:
                            clean_word = word.strip()
                            if (len(clean_word) > 2 and 
                                clean_word not in ['ingredient', 'food', 'item', 'the', 'and', 'with'] and
                                any(keyword in clean_word for category in self.food_keywords.values() for keyword in category)):
                                items.append({
                                    'name': clean_word,
                                    'confidence': 80,
                                    'source': 'gemini'
                                })
                
                return items[:25]
            else:
                logger.warning("No response text from Gemini")
                return []
                
        except Exception as e:
            logger.error(f"Gemini detection error: {e}")
            return []

    def detect_with_clarifai(self, image_data: bytes) -> List[Dict]:
        """Use Clarifai Food Model to detect food items"""
        try:
            if not image_data:
                logger.warning("No image data provided to Clarifai")
                return []
                
            if not CLARIFAI_API_KEY:
                logger.warning("Clarifai API key not configured")
                return []
                
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            url = "https://api.clarifai.com/v2/models/food-item-recognition/outputs"
            headers = {
                "Authorization": f"Key {CLARIFAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": [{
                    "data": {
                        "image": {
                            "base64": image_b64
                        }
                    }
                }]
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                items = []
                
                if (data and data.get('outputs') and 
                    len(data['outputs']) > 0 and 
                    data['outputs'][0].get('data', {}).get('concepts')):
                    
                    for concept in data['outputs'][0]['data']['concepts']:
                        name = concept.get('name', '').lower().strip()
                        confidence = concept.get('value', 0) * 100
                        
                        if name and confidence > 20 and name not in ['food', 'meal', 'dish']:
                            items.append({
                                'name': name,
                                'confidence': min(int(confidence), 95),
                                'source': 'clarifai'
                            })
                
                return items[:20]
            else:
                logger.warning(f"Clarifai API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Clarifai detection error: {e}")
            return []

    def detect_with_blip(self, image_data: bytes) -> List[Dict]:
        """Use Hugging Face BLIP-2 to detect food items"""
        try:
            if not image_data:
                logger.warning("No image data provided to BLIP")
                return []
                
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Use Hugging Face Inference API for BLIP-2
            url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
            headers = {"Authorization": "Bearer hf_demo"}  # Demo token - free tier
            
            payload = {
                "inputs": image_b64,
                "parameters": {
                    "max_length": 200,
                    "do_sample": True
                }
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                items = []
                
                if isinstance(data, list) and len(data) > 0 and data[0].get('generated_text'):
                    description = data[0]['generated_text'].lower()
                    
                    # Extract food ingredients from the description
                    for category, keywords in self.food_keywords.items():
                        for keyword in keywords:
                            if keyword in description:
                                # Check if it's not already added
                                if keyword not in [item['name'] for item in items]:
                                    confidence = 70
                                    # Boost confidence if multiple matches
                                    if description.count(keyword) > 1:
                                        confidence = min(confidence + 10, 90)
                                    
                                    items.append({
                                        'name': keyword,
                                        'confidence': confidence,
                                        'source': 'blip'
                                    })
                
                return items[:15]
            else:
                logger.warning(f"BLIP API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"BLIP detection error: {e}")
            return []

    def normalize_ingredient_name(self, name: str) -> str:
        """Normalize ingredient names for better matching"""
        if not name or not isinstance(name, str):
            return ""
            
        # Remove common words and normalize
        name = re.sub(r'\b(fresh|organic|raw|cooked|frozen|dried|canned|sliced|diced)\b', '', name)
        name = re.sub(r'\b(piece|pieces|slice|slices|cup|cups|lb|lbs|oz)\b', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Handle plurals
        if name.endswith('ies'):
            name = name[:-3] + 'y'
        elif name.endswith('es') and len(name) > 4:
            if name.endswith('ches') or name.endswith('shes'):
                name = name[:-2]
            else:
                name = name[:-1]
        elif name.endswith('s') and len(name) > 3:
            name = name[:-1]
        
        return name

    def merge_detections(self, gemini_items: List[Dict], clarifai_items: List[Dict], 
                        blip_items: List[Dict]) -> Dict:
        """Merge results from all three AI services"""
        # Ensure all inputs are lists (handle None values)
        gemini_items = gemini_items if gemini_items is not None else []
        clarifai_items = clarifai_items if clarifai_items is not None else []
        blip_items = blip_items if blip_items is not None else []
        
        all_items = []
        ingredient_votes = {}
        
        # Combine all items
        for items_list in [gemini_items, clarifai_items, blip_items]:
            if items_list and isinstance(items_list, list):
                all_items.extend(items_list)
        
        if not all_items:
            logger.warning("No items detected by any AI service")
            return {
                'items': [],
                'high_confidence': [],
                'medium_confidence': [],
                'low_confidence': [],
                'total_food_detected': 0,
                'ai_enhanced': False,
                'detection_sources': []
            }
        
        # Group by normalized names and merge
        for item in all_items:
            if not isinstance(item, dict) or not item.get('name'):
                continue
                
            normalized_name = self.normalize_ingredient_name(item['name'])
            if not normalized_name:
                continue
            
            if normalized_name not in ingredient_votes:
                ingredient_votes[normalized_name] = {
                    'names': [],
                    'confidences': [],
                    'sources': [],
                    'total_confidence': 0,
                    'count': 0
                }
            
            ingredient_votes[normalized_name]['names'].append(item['name'])
            ingredient_votes[normalized_name]['confidences'].append(item.get('confidence', 50))
            ingredient_votes[normalized_name]['sources'].append(item.get('source', 'unknown'))
            ingredient_votes[normalized_name]['total_confidence'] += item.get('confidence', 50)
            ingredient_votes[normalized_name]['count'] += 1
        
        # Create final ingredient list
        final_items = []
        for normalized_name, data in ingredient_votes.items():
            if data['count'] == 0:
                continue
                
            # Use most common name
            name_counts = Counter(data['names'])
            best_name = name_counts.most_common(1)[0][0]
            
            # Calculate boosted confidence for multiple AI agreement
            avg_confidence = data['total_confidence'] / data['count']
            unique_sources = len(set(data['sources']))
            
            # Boost confidence for multiple AI detections
            if unique_sources >= 3:
                final_confidence = min(avg_confidence + 25, 98)
                source = 'all_three_ais'
            elif unique_sources == 2:
                final_confidence = min(avg_confidence + 15, 95)
                source = 'two_ais'
            else:
                final_confidence = avg_confidence
                source = data['sources'][0] if data['sources'] else 'unknown'
            
            final_items.append({
                'name': best_name,
                'confidence': int(final_confidence),
                'source': source,
                'detection_count': data['count'],
                'detected_by': list(set(data['sources']))
            })
        
        # Sort by confidence
        final_items.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Categorize by confidence levels
        high_confidence = [item for item in final_items if item['confidence'] >= 70]
        medium_confidence = [item for item in final_items if 40 <= item['confidence'] < 70]
        low_confidence = [item for item in final_items if item['confidence'] < 40]
        
        return {
            'items': [item['name'] for item in final_items],
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'total_food_detected': len(final_items),
            'ai_enhanced': len(final_items) > 0,
            'detection_sources': ['gemini', 'clarifai', 'blip2']
        }

    def get_recipe_details(self, recipe_id: int) -> Dict:
        """Get detailed recipe information including nutrition"""
        try:
            if not SPOONACULAR_API_KEY:
                logger.warning("Spoonacular API key not configured")
                return {}
                
            url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
            params = {
                'apiKey': SPOONACULAR_API_KEY,
                'includeNutrition': True
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                recipe_data = response.json()
                
                # Extract nutrition info
                nutrition = {}
                if 'nutrition' in recipe_data and 'nutrients' in recipe_data['nutrition']:
                    for nutrient in recipe_data['nutrition']['nutrients']:
                        name = nutrient.get('name', '').lower()
                        if name in ['calories', 'fat', 'protein', 'carbohydrates', 'fiber', 'sugar']:
                            nutrition[name] = {
                                'amount': nutrient.get('amount', 0),
                                'unit': nutrient.get('unit', '')
                            }
                
                # Extract detailed recipe info
                detailed_recipe = {
                    'id': recipe_data.get('id'),
                    'name': recipe_data.get('title', ''),
                    'description': recipe_data.get('summary', '').replace('<b>', '').replace('</b>', ''),
                    'image': recipe_data.get('image', ''),
                    'ready_in_minutes': recipe_data.get('readyInMinutes', 0),
                    'servings': recipe_data.get('servings', 1),
                    'source_url': recipe_data.get('sourceUrl', ''),
                    'spoonacular_url': recipe_data.get('spoonacularSourceUrl', ''),
                    'instructions': [],
                    'ingredients': [],
                    'nutrition': nutrition,
                    'diet_types': recipe_data.get('diets', []),
                    'cuisines': recipe_data.get('cuisines', []),
                    'dish_types': recipe_data.get('dishTypes', [])
                }
                
                # Extract instructions
                if 'analyzedInstructions' in recipe_data:
                    for instruction_group in recipe_data['analyzedInstructions']:
                        if 'steps' in instruction_group:
                            for step in instruction_group['steps']:
                                detailed_recipe['instructions'].append({
                                    'step': step.get('number', 0),
                                    'instruction': step.get('step', '')
                                })
                
                # Extract ingredients
                if 'extendedIngredients' in recipe_data:
                    for ingredient in recipe_data['extendedIngredients']:
                        detailed_recipe['ingredients'].append({
                            'name': ingredient.get('name', ''),
                            'amount': ingredient.get('amount', 0),
                            'unit': ingredient.get('unit', ''),
                            'original': ingredient.get('original', '')
                        })
                
                return detailed_recipe
            else:
                logger.warning(f"Spoonacular recipe details API returned status {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Recipe details error: {e}")
            return {}

    def get_recipe_suggestions(self, ingredients: List[str], max_recipes: int = 8) -> Dict:
        """Get recipe suggestions with enhanced details"""
        try:
            if not SPOONACULAR_API_KEY:
                logger.error("Spoonacular API key not configured")
                return {'recipes': [], 'error': 'Recipe service not configured'}
                
            if not ingredients:
                return {'recipes': [], 'error': 'No ingredients provided'}
            
            # Clean ingredients list
            cleaned_ingredients = [ing.strip().lower() for ing in ingredients if ing and ing.strip()]
            if not cleaned_ingredients:
                return {'recipes': [], 'error': 'No valid ingredients provided'}
                
            ingredients_str = ','.join(cleaned_ingredients[:20])  # Limit to 20 ingredients
            
            url = "https://api.spoonacular.com/recipes/findByIngredients"
            params = {
                'apiKey': SPOONACULAR_API_KEY,
                'ingredients': ingredients_str,
                'number': max_recipes,
                'limitLicense': True,
                'ranking': 2,  # Maximize used ingredients
                'ignorePantry': False
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                recipes_data = response.json()
                
                if not recipes_data:
                    return {
                        'recipes': [],
                        'ready_to_cook_count': 0,
                        'average_match_percentage': 0,
                        'total_recipes_found': 0,
                        'search_ingredients': cleaned_ingredients
                    }
                
                processed_recipes = []
                ready_to_cook_count = 0
                total_match_percentage = 0
                
                for recipe in recipes_data:
                    if not recipe:
                        continue
                        
                    # Get detailed recipe information
                    recipe_details = self.get_recipe_details(recipe.get('id')) if recipe.get('id') else {}
                    
                    # Extract basic recipe information
                    used_ingredients = [ing['name'].lower() for ing in recipe.get('usedIngredients', [])]
                    missed_ingredients = [ing['name'].lower() for ing in recipe.get('missedIngredients', [])]
                    
                    used_count = len(used_ingredients)
                    missed_count = len(missed_ingredients)
                    total_ingredients = used_count + missed_count
                    
                    match_percentage = int((used_count / total_ingredients * 100)) if total_ingredients > 0 else 0
                    ready_to_cook = missed_count <= 2
                    
                    if ready_to_cook:
                        ready_to_cook_count += 1
                    
                    total_match_percentage += match_percentage
                    
                    # Determine difficulty
                    if missed_count == 0:
                        difficulty = "Easy"
                    elif missed_count <= 2:
                        difficulty = "Medium" 
                    else:
                        difficulty = "Hard"
                    
                    processed_recipe = {
                        'id': recipe.get('id'),
                        'name': recipe.get('title', ''),
                        'description': recipe_details.get('description', '')[:200] + '...' if recipe_details.get('description') else '',
                        'image': recipe.get('image', ''),
                        'used_ingredients': used_ingredients,
                        'missed_ingredients': missed_ingredients,
                        'used_ingredient_count': used_count,
                        'missed_ingredient_count': missed_count,
                        'match_percentage': match_percentage,
                        'ready_to_cook': ready_to_cook,
                        'difficulty': difficulty,
                        'spoonacular_score': recipe.get('likes', 0),
                        'ready_in_minutes': recipe_details.get('ready_in_minutes', 0),
                        'servings': recipe_details.get('servings', 1),
                        'nutrition': recipe_details.get('nutrition', {}),
                        'diet_types': recipe_details.get('diet_types', []),
                        'cuisines': recipe_details.get('cuisines', []),
                        'source_url': recipe_details.get('source_url', ''),
                        'spoonacular_url': recipe_details.get('spoonacular_url', ''),
                        'full_recipe': recipe_details  # Include full recipe for detailed view
                    }
                    
                    processed_recipes.append(processed_recipe)
                
                # Sort by match percentage and readiness
                processed_recipes.sort(key=lambda x: (x['match_percentage'], -x['missed_ingredient_count']), reverse=True)
                
                avg_match = int(total_match_percentage / len(processed_recipes)) if processed_recipes else 0
                
                return {
                    'recipes': processed_recipes,
                    'ready_to_cook_count': ready_to_cook_count,
                    'average_match_percentage': avg_match,
                    'total_recipes_found': len(processed_recipes),
                    'search_ingredients': cleaned_ingredients
                }
            else:
                logger.error(f"Spoonacular API error: {response.status_code} - {response.text}")
                return {'recipes': [], 'error': f'Recipe service returned error: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Recipe suggestion error: {e}")
            return {'recipes': [], 'error': str(e)}

# Initialize the service
food_detector = FoodDetectionService()

@app.route('/detect_foods', methods=['POST'])
def detect_foods():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image data
        image_data = file.read()
        
        if not image_data:
            return jsonify({'error': 'Empty file uploaded'}), 400
        
        # Run all three AI detections with proper error handling
        logger.info("Starting food detection with Gemini, Clarifai, and BLIP-2...")
        
        gemini_results = food_detector.detect_with_gemini(image_data)
        logger.info(f"Gemini detected {len(gemini_results) if gemini_results else 0} items")
        
        clarifai_results = food_detector.detect_with_clarifai(image_data)
        logger.info(f"Clarifai detected {len(clarifai_results) if clarifai_results else 0} items")
        
        blip_results = food_detector.detect_with_blip(image_data)
        logger.info(f"BLIP-2 detected {len(blip_results) if blip_results else 0} items")
        
        # Merge all results
        final_results = food_detector.merge_detections(
            gemini_results, clarifai_results, blip_results
        )
        
        logger.info(f"Final merged results: {final_results['total_food_detected']} unique ingredients")
        
        return jsonify(final_results)
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/suggest_meals', methods=['POST'])
def suggest_meals():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        ingredients = data.get('ingredients', [])
        max_recipes = data.get('max_recipes', 8)
        
        if not ingredients:
            return jsonify({'error': 'No ingredients provided'}), 400
        
        logger.info(f"Getting recipe suggestions for {len(ingredients)} ingredients")
        
        results = food_detector.get_recipe_suggestions(ingredients, max_recipes)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Recipe suggestion error: {e}")
        return jsonify({'error': f'Recipe suggestion failed: {str(e)}'}), 500

@app.route('/recipe/<int:recipe_id>', methods=['GET'])
def get_recipe_detail(recipe_id):
    """Get detailed recipe information"""
    try:
        recipe_details = food_detector.get_recipe_details(recipe_id)
        
        if recipe_details:
            return jsonify(recipe_details)
        else:
            return jsonify({'error': 'Recipe not found'}), 404
            
    except Exception as e:
        logger.error(f"Recipe detail error: {e}")
        return jsonify({'error': f'Failed to get recipe details: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'services': {
            'gemini': bool(os.getenv('GOOGLE_GEMINI_API_KEY')),
            'clarifai': bool(os.getenv('CLARIFAI_API_KEY')),
            'spoonacular': bool(os.getenv('SPOONACULAR_API_KEY')),
            'blip2': True
        }
    })

@app.route('/')
def root():
    return jsonify({'message': 'MealMapper API is running', 'status': 'healthy'})

if __name__ == '__main__':
    # Check for required environment variables
    required_vars = ['GOOGLE_GEMINI_API_KEY', 'CLARIFAI_API_KEY', 'SPOONACULAR_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        print(f"Please set the following environment variables: {missing_vars}")
    else:
        logger.info("All environment variables configured. Starting server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
