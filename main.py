from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os

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
USER_ID = "clarifai"  # Default user ID
APP_ID = "main"       # Default app ID
MODEL_ID = "food-item-recognition"  # Try this model ID
# Alternative: MODEL_ID = "general"

@app.post("/detect_foods")
async def detect_foods(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        image_bytes_base64 = base64.b64encode(contents).decode("utf-8")

        # Clarifai API headers
        headers = {
            "Authorization": f"Key {CLARIFAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # Updated API endpoint structure
        url = f"https://api.clarifai.com/v2/users/{USER_ID}/apps/{APP_ID}/models/{MODEL_ID}/outputs"
        
        payload = {
            "inputs": [
                {
                    "data": {
                        "image": {
                            "base64": image_bytes_base64
                        }
                    }
                }
            ]
        }

        print(f"Making request to: {url}")  # Debug logging
        response = requests.post(url, json=payload, headers=headers)
        print(f"Response status: {response.status_code}")  # Debug logging
        
        data = response.json()
        print(f"Response data: {data}")  # Debug logging

        # Check for errors
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Clarifai API error: {data}")

        if data.get("status", {}).get("code") != 10000:
            error_msg = data.get("status", {}).get("description", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Clarifai error: {error_msg}")

        # Extract concepts
        outputs = data.get("outputs", [])
        if not outputs:
            return {"items": [], "message": "No outputs received from Clarifai"}

        concepts = outputs[0].get("data", {}).get("concepts", [])
        print(f"All concepts detected: {len(concepts)}")
        for concept in concepts[:10]:  # Print top 10 for debugging
            print(f"  {concept['name']}: {concept['value']:.3f}")
        
        # Get ALL items with very low confidence threshold for maximum detection
        all_detected = []
        
        for concept in concepts:
            confidence = concept.get("value", 0)
            # Include almost everything - very low threshold
            if confidence > 0.1:  # Very low threshold to catch everything
                all_detected.append({
                    "name": concept["name"],
                    "confidence": round(confidence * 100, 1)
                })
        
        # Sort by confidence (highest first)
        all_detected.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Separate into confidence tiers for display
        high_confidence = [item for item in all_detected if item["confidence"] > 70]
        medium_confidence = [item for item in all_detected if 40 <= item["confidence"] <= 70]
        low_confidence = [item for item in all_detected if 10 <= item["confidence"] < 40]
        
        all_items = all_detected
        
        return {
            "items": [item["name"] for item in all_items],
            "high_confidence": [item for item in high_confidence],
            "medium_confidence": [item for item in medium_confidence], 
            "low_confidence": [item for item in low_confidence],
            "detailed_items": all_items[:50],  # Limit to top 50 to avoid overwhelming
            "total_detected": len(all_items),
            "total_concepts": len(concepts)
        }

    except Exception as e:
        print(f"Error: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "MealMapper API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}