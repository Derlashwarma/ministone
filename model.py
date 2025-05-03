from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import os
import requests
import json
import re
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), 'api_keys', '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError(f".env file not found at path: {dotenv_path}")

class MachineLearningModel:
    def __init__(self):
        self.model = tf.lite.Interpreter(model_path='./recycling_model.tflite')
        self.model.allocate_tensors()
        
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        
    def preprocess_image(self, img_bytes, target_size=(224, 224)):
        img = Image.open(BytesIO(img_bytes)).resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img_bytes):
        labels = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
        input_data = self.preprocess_image(img_bytes)
        
        self.model.set_tensor(self.input_details[0]['index'], input_data)
        self.model.invoke()
        
        output_data = self.model.get_tensor(self.output_details[0]['index'])
        
        predicted_class = np.argmax(output_data)
        return labels[predicted_class]

class LLMAIEngine:
    
    @staticmethod
    def get_ai_feedback(item):
        api_key = os.getenv("CHAT_API_KEY")
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Refined prompt to force clean JSON output
        formatted_prompt = f"""
                Return only a raw JSON object containing the following fields given.
                Do not include explanation, markdown, comments, or any extra text.
                Strictly return only valid JSON. The user prompt is:

                Give me what I can do with this item{item}
                
                return exactly with this format:
                safe_waste_disposal: (description of safe waste disposal),
                recyclable_method: (if recyclable how to recycle),
                item_description: (what are the uses for the item and it's description)
            """

        payload = {
            "model": "meta-llama/llama-4-maverick:free",
            "messages": [
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        try:
            data = response.json()
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"OpenRouter API Error {response.status_code}: {data}")

        try:
            result_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise Exception(f"Unexpected response format: {data}")

        match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if match:
            result_text = match.group(1)

        try:
            result_json = json.loads(result_text.strip())
            return result_json
        except json.JSONDecodeError:
            raise Exception(f"Failed to parse JSON from model output:\n{result_text}")
        