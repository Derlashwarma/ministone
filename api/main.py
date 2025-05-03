from fastapi import FastAPI, File, UploadFile, Form
from io import BytesIO
from typing import Dict
from PIL import Image
from model import MachineLearningModel, LLMAIEngine

app = FastAPI()
ml_model = MachineLearningModel()

@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.post('/api/classify')
async def predict_image(file: UploadFile = File(...)) -> Dict[str, str]:
    image_bytes = await file.read()
    prediction = ml_model.predict(image_bytes)
    return {"prediction": prediction}

@app.post('/api/LLM-response')
async def get_ai_response(item: str = Form(...)):
    description = LLMAIEngine.get_ai_feedback(item)
    return {"response": description}