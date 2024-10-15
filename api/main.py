from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from functions.process_audio import model_predict
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2Templates to serve HTML pages
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    # Serve the HTML page with a file upload form
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_model", response_class=HTMLResponse)
async def predict_model(request: Request, audio_file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_location = "./audios/audio_to_process.wav"
        with open(file_location, "wb") as buffer:
            buffer.write(await audio_file.read())

        # Process the file with your prediction logic
        prediction = model_predict()

        # Respond with prediction in HTML
        return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction, "success": True})
    except Exception as err:
        return templates.TemplateResponse("result.html", {"request": request, "prediction": err, "success": False})
