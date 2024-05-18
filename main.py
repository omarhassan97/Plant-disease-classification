from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn


app  = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./model1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
def read_file_as_img(data) -> np.array:
    image = Image.open(BytesIO(data))
    image_array = np.array(image)
    return image_array


@app.get("/")
async def ping():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes = await file.read()
    image_array = read_file_as_img(bytes)
    img_batch = np.expand_dims(image_array,0)
    prediction = MODEL.predict(img_batch)
    c_name = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return {"class": c_name,"confidence":confidence }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port = 8000)
