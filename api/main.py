from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
app = FastAPI()



MODEL = tf.keras.models.load_model("/home/ines/PycharmProjects/models/1")
# beta_model = tf.keras.models.load_model("/home/ines/PycharmProjects/models/2")
CLASS_NAMES = ['Tomato_Early_blight', 'Tomato_healthy']

@app.get("/ping")
async def ping():
    return "hello"



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        "class" : predicted_class,
        'confidence' : float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)