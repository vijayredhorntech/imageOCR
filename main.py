from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import cv2
import numpy as np

app = FastAPI()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

@app.post("/ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    # Read image as bytes
    image_bytes = await file.read()
    
    # Convert bytes data to numpy array for OpenCV
    image_np = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image using OpenCV
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Perform OCR using EasyOCR
    result = reader.readtext(image, detail=0)
    
    # Return the recognized text
    return JSONResponse(content={"text": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1000)  # Explicitly bind to port 1000
