from fastapi import FastAPI, File, UploadFile, Form
from typing import Union, Annotated
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import aiofiles
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from process import show_image, show_monotone_image
from canny import cannyMask
from edge_detection import laplacianMask

import argparse
from skimage.io import imread, imsave
from inpainterModule import Inpainter
# uvicorn main:app --reload

app = FastAPI()

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileUpload(BaseModel):
    files: List[bytes]  # Use bytes for file data
    
def inpainter():
    image = imread("resources/image.jpg")
    mask = imread("resources/mask.jpg", as_gray=True)

    output_image = Inpainter(
        image,
        mask,
        patch_size=9,
        plot_progress=True
    ).inpaint()
    imsave("resources/imageEdit.jpg", output_image, quality=100)
    

@app.post("/")
# async def create_upload_file(datas: dict) -> dict:
async def create_upload_file(mode: Annotated[str, Form()], image : UploadFile = File(...)) -> dict:
    async with aiofiles.open("./resources/image.jpg", 'wb') as out_file:
        content = await image.read()  # async read
        await out_file.write(content)  # async write
    if mode == "laprician":
        laplacianMask()
        inpainter()
    elif mode == "canny":
        cannyMask()
        inpainter()

    return {"filename": image.filename}



app.mount("/resources", StaticFiles(directory="resources"), name='images')
 
@app.get("/pic", response_class=HTMLResponse)
def serve():
    return """
    <html>
        <head>
            <title>Process Result</title>
        </head>
        <body>
            <div>
                <h1>Here's your image</h1>
            </div>
            <img src="resources/imageEdit.jpg">
        </body>
        
        <style>
        h1{
            color:#FF3333
        }
        div{
            backgroundColor:#000055
        }
        </style>
    </html>
    """