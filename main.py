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

import argparse
from skimage.io import imread, imsave
from inpainterModule import Inpainter

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
        inpainter()
        print("received!")
    elif mode == "canny":
        # show_image()
        print("received!")

    return {"filename": image.filename}



app.mount("/resources", StaticFiles(directory="resources"), name='images')
 
@app.get("/pic", response_class=HTMLResponse)
def serve():
    return """
    <html>
        <head>
            <title>Helppppppp!</title>
        </head>
        <body>
        <h1>Here's your image</h1>
        <img src="resources/image.jpg">
        </body>
        <style>
        h1{
            color:#FF3333
        }
        </style>
    </html>
    """