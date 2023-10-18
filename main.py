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
from insert_part import repairImage
from prewitt import prewitt
from robert import robert
from sorbel import sorbel

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
    image = imread("upload/crop.jpg")
    mask = imread("upload/mask.jpg", as_gray=True)

    output_image = Inpainter(
        image,
        mask,
        patch_size=9,
        plot_progress=True
    ).inpaint()
    imsave("upload/imageEdit.jpg", output_image, quality=100)
    

@app.post("/")
async def create_upload_file(mode: Annotated[str, Form()], pos: Annotated[str, Form()], size: Annotated[str, Form()], image : UploadFile = File(...)) -> dict:
    async with aiofiles.open("./upload/image.jpg", 'wb') as out_file:
        content = await image.read()
        await out_file.write(content)
        image_pic = cv2.imread("./upload/image.jpg",1)
        print(f"MAIN: image shape: {image_pic.shape}")
    size = size.split(",")
    width, height = int(round(float(size[0]), 0)), int(round(float(size[1]), 0))
    print(f"size {width} {height}")
    pos = pos.split(",")
    posx, posy = int(round(float(pos[0]), 0)), int(round(float(pos[1]), 0))
    
    # async with aiofiles.open("./upload/crop.jpg", 'wb') as out_file:
    #     content = await crop.read()
    #     await out_file.write(content)
    #     crop_pic = cv2.imread("./upload/crop.jpg",1)
    #     print(f"crop shape: {crop_pic.shape}")
    # print("saved all component...\n...")
    # print(f"crop pos: ({pos})")
    
    # สร้าง crop ปลอม(ใช้ไม่ได้กับทุกรูปเพราะขนาดแต่ละรูปไม่เท่ากัน)
    img2 = cv2.imread("./upload/image.jpg")
    crop = img2[posy : posy + height, posx : posx + width]
    cv2.imwrite("./upload/crop.jpg", crop)
    print(f"MAIN: crop shape: {crop.shape}")
    print("MAIN: finish crop")
    if mode == "canny":
        cannyMask()
    elif mode == "laplacian":
        laplacianMask()
    elif mode == "prewitt":
        prewitt("./upload/crop.jpg", "./upload/mask.jpg", 10)
    elif mode == "robert":
        robert("./upload/crop.jpg", "./upload/mask.jpg", 20)
    elif mode == "sorbel":
        sorbel("./upload/crop.jpg", "./upload/mask.jpg", 30)
    inpainter()
    repairImage(posy, posx)
    print("!!!finished process!!!\n...\n......returning page into front")
    return {"filename": "success laew"}



app.mount("/upload", StaticFiles(directory="upload"), name='images')
 
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
            <img src="upload/repaired.jpg">
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