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

def show_image():
    # Set up data to send to mouse handler
    data = {}
    img = cv2.imread("testing.jpeg", 1)
    print("read success!!")

    cv2.imshow("Image", img)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    # points = np.uint16(data['lines'])

    return "Hello from the hell"
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
async def create_upload_file(mode: Annotated[str, Form()], pos: Annotated[str, Form()], image : UploadFile = File(...), crop : UploadFile = File(...)) -> dict:
    async with aiofiles.open("./upload/image.jpg", 'wb') as out_file:
        content = await image.read()
        await out_file.write(content)
    async with aiofiles.open("./upload/crop.jpg", 'wb') as out_file:
        content = await crop.read()
        await out_file.write(content)
    if mode == "laprician":
#         img2 = cv2.imread("./upload/image.jpg",1)
        print(f"crop area:{img2.shape}")
        
        # สร้าง crop ปลอม(ใช้ไม่ได้กับทุกรูปเพราะขนาดแต่ละรูปไม่เท่ากัน)
#         crop = img2[70 : 70 + 70, 160 : 160 + 70]
#         cv2.imwrite("./upload/crop.jpg", crop)
#         print("finish crop")
        
        cannyMask()
        inpainter()
        repairImage("70,160")
        # repairImage(pos)
    elif mode == "canny":
#         img2 = cv2.imread("./upload/image.jpg",1)
        print(f"crop area:{img2.shape}")
        
        # สร้าง crop ปลอม
#         crop = img2[70 : 70 + 70, 160 : 160 + 70]
#         cv2.imwrite("./upload/crop.jpg", crop)
#         print("finish crop")
        
        cannyMask()
        inpainter()
        repairImage("70,160")
        # repairImage(pos)

    return {"filename": crop.filename}



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