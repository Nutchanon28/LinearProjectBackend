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
    img = cv2.imread("image.png", 1)
    print("read success!!")

    cv2.imshow("Image", img)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])

    return "Hello from the hell"

class FileUpload(BaseModel):
    files: List[bytes]  # Use bytes for file data

@app.post("/")
# async def create_upload_file(datas: dict) -> dict:
async def create_upload_file(mode: Annotated[str, Form()], image : UploadFile = File(...)) -> dict:
    async with aiofiles.open("./image.png", 'wb') as out_file:
        content = await image.read()  # async read
        await out_file.write(content)  # async write
    show_image()

    return {"filename": image.filename}



app.mount("/sample", StaticFiles(directory="sample"), name='images')
 
@app.get("/pic", response_class=HTMLResponse)
def serve():
    return """
    <html>
        <head>
            <title></title>
        </head>
        <body>
        <img src="sample/beach.png">
        <h1>Hello World</h1>
        </body>
        <style>
        h1{
            color:#FF3333
        }
        </style>
    </html>
    """