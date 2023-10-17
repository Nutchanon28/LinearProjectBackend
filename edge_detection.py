# pip install pillow
from PIL import Image, ImageFilter
import os

directory = "test"


def generateDetection(file_path, dest_dir):
    
    print("LAPLACIAN: start detecting edge...\n...")
    # ต้อง run code ใน directory เดียวกับโปรแกรมไม่งั้นมันจะไม่เจอไฟล์
    # img = Image.open(r"../test/apple_85.jpg")
    img = Image.open(file_path)
    color_img = img

    # convert to grayscale
    img = img.convert("L")

    # Calculating Edges using the passed laplacian Kernel
    final = img.filter(
        ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0)
    )

    file_name = file_path.split("/")[-1]
    final.save(f"./{dest_dir}/mask_edge.jpg")
    print("LAPLACIAN: saved edges...\n...")

    edges = {}
    threshold = 30

    # หลังจากเอาภาพผ่าน filter dot product มันจะได้ edge ขาวดำมา (ค่าสี 0 (ดำ) - 255(ขาว)) ค่าที่เป็นขอบจะเป็นสีขาว มีค่ามาก
    # ขอบของภาพที่เป็นขอบของภาพจริงๆจะถูกนับเป็นขอบด้วย เลยต้องเอา แถว/หลักที่ 1 กับ แถว/หลักสุดท้าย ออก

    for y in range(1, final.height - 1):
        found_left = False
        right_edge = 0

        for x in range(1, final.width - 1):
            if (
                final.getpixel((x, y)) > threshold
            ):  # ถ้าค่าสีใน pixel (x,y) สูงกว่า threshold ให้ถือว่าเป็นขอบ
                if not found_left:
                    edges[y] = [x]  # เก็บขอบซ้ายของแถว y
                    found_left = (
                        True  # ขอบซ้ายของแถว y คือ (x,y) อันแรกในแถว y ที่มากกว่า 10
                    )
                right_edge = x

        if y in edges:
            edges[y].append(
                right_edge
            )  # เก็บขอบขวาของแถว y ขอบขวาคือ (x,y) อันสุดท้ายในแถว y ที่มากกว่า 10

    # print(edges)

    final = final.convert("RGB")

    for y in range(0, final.height):
        for x in range(0, final.width):
            if y in edges and (
                x >= edges[y][0] and x <= edges[y][1]
            ):  # ถ้า x อยู่ระหว่างขอบซ้ายและขอบขวาของแถว y แสดงว่ามันอยู่ในขอบ
                color_img.putpixel(
                    (x, y), (255, 255, 255)
                )  # แทนที่ด้วยสีเขียว rgb(255, 255, 255)
            else:
                color_img.putpixel((x, y), (0, 0, 0))
                # ถ้าไม่เป็นขอบ ให้แทนทีด้วยสีดำ

    color_img.save(f"./{dest_dir}/mask.jpg")
    print("LAPLACIAN: pushed mask to ./upload!!")

def laplacianMask():
    generateDetection(r"./upload/crop.jpg", "upload")

# wrong_extension = ["xml", "DS_Store"]

# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     if os.path.isfile(f) and filename.split(".")[1] not in wrong_extension:
#         # print(f)
#         generateDetection(f, "result3")