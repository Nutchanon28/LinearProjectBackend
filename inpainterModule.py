import numpy as np
import matplotlib.pyplot as plt
import time 
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from PIL import Image as im
import cv2


class Inpainter():
    def __init__(self, image, mask, patch_size=9, plot_progress=False):
        self.image = image.astype('uint8')
        print(f"self.image shape{self.image.shape}") 
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.priority = None

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        while keep_going:
            self._find_front() #หาจุดที่เป็น edge ของ mask
            if self.plot_progress:
                self._plot_image()

            self._update_priority() #ขั้นตอนนี้ยังงงๆอยู่ว่าแต่ละตัวมันแทนอะไร

            target_pixel = self._find_highest_priority_pixel() #เลือกจุดที่มี priority สูงสุด เพื่อทำการแทนที่จุดนั้น
            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel) #เลือกจุดที่เหมาะสมที่สุดในการแทนที่
            print('Time to find best: %f seconds'
                  % (time.time()-find_start_time))

            self._update_image(target_pixel, source_patch) #แทยที่จุดนั้น

            keep_going = not self._finished()

        print('Took %f seconds to complete' % (time.time() - start_time))
        return self.working_image

    def _validate_inputs(self):
        print("I received and validating")
        print(f"image.shape = {self.image.shape[:2]} mask.shape = {self.mask.shape}")
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)  # TODO: check if this is necessary

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]

        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([height, width])

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

        #cv2.imwrite("./fillProcess/front_updating.jpg", self.front)
        """cv2.imwrite("./mask.jpg", self.working_mask * 255)
        cv2.imwrite("./image.jpg", self.working_image * 255)
        cv2.imwrite("./confidence.jpg", self.confidence * 255)"""

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8') #จะได้ binary image ที่ขอบของ mask เป็นสีขาว 
        cv2.imwrite("./fillProcess/front_updating.jpg", self.front * 255)
        cv2.imwrite("./fillProcess/mask_updating.jpg", self.working_mask * 255)
        cv2.imwrite("./fillProcess/image_updating.jpg", self.working_image)
        #cv2.imwrite("./fillProcess/image_updating4.jpg", self.image)
        # TODO: check if scipy's laplace filter is faster than scikit's

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front
        #print(f"anto {self.confidence.shape} * {self.data.shape} * {self.front.shape}")
        cv2.imwrite("./fillProcess/priority_updating.jpg", self.priority * 10000)

    def _update_confidence(self):
        """"
        ex. patch size = 9, y = 10, x = 10
            half-patch = 4
            new_confidence[y, x] มันจะเอา confidence[y - 4 : y + 4 + 1, x - 4 : x + 4 + 1] -> บริเวณสี่เหลี่ยมรอบๆ xy มาเฉลี่ยกับ
            ขนาดของ patch
            front คือขอบของ mask 

            แต่ละรอบของการทำงานใน method นี้ confidence ที่ถูก update จะเอามาจาก frontตรงที่เป็น 1 คือ ขอบๆของ working_mask
            ขอบๆของ mask จะค่อยๆจางลง 
        """
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1) # [[y, x] where front_positions[y][x] == 1]
        #argwhere จะ return [row, col]  ของตำแหน่งที่ไม่เป็น 0, ใช้หาตำแหน่งที่ต้องการโดย argwhere(เงื่อนไข)
        for point in front_positions:
            patch = self._get_patch(point) #[[top, bottom], [left, right]]
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch) #(จำนวนของจุดที่ไม่ใช่ target mask ใน patch ที่ center = (point[0], point[1])) / patch size

        self.confidence = new_confidence 
        """
        ตำแหน่งที่มี confidence มาก คือ ตำแหน่งที่มีจุดที่ไม่ได้อยู่ใน mask มาก 
        น่าจะหมายถึงตำแหน่งที่มีจุดที่หายไปน้อยที่สุด -> มีความมั่นใจในการแทนที่ตำแหน่งนี้มากกว่าตำแหน่งที่หายไปเยอะๆ
        """
        cv2.imwrite("./fillProcess/confidence.jpg", self.confidence * 255)

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal*gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # To be sure to have a greater than 0 data

        cv2.imwrite("./fillProcess/data_updating.jpg", self.data * 1000)

    def _calc_normal_matrix(self):
        """
        เอา working_mask ไปทำ convolution กับ x_kernel กับ y_kernel 

        kernel ดูคล้ายๆของ  sobel

        """
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal)) #normal.shape = (height, width, 2)
        
        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1 #ตัวไหนเป็น 0 ให้มีค่าเป็น 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image))) # grandient.shape = (2, height, width)
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2) # grandient_val.shape = (height, width)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1) 
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(), #returns indices of the max element of the array
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel):
        """
        
        """
        target_patch = self._get_patch(target_pixel) #บริเวณที่ต้องการแทนที่
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image) #convert to lab image ทำไม...
        cv2.imwrite("./fillProcess/lab_image.jpg", self.working_image)

        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                """
                ดูทุก patch ที่มุมซ้ายบนอยู่ตำแหน่ง (y, x)
                เพื่อหา patch ที่เหมาะที่สุดสำหรับนำไปใช้แทน target_patch
                """
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if self._patch_data(self.working_mask, source_patch) \
                   .sum() != 0:
                    continue #ถ้า mask บริเวณนั้นมีส่วนที่เป็นสีขาว -> เป็นบริเวณที่มีจุดที่ยังไม่ได้แทนที่อยู่ -> continue

                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _update_image(self, target_pixel, source_patch):
        """
        source_patch คือ patch ที่คล้ายกับ target_patch ที่สุด
        """
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]] #pixels_positions คือตำแหน่งที่เป็นสีขาวของ mask ที่อยู่ใน target_patch
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence #เปลี่ยนค่า confidence ของ pixel_positions เป็นค่าconfidence ที่ตำแหน่ง target_pixel


        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask) #ทำให้เป็น 3 tensors
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask) # calculate new data

        #update image
        self._copy_to_patch(
            self.working_image,
            target_patch,
            new_data
        )
        #remove จุดสีขาว ใน mask ที่บริเวณ target_patch
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )

    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch

    def _calc_patch_difference(self, image, target_patch, source_patch):
        """
            ดูทุก patch ใน original image กับ target_patch
            หา square distance กับ euclidean distance ระหว่างทั้งสอง patch
        """
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask) #ทำให้ mask มี 3 tensor -> r,g,b
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data)**2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance

    def _finished(self):
        """
        finish เมื่อไม่มีจุดขาวใน mask แล้ว
        """
        height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        total = height * width
        print('%d of %d completed' % (total-remaining, total))
        return remaining == 0

    @staticmethod
    def _patch_area(patch):
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        #copy ให้ dest บริเวณ dest_patch มีค่าเท่ากับ data
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
