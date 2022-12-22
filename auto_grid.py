import cv2, glob, os
import numpy as np


class AutoGrid():
    def __init__(self, box_size=256, x_box_num=None, y_box_num=None):
        self.box_size = box_size
        self.x_box_num = x_box_num
        self.y_box_num = y_box_num

        self.canvas = np.zeros((self.y_box_num*self.box_size, self.x_box_num*self.box_size, 3))
    
    def add(self, img, x_idx, y_idx, x_box_n, y_box_n):
        img_resized = cv2.resize(img, (x_box_n*self.box_size, y_box_n*self.box_size))
        self.canvas[y_idx*self.box_size:(y_idx+y_box_n)*self.box_size, x_idx*self.box_size:(x_idx+x_box_n)*self.box_size, :] = img_resized
    
    def to_3ch(self, gray_img):
        return gray_img[:, :, None].repeat(3, axis=2)
    
    def save(self, path):
        cv2.imwrite(path, self.canvas)

