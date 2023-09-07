import sys, os
sys.path.append(os.pardir)
import cv2
import numpy as np

#load_img : 불러온 이미지를 반환
def load_img(fname):
    img = cv2.imread(fname)
    
    return img

#scaling_img : 이미지를 확대
def scaling_img(img, x, y):
    pass

#rotate_img : 이미지를 회전
def rotate_img(img):
    pass

#translate_img : 이미지 이동
def translate_img(img):
    pass