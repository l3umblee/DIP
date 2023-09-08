import sys, os
sys.path.append(os.pardir)
import cv2
import numpy as np
import math
from PIL import Image

#load_img : 불러온 이미지를 반환
def load_img(fname):
    img = cv2.imread(fname)
    
    return img

#scaling_img : 이미지를 확대
def scaling_img(img, x, y):
    ori_x = img.shape[0]
    ori_y = img.shape[1]
    
    trans_np = np.array([[x, 0], [0, y]])

    last_np = np.array([ori_x-1, ori_y-1])    
    max_np = np.dot(trans_np, last_np)
    modified_np = np.zeros((max_np[0]+1, max_np[1]+1, 3))
    
    for xidx in range(ori_x):
        for yidx in range(ori_y):
            tmp_np = np.dot(trans_np, np.array([xidx, yidx]))
            tmp_x = tmp_np[0]
            tmp_y = tmp_np[1]
            modified_np[tmp_x][tmp_y][:] = img[xidx][yidx][:]
    
    modi_img = Image.fromarray(modified_np.astype(np.uint8))
    modi_img.save('Scaled_img.png', 'PNG')


#rotate_img : 이미지를 회전
def rotate_img(img, degree):
    theta = math.pi * (degree/180)

    ori_x = img.shape[0]
    ori_y = img.shape[1]

    trans_np = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    modi_x = ori_x*(math.cos(theta) + math.sin(theta))
    modi_x = math.ceil(modi_x)
    modi_y = ori_y*(math.cos(theta) + math.sin(theta))
    modi_y = math.ceil(modi_y)

    modified_np = np.zeros((modi_x, modi_y, 3))

    for xidx in range(ori_x):
        for yidx in range(ori_y):
            tmp_np = np.dot(trans_np, np.array([xidx, yidx]))
            #todo : 회전했을 때 좌표가 음수가 되는 문제 해결해야 함.
            
            tmp_x = math.floor(tmp_np[0] + ori_x*math.sin(theta))
            tmp_y = math.floor(tmp_np[1])

            modified_np[tmp_x][tmp_y][:] = img[xidx][yidx][:]
    
    modi_img = Image.fromarray(modified_np.astype(np.uint8))
    modi_img.save('Rotated_img.png', 'PNG')

#translate_img : 이미지 이동
def translate_img(img):
    pass