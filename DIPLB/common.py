import sys, os
sys.path.append(os.pardir)
import cv2
import numpy as np
import math
from PIL import Image
from util import *

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
    modi_img.save('imgs/Scaled_img.png', 'PNG')


#rotate_img : 이미지를 회전 (grid 변환을 이용)
def rotate_img(img, degree):
    degree = degree % 360
    theta = math.pi * (degree/180)

    ori_x = img.shape[0]
    ori_y = img.shape[1]

    trans_np = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    modi_x = ori_x*math.cos(theta) + ori_y*math.sin(theta)
    modi_x = math.ceil(modi_x)
    modi_y = ori_y*math.cos(theta) + ori_x*math.sin(theta)
    modi_y = math.ceil(modi_y)

    modified_np = np.zeros((modi_x, modi_y, 3))
    
    x_max = 0
    y_max = 0

    square_np = np.array([[0, 0], [ori_x, 0], [0, ori_y], [ori_x, ori_y]])
    for idx in square_np:
        max_np = np.dot(trans_np, idx)
        if max_np[0] < 0:
            x_max = abs(max_np[0]) if abs(max_np[0]) > x_max else x_max
        if max_np[1] < 0:
            y_max = abs(max_np[1]) if abs(max_np[1]) > y_max else y_max

    for xidx in range(ori_x):
        for yidx in range(ori_y):
            tmp_np = np.dot(trans_np, np.array([xidx, yidx]))
            #todo : 회전했을 때 좌표가 음수가 되는 문제 해결해야 함.

            tmp_x = round(tmp_np[0] + x_max)
            tmp_y = round(tmp_np[1] + y_max)

            modified_np[tmp_x][tmp_y][:] = img[xidx][yidx][:]
                        
    modi_img = Image.fromarray(modified_np.astype(np.uint8))
    modi_img.save('imgs/Rotated_img.png', 'PNG')

#rotate_img_rough : 이미지 회전의 homework 버전
def rotate_img_rough(img, degree, clock_wise=True):
    degree = degree % 360
    theta = math.pi * (degree/180)

    ori_x = img.shape[0]
    ori_y = img.shape[1]

    if clock_wise:
        trans_np = np.array([
            [math.cos(theta), math.sin(theta)], 
            [-math.sin(theta), math.cos(theta)]])
    else:
        trans_np = np.array([
            [math.cos(theta), -math.sin(theta)], 
            [math.sin(theta), math.cos(theta)]])


    modified_np = np.zeros((ori_x, ori_y, 3))

    for xidx in range(ori_x):
        for yidx in range(ori_y):
            tmp_np = np.dot(trans_np, np.array([xidx, yidx]))

            tmp_x = int(tmp_np[0])
            tmp_y = int(tmp_np[1])

            if tmp_x > 0 and tmp_y > 0 and tmp_x < ori_x and tmp_y < ori_y: 
                modified_np[tmp_x][tmp_y][:] = img[xidx][yidx][:]
                        
    modi_img = Image.fromarray(modified_np.astype(np.uint8))
    modi_img.save('imgs/Rotated_img.png', 'PNG')

def backward_rotate(img, degree, clock_wise=True):
    degree = degree % 360
    theta = math.pi * (degree/180)

    ori_x = img.shape[0]
    ori_y = img.shape[1]

    #역행렬
    if clock_wise:
        trans_np = np.array([
            [math.cos(theta), -math.sin(theta)], 
            [math.sin(theta), math.cos(theta)]])
    else:  
        trans_np = np.array([
            [math.cos(theta), math.sin(theta)], 
            [-math.sin(theta), math.cos(theta)]])

    modified_np = np.zeros((ori_x, ori_y, 3))

    for xidx in range(ori_x):
        for yidx in range(ori_y):
            tmp_np = np.dot(trans_np, np.array([xidx, yidx]))

            tmp_x = int(tmp_np[0])
            tmp_y = int(tmp_np[1])

            if tmp_x > 0 and tmp_y > 0 and tmp_x < ori_x and tmp_y < ori_y:
                modified_np[xidx][yidx][:] = img[tmp_x][tmp_y][:]
    
    modi_img = Image.fromarray(modified_np.astype(np.uint8))
    modi_img.save('imgs/Rotated_img.png', 'PNG')

#translate_img : 이미지 이동
def translate_img(img, x_move, y_move):
    ori_x = img.shape[0]
    ori_y = img.shape[1]

    modified_np = np.zeros(((ori_x+x_move), (ori_y+y_move), 3))

    for xidx in range(ori_x):
        for yidx in range(ori_y):
            modified_np[xidx+x_move][yidx+y_move][:] = img[xidx][yidx][:]

    modi_img = Image.fromarray(modified_np.astype(np.uint8))
    modi_img.save('imgs/Translated_img.png', 'PNG')

def histogram_equalization(img):
    H, W = img.shape
    im2col_img = np.reshape(img, (1, -1))
    img_hist = np.zeros(256)

    for i in im2col_img[0]:
        img_hist[i] += 1

    for i in range(1, 256):
        img_hist[i] += img_hist[i-1]
    
    img_hist = img_hist * (255/(H*W))
    img_hist = np.asarray(img_hist, dtype=int)
    result_img = np.zeros_like(im2col_img)

    for i in range(H*W):
        result_img[0][i] = img_hist[im2col_img[0][i]]
    result_img = np.reshape(result_img, (H, W))
    return result_img

#gaussian_filter : only 2D gray image
def gaussian_filter(img, fsize):
    H, W = img.shape
    
    s = fsize/6 #Rule of thumb for Gaussian

    x = np.arange(-(fsize//2), (fsize//2)+1)
    y = np.array((1/(np.sqrt(2*np.pi)*s))*np.exp(-(x**2 / (2*s**2))))
    y /= y.sum() #1D Gaussian filter (vector)
    
    Gx_2d = np.outer(y, y)
    
    stride = 1
    padding = 1

    oh = int((H + 2*padding - fsize)/stride + 1)
    ow = int((W + 2*padding - fsize)/stride + 1)

    im2col_out = im2col(img, 3, 3, 1, 1)

    out = np.dot(im2col_out, Gx_2d.reshape(-1, 1))
    out = out.reshape(H, W)
    out = np.asarray(out, dtype=np.uint8)
    return out