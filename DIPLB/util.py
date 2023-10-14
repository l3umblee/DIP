import numpy as np
import cv2

#im2col : only for 1 gray image (batch_size = 1, channel = 1)
def im2col(input_data, filter_h, filter_w, stride=1, pad=1):
    H, W = input_data.shape
    oh = (H + 2*pad - filter_h) // stride + 1
    ow = (W + 2*pad - filter_w) // stride + 1

    img = np.pad(input_data, [(pad, pad), (pad, pad)], 'constant')
    col = np.zeros((filter_h, filter_w, oh, ow))

    for y in range(filter_h):
        ymax = y + stride*oh
        for x in range(filter_w):
            xmax = x + stride*ow
            col[y,x,:,:] = img[y:ymax:stride, x:xmax:stride]

    col = col.transpose(2, 3, 0, 1).reshape(oh*ow, -1)
    return col
