import sys, os
sys.path.append(os.pardir)
from DIPLB.common import load_img, scaling_img, rotate_img

fname = "imgs/testimg_black.png"

img = load_img(fname)

#scaling_img(img, 2, 2)
rotate_img(img, 90)