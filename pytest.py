import sys, os
sys.path.append(os.pardir)
from DIPLB.common import load_img, scaling_img, rotate_img, translate_img, rotate_img_rough, backward_rotate

fname = "imgs/testimg_black.png"

img = load_img(fname)

# scaling_img(img, 2, 2)
#rotate_img_rough(img, 30)
#rotate_img(img, 30)
#translate_img(img, 10, 20)
backward_rotate(img, 30)