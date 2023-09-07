import sys, os
sys.path.append(os.pardir)
from DIPLB.common import load_img

fname = "imgs/testimg.png"

img = load_img(fname)

print(img.shape)