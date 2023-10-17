import sys, os
sys.path.append(os.pardir)
from DIPLB.common import *

fname = "imgs/test.jpg"

input_img = load_img(fname, 0)
#output_img = gaussian_filter(input_img, 7)
output_img = edge_detection(input_img)
show_img(output_img)