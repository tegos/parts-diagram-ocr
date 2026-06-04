# from PIL import Image
import os

from pytesser3 import *

image_file = '../images/22.png'
p = os.path.join(os.path.dirname(__file__), image_file)
print p
im = Image.open(image_file)
text = image_to_string(im, psm=9)
print text
text = image_file_to_string(p, psm=10)
print text
# text = image_file_to_string(image_file, graceful_errors=True)
# print "=====output=======\n"
# print text
