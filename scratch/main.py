import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


f = open("mnist_test.csv","r")
line_text = f.readline()
line_text = f.readline()
a_vec = line_text.split(",")
label=int(a_vec[0])
img_s=28
img = Image.new('RGB', (img_s, img_s), color = 'black')
draw = ImageDraw.Draw(img)
for i in range(img_s):
    for j in range(img_s):
        gray=int(a_vec[i*img_s+j+1])
        img.putpixel((j,i),(gray,gray,gray))
img.save('output.bmp')

