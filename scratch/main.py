import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


f = open("mnist_test.csv","r")
line_text = f.readline()
print(len(line_text))
# img = Image.new('RGB', (32, 32), color = 'black')
# draw = ImageDraw.Draw(img)
# img.putpixel((10,20),(255,0,0,))
# img.save('output.bmp')

