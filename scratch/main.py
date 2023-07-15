import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

img_s=28

def show_img():
    f = open("mnist_test.csv","r")
    line_text = f.readline()
    line_text = f.readline()
    a_vec = line_text.split(",")
    
    img = Image.new('RGB', (img_s, img_s), color = 'black')
    draw = ImageDraw.Draw(img)
    for i in range(img_s):
        for j in range(img_s):
            gray=int(a_vec[i*img_s+j+1])
            img.putpixel((j,i),(gray,gray,gray))
    img.save('output.bmp')
    img.show()


# mlp only in numpy of tree layers
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, x):
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.h1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        self.h2 = np.exp(self.z2) / np.sum(np.exp(self.z2))  # softmax
        result = np.argmax(self.h2, axis=0)
        return result

    def loss_entropy(self, y):
        y = np.eye(self.output_size)[y]
        return -np.sum(y * np.log(self.h2 + 1e-7)) / len(y)

    def backward(self, y):
        y = np.eye(self.output_size)[y]
        delta3 = self.h2 - y
        self.dW2 = np.dot(self.h1.T, delta3)
        self.db2 = np.sum(delta3, axis=0)
        delta2 = np.dot(delta3, self.W2.T) * (self.z1 > 0)
        self.dW1 = np.dot(self.x.T, delta2)
        self.db1 = np.sum(delta2, axis=0)


    def update(self, lr=0.01):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


def read_a_line(f):
    line_text = f.readline()
    a_vec = line_text.split(",")
    label = int(a_vec[0])
    img_data=[]
    for i in range(1,len(a_vec)):
        img_data.append(float(a_vec[i])/255.0)
    return label, img_data

f = open("mnist_test.csv","r")
label, img_data = read_a_line(f)
mlp = MLP(img_s*img_s, 100, 10)
img_data = np.array(img_data)
output = mlp.forward(img_data)
print(output)

