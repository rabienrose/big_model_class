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

def softmax(z):
    z = z - np.max(z)
    return np.exp(z) / np.sum(np.exp(z))

def ReLU(x):
    return np.maximum(0,x)

def dReLU(x):
    return 1 * (x > 0) 

class MLP:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = [256,128]
        self.output_size = 10

        self.W1 = np.random.randn(self.hidden_size[0], self.input_size)
        self.b1 = np.random.randn(self.hidden_size[0],1)
        self.W2 = np.random.randn(self.hidden_size[1], self.hidden_size[0])
        self.b2 = np.random.randn(self.hidden_size[1],1)
        self.W3 = np.random.randn(self.output_size, self.hidden_size[1])
        self.b3 = np.random.randn(self.output_size,1)
        self.lr = 0.001


    def forward(self, x):
        self.x = x
        self.z1 = np.matmul(self.W1, x) + self.b1 # (256, 784) * (784, n) + (256, 1) = (256, n)
        self.a1 = ReLU(self.z1)
        self.z2 = np.matmul(self.W2, self.a1) + self.b2 # (128, 256) * (256, n) + (128, 1) = (128, n)
        self.a2 = ReLU(self.z2)
        self.z3 = np.matmul(self.W3, self.a2) + self.b3 # (10, 128) * (128, n) + (10, 1) = (10, n)
        self.a3 = softmax(self.z3)
        result = np.argmax(self.a3, axis=0)
        return result

    def backward(self, y):
        delta3 = (1/y.shape[1])*(self.a3 - y)
        loss=np.mean((self.a3 - y)**2)
        DW3 = np.matmul(delta3,self.a2.T) # (10, n) * (n, 128) = (10, 128)
        Da2 = np.matmul(delta3.T, self.W3).T # (n, 10) * (10, 128) = (128, n)
        Dz2 = Da2 * dReLU(self.z2) 
        DW2 = np.matmul(Dz2,self.a1.T) # (128, n) * (n, 256) = (128, 256)
        Da1 = np.matmul(Dz2.T, self.W2).T # (n, 128) * (128, 256) = (256, n)
        Dz1 = Da1 * dReLU(self.z1)
        DW1 = np.matmul(Dz1, self.x.T) # (256, n) * (n, 784) = (256, 784)

        Db3 = np.sum(delta3,axis=1,keepdims=True)
        Db2 = np.sum(Dz2,axis=1,keepdims=True)
        Db1 = np.sum(Dz1,axis=1,keepdims=True)
        #update weights
        self.W3 -= self.lr * DW3
        self.W2 -= self.lr * DW2
        self.W1 -= self.lr * DW1
        self.b3 -= self.lr * Db3
        self.b2 -= self.lr * Db2
        self.b1 -= self.lr * Db1
        #print(np.mean(DW3))


def read_a_line(f):
    line_text = f.readline()
    if len(line_text)==0:
        return [], []
    a_vec = line_text.split(",")
    label = np.zeros(10)
    label[int(a_vec[0])] = 1 
    img_data=[]
    for i in range(1,len(a_vec)):
        img_data.append(float(a_vec[i])/255.0)
    return label, img_data


f = open("mnist_test.csv","r")
labels=[]
img_datas=[]
while True:
    label, img_data = read_a_line(f)
    if len(label)==0:
        break
    labels.append(label)
    img_datas.append(img_data)

#get a random batch
batch_size=64
mlp = MLP()
for i in range(100):
    idx = np.random.randint(0, len(labels), batch_size)
    label = np.array(labels)[idx]
    label=label.T
    img_data = np.array(img_datas)[idx]
    img_data=img_data.T
    output = mlp.forward(img_data)
    mlp.backward(label)

