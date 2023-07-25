import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import time

def get_current_time_in_ms():
    return time.time()*1000

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
    z = z - np.max(z, axis = 0, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis = 0, keepdims=True)

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
        self.lr = 0.01
        self.loss_cul=0
        self.accuracy=0

    def forward(self, x, y):
        self.x = x
        self.y = y
        self.z1 = np.matmul(self.W1, x) + self.b1 # (256, 784) * (784, n) + (256, 1) = (256, n)
        self.a1 = ReLU(self.z1)
        self.z2 = np.matmul(self.W2, self.a1) + self.b2 # (128, 256) * (256, n) + (128, 1) = (128, n)
        self.a2 = ReLU(self.z2)
        self.z3 = np.matmul(self.W3, self.a2) + self.b3 # (10, 128) * (128, n) + (10, 1) = (10, n)
        self.a3 = softmax(self.z3)
        self.delta3 = (1/y.shape[1])*(self.a3 - y)
        self.loss_cul=self.loss_cul+np.mean((self.a3 - y)**2)
        output = np.argmax(self.a3, axis=0)
        result=np.zeros(len(output))
        for i in range(len(output)):
            if output[i]==np.argmax(y[:,i]):
                result[i]=1
        self.accuracy=self.accuracy+np.mean(result)
        return result

    def reset_epoch(self):
        self.loss_cul=0
        self.accuracy=0

    def backward(self):
        #t1=get_current_time_in_ms()
        DW3 = np.matmul(self.delta3,self.a2.T) # (10, n) * (n, 128) = (10, 128)
        #t2=get_current_time_in_ms()
        Da2 = np.matmul(self.delta3.T, self.W3).T # (n, 10) * (10, 128) = (128, n)
        Dz2 = Da2 * dReLU(self.z2) 
        DW2 = np.matmul(Dz2,self.a1.T) # (128, n) * (n, 256) = (128, 256)
        Da1 = np.matmul(Dz2.T, self.W2).T # (n, 128) * (128, 256) = (256, n)
        Dz1 = Da1 * dReLU(self.z1)
        DW1 = np.matmul(Dz1, self.x.T) # (256, n) * (n, 784) = (256, 784)

        Db3 = np.sum(self.delta3,axis=1,keepdims=True)
        Db2 = np.sum(Dz2,axis=1,keepdims=True)
        Db1 = np.sum(Dz1,axis=1,keepdims=True)
        #update weights
        self.W3 -= self.lr * DW3
        self.W2 -= self.lr * DW2
        self.W1 -= self.lr * DW1
        self.b3 -= self.lr * Db3
        self.b2 -= self.lr * Db2
        self.b1 -= self.lr * Db1

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

def convert_vectors_2_matrix(vectors):
    out_mat=np.zeros((len(vectors[0]),len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors[0])):
            out_mat[j][i]=vectors[i][j]
    return out_mat

def read_data_from_file(file_name):
    f = open(file_name,"r")
    labels=[]
    img_datas=[]
    t1=get_current_time_in_ms()
    while True:
        label, img_data = read_a_line(f)
        if len(label)==0:
            break
        labels.append(label)
        img_datas.append(img_data)
    t2=get_current_time_in_ms()
    labels=convert_vectors_2_matrix(labels)
    img_datas=convert_vectors_2_matrix(img_datas)
    return labels, img_datas

train_labels, train_img_datas = read_data_from_file("mnist_train.csv")
test_labels, test_img_datas = read_data_from_file("mnist_test.csv")

batch_size=64
mlp = MLP()
epoch=100
for i in range(epoch):
    for j in range(train_labels.shape[1]//batch_size-1):
        start_idx=j*batch_size
        end_idx=start_idx+batch_size
        label = train_labels[:,start_idx:end_idx]
        img_data = train_img_datas[:,start_idx:end_idx]
        output = mlp.forward(img_data, label)
        mlp.backward()
    print("epoch:",i," loss:",mlp.loss_cul/(train_labels.shape[1]//batch_size-1))
    mlp.reset_epoch()
    result = mlp.forward(test_img_datas, test_labels)
    print("epoch:",i," accuracy:",np.mean(result))
    
