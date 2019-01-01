import numpy as np
import struct
import pickle
import matplotlib.pyplot as plt

def load_labels(enter):
    data = enter.read()
    offset = 0
    header = '>II'
    head = struct.unpack_from(header, data, offset)
    #取label的前两个整形数
    offset = struct.calcsize('>II')
    #定位到数据开始的位置
    #解析数据集，>60000B
    labels = struct.unpack_from('>60000B', data, offset)
    labels = np.reshape(labels, [60000])
    #转为一维数组
    return labels


def load_images(enter):
    data = enter.read()
    head = struct.unpack_from('>IIII', data, 0)
    offset = struct.calcsize('>IIII')
    #处理开头，同理

    number = head[1]
    height = head[3]
    #数目、长宽高
    width = head[2]

    bits = number * width * height
    bitsString = '>' + str(bits) + 'B'
    images = struct.unpack_from(bitsString, data, offset)
    images = np.reshape(images, [number, width * height])
    return images


def run():
    data = open('train-labels.idx1-ubyte','rb')
    trainL = load_labels(data)

    data1 = open('train-images.idx3-ubyte','rb')
    trainI = load_images(data1)

    return trainI,trainL
    # 查看前十个数据及其标签以读取是否正确
    #for i in range(10):
    #    print (trainL[i])
    #    plt.imshow(trainI[i], cmap='gray')
    #    plt.show()
    #print ('done')
    #print(trainI)





