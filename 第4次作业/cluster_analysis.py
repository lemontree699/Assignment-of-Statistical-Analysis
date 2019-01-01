import numpy as np
import struct
import matplotlib.pyplot as plt
import math
import random

def images_loading(file): 
	data = file.read()
	head = struct.unpack_from('>IIII', data, 0)
	offset = struct.calcsize('>IIII')
	number = head[1]
	width = head[2]
	height = head[3]

	bits = number * width * height
	bitsString = '>' + str(bits) + 'B'
	images = struct.unpack_from(bitsString, data, offset)
	images = np.reshape(images, [number, width *height])
	return number,images

def labels_loading(file):
    data = file.read()
    offset = 0
    header = '>II'
    head = struct.unpack_from(header, data, offset)
    offset = struct.calcsize('>II')
    labels = struct.unpack_from('>60000B', data, offset)
    labels = np.reshape(labels, [60000])
    return labels

data1 = open(r'.\data\train-images.idx3-ubyte','rb')
_, images = images_loading(data1)
data2 = open(r'.\data\train-labels.idx1-ubyte','rb')
labels = labels_loading(data2)

# 计算欧式距离
def european_distance(pa,pb):
	result = np.sum((pa - pb) ** 2)
	return math.sqrt(result)

def calc_accuracy(cluster_belong_to,labels_of_cluster,labels):
    wrong = 0
    for i in range(0,60000):
        n = cluster_belong_to[i]
        if labels_of_cluster[n] != labels[i]:
            wrong = wrong + 1
    return (60000 - wrong) / 600


k = 10  # 初始类的个数
start_from = random.randint(0,59989)
cluster = images[start_from:(start_from+k),:]  # 随机使用连续的k张图初始化k个类
cluster_belong_to = np.zeros(60000, dtype = np.int)  # 每张图片分别属于哪个类
number_of_iterations = 0  # 迭代次数

while (1):
	# 计算每张图片应该归于哪个类
	for i in range(60000):
		distance = []
		for j in range(k):
			distance.append(european_distance(images[i,:], cluster[j,:]))
		cluster_belong_to[i] = np.argmin(distance)
	# 算出新的类均值
	new_cluster = np.zeros([10,784],dtype = np.int)
	count = np.ones([10,1],dtype = np.int)
	for i in range(60000):
		n = cluster_belong_to[i]
		new_cluster[n,:] = new_cluster[n,:]+ images[i,:]
		count[n,0] = count[n,0] + 1
	for i in range(10):
		for j in range(784):     
			new_cluster[i,j] = new_cluster[i,j] / count[i,0]
	G = np.zeros([10,10],dtype = np.int)  # 计算每个类中标签最多的是哪个
	for i in range(60000):
		G[cluster_belong_to[i],labels[i]] = G[cluster_belong_to[i],labels[i]] + 1
	labels_of_cluster = np.argmax(G, axis = 1)
	# 如果新算出的类均值跟之前的不相等
	if (cluster != new_cluster).any():
		print("Now the number of iterations is", number_of_iterations)
		calc_accuracy(cluster_belong_to,labels_of_cluster,labels)
		number_of_iterations = number_of_iterations + 1
		cluster = new_cluster 
	# 如果新算出的类均值跟之前的相等
	else:
		print("Iterative completed. The final accuracy is", calc_accuracy(cluster_belong_to,labels_of_cluster,labels),'%')
		break

