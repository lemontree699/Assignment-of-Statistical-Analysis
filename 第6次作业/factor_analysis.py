import numpy as np
import pandas as pd
import math

file = pd.read_excel(r'./例6-3.xls')
# print(file.describe())
R = file.corr()  # 样本相关性矩阵
# print(R)

# 求相关性矩阵的特征值
eig_value, eigvector = np.linalg.eig(R)
# print(eig_value)
# print(eigvector)

# eig = pd.DataFrame()
# eig['names'] = file.columns[1:]
# eig['eig_value'] = eig_value
# eig.sort_values('eig_value', ascending = False, inplace = True)
# # print(eig)

m = 3
# 因子载荷矩阵
A = np.array(np.zeros((12,m)))
for i in range(m):
	A[:,i]=math.sqrt(eig_value[i])*eigvector[:,i]
print("m = 3:\n", A)

m = 4
# 因子载荷矩阵
A = np.array(np.zeros((12,m)))
for i in range(m):
	A[:,i]=math.sqrt(eig_value[i])*eigvector[:,i]
print("m = 4:\n", A)

m = 5
# 因子载荷矩阵
A = np.array(np.zeros((12,m)))
for i in range(m):
	A[:,i]=math.sqrt(eig_value[i])*eigvector[:,i]
print("m = 5:\n", A)