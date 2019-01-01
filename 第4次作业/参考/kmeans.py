import readMnist
import numpy as np

def evaluate(C,result,trainL):
    wrong = 0
    for i in range(0,60000):
        n = C[i]
        if result[n] != trainL[i]:
            wrong = wrong + 1
    
    print('Accuracy = ' , (60000 - wrong) / 600,'%')

def Kmeans(trainI,trainL):
    #trainI = 60000 * 784
    #trainL = 1 * 60000
    #K = np.random.randint(0,255,size = [10,784]) 
    K = trainI[190:200,:]
    #K is 10 * 784 matrix which row is eigenvetor
    C = np.zeros(60000,dtype = np.int)
    terms = 0

    while(1):
        
        for i in range(0,60000):
            distance = np.zeros([10,1])

            for j in range(0,10):
                distance[j,0] = np.sqrt(np.sum(np.square(trainI[i,:] - K[j,:])))#conmput the distance

            C[i] = np.argmin(distance)#距离最小值索引
        
        Group = np.zeros([10,10],dtype = np.int)#count

        for i in range(0,60000):
            n = C[i]
            p = trainL[i]
            Group[n,p] = Group[n,p] + 1
        Math_result = np.argmax(Group,axis = 1)

        Miu = np.zeros([10,784],dtype = np.int)#new K
        count = np.ones([10,1],dtype = np.int)#count for computing MIU

        for i in range(0,60000):
            n = C[i]
            Miu[n,:] = Miu[n,:]+ trainI[i,:]#算出均值
            count[n,0] = count[n,0] + 1

        for i in range(0,10):
            for j in range(0,784):            
                Miu[i,j] = Miu[i,j] / count[i,0]
        

        if (K != Miu).any():
            print("Now is terms ",terms)
            evaluate(C,Math_result,trainL)
            terms = terms + 1
            K = Miu #开始新一轮迭代
        else:
            break

    return C
            


if __name__ == '__main__':
    trainI,trainL = readMnist.run()
    result = Kmeans(trainI,trainL)
    print("done!")