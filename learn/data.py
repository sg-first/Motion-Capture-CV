import numpy as np

dataSize=2
x=np.zeros((4,4,dataSize))
y_armL=np.zeros((2,dataSize))
y_armR=np.zeros((2,dataSize))
y_legL=np.zeros((2,dataSize))
y_legR=np.zeros((2,dataSize))

x[:,:,0]=np.array([[0,1,0,1],
                [1,0,0,1],
                [0,1,1,0],
                [0,1,1,0]])
y_armL[:,0]=np.array([1,1])
y_armR[:,0]=np.array([0,0])
y_legL[:,0]=np.array([0,0])
y_legR[:,0]=np.array([0,0])

x[:,:,1]=np.array([[1,0,0,1],
                [1,0,0,1],
                [0,1,1,0],
                [0,1,1,0]])
y_armL[:,1]=np.array([0,0])
y_armR[:,1]=np.array([0,0])
y_legL[:,1]=np.array([0,0])
y_legR[:,1]=np.array([0,0])
