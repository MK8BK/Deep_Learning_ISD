from src.Functions import *
import numpy as np


class DenseLinearLayer:
    def __init__(self, N: int, F: int):
        self.N = N
        self.F = F
        self.W = np.random.randn(self.N, self.F)* (2/(self.F+self.N))
        self.B = np.zeros((self.N,1))
        #if momentum
        #self.HdW = []
        #self.HdB = []

    def forward(self, X: np.array) -> np.array:
        X = np.atleast_2d(X)
        self.X = X
        self.Z = np.dot(self.W,self.X)+self.B 
        return self.Z

    def backward(self, dZ: np.array, lr)->np.array:
        #
        dZ = np.atleast_2d(dZ)
        #dE/dZ = dZ
        #dE/dX = dE/dZ * dZ/dX
        #    Z = WX+B
        #dE/dB = dE/dZ * dZ/dB
        #    Z = 1*B
        dW = np.dot(dZ, self.X.T)#*(1./self.X.shape[1])
        dB = dZ.mean(axis=1, keepdims=True)
        dX = np.dot(self.W.T, dZ)
        
        #DESCENTE
        #if momentum
        #self.HdW, dW = apply_momentum(self.HdW, dW)
        #self.HdB, dB = apply_momentum(self.HdB, dB)
        #self.HdW.append(dW)
        #self.HdB.append(dB)
        
        self.W = self.W - lr*dW
        self.B = self.B - lr*dB
        return dX

class DenseActivatedLayer(DenseLinearLayer):
    def __init__(self, N, F, A):
        super().__init__(N, F)
        self.A = A

    def forward(self, X: np.array) -> np.array:
        #Z = 
        super().forward(X)
        self.A_of_z = self.A.forward(self.Z)
        return self.A_of_z

    def backward(self, dA_of_Z: np.array, lr)->np.array:
        dA_of_Z = np.atleast_2d(dA_of_Z)
        #dE/Z = dE/dA(Z) * dA(Z)/dZ
        #produit hadamard
        dZ = dA_of_Z*self.A.backward(self.Z)
        assert(dZ.shape == dA_of_Z.shape)
        dX = super().backward(dZ, lr) 
        return dX

class OutputLayer(DenseLinearLayer):
    def __init__(self, N, F, C):
        super().__init__(N, F)
        self.C = C

    def forward(self, X):
        super().forward(X)
        self.A_of_z = self.C.forward(self.Z)
        return self.A_of_z

    # use super().forward()
    def backward(self, Y, lr):
        #cost = self.C.compute_cost(Y)
        #, cost 
        dA_of_Z = self.C.backward(Y)
        dX = super().backward(dA_of_Z,lr)
        #print(self.W)
        return dX### dimension errors for now, working on it,


if __name__=="__main__":
    A = [[1,1,1],[1,1,1],[1,1,1]]
    B = [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]
    C = [[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]]
    past = [np.array(A), np.array(B), np.array(C)]
    present = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    present = np.array(present).astype('float64')
    print(apply_momentum(past, present))
#    hdZW.append([[0,0,1],[0,1,1],[1,0,1]])
    print(f"Empty main in : '{__file__[-9:]}'")
