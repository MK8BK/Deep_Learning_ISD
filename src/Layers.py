from src.Functions import *
import numpy as np

class DenseLinearLayer:
    """
        DenseLinearLayer wrapper class
            Fully connected, unactivated layer
    """
    def __init__(self, N, F):
        """
            Constructor uses glorot initialisation
            @param: N : int, number of neurons
            @param: F : int, number of input features 
                        (neurons in previous layer)
        """
        self.N = N
        self.F = F
        #glorot
        self.W = np.random.randn(self.N, self.F)* (2/(self.F+self.N))
        self.B = np.zeros((self.N,1))
        #if momentum
        #self.HdW = []
        #self.HdB = []

    def forward(self, X):
        """
            Performs WX+B given input matrix
            @param: X : an np.array of shape (F, batch_size)
            @return: Z (cached for backward pass), weights*inputs + biases
        """
        X = np.atleast_2d(X)
        self.X = X
        self.Z = np.dot(self.W,self.X)+self.B 
        return self.Z

    def backward(self, dZ, lr):
        """
            Performs the backward pass given gradient matrix
            @param: dZ : an np.array of shape (N, batch_size)
            @action: updates self.W and self.B using SGD
            @return: dX : gradient of error wrt previous layer
                    shape (F, batch_size)
        """
        dZ = np.atleast_2d(dZ)
        dW = np.dot(dZ, self.X.T)#*(1./self.X.shape[1])
        dB = dZ.mean(axis=1, keepdims=True)
        dX = np.dot(self.W.T, dZ)
        
        #if momentum
        #self.HdW, dW = apply_momentum(self.HdW, dW)
        #self.HdB, dB = apply_momentum(self.HdB, dB)
        #self.HdW.append(dW)
        #self.HdB.append(dB)
        
        #DESCENTE
        self.W = self.W - lr*dW
        self.B = self.B - lr*dB
        return dX

class DenseActivatedLayer(DenseLinearLayer):
    """Child of DenseLinearLayer, uses an activation function (R->R)"""
    def __init__(self, N, F, A):
        """
            Constructor, uses DenseLinearLayer constructor
                declares A, activation function for layer instance
            @param: N : int, number of neurons
            @param: F : int, number of input features 
                        (neurons in previous layer)
            @param: A : activation function, 
                        instance of src.Functions.Activation
        """
        super().__init__(N, F)
        self.A = A

    def forward(self, X: np.array) -> np.array:
        """
            Performs A(WX+B) given input matrix
            @param: X : an np.array of shape (F, batch_size)
            @return: A_of_z (cached for backward pass),
                 forward_activation(weights*inputs + biases)
        """
        super().forward(X)
        self.A_of_z = self.A.forward(self.Z)
        return self.A_of_z

    def backward(self, dA_of_Z: np.array, lr)->np.array:
        """
            Performs the backward pass given gradient matrix
            @param: dA_of_Z : an np.array of shape (N, batch_size)
            @action: elementwise activation.backward(Z)*dA_of_Z (Hadamrd Product)
            @action: updates self.W and self.B using super().backward(dZ)
            @return: dX : gradient of error wrt previous layer
                    shape (F, batch_size)
        """
        dA_of_Z = np.atleast_2d(dA_of_Z)
        #produit hadamard
        dZ = dA_of_Z*self.A.backward(self.Z)
        assert(dZ.shape == dA_of_Z.shape)
        dX = super().backward(dZ, lr) 
        return dX

class OutputLayer(DenseLinearLayer):
    def __init__(self, N, F, C):
        """
            Constructor, uses DenseLinearLayer constructor
                declares C, combined cost and activation function for layer
            @param: N : int, number of neurons (same as number of classes)
            @param: F : int, number of input features
                        (neurons in previous layer)
            @param: C : cost and activation function, 
                        Only implemented SoftmaxCrossEntropy for now (works best)
        """
        super().__init__(N, F)
        self.C = C

    def forward(self, X):
        """
            Performs C(WX+B) given input matrix
            @param: X : an np.array of shape (F, batch_size)
            @return: A_of_z (cached for backward pass),
                 forward_activation(weights*inputs + biases)
        """
        super().forward(X)
        self.A_of_z = self.C.forward(self.Z)
        return self.A_of_z

    def backward(self, Y, lr):
        """
            Performs the backward pass given labels matrix
            @param: Y : an np.array of shape (N, batch_size)
            @action: loss/cost function.backward(Y)
            @action: updates self.W and self.B using super().backward(dZ)
            @return: dX : gradient of error wrt previous layer
                    shape (F, batch_size)
        """
        dA_of_Z = self.C.backward(Y)
        dX = super().backward(dA_of_Z,lr)
        return dX

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
