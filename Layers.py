from Functions import *



#Z = WX + B
#W: matrice des poids
# dimensions: (N, F)
#X: matrice des entrees (input features)
# dimensions: (F, S)
#B: vecteur colonne
#   dimensions: (N, 1)

class DenseLinearLayer:
    def __init__(self, N: int, F: int):
        self.N = N
        self.F = F
        self.W = np.random.randn(self.N, self.F) * (2/(self.F+self.N))
        self.B = np.zeros((self.N,1))

    def forward(self, X: np.array) -> np.array:
        X = np.atleast_2d(X)
        assert(X.shape[0]==self.F),\
            f"""Incompatible shapes: W: {self.W.shape} 
                                     X:  {X.shape}"""
        self.X = X
        self.Z = np.dot(self.W,self.X)+self.B 
        return self.Z

    def backward(self, dZ: np.array, lr)->np.array:
        #
        dZ = np.atleast_2d(dZ)
        assert(dZ.shape == self.Z.shape),\
            f"""Incompatible shapes: dZ: {dZ.shape} 
                                     Z:  {self.Z.shape}"""
        #dE/dZ = dZ
        #dE/dX = dE/dZ * dZ/dX
        #    Z = WX+B
        #dE/dB = dE/dZ * dZ/dB
        #    Z = 1*B
        dW = np.dot(dZ, self.X.T)#*(1./self.S)
        dB = dZ.mean(axis=1, keepdims=True)
        dX = np.dot(self.W.T, dZ)
        #DESCENTE
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
        assert(dA_of_Z.shape == self.Z.shape),\
            f"""Incompatible shapes: dA_of_Z: {dA_of_Z.shape} 
                                     A_of_z:  {self.A_of_z.shape}"""
        #dE/Z = dE/dA(Z) * dA(Z)/dZ
        #produit hadamard
        dZ = np.multiply(dA_of_Z, self.A.backward(self.Z))
        assert(dZ.shape == dA_of_Z.shape)
        dX = super().backward(dZ, lr) 
        return dX

class OuputLayer(DenseLinearLayer):
    def __init__(self, N, F, C):
        super().__init__(N, F)
        self.C = C

    def forward(self, X):
        super().forward(X)
        self.A_of_z = self.C.forward(self.Z)
        return self.A_of_z

    # use super().forward()
    def backward(self, Y, lr):
        cost = self.C.compute_cost(Y)
        dA_of_Z = self.C.backward()
        dX = super().backward(dA_of_Z,lr)
        #print(self.W)
        return dX, cost


#class DenseLinearLayer
#    class DenseActivatedLayer
#    class OutputLayer



#    def __init__(self, predicted, real):
#        self.real = real
#        self.predicted = predicted
#        self.type = 'Binary Cross-Entropy'
#    def forward(self):
#        n = len(self.real)
#        loss = np.nansum(-self.real * np.log(self.predicted) - (1 - self.real) * np.log(1 - self.predicted)) / n
#        return np.squeeze(loss)
#    def backward(self):
#        n = len(self.real)
#        return (-(self.real / self.predicted) + ((1 - self.real) / (1 - self.predicted))) / n





#tried inheritance


### dimension errors for now, working on it,


if __name__=="__main__":
    print(f"Empty main in : '{__file__[-9:]}'")