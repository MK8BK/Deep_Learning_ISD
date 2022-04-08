from Functions import *

class DenseLayer:
    """
        The dense fully connected layer class

        Attributes
        ----------
        A : functions.Function object
            The activation function
        N : int
            Number of neurons in layer
        F : int
            Number of features (neurons in the previous layer)
        S : int
            Batch size of input data
        W : np.array
            Weights of the layer, 2d-array of shape (N,F)
        B : np.array
            Biases of the layer, 2d-array of shape (N,1)
        Methods
        -------
        __init__(A, F, N, batch_size):
            Constructor, sets param values, generates random W, zeros B
        forward(X):
            Forward propagation of layer, given input data X -> A(WX+B)
        ##comment later
        __init__(A, F, N, batch_size):
            Backward propagation of layer, given d, genrates random weights, zeros B

    """
    def __init__(self, A: Function, F: int, N: int,
                         batch_size: int=32):
        #fonction d'activation
        self.A = A
        #Neurones dans cette couche
        self.N = N
        #Neurones dans la couche precedente, features
        self.F = F
        #S: nombre d'echantillons, samples
        self.S = batch_size 
        #W: poids associee a chaque neurone (ligne) et feature (colonne)
        self.W = np.random.randn(self.N, self.F) * np.sqrt(2/self.F)
        #B: biais associee a chaque neurone (ligne) (vecteur colonne)
        self.B = np.zeros((self.N,1))

    def forward(self, X: np.array) -> np.array:

        #avoid vectors of shape (n,)
        X = np.atleast_2d(X)
        assert(X.shape[0]==self.F and self.S==X.shape[1]),\
            f"""Incompatible shapes: W: {self.W.shape} 
                                     X:  {X.shape}"""

        #caching input values to the layer
        self.X = X

        #performing WX+b and activation, caching
        self.Z = np.dot(self.W,self.X)+self.B 
        self.A_of_z = self.A.output(self.Z)

        return self.A_of_z


    def backward(self, dA_of_Z: np.array, lr: float=0.01)->np.array:

        #avoid vectors of shape (n,)
        dA_of_Z = np.atleast_2d(dA_of_Z)
        assert(dA_of_Z.shape == self.A_of_z.shape),\
            f"""Incompatible shapes: dA_of_Z: {dA_of_Z.shape} 
                                     A_of_z:  {self.A_of_z.shape}"""

    
        #standard dense layer
        #elementwise (Hadamard) product
        dZ = np.multiply(dA_of_Z, self.A.derive(self.Z))

        #derivative of error with respect to weights
        dW = (1./self.S)*np.dot(dZ, self.X.T)
        #derivative of error with respect to biases
        dB = (1./self.S)*np.sum(dZ, axis=1, keepdims=True)
        #derivative of error with respect to layer input
        #                     (ouptput of previous layer)
        dX = np.dot(self.W.T, dZ)
        #Update weights
        self.W = self.W - lr*dW
        #Update biases
        self.B = self.B - lr*dB

        #implementing momentum

        #implementing 

        #send backward the derivative of the error with respect to input layer
        return dX

#tried inheritance
class OutputLayer(DenseLayer):
    def __init__(self, A: LossFunction, F: int, N: int,
        batch_size: int=32):

        super().__init__(A, F, N, batch_size=batch_size)

    def backward(self, dA_of_Z: np.array, lr: float=1.0)->np.array:
        #avoid vectors of shape (n,)
        dA_of_Z = np.atleast_2d(dA_of_Z)
        assert(dA_of_Z.shape == self.A_of_z.shape),\
            f"""Incompatible shapes: dA_of_Z: {dA_of_Z.shape} 
                                     A_of_z:  {self.A_of_z.shape}"""

        #output layer
        #Derivative (specific to loss function)
        dZ = self.A.derive(self.Z, dA_of_Z)
        #print(dZ.shape)
        #print(self.A_of_z.T.shape)
        #derivative of error with respect to weights
        dW = (1./self.S)*np.dot(dZ, self.X.T)
        #print(dW.shape, "<- here")
        #print(self.W.shape)
        #derivative of error with respect to biases
        dB = (1./self.S)*np.sum(dZ, axis=1, keepdims=True)
        #derivative of error with respect to layer input
        #                     (ouptput of previous layer)
        dX = np.dot(self.W.T, dZ)

        #Update weights
        self.W = self.W - lr*dW
        #Update biases
        self.B = self.B - lr*dB
        #send backward the derivative of the error with respect to input layer
        return dX


### dimension errors for now, working on it,


if __name__=="__main__":
    print(f"Empty main in : '{__file__[-9:]}'")