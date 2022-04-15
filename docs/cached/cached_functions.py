import numpy as np
from typing import Callable, Tuple


def derivative(f: Callable, epsilon: float = 1.0e-6, domain: np.ndarray=np.linspace(-10,10,200)):
    #print(domain, len(list(domain)))
    lst = [(f(x+epsilon)-(f(x-epsilon)))/(2*epsilon) for x in list(domain)]
    return np.array(lst)


def chain_derivative(functions: list[Callable], epsilon: float = 1.0e-6, domaine: np.ndarray=np.linspace(-10,10,200)):
    product = np.ones(len(domaine))
    inner = domaine.copy()
    for f in reversed(functions):
        product *= derivative(f,domain=inner)
        inner = f(inner)
    return list(product)


def MeanAbsoluteError(P, Y):
    return np.mean(np.abs(P - Y))

def MeanSquaredError(P, Y):
    return np.mean(np.power(P - Y, 2))

def RootMeanSquaredError(P, Y):
    return np.sqrt(MeanSquaredError(P,Y))

def standardize(x: np.ndarray):
    return np.exp(x-x.max())



def split_data(X: np.array, Y: np.array, 
                percent_test: int=30) -> tuple[np.array, np.array,
                                                np.array, np.array]:
    
    #assert same number of samples and labels
    assert(X.shape[1]==Y.shape[1]), \
        f"Y labels of {Y.shape[1]} samples dont match X data of {X.shape[1]} samples"

    m = X.shape[1]
    test_samples = int((m*percent_test)/100)
    X_test = np.zeros((X.shape[0], test_samples))
    X_train = X
    Y_test = np.zeros((Y.shape[0], test_samples))
    Y_train = Y
    for i in range(test_samples):
        to_be_removed = listChoice(list(range(X_train.shape[1])))
        X_test[:, i] = X_train[:, to_be_removed]
        X_train = np.delete(X_train, i, 1)
        Y_test[:, i] = Y_train[:, to_be_removed]
        Y_train = np.delete(Y_train, i, 1)

    return X_test, Y_test, X_train, Y_train



def binarize_predictions(p):
    maxvals = np.argmax(p, axis=0)
    p = np.zeros(p.shape)
    for i in range(len(maxvals)):
        p[maxvals[i], i] = 1
    return p





def convolution(img:  np.ndarray, detector: np.ndarray) -> np.ndarray:
    """
        Convolution d'une image par une matrice detectrice
        @param: <img>:np.ndarray ; l'image au format numpy
        @param: <detector>:np.ndarray ; la matrice detectrice au format numpy
        @return: <img_out>:np.ndarray ; la convolution de img par detector
    """
    assert(detector.shape[0]%2==1 and detector.shape[1]%2==1),\
        f"Filtre de taille paire: {detector.shape[0]}x{detector.shape[1]}"
    n, m = img.shape
    d = detector.shape
    h_conv = np.floor(n + - d[0] ).astype(int) + 1
    w_conv = np.floor(m + - d[1] ).astype(int) + 1
    img_out = np.zeros((h_conv, w_conv))
    b = d[0] // 2, d[1] // 2
    center_w = b[0] * 1
    center_h = b[1] * 1
    
    for i in range(h_conv):
        center_x = center_w + i * 1
        indices_x = [center_x + l * 1 for l in range(-b[0], b[0] + 1)]
        for j in range(w_conv):
            center_y = center_h + j * 1
            indices_y = [center_y + l * 1 for l in range(-b[1], b[1] + 1)]
            subimg = img[indices_x, :][:, indices_y]
            img_out[i][j] = np.sum(np.multiply(subimg, detector))

    return img_out


def pool(img: np.ndarray, pool_shape: Tuple[int,int]=(2,2),
                         pool_type: str="max")->np.ndarray:
    """
        poo
    """
    h, w = pool_shape
    #assert(img.shape[0]%h==0 and img.shape[1]%w==0)
    nimg = []
    for i in range(0,img.shape[0],h):
        nimg.append([])
        for j in range(0,img.shape[1],w):
            if pool_type=="max":
                val = np.max(img[i:i+h,j:j+w])
            elif pool_type=="average":
                if(img.dtype == np.dtype('int32')):
                    val = int(np.average(img[i:i+h,j:j+w]))
                else:
                    val = np.mean(img[i:i+h,j:j+w])
            nimg[-1].append(val)
    return np.array(nimg)


if __name__=="__main__":
    pass
