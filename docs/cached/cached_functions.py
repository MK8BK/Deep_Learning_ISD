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