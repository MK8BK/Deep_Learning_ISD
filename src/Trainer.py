from src.LoadData import *
from src.Functions import *
from src.Layers import *
from src.NeuralNetwork import *

def train_on_subset(net, path_str, epochs, batch_size, equilibrium=False,
			lr=0.01, test_on_all=False, iter_test=20):
	
	accuracies = [0]
	costs = []
	length_of_string = len(str(epochs))
	
	if test_on_all:
		X, Y = load_data_set(path_str)
	
	for i in range(epochs):
		x, y = load_training_set(path_str, batch_size=batch_size, equilibrium=equilibrium)
		res = net.forward(x)
		net.backward(y)
		
		if not test_on_all:
				cost = compute_cost(res, y)
				accuracy = percent_good(res, y)
				accuracies.append(accuracy)
				costs.append(cost)

		if i%iter_test==0:
			if test_on_all:
				res = net.forward(X)
				cost = compute_cost(res, Y)
				accuracy = percent_good(res, Y)
				accuracies.append(accuracy)
				costs.append(cost)
			print(f"Iteration: {i:{length_of_string}} | Accuracy: {accuracy} %")
	return accuracies, costs

#def train():
