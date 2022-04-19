from src.LoadData import *
from src.Functions import *
from src.Layers import *
from src.NeuralNetwork import *

def train_on_subset(net, path_str, epochs, batch_size, equilibrium=False,
			lr=0.01, test_on_all=False, iter_test=20):
	"""
		The main training funtion
		@param: net : a NeuralNetwork object from src.NeuralNetwork
		@param: path_str : a string path to the data set folder
		@param: epochs : the number of training iterations
		@param: batch_size : the number of images per training
		@param: equilibrium : optional bool=False : divide batch_size equally per class
		@param:	lr : optional float=0.01 : learning rate
		@param: test_on_all : optional bool=False : test on whole data set (38400)
		@param: iter_test : optional int=20 : test every iter_test iterations, print
		@return: accuracies: list[float] : ordered percentages of testing (float 0-100.)
		@return: costs : list[float] : ordered costs of testing 
			len(accuracies)=len(costs)+1 (accuracies initialised with 0)

	"""

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
			print(f"Iteration: {i:{length_of_string}} | Accuracy: {accuracy:.2f} %")
	return accuracies, costs

#def train():
