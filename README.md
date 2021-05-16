# neural_network_from_scratch
A simple neural network implementation in python to understand the working.


about the dataset:
  - LBW (Low birthweight) classification
about the model:

a] 3 layers. Input, hidden, and output, 1 layer each.
b] conditions for the best number of perceptrons in hidden layer:
    	The number of hidden neurons should be between the size of the input layer and the size of the output layer.
    	The number of hidden neurons should be less than twice the size of the input layer.
    	citation: https://www.heatonresearch.com/2017/06/01/hidden-layers.html
        so our choice is 11, since it gave us better results.
  ouptut layer has 1 perceptron again.
c] mini batch processing is done in loop here. The weight matrix 1
   dimensions no.of features* no. of hidden layer neurons, i.e, 9*11 here
   weight matrtix 2 has dimentions 11*1.
   bias is 0 , this combined with the activation functions used below gave us the best results among the options we had.
d] input to hidden has leaky relu as activation, hiddden to output has sigmoid.
e] loss funtions is mean squared error. this can be seen in function backprop, where new_weights1 variable calculation
    has a term 2*(self.y - self.out) * sigmoid_derivative(self.out), which is the expression given by differentiating the
    loss function.
f] learning rate for our model is 0.01.
 
-and finally our testing accuracy lands at 86.2%.

uniqeuness in our model:
	-we have done mini batch processing in loop. this allows our model to look at the training dataset in smaller parts and then minimize the errors accordingly
 	and is computationally efficient as all computer resources are not being used to process a single sample rather are being used for all training samples
 

steps to run files:

	- all codes are written in python3
	-running preprocess.py will give the processed.csv
	-running the neural_network.py after that will give the performance metrics of our model.
	-after the execution , there will 7 set of metrics displayed. 6 of training 6 batches, and the last one is testing, all of which are named accordingly in the output. 
	-the required datasets are in the same folder, so normal running is fine.
	-but to run the preprocess.py code, delete, or place the processed.csv elsewhere, since preprocess.py cannot create a file with the
 	name 'processed.csv' when a file of that name is present and it gives error
  
 
