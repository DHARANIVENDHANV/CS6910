# CS6910 ASSIGNMENT 1
### Dharanivendhan v (ed22s006)
#### Instructions to train and evaluate neural network models:
1. Install the libraries required for the project using the command below:
[pip install -r requirements.txt]

  2.There are two ways to use the notebook [DL_Assignment1.ipynb](https://github.com/DHARANIVENDHANV/CS6910/blob/master/ASSIGNMENT%201/DL_Assignment1.ipynb) to train a neural network model for image classification on the Fashion-MNIST dataset using categorical cross-entropy loss:

a. To use the best values for hyperparameters obtained from the wandb sweeps functionality, skip running cells in the section titled "Hyperparameter tuning using Sweeps" and run all other cells of the notebook to train the model. The final model will be trained on the entire training set and evaluated on the test set.

b. If you want to perform the hyperparameter manually feed it yourself, run the entire notebook.

3.To train the neural network model using mean squared error loss function for Fashion-MNIST image classification use the notebook #Assignment1_MSE#.Instructions apply same as above

4.To train the MNIST dataset for the best three hyper-parameter configurations, access the notebook #assignment_1_MNIST.
5.

Note:please remember to change the name of the project in the corresponding line of code whenever you need to log in wandb

#### Link to the project report:
-----

#### project explanation:
1.#assign# code contains all the necessary functions for neural network model for training Fashion-MNIST datasets for image classification.It contains all the optimizers such as stochastic,momentum,RMSprop,adam,nadam.This code suits for MNIST dataset too 

2.#deeplearningass# code contains feedforward neural network(Qn.3) for specifying number of neurons per hidden layer 

3.#deeplearningass1# code contains all the sweep operations performed by wandb for cross entropy(Qn.4) and mean squared error loss functions(Qn.8) for Fashion-MNIST datasets.This code also generates plots such as scatterplot for validation accuracy(Qn.5) and parallel co-ordinate plot, corelation summary(Qn.6).It also contains cofusion matrix plot for the best train and validation accuracy(Qn.7) 

4.#ass1MNIST# code contains the best three hyper-parameter configurations recommendation for MNIST datasets(Qn.10)

#### Neural_Network framework:
The code is developed from scratch without any libraries used for training models.This works for multiclass or singleclass classification problem as the last layer could take activation function accordingly as softmax or sigmoid.For the hyper-parameter search user needs to use functions Neural_Network()
Neural_Network()functions take on the train and validation data from the splitted Fashion-MNIST or MNIST data along with the hyper-parameters to a train a neural network model specified by num_neurons and num_hidden.Code is very flexible in selecting hyper-parameters configurations mentioned below:
- learnig_rate : Learning rate
- activation_function : activation functions to be used for hidden layers and output layer is automatically selected accordingly as softmax or sigmoid
- init_mode : initailization mode (Random_normal,Random_uniform,Xavier) for weights
- optimizer : stochastic,momentum,RMS,adam,nadam
- Batch_size : minibatch size
- loss : loss function ('mse','cross entropy')
- interations : epochs 
- lamb : lambda for L2 regularization of weights
- num_hidden : number of hidden layers
- num_neurons : number of neurons in every hidden layers

#assignqn3# code provides a freedom to specify the number of neurons in every hidden layers(Qn.3) but for sake of simplicity and as per the instructions in Qn.6 this was not utilised in wandb plots
The function returns 'Parameters',it is a dictionary which contains all the weights and biases after the end of iterations.



