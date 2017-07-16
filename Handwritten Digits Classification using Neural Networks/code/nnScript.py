import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import time
import sys

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigmoidVal = 1.0 / (1.0 + np.exp(-1.0 * z))
    return sigmoidVal


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    # loads the MAT object as a Dictionary
    mat = loadmat('S:/2ndSem/ML/1-Assignment/mnist_all(1)')

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]


    # Feature selection
    # Your code here.

    #We first check that what all column indices have same values.
    useful_col = np.all(train_data == train_data[0,:], axis=0)
    col_indices = []
    selected_features = []
    for i in range(0,useful_col.shape[0]):
        if (useful_col[i] == True):
            col_indices.append(i)
        else:
            selected_features.append(i)

    print(len(col_indices))
    #Now we delete those columns which were collected in col_indices in above loop
    train_data=np.delete(train_data, col_indices , axis=1)
    validation_data=np.delete(validation_data, col_indices, axis=1)
    test_data=np.delete(test_data, col_indices, axis=1)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    #global count
    #count = count+1

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Your code here

    # initializing scalar value of error function
    obj_val = 0
    # to store number of training examples
    num_data = training_data.shape[0]

    # Adding an extra row of zeroes to make it compatible for dot product with weight matrix
    training_data = np.append(training_data, np.zeros([len(training_data), 1]), 1)

    ''' Feedforward propogation .... START '''

    # output coming out from hidden layer before activation
    output_mid_before_activation = np.dot(training_data, w1.T)
    # applying activation sigmoid function on above output
    output_mid_after_activation = sigmoid(output_mid_before_activation)

    # Adding an extra row of zeroes to make it compatible for dot product with weight matrix
    output_mid_after_activation = np.append(output_mid_after_activation, np.zeros([len(output_mid_after_activation), 1]), 1)

    # output coming out from output layer before activation
    final_output_before_activation = np.dot(output_mid_after_activation, w2.T)
    # applying activation sigmoid function on above output
    final_output_after_activation = sigmoid(final_output_before_activation)

    ''' Feedforward propogation .... END '''

    # Matrix to store actual output of each training sample, i.e. the true label
    actual_output = np.zeros((training_label.shape[0], 10));
    for i in range(num_data):
        # Here we set the value of the column corresponding to true label to 1, others remain 0 ::::: K-encoding
        actual_output[i][training_label[i]] = 1;

    ''' Back propogation calculation .... START '''

    # Difference between output value and actual value.
    diff = final_output_after_activation - actual_output

    #As per the equation (9) in assignment pdf
    error_at_output_layer = np.dot(diff.T, output_mid_after_activation)

    summation_term = np.dot(diff, w2)
    term1 = (1-output_mid_after_activation)*output_mid_after_activation
    temp = term1 * summation_term

    #Using above values, we find the value as per the equation (12) in assignment pdf
    error_at_hidden_layer = np.dot(training_data.T, temp)

    ''' Back propogation calculation .... END '''

    ''' Regularization .... START '''

    #remove the last column
    error_at_hidden_layer = np.delete(error_at_hidden_layer, -1, axis=1)
    # As per the equation (17) in assignment pdf calc grad_w1
    grad_w1 = (error_at_hidden_layer.T + (lambdaval*w1))/num_data
    # As per the equation (16) in assignment pdf calc grad_w2
    grad_w2 = (error_at_output_layer + (lambdaval*w2))/num_data

    # As per the equation (7) in assignment pdf
    error_func = -(actual_output*np.log(final_output_after_activation) + (1-actual_output)*np.log(1-final_output_after_activation))
    # As per the equation (6) in assignment pdf
    error_func_final = np.sum(error_func/num_data)

    # As per the equation (15) in assignment pdf
    obj_val = error_func_final + (lambdaval/(2*num_data))*(np.sum(np.square(w1)) + np.sum(np.square(w2)))

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
   # print(count)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    # Adding an extra row of zeroes to make it compatible for dot product with weight matrix
    data = np.append(data, np.zeros([len(data), 1]),1)

    ''' Feedforward propogation .... START '''

    # output coming out from hidden layer before activation
    output_mid_before_activation = np.dot(data, w1.T)
    # applying activation sigmoid function on above output
    output_mid_after_activation = sigmoid(output_mid_before_activation)

    # Adding an extra row of zeroes to make it compatible for dot product with weight matrix
    output_mid_after_activation = np.append(output_mid_after_activation,np.zeros([len(output_mid_after_activation), 1]),1)

    # output coming out from output layer before activation
    final_output_before_activation = np.dot(output_mid_after_activation,w2.T)
    # applying activation sigmoid function on above output
    final_output_after_activation = sigmoid(final_output_before_activation)

    ''' Feedforward propogation .... END '''

    for i in range(0,data.shape[0]):
        #Now for each data(row of our matrix), we find the maximum valued column, which will be our predicted label for that sample
        labels = np.append(labels, np.argmax(final_output_after_activation[i]))

    return labels

"""**************Neural Network Script Starts here********************************"""
#for j in range(4,24,4):
#for i in range(0, 85, 5):
start_time = time.time();
train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter

lambdaval = 15

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.
count = 0
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset
#print("Result with hindden nodes : 50 and lambdavalue", i)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

obj = [selected_features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj, open('params.pickle', 'wb'))

end_time = time.time()
print("Time taken : ", end_time-start_time)