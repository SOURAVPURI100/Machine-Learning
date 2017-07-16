import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import datetime
import pickle


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    #######################################################################################################################
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    bias = np.ones((n_data, 1))
    # bias = np.transpose(bias)
    train_data = np.append(train_data, bias, 1)

    weight = np.reshape(initialWeights, (initialWeights.shape[0], 1))  # reshaping from (716,) to (716,1)

    theta = sigmoid(np.dot(train_data, weight))  # computing theta 50000 x 1
    # theta = np.transpose(theta)

    Y = np.transpose(labeli)  # 1 x 50000
    #  theta = np.transpose(theta) # N cross 1
    # theta = np.reshape(theta,(theta.shape[0],1))

    error1 = np.dot(Y, np.log(theta)) + np.dot((1.0 - Y), np.log((1.0 - theta)))
    error = (-np.sum(error1))/n_data
    #print(error)

    # error_grad = np.transpose(np.dot(((np.transpose(theta)) - Y), train_data))
    error_grad = np.dot(np.transpose(train_data), (theta - labeli))
    error_grad = np.transpose((error_grad/n_data))

    error_grad = (np.sum(error_grad, axis=0))
    #######################################################################################################################
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    #######################################################################################################################
    label = np.zeros((data.shape[0], 1))

    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones((data.shape[0], 1))
    #bias = np.transpose(bias)
    data = np.append(data, bias, 1)

    val = np.dot(data, W)             # Nx10
    val = sigmoid(val)
    for i in range(0, val.shape[0]):
        label[i][0] = np.argmax(val[i])
    #######################################################################################################################
    return label


def mlrObjFunction(initialWeights_b, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    #######################################################################################################################
    train_data, Y = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error_grad = np.zeros((n_feature + 1, n_class))

    # Add the  bias term
    bias = np.ones((n_data, 1))
    train_data = np.append(train_data, bias, 1)

    no_rows = int(initialWeights_b.shape[0]/n_class)

    initialWeights_b = np.reshape(initialWeights_b, (no_rows, n_class))  # reshaping from (7160,) to (716,10)
    # Compute Theta N-K matrix
    Theta_NK = np.exp(np.dot(train_data, initialWeights_b)/1000)
    Theta_sum = np.sum(Theta_NK, axis=1)
    Theta_NK = Theta_NK / Theta_sum[:, None]

    # Compute error value
    Theta_NK = np.add(Theta_NK, 0.00000001)
    error = - np.sum((np.multiply(Y, np.log(Theta_NK))))

    # Compute error Grad matrix
    temp_subtract = np.subtract(Theta_NK, Y)
    error_grad = np.dot(np.transpose(train_data), temp_subtract)
    error_grad = error_grad.flatten()
    #######################################################################################################################
    # HINT: Do not forget to add the bias term to your input data
    #print(error)
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    #######################################################################################################################
    n_data = data.shape[0]
    bias = np.ones((n_data,1))
    data = np.append(data, bias, 1)

    val = np.dot(data, W)        #N*D+1 , D+1*K = N*K

    label = np.argmax(val, axis=1)
    label = np.reshape(label,(label.shape[0], 1))  # reshaping from (50000,) to (50000,1)
    #######################################################################################################################
    # HINT: Do not forget to add the bias term to your input data

    return label


####################################################SVM Calculation#################################################
def SVM_Calc(test_data, test_label, train_data, train_label, validation_data, validation_label, kernelSvm, type):

    if(kernelSvm == 'linear'):
        print('Starting linear ...')
        clf = SVC(kernel=kernelSvm)
        clf.fit(train_data, train_label.ravel())

        accuracy_score_train = str(100*clf.score(train_data, train_label))
        print('Train data Accuracy - linear : ' + accuracy_score_train + '%')
        accuracy_score_test = str(100*clf.score(test_data, test_label))
        print('Test data Accuracy - linear : ' + accuracy_score_test + '%')
        accuracy_score_validation = str(100*clf.score(validation_data, validation_label))
        print('Test data Accuracy - linear : ' + accuracy_score_validation + '%')

    elif(kernelSvm == 'rbf'):


        if(type == 'default'):
            print('Starting rbf default ...')
            clf = SVC(kernel=kernelSvm)
            clf.fit(train_data, train_label.ravel())

            accuracy_score_train = str(100*clf.score(train_data, train_label))
            print('Train data Accuracy - rbf - default : ' + accuracy_score_train + '%')
            accuracy_score_test = str(100*clf.score(test_data, test_label))
            print('Test data Accuracy - rbf - default : ' + accuracy_score_test + '%')
            accuracy_score_validation = str(100*clf.score(validation_data, validation_label))
            print('Test data Accuracy - rbf - default : ' + accuracy_score_validation + '%')


        elif(type == 'gamma'):
            print('Starting rbf gamma=1, other parameters as default ...')
            clf = SVC(gamma=1, kernel=kernelSvm)
            clf.fit(train_data, train_label.ravel())

            accuracy_score_train = str(100*clf.score(train_data, train_label))
            print('Train data Accuracy - rbf - gamma=1 : ' + accuracy_score_train + '%')
            accuracy_score_test = str(100*clf.score(test_data, test_label))
            print('Test data Accuracy - rbf - gamma=1 : ' + accuracy_score_test + '%')
            accuracy_score_validation = str(100*clf.score(validation_data, validation_label))
            print('Test data Accuracy - rbf - gamma=1 : ' + accuracy_score_validation + '%')

        elif(type == 'c'):
            print('Starting rbf with various values of C ...')

            for i in range(0, 11):
                if(i==0):
                    cVal = 1
                else:
                    cVal = i*10
                print('Starting rbf c ... '+str(cVal))
                clf = SVC(C=(cVal), kernel=kernelSvm)
                clf.fit(train_data, train_label.ravel())

                accuracy_score_train = str(100*clf.score(train_data, train_label))
                print('Train data Accuracy - rbf - C=' + str(cVal) + ' : ' + accuracy_score_train + '%')
                accuracy_score_test = str(100*clf.score(test_data, test_label))
                print('Test data Accuracy - rbf - C=' + str(cVal) + ' : ' + accuracy_score_test + '%')
                accuracy_score_validation = str(100*clf.score(validation_data, validation_label))
                print('Test data Accuracy - rbf - C=' + str(cVal) + ' : ' + accuracy_score_validation + '%')

#######################################################################################################################



"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
print('STARTING............')

print('TIME ******************* '+str(datetime.datetime.now()))


# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}

for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    #print(i)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

print('TIME ******************* '+str(datetime.datetime.now()))

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
#######################################################################################################################
SVM_Calc(test_data, test_label, train_data, train_label, validation_data, validation_label, 'linear', 'default')
SVM_Calc(test_data, test_label, train_data, train_label, validation_data, validation_label, 'rbf', 'default')
SVM_Calc(test_data, test_label, train_data, train_label, validation_data, validation_label, 'rbf', 'gamma')
SVM_Calc(test_data, test_label, train_data, train_label, validation_data, validation_label, 'rbf', 'c')
#######################################################################################################################


print('TIME ******************* '+str(datetime.datetime.now()))

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

print('TIME ******************* '+str(datetime.datetime.now()))


f1 = open('params.pickle', 'wb')
pickle.dump(W, f1)
f1.close()

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
