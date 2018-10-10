##
# Project 1, starter code part b
#
# Read the data from the file ‘california_housing.data’.
# Each data sample is a row of 9 values: 8 input attributes and the
#  median housing price as targets. Divide the dataset at 70:30 ratio
# for validation and testing datasets. For selection of the best models,
# use 5-fold cross-validation on the validation data. The performances should
# be evaluated on the test data.

#3-layer feedforward neural network: 1 input layer, hidden layer with 30 neurpns
# and having ReLu activation function, and a linear output layer.
## mini-batch gradient descent: batch size 32
# L2 regularisation with weight decay parameter beta= 10^-3
# learning rate alpha: 10^-7

import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn import model_selection
from sklearn import preprocessing
import random

import os
os.chdir('/Users/Charlene/Desktop/REP 4/Neural Networks and Deep Learning/Assignment')

# to create plots folder
if not os.path.isdir('figures'):
    print('create figures folder')
    os.makedirs('figures')

NUM_FEATURES = 8
no_labels=1

epochs = 1000 #change the number of epochs when necessary
learning_rate = 10**-7
batch_size = 32
num_neuron = 30 # number of neurons in hidden layer
beta=10**-3
seed = 10
np.random.seed(seed)

print('lr: {}, decay parameter: {}'.format(learning_rate,beta ))

def train(para):
    # Build the graph for the deep net
    # for the hidden layer
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
    # np.asmatrix interpret input as a matrix
    Y_data = (np.asmatrix(Y_data)).transpose()

    print('X_data: {}'.format(X_data))
    print('Y_data: {}'.format(Y_data))


    idx = np.arange(X_data.shape[0])
    # randomly shuffle the idx
    np.random.shuffle(idx)
    X_data, Y_data = X_data[idx], Y_data[idx]

    # experiment with small databases
    X_data, Y_data = X_data[:1000], Y_data [:1000]
    trainX, testX, trainY, testY = model_selection.train_test_split(X_data, Y_data, test_size=0.3, random_state=42)

    err=[] #error for different folds

    trainY_ = trainY.reshape(len(trainY), no_labels)
    testY_ = testY.reshape(len(testY), no_labels)

    # standardise the input data
    scaler = preprocessing.StandardScaler()
    trainX_ = scaler.fit_transform(trainX)
    testX_ = scaler.transform(testX)

    print('Training data: {}'.format(trainX_))
    print('Testing data: {}'.format(testX_))

    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, no_labels])

    with tf.name_scope('hidden'):
        w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(float(NUM_FEATURES))), name='weights')
        b1 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases')
        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    with tf.name_scope('linear'):
        w2 = tf.Variable(tf.truncated_normal([num_neuron, no_labels], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
        b2= tf.Variable(tf.zeros([no_labels]), name='biases')
        prediction=tf.matmul(h1,w2)+b2

    # tabulate the loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - prediction), axis=1))
    # loss with regularisation term
    regularization=tf.nn.l2_loss(w1)+ tf.nn.l2_loss(w2)

    losswithreg= tf.reduce_mean(loss+para*regularization)

    #Create the gradient descent optimizer with the given learning rate.
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(losswithreg)

    N=len(trainX_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_err=[]

        batch_train_err=[]
        # Do we need a batch testing error?
        for i in range(epochs):
            idx3 = np.arange(trainX_.shape[0])
            np.random.shuffle(idx3)
            trainXX=trainX_[idx3]
            trainYY=trainY_[idx3]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end]})
                batch_train_err.append(loss.eval(feed_dict={x: trainXX[start:end], y_: trainYY[start:end]}))

            train_err.append(sum(batch_train_err)/len(batch_train_err))

        test_err=loss.eval(feed_dict={x: testX_, y_: testY_}) #obtain error at the end? Because you are not training with the test data right?

        predicted = sess.run([prediction], feed_dict={x: testX_, y_: testY_})
        predicted = np.reshape(predicted, (len(testX), 1))

    index = np.arange(testX.shape[0])
    # randomly shuffle the idx
    np.random.shuffle(index)
    predicted,testY_ = predicted[index], testY_[index]

    print('first 50: {}'.format(predicted[:50]))
    plt.figure(2)
    plt.plot(np.arange(50),predicted[:50],marker='o', label='predicted value',linestyle='None')
    plt.plot(range(50), testY_[:50], marker='x', label='targeted value',linestyle='None')
    plt.xlabel('Number of samples')
    plt.ylabel('Mean housing prices')
    plt.legend()
    plt.show()

    return test_err,train_err

def main():

    testerr,trainerr=train(beta)
    print('mean error is: {}'.format(testerr))

    #plot training error
    plt.figure(3)
    plt.plot(range(epochs), trainerr, label = 'train error')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Mean Square Validation Error')
    # also the values for y is very high
    plt.title('GD Learning')
    plt.legend()
    plt.show()



if __name__ == '__main__':
  main()