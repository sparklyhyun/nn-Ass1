#
# Project 1, starter code part b
#
# Find Optimal Learning Rate for 3 layer network designed using 5-fold cross-validation

# Question 1

import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn import model_selection
from sklearn import preprocessing
import multiprocessing as mp

import os
os.chdir('/Users/Charlene/Desktop/REP 4/Neural Networks and Deep Learning/Assignment')

# to create plots folder
if not os.path.isdir('figures'):
    print('create figures folder')
    os.makedirs('figures')

NUM_FEATURES = 8
no_labels=1

epochs = 10
batch_size = 32
num_neuron = 30 # number of neurons in hidden layer
beta=10**-3
seed = 10
np.random.seed(seed) # 5 fold cross validation
no_folds=5

def train(learning_rate):

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
    X_data, Y_data = X_data[:1000], Y_data[:1000] #experiment with small datasets

    fold_data=int(0.2* len(X_data))
    print('fold_data: %d'%fold_data)

    error=[]
    for fold in range (no_folds):
        start, end=fold*fold_data,(fold+1)*fold_data
        testX, testY = X_data[start:end], Y_data[start:end]
        trainX= np.append(X_data[:start], X_data[end:], axis=0)
        trainY= np.append(Y_data[:start], Y_data[end:], axis=0)

        err=[] #error for different folds

        trainY_ = trainY.reshape(len(trainY), no_labels)
        testY_ = testY.reshape(len(testY), no_labels)

        # standardise the input data
        scaler = preprocessing.StandardScaler()
        trainX_ = scaler.fit_transform(trainX)
        testX_ = scaler.transform(testX)

        # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, no_labels])

        with tf.name_scope('hidden'):
            w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(float(NUM_FEATURES))), name='weights')
            b1 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases')
            h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        with tf.name_scope('linear'):
            w2=tf.Variable(tf.truncated_normal([num_neuron, no_labels], stddev=1.0/np.sqrt(float(num_neuron))), name='weights')
            b2= tf.Variable(tf.zeros([no_labels]), name='biases')
            y=tf.matmul(h1,w2)+b2

        # tabulate the loss
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))
        # loss with regularisation term
        regularization=tf.nn.l2_loss(w1)+ tf.nn.l2_loss(w2)

        losswithreg= tf.reduce_mean(loss+beta*regularization)

        #Create the gradient descent optimizer with the given learning rate.
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(losswithreg)

        N=len(trainX_)
        idx=np.arange(N)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            test_err = []

            for i in range(epochs):
                np.random.shuffle(idx)
                trainXX=trainX_[idx]
                trainYY=trainY_[idx]

                for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    train_op.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end]})

                test_err.append(loss.eval(feed_dict={x: testX_, y_: testY_})) #error per epoch

        error.append(test_err) #obtain error for different folds

    return error

def main():

    # test out different learning rates
    no_threads=mp.cpu_count()
    learning_rate = [0.5*10**-6, 10 ** -7, 0.5*10**- 8, 10**-9, 10 ** -10]
    p=mp.Pool(processes=no_threads)
    error= p.map(train, learning_rate)
    cv_err = np.mean(np.array(error), axis=0)  # get  mean error
    print('the CV errors are: {}'.format(cv_err))

    # plot learning curves
    plt.figure(1)
    for i in range(len(learning_rate)):
        plt.plot(range(epochs), cv_err[i], label= 'Learning Rate= {}'.format(learning_rate[i]))

    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean CV Error Across Folds') # the mean square error is high because it is a squared error.
    # also the values for y is very high
    plt.title('GD Learning')
    plt.legend()
    plt.show()


    #The optimal learning rate is 0.5* 10**-6

    #Define optimal learning rate as one that gives the lowest mean CV error?

    bestlr=learning_rate[np.argmin(np.min(cv_err, axis=1))]
    print('learning rate that gives lowest error: {}'.format(bestlr))

    idx=learning_rate.index(bestlr)

    plt.figure(2)
    plt.plot(range(epochs), cv_err[idx], label='Learning Rate= {}'.format(learning_rate[idx]))
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean CV Error Across Folds') # the mean square error is high because it is a squared error.
    # also the values for y is very high
    plt.title('GD Learning')
    plt.legend()
    plt.show()


if __name__ == '__main__':
  main()
