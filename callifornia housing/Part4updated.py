# Question: No need to do 5 fold here because not selecting between models?

# Question: Check on the error: Just need to take test error across epochs or? 

import tensorflow as tf
import numpy as np
import multiprocessing as mp
from sklearn import preprocessing
from sklearn import model_selection
import pylab as plt


import os
os.chdir('/Users/Charlene/Desktop/REP 4/Neural Networks and Deep Learning/Assignment')


# to create plots folder
if not os.path.isdir('figures'):
    print('create figures folder')
    os.makedirs('figures')


NUM_FEATURES=8
no_labels=1
learning_rate=10**-9
epochs = 100
neurons_firsthiddenlayer=100
hidden_neurons=20
beta=10**-3

seed = 10
np.random.seed(seed)

no_folds=5
batch_size=32

tf.logging.set_verbosity(tf.logging.ERROR)

# Define the Networks
def network_3_layer(x,weights, biases,keep_prob):
    # hidden layer 1: relu layer
    h1=tf.nn.relu(tf.matmul(x, weights['w1']) + biases['b1'])
    h1_dropout=tf.nn.dropout(h1, keep_prob)

    #output layer: linear layer
    output3=tf.matmul(h1_dropout, weights['out'])+biases['out']

    return output3

def network_4_layer(x,weights, biases, keep_prob):

    # hidden layer 1: relu layer
    h1=tf.nn.relu(tf.matmul(x, weights['w1']) + biases['b1'])
    h1_dropout=tf.nn.dropout(h1, keep_prob)

    #hidden layer 2: relu layer
    h2=tf.nn.relu(tf.matmul(h1_dropout, weights['w2'])+biases['b2'])
    h2_dropout=tf.nn.dropout(h2, keep_prob)

    #output layer: linear layer
    output4=tf.matmul(h2_dropout, weights['out'])+biases['out']

    return output4


def network_5_layer(x, weights, biases, keep_prob):

    #hidden layer 1: relu layer
    h1=tf.nn.relu(tf.matmul(x,weights['w1'])+biases['b1'])
    h1_dropout=tf.nn.dropout(h1, keep_prob)

    #hidden layer 2: relu layer
    h2=tf.nn.relu(tf.matmul(h1_dropout, weights['w2']) +biases['b2'])
    h2_dropout=tf.nn.dropout(h2, keep_prob)

    #hidden layer 3: relu layer
    h3=tf.nn.relu(tf.matmul(h2_dropout, weights['w3'])+ biases['b3'])
    h3_dropout=tf.nn.dropout(h3, keep_prob)

    #output layer: linear layer
    output5=tf.matmul(h3_dropout, weights['out'])+biases['out']

    return output5

def train(prob):

    # load datasets
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
    X_data, Y_data = X_data[:1000], Y_data[:1000]  # experiment with small datasets
    trainX, testX, trainY, testY = model_selection.train_test_split(X_data, Y_data, test_size=0.3, random_state=42)

    tf.set_random_seed(seed)  # set the graph random seed

    # standardise the input data
    scaler = preprocessing.StandardScaler()
    trainX_ = scaler.fit_transform(trainX)
    testX_ = scaler.transform(testX)

    tf.set_random_seed(seed)

    x= tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_= tf.placeholder(tf.float32, [None, no_labels])
    keep_prob=tf.placeholder(tf.float32)

    # Initialising the weights and biases with hidden Relu layers and outer linear layer

    weights={
        #from input to hidden layer 1
        'w1' : tf.Variable(tf.truncated_normal([NUM_FEATURES,neurons_firsthiddenlayer], stddev=1.0/np.sqrt(float(NUM_FEATURES)))),

        #from hidden layer 1 to hidden layer 2
        'w2' : tf.Variable(tf.truncated_normal([neurons_firsthiddenlayer, hidden_neurons], stddev=1.0/np.sqrt(float(neurons_firsthiddenlayer)))),

        # from hidden layer 2 to hidden layer 3
        'w3' : tf.Variable(tf.truncated_normal([hidden_neurons, hidden_neurons], stddev=1.0/np.sqrt(float(hidden_neurons)))),

        #from last hidden layer to output layer
        'out' : tf.Variable(tf.truncated_normal([hidden_neurons, no_labels], stddev=1.0/np.sqrt(float(hidden_neurons))))

    }

    biases= {
        # from input to hidden layer 1
        'b1': tf.Variable(tf.zeros([neurons_firsthiddenlayer])),

        # from hidden layer 1 to hidden layer 2
        'b2': tf.Variable(tf.zeros([hidden_neurons])),

        # from hidden layer 2 to hidden layer 3
        'b3': tf.Variable(tf.zeros([hidden_neurons])),

        # from hidden layer 3 to output layer
        'out': tf.Variable(tf.zeros([no_labels]))

    }

    y3=network_4_layer(x,weights,biases,keep_prob)

    y4=network_4_layer(x,weights,biases,keep_prob)

    y5=network_5_layer(x,weights,biases,keep_prob)

    # Cost function for 3 layer network
    loss3 = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y3), axis=1))
    regularization3 = tf.nn.l2_loss(weights['w1']) +tf.nn.l2_loss(weights['out'])
    losswithreg3=tf.reduce_mean(loss3+ beta* regularization3)

    # Cost function for 4 layer network
    loss4 = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y4), axis=1))
    regularization4 = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2'])+tf.nn.l2_loss(weights['out'])
    losswithreg4=tf.reduce_mean(loss4+ beta* regularization4)

    # Cost function for 5 layer network
    loss5 = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y5), axis=1))
    regularization5 = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2'])+tf.nn.l2_loss(weights['w3'])+tf.nn.l2_loss(weights['out'])
    losswithreg5=tf.reduce_mean(loss5+ beta* regularization5)

    #Create the gradient descent algorithm with the given learning rate

    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    train_op3= optimizer.minimize(loss3)
    train_op4= optimizer.minimize(loss4)
    train_op5= optimizer.minimize(loss5)


    N = len(trainX_)
    idx = np.arange(N)
    test_err=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_err3, test_err4, test_err5=[],[], []
        for i in range(epochs):
            np.random.shuffle(idx)
            trainXX = trainX_[idx]
            trainYY = trainY[idx]

            #Do we need test error in batch? SINCE THEY ARE TEST ERRORS? Or just for each epoch
            #need to do in batches
            #Another thing:
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op3.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end], keep_prob: prob})

                train_op4.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end], keep_prob:prob})

                train_op5.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end], keep_prob:prob})


            #for multiiple epochs: Mean batch error
            test_err3.append(loss3.eval(feed_dict={x: testX_, y_: testY, keep_prob: prob}))
            test_err4.append(loss4.eval(feed_dict={x: testX_, y_: testY, keep_prob:prob}))
            test_err5.append(loss5.eval(feed_dict={x: testX_, y_: testY, keep_prob: prob}))

    test_err=[test_err3, test_err4, test_err5]

    return test_err3, test_err4, test_err5

def main():
    no_threads=mp.cpu_count()
    prob = [0.9, 1.0] # keeping probability for dropout
    p=mp.Pool(processes=no_threads)
    test_err= p.map(train, prob)

    test_errwdropout=test_err[0]
    test_errwodropout=test_err[1]

    # plot learning curves
    plt.figure(1)
    plt.plot(range(epochs), test_errwdropout[0], label='3 layers Neural Network')
    plt.plot(range(epochs), test_errwdropout[1], label='4 layers Neural Network')
    plt.plot(range(epochs), test_errwdropout[2], label='5 layers Neural Network')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Test Errors with Dropouts')  # the mean square error is high because it is a squared error.
    # also the values for y is very high
    plt.title('GD Learning')
    plt.legend()
    plt.show()

    # plot learning curves
    plt.figure(1)
    plt.plot(range(epochs), test_errwodropout[0], label='3 layers Neural Network')
    plt.plot(range(epochs), test_errwodropout[1], label='4 layers Neural Network')
    plt.plot(range(epochs), test_errwodropout[2], label='5 layers Neural Network')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Test Errors without Dropouts')  # the mean square error is high because it is a squared error.
    # also the values for y is very high
    plt.title('GD Learning')
    plt.legend()
    plt.show()

if __name__ == '__main__':
  main()



