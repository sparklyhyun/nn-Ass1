import math
import tensorflow as tf
import numpy as np
import pylab as plt
import os
import time

#create plots folder, remove comment later
'''
if not os.path.isdir('plots'):
    print('create figures folder')
    os.makedirs('plots')
'''

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons = 10   #hidden layer neurons
seed = 10
np.random.seed(seed)
decay_param = 10**-6

#read train data
train_input = np.loadtxt('sat.trn',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)                      
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))  
train_Y[train_Y == 7] = 6                                                                                    

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES)) 
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1

print('train data read')

#read test data
test_input = np.loadtxt('sat.tst', delimiter = ' ')
testX, test_Y = test_input[:, :36], test_input[:, -1].astype(int)
testX = scale(testX, np.min(testX, axis = 0), np.max(testX, axis = 0))
test_Y[test_Y == 7] = 6

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix, K 

print('test data read')

# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]
testX = testX[:1000]
testY = testY[:1000]

# model input & output , x = input, y_ = output
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])        
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])        

#Initializing weights and biases for hidden perceptron layers & output softmax layer
#h1 - for 1st hidden perceptron layer, h2 - for 2nd hidden perceptron layer
weights = {
    #from input to hidden layer1
    'w1' : tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons],
                                   stddev  = 1.0/math.sqrt(float(NUM_FEATURES)))),
    #from hidden layer 1 to hidden layer 2
    'w2' : tf.Variable(tf.truncated_normal([num_neurons, num_neurons],
                                   stddev  = 1.0/math.sqrt(float(NUM_FEATURES)))),
    #from hidden layer 2 to output layer
    'out' : tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES],
                                   stddev  = 1.0/math.sqrt(float(NUM_FEATURES)))) 
    }

biases = {
    #from input to hidden layer 1
    'b1' : tf.Variable(tf.zeros([num_neurons])),
    #from hidden layer 1 to hidden layer 2
    'b2' : tf.Variable(tf.zeros([num_neurons])),
    #from hidden layer 2 to output layer
    'out' : tf.Variable(tf.zeros([NUM_CLASSES]))
    }

def network_4_layer(x):
    hidden1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    hidden1_out = tf.sigmoid(hidden1)
    hidden2 = tf.add(tf.matmul(hidden1_out, weights['w2']), biases['b2'])
    hidden2_out = tf.sigmoid(hidden2)
    output = tf.matmul(hidden2_out, weights['out']) + biases['out']
    return output #need to apply softmax later

def network_3_layer(x):
    hidden1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    hidden1_out = tf.sigmoid(hidden1)
    output = tf.matmul(hidden1_out, weights['out']) + biases['out']
    return output #need to apply softmax later 

logits3 = network_3_layer(x)
print('here') 
logits4 = network_4_layer(x)

#Cost function for 3 layer network (softmax)
cross_entropy3 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits3) 
loss3 = tf.reduce_mean(cross_entropy3)


#Cost function for 4 layer network (softmax)
cross_entropy4 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits4) 
loss4 = tf.reduce_mean(cross_entropy4)

#implementing l2 regularizer for 3 layer network
regularizer3 = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['out'])
loss3 = tf.reduce_mean(loss3 + decay_param * regularizer3)


#implementing l2 regularizer for 4 layer network 
regularizer4 = tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['out'])
loss4 = tf.reduce_mean(loss4 + decay_param * regularizer4) 


# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op3 = optimizer.minimize(loss3)
train_op4 = optimizer.minimize(loss4)

correct_prediction3 = tf.cast(tf.equal(tf.argmax(logits3, 1), tf.argmax(y_, 1)), tf.float32)
accuracy3 = tf.reduce_mean(correct_prediction3)

correct_prediction4 = tf.cast(tf.equal(tf.argmax(logits4, 1), tf.argmax(y_, 1)), tf.float32)
accuracy4 = tf.reduce_mean(correct_prediction4)

# running training & testing 
err_= [[],[]]
err_batch = []
acc_test = [[],[]]
training_time = [[],[]]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #need to change the batch size 
    trainX_batch = [trainX[i * batch_size : (i +1) * batch_size]
                    for i in range((trainX.shape[0] + batch_size -1)// batch_size)]
    trainY_batch = [trainY[i * batch_size : (i +1) * batch_size]
                    for i in range((trainY.shape[0] + batch_size -1)// batch_size)]

    #train 3 layer network first
    for i in range(epochs):
        #batch
        if i == 1:
            start = time.time() 
        for j in range(len(trainX_batch)):
            train_op3.run(feed_dict={x: trainX_batch[j], y_: trainY_batch[j]})
            err_batch.append(loss3.eval(feed_dict = {x: trainX_batch[j], y_: trainY_batch[j]}))

        if i==1:
            training_time[0] = time.time() - start
            print('time taken to train 1 epoch %f' %(training_time[0]) )
        
        err_[0].append(sum(err_batch)/len(err_batch))
        err_batch[:] = []

        #test
        acc_test[0].append(accuracy3.eval(feed_dict = {x: testX, y_:testY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g error:%g'%(i, acc_test[0][i], err_[0][i]))
    print('3 layer done')
    
    #train 4 layer network
    for i in range(epochs):
    #batch
        if i == 1:
            start = time.time() 
        for j in range(len(trainX_batch)):
            train_op4.run(feed_dict={x: trainX_batch[j], y_: trainY_batch[j]})
            err_batch.append(loss4.eval(feed_dict = {x: trainX_batch[j], y_: trainY_batch[j]}))

        if i==1:
            training_time[1] = time.time() - start
            print('time taken to train 1 epoch %f' %(training_time[1]) )
        
        err_[1].append(sum(err_batch)/len(err_batch))
        err_batch[:] = []

        #test
        acc_test[1].append(accuracy4.eval(feed_dict = {x: testX, y_:testY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g error:%g'%(i, acc_test[1][i], err_[1][i]))
    
    print('learning & testing done')


# plot Q2 - training errors against no. of epoch
plt.figure(1)
plt.plot(range(epochs), err_[0])
plt.plot(range(epochs), err_[1])
plt.xlabel(str(epochs) + 'iterations')
plt.ylabel('classification error')
plt.legend(['3 layer network', '4 layer network'])
plt.title('Q2. training error')
#plt.savefig('plots/Qn2(1).png)

#plot Q2 - test accurcy against no. of epoch
plt.figure(2)
plt.plot(range(epochs), acc_test[0])
plt.plot(range(epochs), acc_test[1])
plt.xlabel(str(epochs) + 'iterations')
plt.ylabel('test accuracy')
plt.legend(['3 layer network', '4 layer network'])
plt.title('Q2. test accuracy')
#plt.savefig('plots/Qn2(2).png')


plt.show()
