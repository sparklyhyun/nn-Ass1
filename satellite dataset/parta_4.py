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
epochs = 5000
batch_size = 32
num_neurons = 15   #hidden layer neurons 
seed = 10
np.random.seed(seed)
decay_param = [0, 10**-3, 10**-6, 10**-9, 10**-12]  

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
  

#Initializing weights and biases for hidden perceptron layer & output softmax layer 
#from input to hidden layer
W = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons],
                                   stddev  = 1.0/math.sqrt(float(NUM_FEATURES))), name = 'W') 
b = tf.Variable(tf.zeros([num_neurons]), name = 'b')

#from hidden to output layer
V = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES],
                                    stddev = 1.0 / math.sqrt(float(num_neurons)), name = 'V'))
c = tf.Variable(tf.zeros([NUM_CLASSES]), name = 'c')


'''model input & output'''
# model input & output , x = input, y_ = output
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])        #array - shape of the placeholder, none - 1st dimension can be of any size 
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])        #for output, need one more layer in between

# placeholder for hidden layer neurons
z = tf.placeholder(tf.float32, [None, num_neurons])
# placeholder for K
k = tf.placeholder(tf.float32, train_Y.shape)

#from input to hidden perceptron layer
z = tf.matmul(x,W) + b   #syaptic input to hidden layer, 10 x 6 
h = tf.nn.sigmoid(z)        #perceptron, thus sigmod function, 10 x 6

#from hidden to output softmax layer 
u = tf.matmul(h, V) + c
p = tf.exp(u)/tf.reduce_sum(tf.exp(u), axis = 1, keepdims = True)
y = tf.argmax(p, axis = 1)

#Cost function (softmax)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u) 
loss = tf.reduce_mean(cross_entropy)

#implementing l2 regularizer - or just use reduce_mean?
'''
regularizer = tf.nn.l2_loss(V)
loss = tf.reduce_mean(loss + decay_param * regularizer) #loss_V

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
'''

# running training & testing 
err_= [[],[],[],[],[]]
err_batch = []
acc_test = [[],[],[],[],[]]
training_time = [[],[],[],[],[]]
print('test once per epoch')

def batch_training(batch_size, a, d):
    #implementing l2 regularizer - or just use reduce_mean?
    regularizer = tf.nn.l2_loss(V)
    loss = tf.reduce_mean(loss + decay_param[d] * regularizer) #loss_V

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    print('batch start')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trainX_batch = np.array_split(trainX, batch_size)
        trainY_batch = np.array_split(trainY, batch_size)
        for i in range(epochs):
            #batch'
            if i == 1:
                start = time.time() #time taken to train 1 epoch
            for j in range(len(trainX_batch)):
                train_op.run(feed_dict={x: trainX_batch[j], y_: trainY_batch[j]})
                err_batch.append(loss.eval(feed_dict = {x: trainX_batch[j], y_: trainY_batch[j]}))

            if i==1:
                training_time[a] = time.time() - start
                print('time taken to train 1 epoch %f' %(training_time[a]) )
            
            err_[a].append(sum(err_batch)/len(err_batch))
            err_batch[:] = []

            #test
            acc_test[a].append(accuracy.eval(feed_dict = {x: testX, y_:testY}))

            if i % 1000 == 0:
                print('iter %d: accuracy %g error:%g'%(i, acc_test[a][i], err_[a][i]))
        print('learning & testing done')
    return 

batch_training(batch_size, 0, 0) #decay_param = 0
batch_training(batch_size, 1, 1) #decay_param = 0**-3
#batch_training(batch_size, 2, 2) #decay_param = 10**-6
#batch_training(batch_size, 3, 3) #decay_param = 10**-9
#batch_training(batch_size, 4, 4) #decay_param = 10**-12

# plot Q2 - training errors against no. of epoch
plt.figure(1)
plt.plot(range(epochs), err_[0])
plt.plot(range(epochs), err_[1])
#plt.plot(range(epochs), err_[2])
#plt.plot(range(epochs), err_[3])
#plt.plot(range(epochs), err_[4])
plt.xlabel(str(epochs) + 'iterations')
plt.ylabel('classification error')
#plt.legend(['decay_param = 0', 'decay_param = 10**-3', 'decay_param = 10**-6', 'decay_param = 10**-9', 'decay_param = 10**-12'])
plt.legend(['decay_param = 0', 'decay_param = 10**-3'])
plt.title('Q2. training error')
#plt.savefig('plots/Qn2(1).png)

#plot Q2 - test accurcy against no. of epoch
plt.figure(2)
plt.plot(range(epochs), acc_test[0])
plt.plot(range(epochs), acc_test[1])
#plt.plot(range(epochs), acc_test[2])
#plt.plot(range(epochs), acc_test[3])
#plt.plot(range(epochs), acc_test[4])
plt.xlabel(str(epochs) + 'iterations')
plt.ylabel('test accuracy')
#plt.legend(['decay_param = 0', 'decay_param = 10**-3', 'decay_param = 10**-6', 'decay_param = 10**-9', 'decay_param = 10**-12'])
plt.legend(['decay_param = 0', 'decay_param = 10**-3'])
plt.title('Q2. test accuracy')
#plt.savefig('plots/Qn2(2).png')

#plot Q2 - training tme against each bath size
plt.figure(3)
plt.plot(batch_size, training_time)
plt.xlabel('batch')
plt.ylabel('training time')
#plt.legend(['batch size = 4', 'batch size = 8'])
plt.title('Q2. training time')
#plt.savefig('plots/Qn2(2).png') 


plt.show()
