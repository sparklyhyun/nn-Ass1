import math
import tensorflow as tf
import numpy as np
import pylab as plt

import os

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
epochs = 2000
batch_size = 32         #per batch
num_neurons = 10    #hidden layer neurons 
seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('sat.trn',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)                      # X -> 36 features, Y -> output
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))           # appropriate scaling
train_Y[train_Y == 7] = 6                                                                                    # output 7 -> class 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES)) #shape[0] - no. of rows in Y, num_classes - 6 columns
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix, K

print('y: %s'%(train_Y))

# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

n = trainX.shape[0]         #no. of rows



'''Initializing weights and biases for hidden perceptron layer & output softmax layer '''
#from input to hidden layer (output 36 x 10)
W = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons],
                                   stddev  = 1.0/math.sqrt(float(NUM_FEATURES))), name = 'W') 
b = tf.Variable(tf.zeros([num_neurons]), name = 'b')

#from hidden to output layer (output 10 x 6)
V = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES],
                                    stddev = 1.0 / math.sqrt(float(num_neurons)), name = 'V')) #np or maths?
c = tf.Variable(tf.zeros([NUM_CLASSES]), name = 'c')


'''model input & output'''
# model input & output , x = input, y_ = output
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])        #array - shape of the placeholder, none - 1st dimension can be of any size 
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])        #for output, need one more layer in between

# placeholder for hidden layer neurons
z = tf.placeholder(tf.float32, [None, num_neurons])

#from input to hidden perceptron layer
z = tf.matmul(x,W) + b   #syaptic input to hidden layer, 10 x 6 
h = tf.nn.sigmoid(z)        #perceptron, thus sigmod function, 10 x 6

#from hidden to output layer, output = y, output softmax layer 
u = tf.matmul(h, V) + c

'''
p = tf.exp(u)/tf.reduce_sum(tf.exp(u), axis = 1, keepdims = True)       #keepdims - keep dimension of tensor
y = tf.argmax(p, axis = 1) #returns type int64
'''
print(u)



##Cost function, need to be modified later
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u) #logits = p worked for a while 
loss = tf.reduce_mean(cross_entropy)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

print('here')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    for i in range(epochs):
        train_op.run(feed_dict={x: trainX, y_: trainY})
        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g'%(i, train_acc[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.title('GD Learing')
#plt.savefig('plots/Qn1.png')
plt.show()

# plot Q2

# plot Q3

# plot Q4

# plot Q5

