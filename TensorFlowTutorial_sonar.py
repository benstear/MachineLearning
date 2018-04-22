#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# TENSORFLOW tutorial on youtube
# datasets/Sonar.csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




#sonarData = open("/Users/dawnstear/desktop/datasets-master/Sonar.csv","r+")
#print sonarData.read() # <-- to view

# Reading the dataset
def read_dataset():
    df = pd.read_csv("/Users/dawnstear/desktop/datasets-master/Sonar.csv",header=None)
    #print(len(df.columns)) # see how many cols
    X = df[df.columns[0:60]].values # 0-59
    y = df[df.columns[60]] # labels 0 or 1, rock or mine

    # one-hot encoding, only one type active at a time
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    print(X.shape)
    return (X,Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode


# Read dataset,  X is features, Y is onehotencoded labels
X,Y = read_dataset()

# Shuffle the data set
X,Y = shuffle(X,Y, random_state = 1) # must shuffle bc its in order of mines then rocks
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.20, random_state=415)
# 20% test, 80% train^^

''' look at shape of data...
print(train_x.shape)
print(train_y.shape)
print(test_x.shape) '''

# Define input parameters and variables
learning_rate = 0.3
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float) # costfunction
n_dim = X.shape[1] # number of columns
n_class = 2 # 2 classes, mine or rock
model_path = "desktop/asdf"  # where to store graph

# Define the number of hidden layers and nodes per layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32,[None, n_dim]) # inputs
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class]) # None means it can be any value



# Now create model
def multilayer_perceptron(x,weights,biases):
    # hidden layer with sigmoid activation fxn
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1']) # x*W + b
    layer_1 = tf.nn.sigmoid(layer_1) 
    
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2']) # input is layer_1 output
    layer_2 = tf.nn.sigmoid(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']), biases['h3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    # ReLU activation fxn
    layer_4 = tf.add(tf.matmul(layer_3,weights['h4']), biases['h4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4,weights['out']) + biases['out']
    return out_layer


#  Define weights and biases for each layer
weights = {
        'h1': tf.Variable(truncated_normal([n_dim, n_hidden_1])),    # 60x60
        'h2': tf.Variable(truncated_normal([n_hidden_1, n_hidden_2])), # 60x60
        'h3': tf.Variable(truncated_normal([n_hidden_2, n_hidden_3])), # 60x60
        'h4': tf.Variable(truncated_normal([n_hidden_3, n_hidden_4])), # 60x60
        'out': tf.Variable(truncated_normal([n_hidden_4, n_class])) # 60x2
        }

biases = { 
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1])), # 60x1
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])), # 60x1
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3])), # 60x1
        'b4': tf.Variable(tf.truncated_normal([n_hidden_4])), # 60x1
        'out': tf.Variable(tf.truncated_normal([n_class])),   # 2x1
        }

# Initialize all variables
init = tf.global_variables_initializer()

# Save
saver = tf.train.Saver()

# Call model
y = multilayer_perceptron(x,weights,biases)

# Define cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# Calculate the cost and accuracy for each epoch
mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history,cost)
    cost_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y,feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y-test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y: train_y}))
    accuracy_history.append(accuracy)
    
a    
save_path = saver.save(sess,model_path)
print("Moddel saved in file: %s" % save_path)

plt.plot(mse_history,'r')
plt.show()
plt.plot(accuracy_history)
plt.show()


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

pred_y = sess.run(y,feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y-test_y))

    

'''
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
# print(node1,node2)  # this will only print an abstraction
# we need to run it in a session

sess = tf.Session() # session object
print(sess.run([node1, node2]))
sess.close() # remember to close sess
with tf.Session() as sess:  # doing it with  with closes sess for us
    output = sess.run([node1, node2]))
    print(output)
    File_Writer = tf.summary.FileWriter('filepath',sess.graph)

# if we want our graph to accept external input, aka we 
# dont want constants... use placeholders instead
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

sess = tf.Session()
print(sess.run(adder_node,{a:[1,3],b:[2,4]})) # output = [3 7]    1+2 and 3+4

# LINEAR MODEL::::
# if we want to train our graph, we use variables
W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)  # holds input values
linear_model = W*x + b

# to initialize variables...
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # to initialize var's
print(sess.run(linear_model,{x:[1,2,3,4]}))
    
# how to evaluate the loss/error?

# use y to hold our output/prediction
y = tf.placeholder(tf.float32) # output of model
squared_deltas = tf.square(linear_model - y) # use linear regression loss model
loss = tf.reduce_sum(squared_deltas) # sums each iter's loss
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

# setting W = -1 and b = 1 (with the current x and y vals) our loss is 0,
# but what if we want the computer to reduce loss for us?
# for that we need optimizers -- gradient descent!!!

#  Batch Grad Descent J(w) = .5* SUM (target-output)^2
optimizer = tf.train.GradientDescentOptimizer(0.01) # 0.01 is learning rate
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    
print(sess.run([W,b])) # W = -0.99999   b = 0.99999    
'''

