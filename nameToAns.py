
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import csv

# In[2]:



# In[3]:

birdDict = dict()
countB = 0
birdDict['0'] = np.zeros(2,dtype=np.int64)
birdDict['0'][0] = 1
birdDict['1'] = np.zeros(2,dtype=np.int64)
birdDict['1'][1] = 1

# In[15]:

y_file = open('output.csv','r')
y_data = []
y_data_dict =[]
y_train = []

for x in y_file:
    y_data.append(x)

for x in range(len(y_data)):
    y_data[x] = y_data[x].split('\n')[0]

for x in y_data:
    y_data_dict.append(birdDict[x])

tempY = np.zeros(shape=(2))

for x in y_data_dict:
    tempY = np.row_stack((tempY,x))
y_train = np.delete(tempY,0,0)


# In[15]:

test_y_file = open('testOutput.csv','r')
test_y_data = []
test_y_data_dict =[]
y_test = []

for x in test_y_file:
    test_y_data.append(x)

for x in range(len(test_y_data)):
    test_y_data[x] = test_y_data[x].split('\n')[0]

for x in test_y_data:
    test_y_data_dict.append(birdDict[x])

tempY = np.zeros(shape=(2))

for x in test_y_data_dict:
    tempY = np.row_stack((tempY,x))
y_test = np.delete(tempY,0,0)


# In[26]:

x_file = open('input.csv','r')
x_data = []
x_data_dict =[]
x_train = []

for x in x_file:
    x_data.append(x)
for x in range(len(x_data)):
    x_data[x] = x_data[x].split('\n')[0]
for x in x_data:
    templist = []
    for y in x.split(','):
        templist.append(float(y))
    x_data_dict.append(templist)

tempX =np.zeros(shape=(12))

for d in x_data_dict:
    tempX = np.row_stack((tempX,d))
x_train = np.delete(tempX,0,0)


# In[26]:

test_x_file = open('testInput.csv','r')
test_x_data = []
test_x_data_dict =[]
x_test = []

for x in test_x_file:
    test_x_data.append(x)
for x in range(len(test_x_data)):
    test_x_data[x] = test_x_data[x].split('\n')[0]
for x in test_x_data:
    templist = []
    for y in x.split(','):
        templist.append(float(y))
    test_x_data_dict.append(templist)

tempX =np.zeros(shape=(12))

for d in test_x_data_dict:
    tempX = np.row_stack((tempX,d))
x_test = np.delete(tempX,0,0)


# In[34]:

def add_layer(inputs, in_size, out_size, activation_function=None ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs

def compute_accutacy(v_x,v_y):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_x, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_y,1))
    accurcay = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accurcay,feed_dict={xs:v_x,ys:v_y})
    return result


#create placeholder
xs = tf.placeholder(tf.float32,[None,12])
ys = tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32)

l1 = add_layer(xs, 12, 60, activation_function= tf.nn.tanh)
prediction = add_layer(l1, 60, 2, activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum((ys * tf.log(prediction)),reduction_indices=[1]))
#loss = tf.reduce_mean(tf.square(prediction - ys))

# In[30]:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# In[35]:

init = tf.initialize_all_variables()
sess = tf.Session()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10050):
        sess.run(train_step, feed_dict={xs:x_train, ys:y_train, keep_prob:0.5})
        if i % 100 == 0:
            #print(sess.run(prediction, feed_dict={xs:x_data, ys:xy_data_}))
            #prediction_a = tf.argmax(prediction,1)
            #print(prediction_a.eval(session=sess, feed_dict={xs:test_x_data}))
            #correct_ = tf.equal(tf.argmax(prediction,1), np.argmax(y_data,1))
            #acc = tf.reduce_mean(tf.cast(correct_,"float"))
            #print("ans: ")
            #print(acc.eval(session=sess, feed_dict={xs:test_x_data}))
            #result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            #writer.add_summary(result,i)
            #str(compute_accutacy(test_x_data,test_xy_data_))
            print("train_loss: ", sess.run(loss, feed_dict={xs:x_train, ys:y_train, keep_prob: 1}))
            print("test_loss", sess.run(loss, feed_dict={xs:x_test, ys:y_test, keep_prob: 1}))
            #print("steps: ",i)
            print("step: " + str(i) + "   accuracy: " + str(compute_accutacy(x_test,y_test)))
