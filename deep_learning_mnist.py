import tensorflow as tf

# start interactive session
sess = tf.InteractiveSession()

# The MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# Initial Parameters
width = 28 # width of image in pixels
height = 28 # height of image in pixels
flat = width * height # number of pixels in one image
class_output = 10 # number of possible classifications for the problem

# Input and output, create placeholders for input and output
x = tf.placeholder(tf.float32, shape = [None, flat])
y_ = tf.placeholder(tf.float32, shape = [None, class_output])

# converting images of the data set to tensors
x_image = tf.reshape(x, [-1, 28, 28, 1])
x_image

# Covolutional layer one
weight_convolve = tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1))
# need 32 biases for 32 inputs
biases_convolve = tf.Variable(tf.constant(0.1, shape = [32]))

convolve1 = tf.nn.conv2d(x_image, weight_convolve, strides = [1,1,1,1],
                         padding = 'SAME') + biases_convolve
            
# Applying the ReLu activation function f(x) = max (0,x)
h_conv1 = tf.nn.relu(convolve1)

# Apply max pooling
convolution1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides = [1,2,2,1],
                              padding = 'SAME')
convolution1
#First layer completed

# starting the second layer CONVOLUTIONAL LAYER 2
weight_convolve = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1))
# need 64 biases for 64 outputs
biases_convolve = tf.Variable(tf.constant(0.1, shape = [64]))  

convolve2 = tf.nn.conv2d(convolution1, weight_convolve, strides=[1, 1, 1, 1],
                        padding='SAME') + biases_convolve      

# Applying relu
h_conv2 = tf.nn.relu(convolve2)  

# Applying max pooling
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], 
                       strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv2                       


# Fully connected layer
# flattening second layer

layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])

# weights and biases between layers 2 and 3
w_fc =  tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
# need 1024 biases for 1024 outputs
b_fc = tf.Variable(tf.constant(0.1, shape = [1024]))                         

# Matrix multiplication(applying weights and biases)
fc = tf.matmul(layer2_matrix, w_fc) + b_fc

# Apply the relu activation
h_fc = tf.nn.relu(fc)
h_fc

# Dropout layer to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc, keep_prob)
layer_drop

# softmax layer
# 1024 neurons
weight_last = tf.Variable(tf.truncated_normal([1024, 10], stddev = 0.1))
bias_last = tf.Variable(tf.constant(0.1, shape = [10]))

# matrix multiplication - applying weights and biases
fc = tf.matmul(layer_drop, weight_last) + bias_last

# Apply the softmax activation function
y_CNN = tf.nn.softmax(fc)
y_CNN

# Define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), 
                                              reduction_indices=[1]))

# defining the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# defining correct prediction
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))

# defining accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run session and train
sess.run(tf.global_variables_initializer())

# Training the model
for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1],
                                                  keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Evaluating the model in the test set
# evaluating in batches to avoid out-of-memory issues

n_batches = mnist.test.images.shape[0] #50
print(mnist.test.images.shape[0])
cumulative_accuracy = 0.0
batch = mnist.test.next_batch(50)
print(batch[0])
print(batch[1])

for index in range(n_batches):
    batch = mnist.test.next_batch(50)
    cumulative_accuracy += accuracy.eval(feed_dict={x:batch[0],
                                                    y_:batch[1],
                                                    keep_prob : 1.0})
print('test_accuracy {}'.format(cumulative_accuracy / n_batches))




                        
                         
                         
                         
        
                         
                         
                         
                         