import cPickle as pickle
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def train_NN():
    model = {}

    # Set up placeholders for x and y_
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Shape the image so that it matches with the parameters
    x_image = tf.reshape(x, [-1,28,28,1])

    # Set up the parameters & computation for the first convolutional + pooling layers
    W = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
    b = tf.Variable(tf.random_normal([10], stddev=0.01))
    o = tf.nn.relu(tf.matmul(x, W) + b)
    y = tf.nn.softmax(o)

    # Set up the loss function we're minimizing
    NLL = -tf.reduce_sum(y_*tf.log(y))

    # Set up the optimizer method, note that the Adam update rule is being used instead of
    # basic Gradient Descent
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(NLL)

    # Set up the logic for what a correct prediction is, and
    # classification accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())

    # Do mini batch gradient descent
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict = {x: batch[0], y_ : batch[1]})

    # Print the accuracy on the validation data
    print(accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
    model['W'] = sess.run(W)
    model['b'] = sess.run(b)

    with open('basic_model.pkl', 'w') as fp:
        pickle.dump(model, fp)

def get_visuals():
    with open('basic_model.pkl', 'r') as fp:
        model = pickle.load(fp)
    visualizeWeights(model['W'], 'shit/')

def visualizeWeights(W, save_dir):
    for i in xrange(10):
        fig = plt.figure()
        ax = fig.gca()
        gray_scale_img = W[:,i].reshape((28, 28))
        heatmap = ax.imshow(gray_scale_img, cmap = mpl.cm.coolwarm)
        fig.colorbar(heatmap, shrink = 0.5, aspect=5)
        plt.show()
        #file_name = save_dir + "heatmap"+ str(i) + ".pdf"
        #fig.savefig(file_name)

def adversarial(inp, faker_class):
    with open('basic_model.pkl', 'r') as fp:
        model = pickle.load(fp)

    this_x = np.reshape(inp,(1, 784))

    #this_x += 1000*model['W'][:,faker_class]

    x = tf.placeholder(tf.float32, shape=[None, 784])

    x_image = tf.reshape(x, [-1,28,28,1])

    # Set up the parameters & computation for the first convolutional + pooling layers
    W = model['W']
    b = model['b']
    o = tf.nn.relu(tf.matmul(x, W) + b)
    y = tf.nn.softmax(o)

    #gradients = tf.nn.relu(tf.gradients(o, x))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    '''
    grads = sess.run(gradients, feed_dict={x:this_x})
    print grads[0].shape
    this_x += 1*grads[0]
    '''
    prob = sess.run(y, feed_dict={x: this_x})
    return np.argmax(prob)

#train_NN()
#get_visuals()
print adversarial(mnist.train.images[0], 9)
print np.argmax(mnist.train.labels[0])
