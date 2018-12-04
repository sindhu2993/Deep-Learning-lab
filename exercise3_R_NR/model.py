import tensorflow as tf
import numpy as np

class Model:
    
    def __init__(self, history_length=1, learning_rate=3e-1):


        self.learning_rate = learning_rate 
        # variable for input and labels
        self.x_input = tf.placeholder(tf.float32, shape = [None, 96, 96, history_length], name = "x_input")
        self.y_label = tf.placeholder(tf.float32, shape = [None,3], name = "y_label")

       
         # TODO: Define network
        
	#layer 1
        self.conv1 = tf.layers.conv2d(self.x_input, filters=32, kernel_size= [4, 4], strides = 2, padding='same', activation = tf.nn.relu)		
        #print('convolution shape{}'.format(self.conv1.shape), end= '\n')
        #self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[ 2, 2], strides=1)
        #print('pool1 shape',pool1.shape)
		
	#layer 2
        self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size= [3, 3], strides = 2, padding='same', activation = tf.nn.relu)
        #self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=1)
        
        
	#layer 3
        self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size= [3, 3], strides = 2, padding='same', activation = tf.nn.relu)
	#self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=1)

        self.conv4 = tf.layers.conv2d(self.conv3, filters=64, kernel_size= [3, 3], strides = 1, padding='same', activation = tf.nn.relu)

        self.layer_flat = tf.contrib.layers.flatten(self.conv4)
        #layer_flat = tf.reshape(pool2, [-1, 94 * 94 * 32])
        self.layer_fc4 = tf.layers.dense(inputs=self.layer_flat, units=512, activation=tf.nn.relu)

        self.layer_fc5 = tf.layers.dense(inputs=self.layer_fc4, units=512, activation=tf.nn.relu)

        self.fc2_drop = tf.nn.dropout(self.layer_fc5, 0.8)

        self.layer_fc6 = tf.layers.dense(inputs=self.fc2_drop, units=64, activation=tf.nn.relu)

        self.logits = tf.layers.dense(inputs=self.layer_fc6, units=3)

	# TODO: Loss and optimizer		

        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.logits, labels=self.y_label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.y = tf.nn.softmax(self.logits)
        self.prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_label,1))

       
	# TODO: Start tensorflow session		 
        self.sess = tf.Session()        
        
        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
