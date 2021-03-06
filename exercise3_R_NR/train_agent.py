from __future__ import print_function

from time import time
import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation
import tensorflow as tf

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    print('shape of train data:::::::::',X_train.shape[0])
    print('shape of valid data::::::::::::',X_valid.shape[0])
  
        
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.
	
    X_train = rgb2gray(X_train[0:31500])
    X_valid = rgb2gray(X_valid)
    X_train = np.expand_dims(X_train,axis=3)
    X_valid = np.expand_dims(X_valid,axis=3)

    print('X_train after rgb2gray',X_train.shape)
    # history:
    '''
    X_train_history = np.zeros((X_train.shape[0]-history_length+1, history_length, X_train.shape[1], X_train.shape[2],X_train.shape[3]))
    for i in range(X_train_history.shape[0]):
        X_train_history[i] = X_train[i:i+history_length,:,:]
    #X_train_history = X_train_history.transpose(0,2,3,1)
    
    X_valid_history = np.zeros((X_valid.shape[0]-history_length+1, history_length, X_valid.shape[1], X_valid.shape[2],X_valid.shape[3]))
    for i in range(X_valid_history.shape[0]):
        X_valid_history[i] = X_valid[i:i+history_length,:,:]
    #X_valid_history = X_valid_history.transpose(0,2,3,1)
    '''
    
    # discretize actions
    y_train_id = np.zeros(y_train.shape[0])
    #y_valid_id = np.zeros(y_valid.shape[0])
    y_train = y_train[0:31500]
    for i in range(y_train.shape[0]):
        y_train_id[i] = action_to_id(y_train[i])
    #for j in range(y_valid.shape[0]):
     #   y_valid_id[j] = action_to_id(y_valid[j])
   
   
    y_train = y_train_id
    #y_valid = y_valid_id
   
    #y_train_id = y_train_id.astype(int)
    #y_valid_id = y_valid_id.astype(int)
    
   # print("y_train_id first element discrete :  ",y_train_id[0])

   # y_train_c = id_to_action(y_train_id)
    
   # print("y_train_c first element continuous :  ",y_train_c[0])

    #y_train = one_hot(y_train_id)
    #y_valid = one_hot(y_valid_id)
   
    #print("the number of ::::::::::",y_train[1:5])
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
	# have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, Y_train, X_valid, Y_valid, epochs , batch_size, lr, history_length=1,model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(learning_rate=lr, history_length=history_length)
    print('exit from model')

    #y = tf.nn.softmax(agent.logits)
    acc = []
    acc_valid = []

    train_cost = np.zeros(epochs)
    valid_cost = np.zeros(epochs)
    
    correct_prediction = agent.prediction 

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
   
    tensorboard_eval = Evaluation(tensorboard_dir)

    tf.reset_default_graph()
    with agent.sess as sess:
        sess.run(tf.global_variables_initializer())
        print('inside session')
        start_time = time()
    
        for epoch in range(epochs):
            correctsum = 0
            #x_train_batchsize = 16
            x_train_batchsize = len(X_train)//batch_size
            print("X_train_batchsize :::::", x_train_batchsize)
            
            correctsum_val = 0
            #x_val_batchsize = 12
            x_val_batchsize = len(X_valid)//batch_size 
        	
            train_cost = np.zeros((epochs))
            valid_cost = np.zeros((epochs))


            for i  in range (x_train_batchsize):
                x_batch = X_train[i*batch_size:(i+1)*batch_size]
                y_batch = Y_train[i*batch_size:(i+1)*batch_size]
                y_batch = id_to_action(y_batch)

                #_, loss = sess.run([agent.optimizer, agent.cost],feed_dict={agent.x_input:x_batch, agent.y_label: y_batch})

                
            # training 
            for i  in range(x_train_batchsize):

                x_batch = X_train[i*batch_size:(i+1)*batch_size]
                y_batch = Y_train[i*batch_size:(i+1)*batch_size]
                
                y_batch = id_to_action(y_batch)
                batch_correct_count = sess.run(accuracy,feed_dict={agent.x_input:x_batch, agent.y_label: y_batch})
                correctsum += batch_correct_count
                _, loss = sess.run([agent.optimizer, agent.cost],feed_dict={agent.x_input:x_batch, agent.y_label: y_batch})
                train_cost[epoch] += sess.run(agent.cost, feed_dict={agent.x_input: x_batch, agent.y_label: y_batch})
                print('iteration {} , epoch {}, train cost {:.2f}'.format(i,epoch,train_cost[epoch]))
            
            total_accuracy = correctsum/x_train_batchsize
            acc.append(total_accuracy)
            print('epoch {}, loss {:.2f}, train accuracy {:.2f}%'.format(epoch, loss, total_accuracy*100), end='\n')
        
        
            #validation
            for i  in range(x_val_batchsize):
                x_val = X_valid[i*batch_size:(i+1)*batch_size]
                y_val = Y_valid[i*batch_size:(i+1)*batch_size]
                #y_val = id_to_action(y_val)
                batch_correct_count = sess.run(accuracy,feed_dict={agent.x_input:x_val, agent.y_label: y_val})
                correctsum_val += batch_correct_count
                valid_cost[epoch] += agent.sess.run(agent.cost, feed_dict={agent.x_input:x_val, agent.y_label:y_val})
                print('valid iteration{}, epoch{}, valid cost ::: {:.2f}'.format(i,epoch,valid_cost[epoch]))

            total_accuracy_v = correctsum_val/x_val_batchsize
            acc_valid.append(total_accuracy_v)
            print('epoch {}, validation accuracy {:.2f}%'.format(epoch, total_accuracy_v*100), end='\n')
            
            train_cost[epoch] = train_cost[epoch] / x_train_batchsize
            valid_cost[epoch] = valid_cost[epoch] / x_val_batchsize
            print("[%d/%d]: train_cost: %.4f, valid_cost: %.4f" %(epoch+1, epochs, train_cost[epoch], valid_cost[epoch]))
            print('epoch {},train_cost{:.2f}, validation cost{:.2f}'.format(epoch,train_cost[epoch], valid_cost[epoch]))
            eval_dict = {"train":train_cost[epoch], "valid":valid_cost[epoch]}
            tensorboard_eval.write_episode_data(epoch, eval_dict)
      
        # TODO: save your agent
        agent.save(os.path.join(model_dir, "agent.ckpt"))
        print(model_dir)
        print("Model saved in file: %s" % model_dir)
        agent.sess.close()     
      
    
if __name__ == "__main__":

    # read data    
    X_train, Y_train, X_valid, Y_valid = read_data("./data")
    
    # preprocess data
    X_train, Y_train, X_valid, Y_valid = preprocessing(X_train, Y_train, X_valid, Y_valid)

    # train model (you can change the parameters!)
    train_model(X_train, Y_train, X_valid, Y_valid, history_length=1, epochs=15, batch_size=25, lr=0.0001339)
    
#train_model(X_train, y_train, X_valid, n_minibatches=100000, batch_size=64, lr=0.0001)

 
