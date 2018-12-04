import numpy as np
import tensorflow as tf

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    ##gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    ##gray =  2 * gray.astype('float32') - 1 
    ##return gray 
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes actions
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0

def id_to_action(labels_id):
    """ 
    this method discretizes actions
    """

    classes = 3
    
    '''
    a_c = np.zeros((a.shape[0], nr_classes))

    print ("a_c shape ::::: ", a_c.shape)
    print('checking the a.all function',a.all())

    
    if a.all() == LEFT :
       left = [-1.0, 0.0, 0.0]
       print('in left condition')
       np.append(a_c,left)
    elif a.all() == RIGHT: 
       right = [1.0, 0.0, 0.0]
       print('in right condition')
       np.append(a_c,right)
    elif a.all() == ACCELERATE:
       acc = [0.0, 1.0, 0.0]
       print('in accelerate condition')
       np.append(a_c,acc)
    elif a.all() == BRAKE : 
       brake = [0.0, 0.0, 0.8]
       print('in brake conidition')
       np.append(a_c,brake)
    elif a.all() == STRAIGHT :  
       straight= [0.0, 0.0, 0.0]
       print('in straight condition')
       np.append(a_c,straight)
    '''
   
    labels_action = np.zeros((labels_id.shape[0], classes))
    
    labels_action[labels_id==LEFT] = [-1.0, 0.0, 0.0]
    labels_action[labels_id==RIGHT] = [1.0, 0.0, 0.0]
    labels_action[labels_id==STRAIGHT] = [0.0, 0.0, 0.0]  #accelerate in this case also
    labels_action[labels_id==ACCELERATE] =[0.0, 1.0, 0.0]
    labels_action[labels_id==BRAKE] = [0.0, 0.0, 0.2]
   
   # print('shape of labels_action:::::',labels_action.shape)
   # print ('final a_c matrix::::::::',labels_action)

    return labels_action

    

    ##if (a == LEFT): return [-1.0, 0.0, 0.0]              # LEFT: 1
    ##elif (a == RIGHT): return [1.0, 0.0, 0.0]             # RIGHT: 2
    ##elif (a == ACCELERATE): return [0.0, 1.0, 0.0]        # ACCELERATE: 3
    ##elif (a == BRAKE): return [0.0, 0.0, 0.8]             # BRAKE: 4
    ##else:       
    ##    return [0.0, 0.0, 0.0]                                       # STRAIGHT = 0

