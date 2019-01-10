import numpy as np

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    ##gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    ##gray =  2 * gray.astype('float32') - 1 
    ##return gray 
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))

def id_to_action(labels_id):
    classes = 3
   
    labels_action = np.zeros(classes)
    
    labels_action[labels_id==LEFT] = [-1.0, 0.0, 0.0]
    labels_action[labels_id==RIGHT] = [1.0, 0.0, 0.0]
    labels_action[labels_id==STRAIGHT] = [0.0, 0.0, 0.0]  #accelerate in this case also
    labels_action[labels_id==ACCELERATE] =[0.0, 1.0, 0.0]
    labels_action[labels_id==BRAKE] = [0.0, 0.0, 0.8]

    return labels_action
