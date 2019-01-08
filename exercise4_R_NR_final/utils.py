import numpy as np

STRAIGHT = 0
LEFT = 1
RIGHT = 2
ACCELERATE = 3
BRAKE = 4
LEFT_BRAKE = 5
RIGHT_BRAKE = 6



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



def id_to_action(a):
    """
    this method undoes action_to_id.
    """
    if a == LEFT: return [-1.0, 0.0, 0.0]                         # LEFT: 1
    elif a == RIGHT: return [1.0, 0.0, 0.0]                       # RIGHT: 2
    elif a == ACCELERATE: return [0.0, 1.0, 0.0]                  # ACCELERATE: 3
    elif a == BRAKE: return [0.0, 0.0, 0.2]                       # BRAKE: 4
    elif a == LEFT_BRAKE: return [-1.0, 0.0, 0.2]                 # LEFT_BRAKE: 5
    elif a == RIGHT_BRAKE: return [1.0, 0.0, 0.2]                 # RIGHT_BRAKE: 6 
    else:
        return [0.0,0.0,0.0]                                 # STRAIGHT = 0

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')

