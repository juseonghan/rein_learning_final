import gym 
import numpy as np
import torch 
import configparser
from surg_env import SurgicalRoboticsEnvironment
from ppo_hanjj1 import PPO

# use configparser to choose hyperparameters easily
def get_params(path):
    config = configparser.ConfigParser()
    config.read(path)
    data = config['PARAMS']
    params = {'timesteps': int(data['timesteps']),
              'max_timesteps': int(data['max_timesteps']),
              'gamma': float(data['gamma']),
              'updates_per_iteration': int(data['updates_per_iteration']),
              'lr': float(data['lr']),
              'clip': float(data['clip']),
              'total_timesteps': int(data['total_timesteps'])}
    return params

# save our models, save our losses :) 
def logging(model, losses_A, losses_C, final_rewards):
    torch.save(model.actor.state_dict(), './actor.pt')
    torch.save(model.critic.state_dict(), './critic.pt')
    np.save('./losses_A.npy', losses_A)
    np.save('./losses_C.npy', losses_C)
    np.save('./rewards.npy', final_rewards  )

def main(env, params):

    # define the model
    model = PPO(env, params)

    # train our model to learn the optimal policy
    losses_A, losses_C, final_rewards = model.train(params['total_timesteps'])
    
    # logging 
    logging(model, losses_A, losses_C, final_rewards)


if __name__ == '__main__':
    params = get_params('./config.ini')
    # env = gym.make('BipedalWalker-v3')
    env = SurgicalRoboticsEnvironment()
    main(env, params)