import gym 
from ppo_hanjj1 import PPO
import configparser
from surg_env import SurgicalRoboticsEnvironment

from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.scene import Scene
import rospy
from sensor_msgs.msg import Image
from surgical_robotics_challenge.task_completion_report import TaskCompletionReport, PoseStamped

from PyKDL import Frame, Rotation, Vector
import numpy as np

# Import AMBF Client
from surgical_robotics_challenge.simulation_manager import SimulationManager
import time

def add_break(s):
    time.sleep(s)
    print('-------------')

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

if __name__ == '__main__':
    
    # params (irrelevant but still need to call for sake of framework)
    params = get_params('./config.ini')

    # create environment
    env = SurgicalRoboticsEnvironment()
    
    # make model, load, and run demo
    model = PPO(env, params)
    model.load_models()
    
    simulation_manager = SimulationManager('my_example_client')
    time.sleep(0.5)
    world_handle = simulation_manager.get_world_handle()

    psm1 = PSM(simulation_manager, 'psm1')
    ecm = ECM(simulation_manager, 'CameraFrame')
    scene = Scene(simulation_manager)
    add_break(0.5)
    max_timesteps = 2000

    for k in range(10):
        world_handle.reset()
        print(f'iteration {k}')
        add_break(3.0)
        obs = env.reset()[0]
        for _ in range(max_timesteps):
            action, _ = model.choose_action(obs)
            psm1.servo_jv(action[:-1])
            psm1.set_jaw_angle(action[-1])