import gym 
from gym import spaces
import numpy as np 
from psm_arm import PSM 
from ecm_arm import ECM 
from scene import Scene 
from simulation_manager import SimulationManager

from surgical_robotics_challenge.utils.utilities import *
from surgical_robotics_challenge.kinematics.DH import *

class SurgicalRoboticsEnvironment(gym.Env):

    def __init__(self):
        super(SurgicalRoboticsEnvironment, self).__init__()

        # state is 7 joint readings + xyz of needle
        self.current_state = [0., 0., 0., 0., 0., 0., 0., -0.0207876, 0.0561979, 0.0711725]
        self.goal_state = np.array([ -0.00373974, 0.0441891, 0.0750042])
        self.carrying_needle = False 
        
        # action space is 7 joint velocities
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1, 1, 1, 1]).astype(np.float32),
        )
        # joint0-6 is continuous, joint7 should be discrete. the last 3 are positions of the needle (also continuous)
        low = np.array([np.deg2rad(-91.96), np.deg2rad(-60), -0.0, np.deg2rad(-175), np.deg2rad(-90), np.deg2rad(-85), 0, -0.1, -0.1, -0.1])
        high = np.array([np.deg2rad(91.96), np.deg2rad(60), 0.240, np.deg2rad(175), np.deg2rad(90), np.deg2rad(85), 1, 0.1, 0.1, 0.1])
        self.observation_space = spaces.Box(low, high)

    def reset(self):
        self.current_state = [-0.038131931943846814, -0.033320203374724154, -0.004842678178664233, 0.00018271978478878736, -2.454812238283921e-05, -0.0004151750181335956, 0.0, -0.0207876, 0.0561979, 0.0711725]
        return np.array(self.current_state, dtype=np.float32), None

    def compute_FK(joint_pos, up_to_link):
        kinematics_data = PSMKinematicData()
        if up_to_link > kinematics_data.num_links:
            raise "ERROR! COMPUTE FK UP_TO_LINK GREATER THAN DOF"
        j = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(joint_pos)):
            j[i] = joint_pos[i]

        T_N_0 = np.identity(4)

        for i in range(up_to_link):
            link_dh = kinematics_data.get_link_params(i)
            link_dh.theta = j[i]
            T_N_0 = T_N_0 * link_dh.get_trans()

        return T_N_0
    
    # need to define new state, reward, done
    def step(self, action):
        # action space is 7 long, for each joint velocity
        old_state = self.current_state
        curr_state = self.current_state
        dt = 1 # try tuning this guy
        curr_state[0] += dt * action[0]
        curr_state[1] += dt * action[1]
        curr_state[2] += dt * action[2]
        curr_state[3] += dt * action[3]
        curr_state[4] += dt * action[4]
        curr_state[5] += dt * action[5]
        curr_state[6] += dt * action[6]
        curr_state[6] = np.round(np.clip(curr_state[6], 0, 1))

        if self.carrying_needle:
            curr_state[7:] = self.compute_FK(self.current_state[:7], 7)[:3,3]

        if self._is_valid_position(curr_state):
            self.current_state = curr_state 
        
        reward = -10
        finish = False      
        
        # penalize for illegal picking up and dropping off
        # first if we closed the gripper without getting to needle 
        if curr_state[6] == 1 and old_state[6] == 0 and not self.carrying_needle and not self.got_to_needle():
            reward = -500
        # next if we release gripper without getting to the end 
        if curr_state[6] == 0 and old_state[6] == 1 and self.carrying_needle and not self.got_to_goal():
            print('dropped needle')
            reward = -500
            self.carrying_needle = False 
        # yay! we finished
        if curr_state[6] == 1 and old_state[6] == 0 and not self.carrying_needle and self.got_to_needle():
            print('pick up needle')
            reward = 500
            self.carrying_needle = True 
        if curr_state[6] == 0 and old_state[6] == 1 and self.carrying_needle and self.got_to_goal():
            print('droppped needle')
            reward = 1000
            finish = True 

        return np.array(self.current_state).astype(np.float32), reward, finish, False, {}

    # returns bool whether or not position is good 
    def _is_valid_position(self, pos):
        return self.observation_space.contains(np.array(pos))

    def got_to_needle(self):
        # if self.carrying_needle:
        #     return True 
        
        gripper_pos = self.compute_FK(self.current_state[:7], 7)[:3,3]
        needle_pos = self.current_state[7:]
        if np.linalg.norm(gripper_pos - needle_pos) < 1e-2:
            return True 
        return False 
    
    def got_to_goal(self):
        gripper_pos = self.compute_FK(self.current_state[:7], 7)[:3,3]
        if np.linalg.norm(gripper_pos - self.goal_state) < 1e-2:
            return True 
        return False 

    # render it to the screen... unsure how this is going to work lol
    def render(self):
        pass 

class PSMKinematicData:
    def __init__(self):
        self.num_links = 7

        self.L_rcc = 0.4389  # From dVRK documentation x 10
        self.L_tool = 0.416  # From dVRK documentation x 10
        self.L_pitch2yaw = 0.009  # Fixed length from the palm joint to the pinch joint
        self.L_yaw2ctrlpnt = 0.0  # Fixed length from the pinch joint to the pinch tip
        self.L_tool2rcm_offset = 0.0229 # Distance between tool tip and the Remote Center of Motion at Home Pose

        # PSM DH Params
        # alpha | a | theta | d | offset | type
        self.kinematics = [DH(PI_2, 0, 0, 0, PI_2, JointType.REVOLUTE, Convention.MODIFIED),
                           DH(-PI_2, 0, 0, 0, -PI_2,
                              JointType.REVOLUTE, Convention.MODIFIED),
                           DH(PI_2, 0, 0, 0, -self.L_rcc,
                              JointType.PRISMATIC, Convention.MODIFIED),
                           DH(0, 0, 0, self.L_tool, 0,
                              JointType.REVOLUTE, Convention.MODIFIED),
                           DH(-PI_2, 0, 0, 0, -PI_2,
                              JointType.REVOLUTE, Convention.MODIFIED),
                           DH(-PI_2, self.L_pitch2yaw, 0, 0, -PI_2,
                              JointType.REVOLUTE, Convention.MODIFIED),
                           DH(-PI_2, 0, 0, self.L_yaw2ctrlpnt, PI_2, JointType.REVOLUTE, Convention.MODIFIED)]

        self.lower_limits = [np.deg2rad(-91.96), np.deg2rad(-60), -0.0, np.deg2rad(-175), np.deg2rad(-90), np.deg2rad(-85)]

        self.upper_limits = [np.deg2rad(91.96), np.deg2rad(60), 0.240, np.deg2rad(175), np.deg2rad(90), np.deg2rad(85)]

    def get_link_params(self, link_num):
        if link_num < 0 or link_num > self.num_links:
            # Error
            print("ERROR, ONLY ", self.num_links, " JOINT DEFINED")
            return []
        else:
            return self.kinematics[link_num]