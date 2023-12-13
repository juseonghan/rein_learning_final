import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions

class PPO:

    def __init__(self, env, params):

        # read params from argument
        self.env = env 
        self.timesteps = params['timesteps']
        self.max_timesteps = params['max_timesteps']
        self.gamma = params['gamma']
        self.updates_per_iteration = params['updates_per_iteration']
        self.lr = params['lr']
        self.clip = params['clip']

        # get some space dim info
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # initialize actor-critic things
        self.actor = MLP(self.observation_dim, self.action_dim)
        self.critic = MLP(self.observation_dim, 1)
        self.optim_A = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_C = optim.Adam(self.critic.parameters(), lr=self.lr)

        # PPO uses covariance so we're updating this throughout our learning
        self.variance = torch.full((self.action_dim,), 0.5)
        self.mat = torch.diag(self.variance)


    # main training function
    def train(self, total_timesteps):
        
        # logging lists 
        t, num_iters = 0, 0
        final_rewards = []
        losses_A, losses_C = [], []

        # multiple time steps progresses in one loop so we use while loop instead 
        while t < total_timesteps:

            # logging and keeping track of results 
            observations, actions, log_probs, rewards, lengths = [], [], [], [], []
            t_batch = 0

            # batch loop basically 
            while t_batch < self.timesteps:
                ep_rewards = []
                obs = self.env.reset()[0]
                done = False 

                # main loop to get action, choose action, stepping into that action, and logging
                for t_episode in range(self.max_timesteps):
                    
                    t_batch += 1
                    observations.append(obs)
                    action, log_prob = self.choose_action(obs)
                    obs, reward, done, _, _ = self.env.step(action)

                    ep_rewards.append(reward)
                    actions.append(action)
                    log_probs.append(log_prob)

                    if done:
                        break
                
                lengths.append(t_episode+1)
                rewards.append(ep_rewards)

            # logging and training results 
            final_rewards.append(sum(rewards[-1]) / len(rewards[-1]) )
            observations = torch.Tensor(np.array(observations)).float()
            actions = torch.Tensor(np.array(actions)).float()
            log_probs = torch.Tensor(np.array(log_probs)).float()

            # get the discounted rewards as listed in the original paper. update relevant variables
            discounted_rewards = self.compute_discounted_rewards(rewards)
            t += sum(lengths)
            num_iters += 1

            # evaluation of our policy via critic network
            V, _log_probs = self.advantage(observations, actions)
            A_at_k = discounted_rewards - V.detach()
            A_at_k = (A_at_k - A_at_k.mean()) / (A_at_k.std() + 1e-10)

            # logging, update model based on critic's observation of actor's chosen action
            running_A, running_C = 0., 0.
            for aaa in range(self.updates_per_iteration):
                
                # PPO uses surrogate losses for actor
                rs = torch.exp(_log_probs - log_probs)
                surrogate1 = rs * A_at_k 
                surrogate2 = A_at_k * torch.clamp(rs, 1 - self.clip, 1 + self.clip)

                # loss functions defined in the original paper
                loss_A = -(torch.min(surrogate1, surrogate2)).mean()
                loss_C = nn.MSELoss()(V, discounted_rewards)

                running_A += loss_A.item()
                running_C += loss_C.item()

                # pytorch stuff...
                self.optim_A.zero_grad()
                self.optim_C.zero_grad()
                loss_A.backward()
                loss_C.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
                self.optim_A.step()
                self.optim_C.step()
                V, _log_probs = self.advantage(observations, actions)

            # logging purposes
            if num_iters % 10 == 0:
                print(f'AT TIMESTEP {t}: AVG REWARDS: {sum(rewards[-1])/len(rewards[-1])}, ACTOR_AVG_LOSS: {running_A / (aaa+1)}, CRITIC_AVG_LOSS: {running_C / (aaa+1)}')
                torch.save(self.actor.state_dict(), './actor.pt')
                torch.save(self.critic.state_dict(), './critic.pt')

            losses_A.append(running_A / (aaa + 1))
            losses_C.append(running_C / (aaa + 1))

        return losses_A, losses_C, final_rewards

    # based on the current state, we sample from a normal distribution based on the action chosen 
    # get log prob of our sample from Gaussian and choose action stochastically
    def choose_action(self, observation):
        value = self.actor(observation)
        dist = distributions.MultivariateNormal(value, self.mat)
        sample = dist.sample()
        p = dist.log_prob(sample)

        return sample.detach().numpy(), p.detach()
    
    # evaluate our model via Critic network. we use Gaussian normal again to 
    # evaluate our Actor's chosen action stochastically
    def advantage(self, observations, actions):
        V = self.critic(observations).squeeze()

        val = self.actor(observations)
        dist = distributions.MultivariateNormal(val, self.mat)
        p = dist.log_prob(actions)
        return V, p

    # we compute discounted rewards just like how we usually do
    def compute_discounted_rewards(self, rewards):

        result = []
        for reward in reversed(rewards):
            disc = 0.

            for r in reversed(reward):
                disc = r + self.gamma * disc 
                result.insert(0, disc)
        
        result = torch.Tensor(np.array(result)).float()
        return result 

    # load models but PPO class itself isn't a network so we have to access 
    # our own actor and critic networks and load state dict
    def load_models(self):
        self.actor.load_state_dict(torch.load('./actor.pt'))
        self.critic.load_state_dict(torch.load('./critic.pt'))
        self.actor.eval()
        self.critic.eval()

    # used for our testing, where we are just choosing actions based on our 
    # learned optimal policy 10 times to show the performance
    def run_demos(self):
        num_demos = 10

        for _ in range(num_demos):
            obs = self.env.reset()[0]
            for _ in range(self.max_timesteps):
                    
                self.env.render()
                action, _ = self.choose_action(obs)
                obs, _, done, _, _ = self.env.step(action)

                if done:
                    break
        self.env.close()

# simple MLP class with 2 hidden layers and ReLU activation
class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(MLP, self).__init__()

        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).float()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x 