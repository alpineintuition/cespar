# -*- coding: utf-8 -*

"""coupled_exo_optimization_rl.py: Source code of the Reinforcement Learning for coupled musculoskeletal and exoskeleton systems
This module demonstrates how to use a gym musculoskeletal environment to adjust the impaired gait with exoskeleton actuations
Example:
    The following command should be typed in the terminal to run RL simulations with the coupled musculoskeletal and exoskeleton system. ::
        $ python coupled_exo_optimization_rl.py
"""

__author__ = "Berat Denizdurduran"
__copyright__ = "Copyright 2023, Alpine Intuition SARL"
__license__ = "Apache-2.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"
__acknowledgement__ = "The CESPAR Project is supported by European Union’s Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreement No: 945539 (Human Brain Project SGA-3)"

import sys
from control.osim_HBP_withexo_RL_after_CMAES import L2M2019Env, L2M2019EnvVecEnv
import numpy as np
import argparse
import time

import random
import sys

import pickle
import time
import os

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
from torch.distributions import LogNormal

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
#print(use_cuda)
device   = torch.device("cuda" if use_cuda else "cpu")

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


from multiprocessing_env import SubprocVecEnv

seed = 64
SEED = int(seed)   # Random seed
random.seed(SEED)
np.random.seed(SEED)

difficulty = 0
sim_dt = 0.01

# Script parameters
parser = argparse.ArgumentParser(prog='Reinforcement Learning (With PPO)')
parser.add_argument("-envs",  "--envs",  help="The number of environments launched in parallel during optimization", default=16)
parser.add_argument("-episodes", "--num_episodes", help="Number of simulations of the model", default=100000)
parser.add_argument("-epochs", "--num_epochs", help="Number of times PPO replays one episode", default=250)
parser.add_argument("-tgt_speed","--tgt_speed",      help="Target/Desired speed", default=1.4)
parser.add_argument("-init_speed", "--init_speed",   help="Initial speed of the simulation", default=1.7)
parser.add_argument("-range", "--range",   help="Range of Actuation", default=1)
parser.add_argument("-nodes", "--nodes",   help="Number of Nodes per Hidden Layer", default=32)
parser.add_argument("-c", "--checkpoint", help="Checkpoint to use for initial parameters", default= None)

args = parser.parse_args()

NUM_ENVS            = int(args.envs)
NUM_EPISODES        = int(args.num_episodes)
NUM_EPOCHS          = int(args.num_epochs)
DESIRED_SPEED       = float(args.tgt_speed)       # Desired speed (= Target speed)
INITIAL_SPEED       = float(args.init_speed)      # Initial speed of the simulation
RANGE               = float(args.range)
NUM_NODES           = int(args.nodes)
CHECKPOINT          = args.checkpoint
'''
Write parameters to the optimization folder during optimization so that we know what we were doing in that experiment
'''
CHECKPOINT_PATH = './results/Exp' # Path where the checkpoints are saved
ID = 1

test_path = f"{CHECKPOINT_PATH}_{ID}"
while os.path.isdir(test_path):
    ID += 1
    test_path = f"{CHECKPOINT_PATH}_{ID}"
CHECKPOINT_PATH = test_path
FIGURE_PATH = f"{CHECKPOINT_PATH}/reward_loss.png"
FIGURE_PATH_POS = f"{CHECKPOINT_PATH}/joint_pos.png"
FIGURE_PATH_TORQUES = f"{CHECKPOINT_PATH}/actuators_torques.png"

os.mkdir(CHECKPOINT_PATH)
with open("{}/params.txt".format(CHECKPOINT_PATH), 'w') as f:
    params_summary_txt = '''
NUM_ENVS {}
NUM_EPISODES {}
NUM_EPOCHS {}
DESIRED_SPEED {}
INITIAL_SPEED {}
RANGE {}
NUM_NODES {}
'''.format(NUM_ENVS, NUM_EPISODES, NUM_EPOCHS, DESIRED_SPEED, INITIAL_SPEED, RANGE, NUM_NODES)
    f.write(params_summary_txt)

# Initial position of the model
INIT_POSE = np.array([
    INITIAL_SPEED,              # forward speed
    .5,                         # rightward speed
    9.023245653983965608e-01,   # pelvis height
    2.012303881285582852e-01,   # trunk lean
    0*np.pi/180,                # [right] hip adduct
    -6.952390849304798115e-01,  # hip flex
    -3.231075259785813891e-01,  # knee extend
    1.709011708233401095e-01,   # ankle flex
    0*np.pi/180,                # [left] hip adduct
    -5.282323914341899296e-02,  # hip flex
    -8.041966456860847323e-01,  # knee extend
    -1.745329251994329478e-01]) # ankle flex

# Get Joint Positions and Muscle Activations
with open('./logs/simulation_CMAES/_40_5_second_all_joints.pkl', 'rb') as f:
    joints = pickle.load(f)

with open('./logs/simulation_CMAES/_40_5_second_muscle_act.pkl', 'rb') as f:
    muscle_act = pickle.load(f)

muscle_activities = np.vstack(muscle_act)

# Create environments in parallel
num_envs = NUM_ENVS
def make_env(visualize):
    def _thunk():
        env = L2M2019EnvVecEnv(visualize=visualize, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
        return env
    return _thunk

'''
    Set Linear Layers weights for neural networks
'''
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.01)
        nn.init.constant_(m.bias, 0.1)

'''
    Defines Actor and Critic Networks architecture
'''
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
            nn.Hardtanh(-2.0, 2.0)
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std).data.squeeze()

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        std = std.to(device)
        dist  = Normal(mu, std*0.01)
        return dist, value, std

def plot(frame_idx, rewards):
    plt.figure(figsize=(12,8))
    plt.subplot(111)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig("Results_RL/leg_ppo_{}_{}".format('e-0x', frame_idx))
    plt.close()

def test_env(num_steps, frame_idx):

    envs_test = [make_env(visualize=False) for i in range(1)]
    envs_test = SubprocVecEnv(envs_test)
    state = envs_test.reset()

    with open('./logs/simulation_CMAES/_40_5_second_all_joints.pkl', 'rb') as f:
        joints = pickle.load(f)
    joint_activities = np.vstack(joints)
    hip_activities = joint_activities[0::3]
    knee_activities = joint_activities[1::3]
    ankle_activities = joint_activities[2::3]
    state_hip_r = []
    state_hip_l = []
    state_knee_r = []
    state_knee_l = []
    state_ankle_r = []
    state_ankle_l = []
    action_hip = []
    action_hip_l = []
    action_knee = []
    action_knee_l = []
    action_ankle = []
    action_ankle_l = []
    done = False
    total_reward = 0
    for i in range(num_steps):
        list_all = state
        state_hip_r.append(list_all[0,0])
        state_hip_l.append(list_all[0,3])
        state_knee_r.append(list_all[0,1])
        state_knee_l.append(list_all[0,4])
        state_ankle_r.append(list_all[0,2])
        state_ankle_l.append(list_all[0,5])
        state = torch.FloatTensor(state).to(device)
        dist, value, _ = model_musculo(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.cpu().numpy()
        if i<30:
            action[0][0] = 0.0
            action[0][1] = 0.0
            action[0][2] = 0.0
            action[0][3] = 0.0
            action[0][4] = 0.0
            action[0][5] = 0.0
            next_state, reward, done, _ = envs_test.step(action)
        else:
            next_state, reward, done, _ = envs_test.step(action)

        action_hip.append(action[0][0])
        action_knee.append(action[0][1])
        action_ankle.append(action[0][2])
        action_hip_l.append(action[0][3])
        action_knee_l.append(action[0][4])
        action_ankle_l.append(action[0][5])
        state = next_state
        total_reward -= reward.mean()
        if done:
            break

    fig, ax = plt.subplots(4, 2, figsize=(12,8))
    ax[0, 0].plot(hip_activities[:,0][:num_steps], 'b')
    ax[0, 0].plot(state_hip_r, 'r')
    ax[1, 0].plot(knee_activities[:,0][:num_steps])
    ax[1, 0].plot(state_knee_r, 'r')
    ax[2, 0].plot(ankle_activities[:,0][:num_steps])
    ax[2, 0].plot(state_ankle_r, 'r')
    ax[3, 0].plot(action_hip, 'r')
    ax[3, 0].plot(action_knee, 'b')
    ax[3, 0].plot(action_ankle, 'g')
    ax[0, 1].plot(hip_activities[:,1][:num_steps], 'b')
    ax[0, 1].plot(state_hip_l, 'r')
    ax[1, 1].plot(knee_activities[:,1][:num_steps])
    ax[1, 1].plot(state_knee_l, 'r')
    ax[2, 1].plot(ankle_activities[:,1][:num_steps])
    ax[2, 1].plot(state_ankle_l, 'r')
    ax[3, 1].plot(action_hip_l, 'r')
    ax[3, 1].plot(action_knee_l, 'b')
    ax[3, 1].plot(action_ankle_l, 'g')
    plt.savefig("Results_RL/exo_ppo_states_all_musculo_{}_{}".format('e-0x',frame_idx))
    plt.close()
    envs_test.reset()
    envs_test.close()
    return total_reward

'''
    Calculates Generalized Advantage Estimate
'''
def compute_gae(next_value, rewards, masks, values, gamma=0.9, tau=0.99):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

'''
    Takes random steps/iterations of the episode
'''
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

'''
    Updates current policy by using sampled steps/iterations from ppo_iter
'''
def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    actor_losses = []
    critic_losses = []
    entropies = []
    losses = []
    kl_divs = []
    stds = []
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value, std = model_musculo(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            kl_divs.append(F.kl_div(new_log_probs, old_log_probs, reduction='batchmean'))
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropies.append(entropy)
            losses.append(loss)
            stds.append(std.mean())

            optimizer_musculo.zero_grad()
            loss.backward()
            optimizer_musculo.step()

    actor_losses = torch.stack(actor_losses)
    critic_losses = torch.stack(critic_losses)
    entropies = torch.stack(entropies)
    losses = torch.stack(losses)
    kl_divs = torch.stack(kl_divs)
    stds = torch.stack(stds)
    return actor_losses, critic_losses, entropies, losses, kl_divs, stds

'''
    Saves current best actions, best reward and critic loss with actor, critic and optimizer parameters
'''
def save_model(episode, model_musculo, optimizer_musculo, reward, critic_loss, best_actions):
    critic_loss = critic_loss.cpu().detach().numpy()
    critic_loss = np.mean(critic_loss)
    #print("Episode: {} - Saving model and testing, Current Error: {}, Current Reward: {}".format(episode, critic_loss, reward))
    ppo_model_arm_musculo = {
        'epoch': frame_idx,
        'model_state_dict': model_musculo.state_dict(),
        'optimizer_state_dict': optimizer_musculo.state_dict(),
        'reward': reward,
        'critic_loss': critic_loss,
        'best_actions': best_actions}
    ppo_path =  f"{CHECKPOINT_PATH}/ppo_exo_hips_only_{episode}_{reward}"
    torch.save(ppo_model_arm_musculo, ppo_path)
    return ppo_path

'''
    Load parameters into arctor, critic and optimizer
'''
def load_model(checkpoint_path, actor_critic, optimizer):
    #print('Loading Checkpoint...')
    actor_critic.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer_state_dict"])
    #print('Loaded Model has Reward : {} and Critic Loss : {}'.format(torch.load(checkpoint_path)["reward"], torch.load(checkpoint_path)["critic_loss"]))

    return torch.load(checkpoint_path)["reward"], torch.load(checkpoint_path)["best_actions"]


num_inputs  = 6
num_outputs = 6

checkpoint_path = CHECKPOINT

#Hyper params:
hidden_size      = 32
lr               = 2e-4
betas            = (0.9, 0.999)
num_steps        = 300
mini_batch_size  = 32
ppo_epochs       = 32


start_time = time.time()
model_musculo = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer_musculo = optim.Adam(model_musculo.parameters(), lr=lr)

model_name = None

# Write model architecture and optimizer in params.txt
with open("{}/params.txt".format(CHECKPOINT_PATH), 'a') as f:
    params_summary_txt = '''
NETWORKS_ARCHITECTURES
{}
OPTIMIZER
{}
'''.format(model_musculo, optimizer_musculo)
    f.write(params_summary_txt)

#Load model parameters
loaded_reward = -10000
if checkpoint_path != None:
    loaded_reward, best_actions = load_model(checkpoint_path, model_musculo, optimizer_musculo)
    best_actions_all = best_actions
    model_name = checkpoint_path
    best_model_name = checkpoint_path

frame_idx  = 0
test_rewards = []

episodes_rewards = []
threadone_episodes_rewards = []
best_episodes_rewards = []
episodes_critic_losses = []
episodes_kl = []
episodes_actor_losses = []
episodes_losses = []
episodes_entropies = []
episodes_std = []
#low_torque = LOW_RANGE
#high_torque = HIGH_RANGE
range_steps = NUM_EPISODES
best_ep = 0

for steps in range(range_steps):

    #print("Episode #{}, Best Episode : #{}".format(steps, best_ep))

    envs = [make_env(visualize=False) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    state = envs.reset()

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    stds = []
    entropy = 0
    threadone_episode_reward = 0
    episode_reward = 0

    for i in range(num_steps):

        state = torch.FloatTensor(state).to(device)
        dist, value, _ = model_musculo(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        actions.append(action)
        action = action.cpu().numpy()
        if i < 29: # initial flight time actions are bypassed
            action = action*0.0
        else:
            action = action

        next_state, reward, done, _ = envs.step(action)

        #learning starts
        if i > 29:
            entropy += dist.entropy().mean()
            threadone_episode_reward += reward[0]
            episode_reward += reward
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
        state = next_state
        frame_idx += 1

        if (True in done):
            #next_state, reward, _, _ = envs.step(action)
            break

        if frame_idx % 10 == 0:
            test_reward = test_env(num_steps, frame_idx)
            test_rewards.append(test_reward)
            plot(frame_idx, test_rewards)
            #print("Iter: {} - Testing, Current Error: {}".format(frame_idx, test_reward))


        if frame_idx % 10 == 0:
            #print("Iter: {} - Saving model".format(frame_idx))
            ppo_model_arm_musculo = {
                        'epoch': frame_idx,
                        'model_state_dict': model_musculo.state_dict(),
                        'optimizer_state_dict': optimizer_musculo.state_dict(),
                        'loss': test_rewards}

            torch.save(ppo_model_arm_musculo, "Results_RL/ppo_model_exo100_Ex1_musculo_{}".format(frame_idx))


    episodes_rewards.append(episode_reward)
    envs.reset()
    envs.close()

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value, _ = model_musculo(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values

    actor_losses, critic_losses, entropies, losses, kl_divs, stds = ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
    episodes_actor_losses.append(actor_losses)
    episodes_critic_losses.append(critic_losses)
    episodes_entropies.append(entropies)
    episodes_losses.append(losses)
    episodes_kl.append(kl_divs)
    episodes_std.append(stds)

end_time = time.time()
run_time = end_time - start_time
