import math
import random

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

env_name = "CartPole-v1"
env = gym.make(env_name)


# Clase ActorCritic: separa Actor (política) y Critic (valor).
class ActorCritic(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(n_inputs, hidden_size), nn.ReLU()
            , nn.Linear(hidden_size, hidden_size), nn.ReLU()
            , nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(n_inputs, hidden_size), nn.ReLU()
            , nn.Linear(hidden_size, hidden_size), nn.ReLU()
            , nn.Linear(hidden_size, n_outputs)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(logits=probs)
        return dist, value.squeeze(-1)

class ObsNorm:
    def __init__(self, eps=1e-8):
        self.count = 0
        self.mean = None
        self.var  = None
        self.eps  = eps

    def update(self, x):
        # Asegurarse de que es torch.Tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.flatten()

        if self.mean is None:
            self.mean = x.clone()
            self.var  = torch.ones_like(x)
            self.count = 1
            return

        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var  += delta * (x - self.mean)

    def normalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.flatten()
        std = torch.sqrt(self.var / max(self.count-1, 1) + self.eps)
        return (x - self.mean) / std    # <- devuelve tensor


obs_norm = ObsNorm()

# dist.sample(): muestrea acciones para explorar.
# En evaluación, usar la acción más probable si se desea comportamiento determinista.
# compute_returns: implementa el bootstrapping con V(s_{t+1}) y máscaras.
import matplotlib.pyplot as plt
from IPython.display import clear_output

def compute_returns_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    """
    rewards, masks, values: tensores 1D de longitud T
    next_value: escalar 1D (valor del estado posterior al último paso)
    Devuelve: returns (T), advantages (T)
    """
    T = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32)
    adv = torch.zeros(T, dtype=torch.float32)

    gae = 0.0
    for t in reversed(range(T)):
        v_next = next_value if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * v_next * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        adv[t] = gae
        returns[t] = adv[t] + values[t]
    return returns, adv

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def test_env(vis=False, model=None):
    state, _ = env.reset()
    # state = obs_norm.normalize(state)
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state)
        next_state, reward, terminated, truncated, _ = env.step(int(dist.probs.argmax(dim=-1)))
        # next_state = obs_norm.normalize(next_state)
        done = np.logical_or(terminated, truncated)
        state = next_state
        total_reward += reward

    return total_reward

# --- NUEVO: utilidades de graficado ---
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Exponencial Moving Average para suavizar (opcional)
def ema(x, alpha=0.1):
    if len(x) == 0:
        return []
    y = [x[0]]
    for t in range(1, len(x)):
        y.append(alpha * x[t] + (1 - alpha) * y[-1])
    return y

def plot_losses(frames_hist, actor_hist, critic_hist, total_hist, use_ema=True):
    clear_output(wait=True)
    plt.figure(figsize=(8,4))
    if use_ema:
        plt.plot(frames_hist, ema(actor_hist), label='Actor loss (EMA)')
        plt.plot(frames_hist, ema(critic_hist), label='Critic loss (EMA)')
        plt.plot(frames_hist, ema(total_hist), label='Total loss (EMA)')
    else:
        plt.plot(frames_hist, actor_hist, label='Actor loss')
        plt.plot(frames_hist, critic_hist, label='Critic loss')
        plt.plot(frames_hist, total_hist, label='Total loss')
    plt.xlabel('Frames')
    plt.ylabel('Loss')
    plt.title('Evolución de pérdidas: Actor / Critic / Total')
    plt.legend()
    plt.grid(True)
    plt.show()

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n
hidden_size = 128

print(f"{n_inputs} - {n_outputs} - {hidden_size}")


import torch.nn.functional as F

actor_critic_model = ActorCritic(n_inputs, n_outputs, hidden_size)
optimizer = optim.Adam(actor_critic_model.parameters(), lr=3e-4)

n_steps = 50
max_frames = 100000

state, _ = env.reset()
obs_norm.update(state)
state = obs_norm.normalize(state)

test_rewards = []
frame_idx = 0

frames_hist = []
actor_hist = []
critic_hist = []
total_hist = []

while frame_idx < max_frames:

    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0.0

    for _ in range(n_steps):

        state_t = torch.as_tensor(state, dtype=torch.float32)
        dist, value = actor_critic_model(state_t)

        action = dist.sample()
        action_np = int(action.item())

        next_state, reward, terminated, truncated, _ = env.step(action_np)

        next_state = obs_norm.normalize(next_state)
        done = np.logical_or(terminated, truncated)

        if done:
            state, _ = env.reset()
            state_t = torch.as_tensor(state, dtype=torch.float32)

        log_prob = dist.log_prob(action)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(-1)
        entropy = entropy + dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value.squeeze(-1))
        rewards.append(torch.as_tensor(reward, dtype=torch.float32))
        masks.append(torch.as_tensor(1.0 - done.astype(np.float32), dtype=torch.float32))  # (n_envs,)

        state = next_state
        if done:
            state, _ = env.reset()

        obs_norm.update(state)
        state = obs_norm.normalize(state)

        frame_idx += 1

        if frame_idx % 1000 == 0:
            test_rewards.append(np.mean([test_env(model=actor_critic_model) for _ in range(5)]))
            plot(frame_idx, test_rewards)  # tu función existente

    next_state_t = torch.as_tensor(next_state, dtype=torch.float32)
    obs_norm.update(next_state_t)
    next_state_t = obs_norm.normalize(next_state_t)

    with torch.no_grad():
        _, next_value = actor_critic_model(next_state_t)

    next_value = next_value.squeeze(-1).unsqueeze(-1)

    returns, advantages = compute_returns_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95)

    log_probs = torch.stack(log_probs)  # [T]
    values = torch.stack(values)  # [T]
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(log_probs * adv_norm.detach()).mean()
    critic_loss = (returns - values).pow(2).mean()  # <- sin normalizar
    # critic_loss = F.smooth_l1_loss(values, returns)
    loss = actor_loss + 0.5 * critic_loss - 0.05 * (entropy / len(log_probs))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic_model.parameters(), 0.5)
    optimizer.step()

    # --- NUEVO: registrar y graficar ---
    frames_hist.append(frame_idx)
    actor_hist.append(actor_loss.item())
    critic_hist.append(critic_loss.item())
    total_hist.append(loss.item())



plot_losses(frames_hist, actor_hist, critic_hist, total_hist, use_ema=True)


