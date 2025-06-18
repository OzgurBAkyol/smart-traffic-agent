import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# En iyi cihazı seçen fonksiyon

def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        qvals = value + (advantage - advantage.mean())
        return qvals

class RLAgent:
    def __init__(self, input_size=None, hidden_size=256, output_size=None, memory_size=100000, batch_size=512, num_phases=2):
        self.num_phases = num_phases
        if input_size is None:
            self.input_size = 32 if num_phases == 2 else 70
        else:
            self.input_size = input_size
        if output_size is None:
            self.output_size = num_phases
        else:
            self.output_size = output_size
        self.device = get_best_device()
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        # Dueling DQN
        self.q_network = DuelingDQN(self.input_size, hidden_size, self.output_size).to(self.device)
        self.target_network = DuelingDQN(self.input_size, hidden_size, self.output_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99995
        self.target_update_freq = 5
        self.train_step = 0
    
    def select_action(self, state):
        # Kural tabanlı öncelik: Eğer state vektöründe büyük kuyruk varsa o faza öncelik ver
        if self.num_phases == 6 and len(state) >= 70:
            phase_scores = []
            for phase in range(6):
                # DL modelindeki gibi: faza ait hareketlerin toplam kuyruk ve bekleme süresi
                phase_movements = [
                    [0,1,2,3],    # A
                    [4,5],        # B
                    [10,11,8,12], # C
                    [6,7,8,9],    # D
                    [13,12],      # E
                    [14,15,3,2]   # F
                ][phase]
                score = 0
                for idx in phase_movements:
                    queue = state[idx*4+2]
                    wait = state[idx*4+1]
                    score += 20*queue + 20*wait
                phase_scores.append(score)
            # %5 olasılıkla random, %95 kural tabanlı
            if random.random() > 0.05:
                return int(np.argmax(phase_scores))
        if random.random() < self.epsilon:
            return random.randrange(self.output_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        # Reward clipping
        reward = max(min(reward, 1000), -1000)
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Replay buffer'da tüm geçmişten örnekleme
        batch_size = min(self.batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class PPOActor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class PPOCritic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

class PPOAgent:
    def __init__(self, input_size, output_size, hidden_size=128, gamma=0.99, clip_epsilon=0.2, lr=0.0003, update_steps=5, entropy_coef=0.001):
        self.device = get_best_device()
        self.actor = PPOActor(input_size, hidden_size, output_size).to(self.device)
        self.critic = PPOCritic(input_size, hidden_size).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_steps = update_steps
        self.entropy_coef = entropy_coef
        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), dist.entropy().item()

    def remember(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) == 0:
            return
        states, actions, rewards, next_states, dones, log_probs, entropies, steps = zip(*self.memory)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        entropies = torch.FloatTensor(entropies).to(self.device)
        steps = torch.LongTensor(steps).to(self.device)
        # Reward scaling: ilk 500 adımda ödül/ceza 2 katı
        rewards = torch.where(steps < 500, rewards * 2, rewards)
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        for _ in range(self.update_steps):
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            # Advantage normalization
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = nn.MSELoss()(values, returns)
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
        self.memory = [] 