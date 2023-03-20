import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()
        self.loss = nn.HuberLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if not torch.backends.mps.is_available() else torch.device('mps')
        self.to(self.device)
        
    def forward(self, state):
        q_vals = self.fc(state)
        
        return q_vals
    
class Dueling_DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, val_fc1_dims, val_fc2_dims, 
                 adv_fc1_dims, adv_fc2_dims, n_actions) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.val_fc1_dims = val_fc1_dims
        self.val_fc2_dims = val_fc2_dims
        self.adv_fc1_dims = adv_fc1_dims
        self.adv_fc2_dims = adv_fc2_dims
        self.n_actions = n_actions
        
        self.fc = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
        )
        
        self.val_fc = nn.Sequential(
            nn.Linear(self.fc2_dims, self.val_fc1_dims),
            nn.ReLU(),
            nn.Linear(self.val_fc1_dims, self.val_fc2_dims),
            nn.ReLU(),
            nn.Linear(self.val_fc2_dims, 1)
        )
        
        self.adv_fc = nn.Sequential(
            nn.Linear(self.fc2_dims, self.adv_fc1_dims),
            nn.ReLU(),
            nn.Linear(self.adv_fc1_dims, self.adv_fc2_dims),
            nn.ReLU(),
            nn.Linear(self.adv_fc2_dims, n_actions)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()
        self.loss = nn.HuberLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if not torch.backends.mps.is_available() else torch.device('mps')
        self.to(self.device)
        
    def forward(self, state):
        x = self.fc(state)
        v = self.val_fc(x)
        a = self.adv_fc(x)
        
        q_vals = v + a - a.mean(dim=-1, keepdim=True).expand(-1, self.n_actions)
        
        return q_vals
        
    
class Agent:
    def __init__(self, gamma, lr, input_dims, batch_size, n_actions, 
                 max_mem_size=100000, eps_max=1.0, eps_min=0.01, eps_dec=0.002, double_dqn=True, 
                 dueling_dqn=True, learn_per_target_net_update=50, seed=None) -> None:
        self.gamma = gamma
        self.epsilon = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.input_dims = input_dims
        self.prediction = False
        self.loss = np.inf
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.target_net_update_per_step = learn_per_target_net_update
        self.learn_counter = 0
        self.seed=seed
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        self.Q_eval = DQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, n_actions=n_actions) if not self.dueling_dqn \
                      else Dueling_DQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, val_fc1_dims=256, val_fc2_dims=256, adv_fc1_dims=256, 
                          adv_fc2_dims=256, n_actions=n_actions)
        self.Q_target = DQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, n_actions=n_actions) if not self.dueling_dqn \
                      else Dueling_DQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, val_fc1_dims=256, val_fc2_dims=256, adv_fc1_dims=256, 
                          adv_fc2_dims=256, n_actions=n_actions)
        self._update_target_net()
        
        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_counter % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        
        self.mem_counter += 1
        
    def choose_action(self, obs):
        if self.prediction or np.random.random() > self.epsilon:
            state = torch.tensor(np.array([obs])).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state) if self.double_dqn else self.Q_target.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    def learn(self, episode):
        # record total learn step number
        self.learn_counter += 1
        
        if self.mem_counter < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, size=self.batch_size, replace=False)
        
        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_idx, action_batch]
        q_next = self.Q_target.forward(new_state_batch) if self.double_dqn else self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.loss = loss.item()
        self.Q_eval.optimizer.step()
        
        # self.epsilon = max(self.eps_min, self.eps_max * 1 / (1 + episode))
        self.epsilon = max(self.eps_max - episode * self.eps_dec, self.eps_min)
                      
        if self.learn_counter % self.target_net_update_per_step == 0:
            self._update_target_net()
                        
    def save(self, filename):
        torch.save(self.Q_eval.state_dict(), filename)
        
    def load(self, filename):
        self.Q_eval.load_state_dict(torch.load(filename))