import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from abc import ABC


class AbstractNoisyLayer(nn.Module, ABC):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float,
    ):
        super().__init__()

        self.sigma = sigma
        self.input_features = input_features
        self.output_features = output_features

        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))

        self.register_buffer('epsilon_input', torch.FloatTensor(input_features))
        self.register_buffer('epsilon_output', torch.FloatTensor(output_features))

    def forward(
            self,
            x: torch.Tensor,
            sample_noise: bool = True
    ) -> torch.Tensor:
        if not self.training:
            return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        return nn.functional.linear(x, weight=self.weight, bias=self.bias)

    @property
    def weight(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bias(self) -> torch.Tensor:
        raise NotImplementedError

    def sample_noise(self) -> None:
        raise NotImplementedError

    def parameter_initialization(self) -> None:
        raise NotImplementedError

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        noise = torch.FloatTensor(features).uniform_(-self.bound, self.bound).to(self.mu_bias.device)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


class IndependentNoisyLayer(AbstractNoisyLayer):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float = 0.017,
    ):
        super().__init__(
            input_features=input_features,
            output_features=output_features,
            sigma=sigma
        )

        self.bound = (3 / input_features) ** 0.5
        self.parameter_initialization()
        self.sample_noise()

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * self.epsilon_weight + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_bias + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_bias = self.get_noise_tensor((self.output_features,))
        self.epsilon_weight = self.get_noise_tensor((self.output_features, self.input_features))

    def parameter_initialization(self) -> None:
        self.sigma_bias.data.fill_(self.sigma)
        self.sigma_weight.data.fill_(self.sigma)
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)


class FactorisedNoisyLayer(AbstractNoisyLayer):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float = 0.4, # 0.5,
    ):
        super().__init__(
            input_features=input_features,
            output_features=output_features,
            sigma=sigma
        )

        self.bound = input_features**(-0.5)
        self.parameter_initialization()
        self.sample_noise()

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_output + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def parameter_initialization(self) -> None:
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma * self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.sigma_weight.data.fill_(self.sigma * self.bound)

class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, noisy_net=True) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.noisy_net = noisy_net
        
        self.fc = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims) if not self.noisy_net else FactorisedNoisyLayer(*self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims) if not self.noisy_net else FactorisedNoisyLayer(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions) if not self.noisy_net else FactorisedNoisyLayer(self.fc2_dims, self.n_actions)
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
    
class DuelingDQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, val_fc1_dims, val_fc2_dims, 
                 adv_fc1_dims, adv_fc2_dims, n_actions, noisy_net=True) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.val_fc1_dims = val_fc1_dims
        self.val_fc2_dims = val_fc2_dims
        self.adv_fc1_dims = adv_fc1_dims
        self.adv_fc2_dims = adv_fc2_dims
        self.n_actions = n_actions
        self.noisy_net = noisy_net
        
        self.fc = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims) if not self.noisy_net else FactorisedNoisyLayer(*self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims) if not self.noisy_net else FactorisedNoisyLayer(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
        )
        
        self.val_fc = nn.Sequential(
            nn.Linear(self.fc2_dims, self.val_fc1_dims) if not self.noisy_net else FactorisedNoisyLayer(self.fc2_dims, self.val_fc1_dims),
            nn.ReLU(),
            nn.Linear(self.val_fc1_dims, self.val_fc2_dims) if not self.noisy_net else FactorisedNoisyLayer(self.val_fc1_dims, self.val_fc2_dims),
            nn.ReLU(),
            nn.Linear(self.val_fc2_dims, 1) if not self.noisy_net else FactorisedNoisyLayer(self.val_fc2_dims, 1)
        )
        
        self.adv_fc = nn.Sequential(
            nn.Linear(self.fc2_dims, self.adv_fc1_dims) if not self.noisy_net else FactorisedNoisyLayer(self.fc2_dims, self.adv_fc1_dims),
            nn.ReLU(),
            nn.Linear(self.adv_fc1_dims, self.adv_fc2_dims) if not self.noisy_net else FactorisedNoisyLayer(self.adv_fc1_dims, self.adv_fc2_dims),
            nn.ReLU(),
            nn.Linear(self.adv_fc2_dims, n_actions) if not self.noisy_net else FactorisedNoisyLayer(self.adv_fc2_dims, n_actions)
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
                 dueling_dqn=True, noisy_net=True, prm=True, learn_per_target_net_update=50, seed=None) -> None:
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
        self.noisy_net = noisy_net
        self.target_net_update_per_step = learn_per_target_net_update
        self.learn_counter = 0
        self.seed=seed
        self.np_random = np.random.default_rng(seed=seed)
        self.prm = prm
        if self.prm:
            self.prm_alpha = 0.7
            self.prm_offset = 1e-4
            self.prm_beta = 0.5
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
                    
        self.Q_eval = DQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, n_actions=n_actions, noisy_net=self.noisy_net) if not self.dueling_dqn \
                      else DuelingDQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, val_fc1_dims=256, val_fc2_dims=256, adv_fc1_dims=256, 
                          adv_fc2_dims=256, n_actions=n_actions, noisy_net=self.noisy_net)
        self.Q_target = DQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, n_actions=n_actions, noisy_net=self.noisy_net) if not self.dueling_dqn \
                      else DuelingDQN(lr=self.lr, input_dims=self.input_dims, fc1_dims=256, 
                          fc2_dims=256, val_fc1_dims=256, val_fc2_dims=256, adv_fc1_dims=256, 
                          adv_fc2_dims=256, n_actions=n_actions, noisy_net=self.noisy_net)
        self._update_target_net()
        
        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        if self.prm:
            self.priority_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_counter % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
    
        if self.prm:
            # priority calculation
            with torch.no_grad():
                q_eval = self.Q_eval.forward(torch.as_tensor(state, device=self.Q_eval.device))[0][action].cpu().item()
                if done:
                    q_target = reward
                else:
                    q_target_next = self.Q_target.forward(torch.as_tensor(new_state, device=self.Q_target.device)).max(1)[0]
                    q_target = reward + self.gamma * q_target_next.cpu().item()
                error = abs(q_eval - q_target)
            self.priority_memory[idx] = (error + self.prm_offset) ** self.prm_alpha
        
        self.mem_counter += 1
        
    def choose_action(self, obs):
        if self.prediction or self.noisy_net or self.np_random.random() > self.epsilon:    # no random selection when evaluation
            state = torch.tensor(np.array(obs)).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state) if self.double_dqn else self.Q_target.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = self.np_random.choice(self.action_space)
            
        return action
    
    def learn(self, episode):
        # record total learn step number
        self.learn_counter += 1
        
        if self.mem_counter < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_counter, self.mem_size)
        if self.prm:
            priority_sum = np.sum(self.priority_memory)
            batch = self.np_random.choice(max_mem, p=self.priority_memory[:max_mem] / priority_sum, size=self.batch_size, replace=False)
        else:
            batch = self.np_random.choice(max_mem, size=self.batch_size, replace=False)
        
        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = torch.as_tensor(self.state_memory[batch], device=self.Q_eval.device)
        new_state_batch = torch.as_tensor(self.new_state_memory[batch], device=self.Q_eval.device)
        reward_batch = torch.as_tensor(self.reward_memory[batch], device=self.Q_eval.device)
        terminal_batch = torch.as_tensor(self.terminal_memory[batch], device=self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        if self.prm:
            priority_batch = self.priority_memory[batch]
            weight_batch = (max_mem * priority_batch / priority_sum) ** (-self.prm_beta)
            weight_batch /= weight_batch.max()  # normalize weigths
        
        q_eval = self.Q_eval.forward(state_batch)[batch_idx, action_batch]
        q_next = self.Q_target.forward(new_state_batch) if self.double_dqn else self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        
        if self.prm:
            loss = (torch.as_tensor(weight_batch, device=self.Q_eval.device) * F.huber_loss(q_eval, q_target).to(device=self.Q_eval.device)).mean()
        else:
            loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.loss = loss.item()
        self.Q_eval.optimizer.step()
        
        if not self.noisy_net:
            # self.epsilon = max(self.eps_min, self.eps_max * 1 / (1 + episode))
            self.epsilon = max(self.eps_max - episode * self.eps_dec, self.eps_min)
                      
        if self.learn_counter % self.target_net_update_per_step == 0:
            self._update_target_net()
                        
    def save(self, filename):
        torch.save(self.Q_eval.state_dict(), filename)
        
    def load(self, filename):
        self.Q_eval.load_state_dict(torch.load(filename))
        
    def eval(self):
        self.prediction = True
        
    def train(self):
        self.prediction = False