import numpy as np
from datetime import datetime
from tqdm import tqdm
import os

import gym
from dqn import Agent

from torch.utils.tensorboard import SummaryWriter

TRAIN = False
GUI = True and (not TRAIN)
dqn_type = 'dueling_dqn_huberloss'
game = 'LunarLander-v2'
success_score = 200
consecutive_success_break_count = 30

if __name__ == '__main__':
    env = gym.make(f'{game}', render_mode='human' if GUI else 'rgb_array')
    agent = Agent(gamma=0.99, batch_size=64, n_actions=env.action_space.n, 
                  eps_min=0.01, input_dims=env.observation_space.shape, lr=5e-4, 
                  double_dqn=True, dueling_dqn=True, learn_per_target_net_update=50)
    
    if TRAIN:
        time = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoints_dir = f'./checkpoints/{game}/{time}_{dqn_type}'
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        scores = []
        n_games = 5000
        
        writer = SummaryWriter(f'./tfb/{game}/{time}_{dqn_type}')
        
        step_counter = 0
        
        consecutive_success = 0
        
        for i in range(n_games):
            score = 0
            terminated = truncated = False
            obs, _ = env.reset()
            # each episode
            while (not terminated) and (not truncated):
                step_counter += 1
                
                action = agent.choose_action(obs)
                new_obs, reward, terminated, truncated, _ = env.step(action)
                score += reward
                agent.store_transition(obs, action, reward, new_obs, 
                                    terminated or truncated)
                agent.learn(i)
                obs = new_obs
                
                if step_counter % 5000 == 0:
                    agent.save(f'{checkpoints_dir}/{game}_{dqn_type}_{step_counter}.pth')
                
            scores.append(score)
            
            avg_score_100 = np.mean(scores[-100:])
            avg_score_50 = np.mean(scores[-50:])
            avg_score_10 = np.mean(scores[-10:])
            
            print('episode ', i, 'score %.2f' % score,
                  'average score 100 %.2f' % avg_score_100,
                  'average score 50 %.2f' % avg_score_50,
                  'average score 10 %.2f' % avg_score_10,
                  'epsilon %.2f' % agent.epsilon
                  )

            writer.add_scalar('Loss/train', agent.loss, i)
            writer.add_scalar('Score/train', score, i)
            writer.add_scalar('Score_ave100/train', avg_score_100, i)
            writer.add_scalar('Score_ave50/train', avg_score_50, i)
            writer.add_scalar('Score_ave10/train', avg_score_10, i)
            writer.add_scalar('Epsilon/train', agent.epsilon, i)
            
            # if avg_score_100 > success_score:
            #     agent.save(f'{checkpoints_dir}/{game}_{dqn_type}_success.pth')
            #     break
            
            if i >= 100 and avg_score_100 > success_score:
                consecutive_success += 1
            else:
                consecutive_success = 0
                
            if consecutive_success >= consecutive_success_break_count:
                agent.save(f'{checkpoints_dir}/{game}_{dqn_type}_success.pth')
                break
        
        env.close()
    else:
        test_scores = []
        agent.load('checkpoints/LunarLander-v2/20230319162008_dueling_dqn_huberloss/LunarLander-v2_dueling_dqn_huberloss_success.pth')
        agent.prediction = True
        
        for _ in tqdm(range(100)):
            terminated = truncated = False
            obs, _ = env.reset()
            # each episode
            score = 0
            while (not terminated) and (not truncated):
                action = agent.choose_action(obs)
                new_obs, reward, terminated, truncated, _ = env.step(action)
                env.render()
                
                obs = new_obs
                
                score += reward
            test_scores.append(score)
                
        print(f'reward = {np.mean(test_scores)}')
        env.close()