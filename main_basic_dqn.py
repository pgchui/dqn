import gym
from basic_dqn import Agent
import numpy as np

from torch.utils.tensorboard import SummaryWriter

TRAIN = True

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode='human' if not TRAIN else 'rgb_array')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, 
                  eps_min=0.01, input_dims=[8], lr=0.003)
    
    if TRAIN:
        scores, eps_history = [], []
        n_games = 500
        
        max_score = -np.inf
        
        writer = SummaryWriter('./tfb/basic_dqn')
        
        for i in range(n_games):
            score = 0
            terminated = truncated = False
            obs, _ = env.reset()
            # each episode
            while (not terminated) and (not truncated):
                action = agent.choose_action(obs)
                new_obs, reward, terminated, truncated, _ = env.step(action)
                score += reward
                agent.store_transition(obs, action, reward, new_obs, 
                                    terminated or truncated)
                agent.learn()
                obs = new_obs
            
            if score > max_score:
                max_score = score
                agent.save('lunarlander_basic_dqn_max_score.pth')
                print('Model saved -v-')
                
            scores.append(score)
            eps_history.append(agent.epsilon)
            
            avg_score = np.mean(scores[-100:])
            
            print('episode ', i, 'score %.2f' % score, 
                  'best score %.2f' % max_score,
                  'average score %.2f' % avg_score,
                  'epsilon %.2f' % agent.epsilon
                  )

            writer.add_scalar('Loss/train', agent.loss, i)
            writer.add_scalar('Score/train', score, i)
            writer.add_scalar('Score_ave100/train', avg_score, i)
            writer.add_scalar('Epsilon/train', agent.epsilon, i)
        
        env.close()
    else:
        terminated = truncated = False
        obs, _ = env.reset()
        agent.load('lunarlander_basic_dqn_max_score.pth')
        agent.prediction = True
        # each episode
        while (not terminated) and (not truncated):
            action = agent.choose_action(obs)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            
        env.close()