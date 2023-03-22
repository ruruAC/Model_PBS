# import os
# from matplotlib.style import available 
import numpy as np
import torch
# import gym
from algo import Agent
import argparse
from PBS import workshop_env
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v1', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=10000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=10, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--mini_batch_size', default=400, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=5, type=int, help='update number')
    parser.add_argument('--actor_lr', default=3e-4, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=3e-4, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')
    # parser.add_argument('-batch_size', default=400, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim')
    parser.add_argument('--device', default='cuda:0', type=str, help="cpu or cuda")
    args = parser.parse_args()
    return args

def env_agent_config(cfg, seed=1):
    env = workshop_env()
    n_states = env.state_space
    n_actions = env.action_space
    agent = Agent(n_states, n_actions, cfg)
    if seed != 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    return env, agent


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    env_eval = workshop_env()
    max_score = -500
    result_training_reward = []
    result_testing_reward = []
    result_score = []

    for i_ep in range(cfg.train_eps):
        state, available_action = env.reset()
        # available_action = np.array([[1,1], [1,1]])
        done = False
        ep_reward = 0
        steps = 0
        while (not done and steps <= 8000):
            action, prob, val = agent.choose_action(state,available_action)
            saved_action = [i for i in action]
            if available_action.sum(axis=1)[0] == 0:
                action[0] = None
            if available_action.sum(axis=1)[1] == 0:
                action[1] = None
            next_state, reward, available_action, done = env.step(action)
            steps += 1
            ep_reward += reward
            agent.memory.push(state, saved_action, prob, val, reward, done, available_action)
                # print("learn_step: ", steps)
                # print("ep_reward: ", ep_reward)
            state = next_state
        agent.learn()
        result_training_reward.append(ep_reward)

        if (i_ep + 1) % 1 == 0:
            print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")

        if (i_ep + 1) % cfg.test_eps == 0:
            agent.actor.eval()
            state, available_action = env_eval.reset(save=True)
            done = False
            ep_reward = 0
            kk = 0
            while (not done and kk <= 8000):
                action = agent.test_action(state,available_action)
                if available_action.sum(axis=1)[0] == 0:
                    action[0] = None
                if available_action.sum(axis=1)[1] == 0:
                    action[1] = None
                next_state, reward, available_action, done = env_eval.step(action, save=True)
                ep_reward += reward
                state = next_state
                kk +=1
                if done:
                    score, score1, score2, score3, score4 = env_eval.get_output_score()
                    if score > max_score:
                        df=pd.DataFrame(env_eval.cars_excel, index=[i+1 for i in range(env_eval.cars_excel.shape[0])])
                        df.to_csv("./results/resultour.csv",index=True)

                        max_score = score
                        print("save results")
                    result_score.append([score, score1, score2, score3, score4])

                    print("分数为：", score)

            result_testing_reward.append(ep_reward)

            print(f"测试.............., 回合：{i_ep + 1}，奖励：{ep_reward:.2f}")
            agent.actor.train()

        if (i_ep+1) % 10 ==0:
            training_reward = np.array(result_training_reward)
            testing_reward = np.array(result_testing_reward)
            score = np.array(result_score)

            df1=pd.DataFrame(training_reward)
            df2=pd.DataFrame(testing_reward)
            df3 = pd.DataFrame(score)
            df1.to_csv("./results/reward_training_our.csv",index=False)
            df2.to_csv("./results/reward_testing_our.csv",index=False)
            df3.to_csv("./results/score_our.csv",index=False)
            print("save rewards")

    print('完成训练！')

if __name__ == '__main__':
    cfg = get_args()
    env, agent = env_agent_config(cfg, seed=5)
    train(cfg, env, agent)