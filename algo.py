import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np

class PPOmemory:
    def __init__(self, mini_batch_size, cfg):
        self.gamma = cfg.gamma  # 折扣因子
        self.gae_lambda = cfg.gae_lambda  # GAE参数
        self.states = []  # 状态
        self.actions = []  # 实际采取的动作
        self.probs = []  # 动作概率
        self.vals = []  # critic输出的状态值
        self.rewards = []  # 奖励
        self.dones = []  # 结束标志
        self.available_actions = []

        self.mini_batch_size = mini_batch_size  # minibatch的大小


    def gae_adv(self,):

        # 计算GAE
        vals_arr, rewards_arr, dones_arr = np.array(self.vals), np.array(self.rewards), np.array(self.dones)
        returns = list()
        G = 0

        for r in rewards_arr[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)

        self.advantage = np.array(returns)

        
    def sample(self):
        n_states = len(self.states)  # memory记录数量=20
        self.mini_batch_size = n_states
        batch_start = np.arange(0, n_states, self.mini_batch_size)  # 每个batch开始的位置[0,5,10,15]
        indices = np.arange(n_states, dtype=np.int64)  # 记录编号[0,1,2....19]
        np.random.shuffle(indices)  # 打乱编号顺序[3,1,9,11....18]
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]  # 生成4个minibatch，每个minibatch记录乱序且不重复

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), self.advantage, np.array(self.available_actions), mini_batches

    # 每一步都存储trace到memory
    def push(self, state, action, prob, val, reward, done, available_action):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        self.available_actions.append(available_action)

    # 固定步长更新完网络后清空memory
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Actor, self).__init__()
        self.base1 = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),)
        self.base2 = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),)

        self.out1 = nn.Linear(hidden_dim, n_actions)
        self.out2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state, available_action, test=False):
        mask = (available_action == 0)
        x1 = self.base1(state)
        x1 = self.out1(x1)
        x2 = self.base2(state)
        x2 = self.out2(x2)
        x1[mask[:,0]] = -1e10
        x2[mask[:,1]] = -1e10
        probs_in = F.softmax(x1, dim=1)
        probs_out = F.softmax(x2, dim=1)
        dist_in = Categorical(probs_in)
        dist_out = Categorical(probs_out)
        if test:
            return [probs_in, probs_out]
        else:
            return {'dist_in':dist_in, 'dist_out':dist_out}


class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state):
        value = self.critic(state)
        return value


class Agent:
    def __init__(self, n_states, n_actions, cfg):
        # 训练参数
        self.gamma = cfg.gamma  # 折扣因子
        self.n_epochs = cfg.n_epochs  # 每次更新重复次数
        self.gae_lambda = cfg.gae_lambda  # GAE参数
        self.policy_clip = cfg.policy_clip  # clip参数
        self.device = cfg.device  # 运行设备

        # AC网络及优化器
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.critic = Critic(n_states, cfg.hidden_dim).to(cfg.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # 经验池
        self.memory = PPOmemory(cfg.mini_batch_size,cfg)

    def choose_action(self, state, available_action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 数组变成张量
        available_action = torch.LongTensor(available_action).unsqueeze(0).to(self.device)  # 数组变成张量

        dist = self.actor(state, available_action)  # action分布
        dist_in, dist_out = dist['dist_in'], dist['dist_out']
        value = self.critic(state)  # state value值

        action_in = dist_in.sample()  # 随机选择action
        action_out = dist_out.sample()  # 随机选择action

        prob_in = torch.squeeze(dist_in.log_prob(action_in)).item()  # action对数概率
        prob_out = torch.squeeze(dist_out.log_prob(action_out)).item()  # action对数概率

        action_in = torch.squeeze(action_in).item()
        action_out = torch.squeeze(action_out).item()
        value = torch.squeeze(value).item()
        return [action_in, action_out], [prob_in, prob_out], value

    def test_action(self, state, available_action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 数组变成张量
        available_action = torch.LongTensor(available_action).unsqueeze(0).to(self.device)  # 数组变成张量

        probs = self.actor(state, available_action, test=True)  # action分布
        probs_in, probs_out = probs[0], probs[1]

        action_in = probs_in.argmax()  # 随机选择action
        action_out = probs_out.argmax()  # 随机选择action

        action_in = torch.squeeze(action_in).item()
        action_out = torch.squeeze(action_out).item()
        return [action_in, action_out]

    def learn(self):
        self.memory.gae_adv()
        for _ in range(self.n_epochs):
            # memory中的trace以及处理后的mini_batches，mini_batches只是trace索引而非真正的数据
            states_arr, actions_arr, old_probs_arr, vals_arr,\
                rewards_arr, dones_arr, Gt, available_actions, mini_batches = self.memory.sample()

            # mini batch 更新网络
            values = vals_arr[:]
            Gt = torch.FloatTensor(Gt).to(self.device)
            values = torch.FloatTensor(values).to(self.device)
            for batch in mini_batches:
                states = torch.FloatTensor(states_arr[batch]).to(self.device)
                old_probs = torch.FloatTensor(old_probs_arr[batch]).to(self.device)
                actions = torch.LongTensor(actions_arr[batch]).to(self.device)
                active_mask = torch.FloatTensor(available_actions[batch].sum(axis=2) > 0).to(self.device)
                value_mask = torch.FloatTensor(available_actions[batch].sum(axis=(1,2)) > 0).to(self.device)
                available_actions = torch.LongTensor(available_actions[batch]).to(self.device)  # 数组变成张量
                # mini batch 更新一次critic和actor的网络参数就会变化
                # 需要重新计算新的dist,values,probs得到ratio,即重要性采样中的新旧策略比值
                critic_value = torch.squeeze(self.critic(states))

                dist = self.actor(states, available_actions)
                dist_in, dist_out = dist['dist_in'], dist['dist_out']
                new_probs_in = dist_in.log_prob(actions[:, 0]) # in action
                new_probs_out = dist_out.log_prob(actions[:, 1]) # out action
                prob_ratio_in = new_probs_in.exp() / old_probs[:, 0].exp() # in ratio
                prob_ratio_out = new_probs_out.exp() / old_probs[:, 1].exp() # in ratio

                advantage = Gt[batch] - critic_value.detach()

                # in 
                weighted_probs_in = advantage * prob_ratio_in
                weighted_clip_probs_in = torch.clamp(prob_ratio_in, 1 - self.policy_clip,\
                     1 + self.policy_clip) * advantage
                actor_loss_in = -torch.sum(torch.min(weighted_probs_in, weighted_clip_probs_in) * active_mask[:,0]) / active_mask[:,0].sum()

                
                # in 
                weighted_probs_out = advantage * prob_ratio_out
                weighted_clip_probs_out = torch.clamp(prob_ratio_out, 1 - self.policy_clip,\
                     1 + self.policy_clip) * advantage
                actor_loss_our = -torch.sum(torch.min(weighted_probs_out, weighted_clip_probs_out) * active_mask[:,1]) / active_mask[:,1].sum()
                
                
                actor_loss = actor_loss_in + actor_loss_our
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()


                # critic loss
                critic_loss = torch.sum(((Gt[batch] - critic_value) ** 2) * value_mask) / value_mask.sum()
                critic_loss = critic_loss.mean()
                # total_loss

                # 更新
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
        self.memory.clear()
