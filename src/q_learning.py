# -*- coding:utf-8 -*-

import numpy as np
import random


def epsilon_greedy(q_table, state, epsilon):
    # 生成一个概率
    p = random.uniform()
    if p < epsilon:
        # 如果概率小于epsilopn,则在动作空间中随机选择一个动作
        return random.choice(ACTION_SPACE)
    else:
        # 否则根据贪心策略，选择使得Q值最大的动作
        return np.argmax(q_table[state])


def q_learning(num_iters ):
    # 建立并初始化Q值表(这里采用全0初始化)
    q_table = np.zeros((num_states, num_actions))

    # 迭代
    for _ in xrange(num_iters):
        # 初始化状态
        state = env.reset()
        done = False

        # 直到一局游戏结束
        while not done:
            # 利用Q值表选择要执行的动作
            action = epsilon_greedy(q_table, state, epsilon=0.2)
            # 执行动作
            next_state, reward, done = env.step(action)
            # 更新Q值
            td_target = reward + gamma * np.max(q_table[next_state])
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            # 进入下一个状态
            state = next_state