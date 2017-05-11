# coding:utf-8
import pprint
import numpy as np
import random as rd


class Environment:
    def __init__(self, state, transitions, reward, start, end):
        self.state = state
        self.transitions = transitions
        self.reward = reward

    def outputInfo(self):
        print('状態遷移')
        pprint.pprint(self.transitions)
        print('報酬')
        pprint.pprint(self.reward)
 
    def actionResult(self, now_state, action):
        return (self.transitions[now_state][action], self.reward[now_state][action])


class QAgent:
    def __init__(self, state, actions, init, start, end):
        self.state = state
        self.actions = actions
        # init: Qの初期値
        self.Q = np.zeros((len(state), len(actions))) + init
        # 状態名・行動名と、インデックスの対応（要改良）
        self.qtable = {
                        's1': 0,
                        's2': 1,
                        's3': 2,
                        's4': 3,
                        'a1': 0,
                        'a2': 1
                      }
        # 初期状態
        self.start = start
        # now: 現在位置
        self.now = start
        # 終了状態
        self.end = end
        # 報酬和
        self.reward = 0

    def resetParameter(self):
        self.now = self.start
        self.reward = 0

    def policy(self):
        # 方策
        if pr(0.5):
            return 'a1'
        else:
            return 'a2'

    def update(self, next_state, pred_action, reward, alpha, gamma):
        # Qの更新(Q-learning)
        s_t = self.qtable[self.now]
        s_t1 = self.qtable[next_state]
        a_t = self.qtable[pred_action]
        # Q(s_t+1, a_t+1)が最大値となるa_t+1
        a_t1 = np.argmax(self.Q[s_t1])
        # 次状態が終了状態なら、maxの項は0とする
        if next_state == self.end:
            self.Q[s_t, a_t] \
                    = (1-alpha)*self.Q[s_t, a_t] + alpha*(reward+0)
        else:
            self.Q[s_t, a_t] \
                    = (1-alpha)*self.Q[s_t, a_t] + alpha*(reward+gamma*self.Q[s_t1, a_t1])

        self.now = next_state
        self.reward += reward
        # print(self.Q[0, 0])

    def checkGoal(self):
        # 終了条件判定
        if self.now == self.end:
            return True


def pr(p):

    '''
    return True with probability p
    return False with probability 1-p
    '''

    if rd.random() < p:
        return True
    else:
        return False

