import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
from collections import Counter
from progressbar import ProgressBar
import seaborn as sns


def pr(p):

    '''
    return True with probability p
    return False with probability 1-p
    '''

    if rd.random() < p:
        return True
    else:
        return False


class Agent(object):
    def __init__(self, arm_K, greedy_K):
        self.history = np.array([])
        self.resulthistory = np.array([])
        self.greedy_K = greedy_K
        self.k = arm_K
        # 各腕の期待値
        self.expected = np.zeros(arm_K, dtype=float)
        # 各腕の報酬の信頼区間
        self.confidence = np.zeros(arm_K, dtype=float)
        self.pullnum = np.zeros(arm_K)

    def output(self):
        print(self.history)

    def policy_greedy(self):
        '''
        Greedy alogrithm
        input: expected value
        output: next arm
        '''

        # 初期探索
        # 引いた回数がK未満の腕を引く
        if len(np.where(self.pullnum < self.greedy_K)[0]) > 0:
            return np.where(self.pullnum < self.greedy_K)[0][0]

        nextarm = np.argmax(self.expected)
        return nextarm

    def policy_e_greedy(self, ep):
        '''
        epsilon-Greedy alogrithm
        input: expected value
        output: next arm
        '''

        # 初期探索
        # 引いた回数が1回未満の腕を引く
        if len(np.where(self.pullnum < 1)[0]) > 0:
            return np.where(self.pullnum < 1)[0][0]

        # 確率epでランダムに、1-epで平均が最大の腕を選ぶ
        if pr(ep):
            # 0以上k-1以下からランダムに1本選択
            return rd.randint(0, self.k-1)
        else:
            nextarm = np.argmax(self.expected)
            return nextarm

    def policy_ucb1(self):
        '''
        UCB1 alogrithm
        input: expected value
        output: next arm
        '''

        # 初期探索
        # 引いた回数が1回未満の腕を引く
        if len(np.where(self.pullnum < 1)[0]) > 0:
            return np.where(self.pullnum < 1)[0][0]

        nextarm = np.argmax(self.expected + self.confidence)
        return nextarm

    def addHistory(self, i):
        self.history = np.append(self.history, i)
        self.pullnum[i] += 1

        # 報酬の信頼区間の更新
        self.confidence[i] = np.sqrt((2*np.log(np.sum(self.pullnum))/self.pullnum[i]))

    def updateExpected(self, i, value, result):
        num = Counter(self.history)
        _sum = self.expected[i] * num[i]

        if result:
            self.expected[i] = (_sum+value) / (num[i]+1)
            self.resulthistory = np.append(self.resulthistory, 'o')
        else:
            self.expected[i] = _sum / (num[i]+1)
            self.resulthistory = np.append(self.resulthistory, 'x')


class Bandit(object):
    def __init__(self, arm_K):
        self.k = arm_K
        self.pr_i = []

        if self.k <= 2:
            print('Error: number of arm >= 3')
            exit()

        for i in range(0, self.k):
            if i < (self.k - 2):
                self.pr_i.append(0.3)
            elif i == (self.k - 2):
                self.pr_i.append(0.4)
            else:
                self.pr_i.append(0.6)

        self.pr_i = np.array(self.pr_i)

    def pullArm(self, i):
        '''
        Win: True
        Lose: False
        '''
        return pr(self.pr_i[i])


def pullCycle(_agent):
    nextarm = _agent.policy_ucb1()
    result = bandit.pullArm(nextarm)
    _agent.updateExpected(nextarm, 1, result)
    _agent.addHistory(nextarm)


if __name__ == '__main__':
    # 腕の本数
    arm = 15

    # 初期探索回数
    k = 3

    # エージェント数
    a_num = 1000
    agent = []

    # 繰り返し回数
    times = 3000

    # 腕が引かれた数の総合計
    agents_history = pd.DataFrame({})

    # 各腕が各timeで引かれた割合
    # 行：各time　列：各腕
    arm_analysis = np.zeros((times, arm), dtype=float)

    bandit = Bandit(arm)

    # プログレスバー
    p = ProgressBar()

    for a in range(a_num):
        p.update((a+1)/10)
        agent.append(Agent(arm, k))

        for i in range(times):
            pullCycle(agent[a])
        agents_history[a] = agent[a].history

    p.finish()

    for i in range(times):
        count = Counter(agents_history.T[i])
        for key, value in count.items():
            count[key] = value / a_num
            arm_analysis[i, key] = count[key]

    arm_analysis = pd.DataFrame(arm_analysis)

    arm_analysis.plot(kind='area', stacked=True)
    plt.xlabel('number of plays')
    plt.ylabel('Probability to pick each arm')
    plt.legend(['arm0', 'arm1', 'arm2', 'arm3', 'arm4'],
               bbox_to_anchor=(1.05, 1)
               )
    plt.subplots_adjust(right=0.8)
    plt.show()
