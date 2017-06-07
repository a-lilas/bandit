# coding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Maze:
    def __init__(self, maze, start, goal, reward):
        # 0: 壁, 1:通路
        self.maze = maze
        self.start = start
        self.goal = goal
        self.size = np.shape(self.maze)
        self.reward = reward

    def actionResult(self, now_state, action):
        # return (now_state + action, self.reward[now_state][action])
        reward = 0
        if tuple(np.array(now_state) + action) == self.goal:
            reward = self.reward
        return (np.array(now_state) + action, reward)

    def drawMaze(self, agent_now):
        # plt.imshow(self.maze,
        #            cmap=matplotlib.cm.hot,
        #            interpolation="nearest"
        #            )
        # エージェントの位置を色付け 深いコピーをすること
        agent_map = np.copy(self.maze)
        # float型に変換
        agent_map = agent_map.astype(np.float)
        agent_map[agent_now[0], agent_now[1]] = 0.3
        plt.imshow(agent_map,
                   cmap=matplotlib.cm.hot,
                   interpolation="nearest"
                   )
        plt.axis("off")
        plt.pause(0.0001)


class QAgent:
    def __init__(self, state_size, init, start, end):
        # state_size:状態数
        # 行動(上下左右へインデックスをずらす)
        self.actions = [[-1, 0],
                        [1, 0],
                        [0, -1],
                        [0, 1]]

        # init: Qの初期値
        self.Q = np.zeros((len(self.actions), state_size[0], state_size[1])) + init
        # 初期状態
        self.start = start
        # now: 現在位置
        self.now = start
        # 終了状態
        self.end = end
        # 報酬和
        self.reward = 0
        # 行動回数
        self.times = 0

    def resetParameter(self):
        self.now = self.start
        self.reward = 0
        self.times = 0

    def policy(self):
        # ε-greedy方策を採用
        # ε = 0.1
        eps = 0.1
        if pr(eps):
            return np.random.randint(0, 4)
        else:
            # 特定の状態における，Q[a,s1,s2]を最大にするaを返す
            # index: Q関数の最大値
            # 最大値が複数でた場合はランダムで返す
            a_max = np.where(self.Q[:, self.now[0], self.now[1]] == np.max(self.Q[:, self.now[0], self.now[1]]))
            return np.random.choice(a_max[0])

    def update(self, next_state, pred_action, reward, alpha, gamma):
        # Qの更新(Q-learning)
        s_t = self.now

        s_t1 = next_state
        a_t = pred_action
        # Q(s_t+1, a_t+1)が最大値となるa_t+1
        a_t1_tmp = np.where(self.Q[:, s_t1[0], s_t1[1]] == np.max(self.Q[:, s_t1[0], s_t1[1]]))
        # a_t1が可能な行動かどうかチェック
        a_t1 = np.random.choice(a_t1_tmp[0])

        # print(self.Q[:, s_t1[0], s_t1[1]])
        # 次状態が終了状態なら、maxの項は0とする
        if tuple(next_state) == self.end:
            self.Q[a_t, s_t[0], s_t[1]] \
                    = (1-alpha)*self.Q[a_t, s_t[0], s_t[1]] + alpha*(reward+0)
        else:
            self.Q[a_t, s_t[0], s_t[1]] \
                    = (1-alpha)*self.Q[a_t, s_t[0], s_t[1]] + alpha*(reward+gamma*self.Q[a_t1, s_t1[0], s_t1[1]])

        self.now = next_state
        self.reward += reward
        # print(self.Q[0, 0])

    def checkGoal(self):
        # 終了条件判定
        if tuple(self.now) == self.end:
            return True


def pr(p):

    '''
    return True with probability p
    return False with probability 1-p
    '''

    if np.random.random() < p:
        return True
    else:
        return False


if __name__ == '__main__':
    maze = np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                    [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    ])
    plt.imshow(maze,
               cmap=matplotlib.cm.hot,
               interpolation="nearest"
               )
    plt.axis('off')
    plt.show()
