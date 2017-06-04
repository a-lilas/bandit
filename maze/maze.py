# coding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Maze:
    def __init__(self):
        # 1: 壁, 0:通路
        self.maze = [
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                     [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                     [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                     [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                     [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
                     [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1],
                     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    ]

        self.size = np.shape(self.maze)

    def drawMaze(self):
        plt.imshow(self.maze,
                   cmap=matplotlib.cm.binary,
                   interpolation="nearest"
                   )
        plt.axis("off")
        plt.show()


class QAgent:
    def __init__(self, state, actions, init, start, end):
        self.state = state
        self.actions = actions
        # init: Qの初期値
        self.Q = np.zeros((len(state), len(actions))) + init
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
        # ε-greedy方策を採用
        # ε = 0.1
        if pr(0.1):
            return 
        else:
            return 

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


if __name__ == '__main__':
    maze = Maze()
    maze.drawMaze()