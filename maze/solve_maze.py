# coding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from maze import *


def __main():
    # 0: 壁, 1:通路
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
    start = (1, 1)
    goal = (12, 10)

    env = Maze(maze=maze, start=start, goal=goal, reward=100)
    agent = QAgent(state_size=env.size, init=0, start=env.start, end=env.goal)

    for i in range(50):
        while True:
            while True:
                # 行動A_tの決定(行動空間外の行動が選択された場合はやり直し)
                pred_action = agent.policy()
                # 行動した先が壁かどうかチェック
                # 現在位置+移動量を計算 計算する場合はnp.arrayに，インデックスとして扱う場合はタプルにする
                if env.maze[tuple(np.array(agent.now)+agent.actions[pred_action])] == 1:
                    break

            # 行動後の状態S_t+1と報酬R_t+1が返る
            next_state, reward = env.actionResult(agent.now, agent.actions[pred_action])

            # エージェントにここまでの結果を渡す/Qテーブルの更新
            agent.update(next_state, pred_action, reward, alpha=0.01, gamma=0.8)

            # 終了判定(True:ゴール)
            if agent.checkGoal():
                break

            # Q-learningはここで決定する
            # その次の行動A_t+1を決定
            while True:
                # 行動A_tの決定(行動空間外の行動が選択された場合はやり直し)
                next_action = agent.policy()
                if env.maze[tuple(np.array(agent.now)+agent.actions[next_action])] == 1:
                    break

            # 行動回数
            agent.times += 1

            # 現在位置
            # print(agent.now, i)
            # if i > 20:
            #     # 思いので定期的にAlt+F4などで閉じること
            #     env.drawMaze(agent.now)

            # 行動A_t+1を実行に移す(ループ先頭に戻る)
            pred_action = next_action

        print(i, agent.times)
        agent.resetParameter()

if __name__ == '__main__':
    __main()
