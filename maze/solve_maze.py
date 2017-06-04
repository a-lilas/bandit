# coding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from maze import *


def __main():
    env = Maze()
    # env.drawMaze()
    agent = QAgent(state_size=env.size, init=30, start=env.start, end=env.goal)


    while True:
        while True:
            # 行動A_tの決定(行動空間外の行動が選択された場合はやり直し)
            pred_action = agent.policy()
            # 行動した先が壁かどうかチェック
            # 現在位置+移動量を計算 計算する場合はnp.arrayに，インデックスとして扱う場合はタプルにする
            if env.maze[tuple(np.array(agent.now)+agent.actions[pred_action])] == 0:
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
            if env.maze[tuple(np.array(agent.now)+agent.actions[next_action])] == 0:
                break

        # 現在位置
        print(agent.now, env.maze[tuple(agent.now)], next_action)

        # 行動A_t+1を実行に移す(ループ先頭に戻る)
        pred_action = next_action


if __name__ == '__main__':
    __main()
