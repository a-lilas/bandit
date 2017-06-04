# coding:utf-8
import pprint
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sarsa import *
from q_learning import QAgent


def __main():
    state = ['s1', 's2', 's3', 's4']
    actions = ['a1', 'a2']
    transitions = {
                    's1': {'a1': 's3', 'a2': 's2'},
                    's2': {'a1': 's1', 'a2': 's4'},
                    's3': {'a1': 's4', 'a2': 's1'},
                    's4': {'a1': 0, 'a2': 0}
                  }
    reward = {
                's1': {'a1': 0, 'a2': 1},
                's2': {'a1': -1, 'a2': 1},
                's3': {'a1': 5, 'a2': -100},
                's4': {'a1': 0, 'a2': 0}
             }
    reward2 = {
                's1': {'a1': -100, 'a2': 0},
                's2': {'a1': 0, 'a2': 100},
                's3': {'a1': 300, 'a2': 0},
                's4': {'a1': 0, 'a2': 0}
             }
    start = 's1'
    end = 's4'

    # 環境の構築
    env = Environment(state, transitions, reward, start, end)
    # env.outputInfo()

    if input('sarsa:0, q-learning:1 [0/1]: ') == '0':
        # Sarsaエージェントの初期化
        agent = SarsaAgent(state, actions, 10, start, end)
        plt.title('<Sarsa> Action value function Q')
        fg = 0

    else:
        # Q-learningエージェントの初期化
        agent = QAgent(state, actions, 10, start, end)
        plt.title('<Q-learning> Action value function Q')
        fg = 1

    # Q値の履歴
    Q1 = []
    Q2 = []
    episode = []
    # 報酬の履歴
    reward_trace = []

    for i in range(5000):
        Q1.append(agent.Q[0, 0])
        Q2.append(agent.Q[0, 1])
        episode.append(i)

        while True:
            # 行動A_tの決定
            pred_action = agent.policy()
            # 行動後の状態S_t+1と報酬R_t+1が返る
            next_state, reward = env.actionResult(agent.now, pred_action)

            if fg == 0:
                # Sarsaはここで決定する
                # その次の行動A_t+1の予定を決定
                next_action = agent.policy()
                # エージェントにここまでの結果を渡す/Qテーブルの更新
                agent.update(next_state, pred_action, next_action, reward, alpha=0.01, gamma=0.8)
            else:
                # エージェントにここまでの結果を渡す/Qテーブルの更新
                agent.update(next_state, pred_action, reward, alpha=0.01, gamma=0.8)

            # 終了判定(True:ゴール)
            if agent.checkGoal():
                break

            if fg == 1:
                # Q-learningはここで決定する
                # その次の行動A_t+1を決定
                next_action = agent.policy()

            # 行動A_t+1を実行に移す(ループ先頭に戻る)
            pred_action = next_action

        reward_trace.append(agent.reward)

        # ゴールしたら、エージェントのリセットを行う
        agent.resetParameter()

    # グラフの描画処理
    plt.plot(episode, Q1, 'r', label='Q(s1,a1)')
    plt.plot(episode, Q2, 'b', label='Q(s1,a2)')
    plt.xlabel('episode')
    plt.ylabel('Action value function Q')
    plt.legend()
    plt.show()

#    plt.plot(episode, reward_trace)
#    plt.title('Reward')
#    plt.show()

if __name__ == '__main__':
    __main()