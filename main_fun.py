import gym
from dub_dqn import Agent
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x, scores, epsilons):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2")

    # this is the epsilon plot
    ax.plot(x, epsilons)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Epsilon")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color='C4')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C3')


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dim=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ' + str(i) + 'score %.2f' % score +
              'average score %.2f' % avg_score +
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plot_curve(x, scores, eps_history)
