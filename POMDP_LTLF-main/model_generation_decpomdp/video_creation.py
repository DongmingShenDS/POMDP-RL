import os
from copy import deepcopy
import importlib
import numpy as np
if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties


def plot(shape, states, actions, L, lcmap, reward, goal, save=None):
    """Plot 1 frame between episode[t] and episode[t+1]
    """
    # axis grid initialize
    size = 5
    colormap = plt.cm.RdBu
    color = ['blue', 'red']
    f = FontProperties(weight='bold')
    fontname = 'Times New Roman'
    fontsize = 20
    n_rows, n_cols = shape
    fig = plt.figure(figsize=(size, size))
    plt.rc('text', usetex=True)
    value = np.copy(reward)
    threshold = np.nanmax(np.abs(value)) * 2
    threshold = 1 if threshold == 0 else threshold
    plt.imshow(value, interpolation='nearest', cmap=colormap, vmax=threshold, vmin=-threshold)
    ax = fig.axes[0]
    ax.set_xticks(np.arange(0, n_cols, 1))
    ax.set_yticks(np.arange(0, n_rows, 1))
    ax.set_xticklabels(np.arange(n_cols), fontsize=fontsize)
    ax.set_yticklabels(np.arange(n_rows), fontsize=fontsize)
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.xaxis.tick_top()
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1, alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(bottom='off', left='off')

    # action arrow
    ag1_s, ag2_s = states[0], states[1]
    ag1_a, ag2_a = actions[0], actions[1]
    ag1_i, ag1_j = ag1_s[0], ag1_s[1]
    ag2_i, ag2_j = ag2_s[0], ag2_s[1]
    # agent1
    if ag1_a == 'U':
        plt.arrow(ag1_j, ag1_i, 0, -0.2, head_width=.2, head_length=.15, color=color[0])
    elif ag1_a == 'D':
        plt.arrow(ag1_j, ag1_i - .3, 0, 0.2, head_width=.2, head_length=.15, color=color[0])
    elif ag1_a == 'R':
        plt.arrow(ag1_j - .15, ag1_i - 0.15, 0.2, 0, head_width=.2, head_length=.15, color=color[0])
    elif ag1_a == 'L':
        plt.arrow(ag1_j + .15, ag1_i - 0.15, -0.2, 0, head_width=.2, head_length=.15, color=color[0])
    # agent2
    if ag2_a == 'U':
        plt.arrow(ag2_j, ag2_i, 0, -0.2, head_width=.2, head_length=.15, color=color[1])
    elif ag2_a == 'D':
        plt.arrow(ag2_j, ag2_i - .3, 0, 0.2, head_width=.2, head_length=.15, color=color[1])
    elif ag2_a == 'R':
        plt.arrow(ag2_j - .15, ag2_i - 0.15, 0.2, 0, head_width=.2, head_length=.15, color=color[1])
    elif ag2_a == 'L':
        plt.arrow(ag2_j + .15, ag2_i - 0.15, -0.2, 0, head_width=.2, head_length=.15, color=color[1])

    # inserting labels
    for i in range(n_rows):
        for j in range(n_cols):
            if L[(i, j)] in lcmap:  # 0.24 before
                if L[(i, j)][0] == '1':
                    circle = plt.Circle((j, i + 0.24), 0.2, color=lcmap[L[(i, j)]], alpha=0.5)
                if L[(i, j)][0] == '2':
                    circle = plt.Circle((j, i + 0.24), 0.2, color=lcmap[L[(i, j)]], alpha=0.5)
                plt.gcf().gca().add_artist(circle)
            if L[(i, j)]:
                plt.text(j, i + 0.4, L[(i, j)][0], horizontalalignment='center', color="black", fontproperties=f,
                         fontname=fontname, fontsize=fontsize + 5)
    plt.savefig(save, bbox_inches='tight')


def multi_plot(shape, episode_sa, episode_L, lcmap, reward, goal, animation):
    pad = 5
    if not os.path.exists(animation):
        os.makedirs(animation)
    # create frame figures
    T = len(episode_sa)
    for t in range(T):
        plot(shape, episode_sa[t][0], episode_sa[t][1], episode_L[t], lcmap, reward, goal,
             save=animation + os.sep + str(t).zfill(pad) + '.png')
        plt.close()
    # create video
    os.system(
        '/Users/dongmingshen/ffmpeg -r 2 -i ' + animation + os.sep + '%0' + str(pad) + 'd.png ' + animation + '.mp4')


def main():
    file_name = "results_mid/order/ma_grid14x14_2order1.txt"
    shape = (14, 14)

    # not reward, just to fill-up things
    reward = np.zeros(shape)
    reward[0][0] = 2
    reward[0][1] = 2
    reward[0][2] = 2
    reward[0][3] = 2
    reward[0][4] = 2
    reward[0][5] = 2

    reward[1][0] = 2
    reward[1][1] = 2
    reward[1][2] = 2
    reward[1][3] = 2
    reward[1][4] = 2
    reward[1][5] = 2

    reward[0][7] = 2
    reward[0][8] = 2
    reward[0][9] = 2
    reward[0][10] = 2
    reward[0][11] = 2

    reward[1][7] = 2
    reward[1][8] = 2
    reward[1][9] = 2
    reward[1][10] = 2
    reward[1][11] = 2

    reward[3][0] = 2
    reward[3][1] = 2
    reward[3][2] = 2
    reward[3][3] = 2
    reward[3][4] = 2
    reward[3][5] = 2
    reward[3][6] = 2
    reward[3][7] = 2
    reward[3][8] = 2

    reward[4][0] = 2
    reward[4][1] = 2
    reward[4][2] = 2
    reward[4][3] = 2
    reward[4][4] = 2
    reward[4][5] = 2
    reward[4][6] = 2
    reward[4][7] = 2
    reward[4][8] = 2

    reward[3][10] = 2
    reward[3][11] = 2
    reward[3][12] = 2
    reward[3][13] = 2
    reward[1][13] = 2
    reward[2][13] = 2

    # pick a k for the demo, Lines[start] corresponds to the start line of the picked k
    k = 40
    line_mark = "==========[Running Simulation at k={},".format(k)
    len_mark = len(line_mark)
    with open(file_name, 'r') as file:
        Lines = [line.rstrip() for line in file]  # all lines from the file in a list
        length = len(Lines)
    start = -1
    for i in range(length):
        if Lines[i][0:len_mark] == line_mark:
            start = i
            break
    if start == -1:
        print("wrong k, not in range")
        return
    print(file_name)
    print(Lines[start])

    # set up GOAL for agents, assign to rewards
    line = Lines[start + 1 * 2]
    ag1_g = line[line.index("S:"):][23:27]
    ag2_g = line[line.index("S:"):][31:35]
    ag1_goal = eval(ag1_g)
    ag2_goal = eval(ag2_g)
    goal = np.zeros(shape)
    reward[ag1_goal[0]][ag1_goal[1]] = 0
    reward[ag2_goal[0]][ag2_goal[1]] = 0

    # start the episode, 20 rounds to save time
    L = {}
    lcmap = {
        ('1',): 'blue',
        ('2',): 'red',
    }
    episode_sa = [None] * 300  # states & actions
    episode_L = [None] * 300  # labels
    for i in range(len(episode_sa) + 1):  # 1 more than the episode
        if i == 0:
            continue  # skip the header line
        line = Lines[start + i * 2]  # the current line working on (*2 to skip belief)
        # state
        ag1_l = line[line.index("S:"):][6:10]
        ag2_l = line[line.index("S:"):][14:18]
        ag1_state = eval(ag1_l)
        ag2_state = eval(ag2_l)
        # goal
        ag1_g = line[line.index("S:"):][23:27]
        ag2_g = line[line.index("S:"):][31:35]
        goals = eval(ag1_g), eval(ag2_g)
        # labels, to show goal
        for x in range(shape[0]):
            for y in range(shape[1]):
                L[(x, y)] = ()
        L[(eval(ag1_g)[0], eval(ag1_g)[1])] = ('1',)
        L[(eval(ag2_g)[0], eval(ag2_g)[1])] = ('2',)
        episode_L[i - 1] = deepcopy(L)
        # actions
        a = line[line.index("A:") + 2: line.index(", R:")]  # A:...
        actions = eval(a)
        pers = dict(actions[0]), dict(actions[1])  # prescription for ag1 & ag2
        ag1_action = pers[0][goals[0]]
        ag2_action = pers[1][goals[1]]
        # create states and actions (for both agents)
        states = ag1_state, ag2_state
        actions = ag1_action, ag2_action
        episode_sa[i - 1] = (states, actions)
    multi_plot(shape, episode_sa, episode_L, lcmap, reward, goal, './video_new/ma_grid14x5_2order1')
    exit(0)


if __name__ == '__main__':
    main()
