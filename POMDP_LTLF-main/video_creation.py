import os
from copy import deepcopy
import importlib
import numpy as np
if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties


def plot(shape, pos, action, L, lcmap, reward, save=None):
    """Plot 1 frame between episode[t] and episode[t+1]
    """
    # axis grid initialize
    size = 5
    colormap = plt.cm.RdBu
    color = 'black'
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
    i, j = pos[0], pos[1]
    if action == 'U':
        plt.arrow(j, i, 0, -0.2, head_width=.2, head_length=.15, color=color)
    elif action == 'D':
        plt.arrow(j, i - .3, 0, 0.2, head_width=.2, head_length=.15, color=color)
    elif action == 'R':
        plt.arrow(j - .15, i - 0.15, 0.2, 0, head_width=.2, head_length=.15, color=color)
    elif action == 'L':
        plt.arrow(j + .15, i - 0.15, -0.2, 0, head_width=.2, head_length=.15, color=color)

    # inserting labels
    for i in range(n_rows):
        for j in range(n_cols):
            if L[(i, j)] in lcmap:  # 0.24 before
                circle = plt.Circle((j, i + 0.24), 0.2, color=lcmap[L[(i, j)]], alpha=0.5)
                plt.gcf().gca().add_artist(circle)
            if L[(i, j)]:
                plt.text(j, i + 0.4, L[(i, j)][0], horizontalalignment='center', color="black", fontproperties=f,
                         fontname=fontname, fontsize=fontsize + 5)
    plt.savefig(save, bbox_inches='tight')


def multi_plot(shape, episode_sa, episode_L, lcmap, reward, count, animation):
    pad = 5
    if not os.path.exists(animation):
        os.makedirs(animation)
    # create frame figures
    T = count
    for t in range(T):
        print(episode_sa[t][0], episode_sa[t][1], episode_L[t])
        plot(shape, episode_sa[t][0], episode_sa[t][1], episode_L[t], lcmap, reward,
             save=animation + os.sep + str(t).zfill(pad) + '.png')
        plt.close()
    # create video
    os.system(
        '/Users/dongmingshen/ffmpeg -r 2 -i ' + animation + os.sep + '%0' + str(pad) + 'd.png ' + animation + '.mp4')


def main():
    file_name = "result/p_sequence_gridgoal/p_sequence_gridgoal.txt"
    shape = (4, 4)

    # not reward, just to fill-up things
    reward = np.zeros(shape)
    # reward[1][2] = 2
    # reward[2][0] = 2
    # reward[3][0] = 2
    # reward[4][4] = 2

    # pick a k for the demo, Lines[start] corresponds to the start line of the picked k
    k = 10
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
    line = Lines[start + 1]

    # start the episode, 20 rounds to save time
    L = {}
    "vid_p_sequence_dfagoal2"
    # L[(2, 2)] = ('c',)
    # L[(0, 3)] = ('a',)
    # L[(3, 0)] = ('b',)
    # lcmap = {
    #     ('c',): 'red',
    #     ('a',): 'blue',
    #     ('b',): 'green',
    # }
    "sequence_dfagoal / sequence_gridgoal"
    # L[(3, 3)] = ('G',)
    # L[(0, 3)] = ('a',)
    # L[(3, 0)] = ('b',)
    "vid_p_sequence_gridgoal2"
    # L[(3, 3)] = ('G',)
    # L[(2, 2)] = ('c',)
    # L[(0, 3)] = ('a',)
    # L[(3, 0)] = ('b',)
    "vid_p_sequence_gridgoal2"
    lcmap = {
        ('c',): 'red',
        ('a',): 'blue',
        ('b',): 'green',
    }
    episode_sa = [None] * 170  # states & actions
    episode_L = [None] * 170  # labels
    counter = 0
    for i in range(len(episode_sa) + 1):  # 1 more than the episode
        if i == 0:
            continue  # skip the header line
        line = Lines[start + i]  # the current line working on (*2 to skip belief)
        # state
        s = eval(line[line.index("state: ") + 6:line.index(", observation:")])[0][0]
        ag_state = s
        # stop
        if ag_state[0] > shape[0] - 1 or ag_state[1] > shape[1] - 1:
            break
        counter += 1
        # labels, to show goal
        for x in range(shape[0]):
            for y in range(shape[1]):
                L[(x, y)] = ()
        "sequence_dfagoal / sequence_gridgoal"
        # L[(3, 3)] = ('G',)
        # L[(0, 3)] = ('a',)
        # L[(3, 0)] = ('b',)
        "vid_p_sequence_dfagoal2"
        # L[(2, 2)] = ('c',)
        # L[(0, 3)] = ('a',)
        # L[(3, 0)] = ('b',)
        "vid_p_sequence_gridgoal2"
        # L[(3, 3)] = ('G',)
        # L[(2, 2)] = ('c',)
        # L[(0, 3)] = ('a',)
        # L[(3, 0)] = ('b',)
        episode_L[i - 1] = deepcopy(L)
        # actions
        ag_action = line[line.index("action: "):][8:9]
        # create state and action
        episode_sa[i - 1] = (ag_state, ag_action)
    multi_plot(shape, episode_sa, episode_L, lcmap, reward, counter, './video_new/sequence_gridgoal')
    exit(0)


if __name__ == '__main__':
    main()
