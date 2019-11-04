# %%
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding



seed = 1
np.random.seed(seed)


class shepherd():
    def __init__(self):
        self.coords = np.random.uniform(0, 0.1, 2).reshape(1, 2) * 150 / scale
        self.headings = [np.copy(self.coords)]
        self.history = []


class sheep():
    def __init__(self, id):
        self.id = id
        self.coords = np.random.uniform(0.75, 1, 2).reshape(1, 2) * 150 / scale
        self.headings = [np.copy(self.coords)]
        self.history = []

scale =1
NUM_SHEEP = 50
NUM_NEIGHBOURS = 30
agent_shepherd_repulsion_threshold = 65
inter_agent_repulsion_threshold = 2
relative_agent_repulsion = 2
relative_agent_attraction = 1.05
relative_shepherd_repulsion = 1
relative_inertia = 0.5
relative_angular_noise = 0.3
agent_displacement = 1
grazing_moving_prob = 0.05
shepherd_displacement = 1.5
agent_distances = np.zeros((NUM_SHEEP, NUM_SHEEP))


agents = [sheep(i) for i in range(NUM_SHEEP)]
driver = shepherd()


TIMESTEPS = 100


def get_gcm():
    coords = [i.coords for i in agents]
    return (sum(coords) / len(coords))


def n_nearest(i):
    return (agent_distances[i, :].argsort()[1:NUM_NEIGHBOURS + 1])


def update_distances():
    global agent_distances
    coords = np.asarray([i.coords for i in agents]).reshape(-1, 2)
    agent_distances = cdist(coords, coords)


def get_lcm(i):
    neighbours = n_nearest(i)
    coords = [agents[i].coords for i in neighbours]
    return (sum(coords) / len(coords))


def inter_agent_repulsion(i):
    neighbours = np.where(agent_distances[i] < inter_agent_repulsion_threshold)[0]
    force = np.zeros((1, 2))
    for j in neighbours:
        if (j == i):
            continue
        diff_vector = agents[i].coords - agents[j].coords
        force += normalize(diff_vector)
    return (force)


def get_error():
    return (np.zeros((1, 2)))


## Sheherd functions


def shepherd_agent_repulsion(i):
    if (np.linalg.norm(agents[i].coords - driver.coords) <= agent_shepherd_repulsion_threshold):
        return (agents[i].coords - driver.coords)
    return (np.zeros((1, 2)))


def shepherd_proximity_check():
    for i in range(NUM_SHEEP):
        if (np.linalg.norm(agents[i].coords - driver.coords) <= 3 * inter_agent_repulsion_threshold):
            return (True)
    return (False)


def furthest_from_gcm():
    gcm = get_gcm()
    distances = [np.linalg.norm(agents[i].coords - gcm) for i in range(NUM_SHEEP)]
    radius = inter_agent_repulsion_threshold * (NUM_SHEEP) ** (2 / 3)
    if (sum(np.asarray(distances) <= radius) == NUM_SHEEP):
        return (-1)
    return (np.asarray(distances).argmax())


def normalize(v):
    if isinstance(v, list):
        denom = (v[0] ** 2 + v[1] ** 2) ** 0.5
        if (denom == 0):
            return (v)
        v = [v[0] / denom, v[1] / denom]
        return (v)
    denom = (v[0, 0] ** 2 + v[0, 1] ** 2) ** 0.5
    if (denom == 0):
        return (v)
    return (v / denom)


def visualize():
    print("SHEPHERD COORDS", driver.coords)
    # plt.xlim([-100, 100])
    # plt.ylim([-100, 100])
    df1 = pd.DataFrame(np.asarray(driver.coords).reshape(-1, 2))
    sns.scatterplot(x=df1[0], y=df1[1])
    plt.show()
    # plt.xlim([-100, 100])
    # plt.ylim([-100, 100])
    df2 = pd.DataFrame(np.asarray([i.coords for i in agents]).reshape(-1, 2))
    sns.scatterplot(x=df2[0], y=df2[1])
    plt.show()


def visualize_gcm():
    arr = pd.DataFrame(np.asarray([get_gcm(), driver.coords]).reshape(-1, 2))
    sns.scatterplot(x=arr[0], y=arr[1])
    plt.show()


def plot_individual_trajectories():
    # plt.xlim([-200, 200])
    # plt.ylim([-200, 200])
    # df1 = pd.DataFrame(np.asarray(driver.history).reshape(-1,2))
    # sns.scatterplot(x=df1[0],y=df1[1])
    # df1
    for s in range(10):
        # plt.xlim([-200, 200])
        # plt.ylim([-200, 200])
        df2 = pd.DataFrame(np.asarray(agents[s].history).reshape(-1, 2))
        sns.scatterplot(x=df2[0], y=df2[1])
        plt.show()
        # df2


def visualize_trajectory():
    sheph = np.asarray(driver.history).reshape(-1, 2)
    start_agents = np.asarray([i.history[0] for i in agents]).reshape(-1, 2)
    end_agents = np.asarray([i.history[-1] for i in agents]).reshape(-1, 2)
    journey_agents = np.asarray([i.history[1:-1] for i in agents]).reshape(-1, 2)
    colours = []
    Xs = []
    Ys = []
    for i in sheph:
        Xs.append(i[0])
        Ys.append(i[1])
        colours.append('red')
    for i in start_agents:
        Xs.append(i[0])
        Ys.append(i[1])
        colours.append('green')
    for i in end_agents:
        Xs.append(i[0])
        Ys.append(i[1])
        colours.append('yellow')
    #     for i in journey_agents:
    #         Xs.append(i[0])
    #         Ys.append(i[1])
    #         colours.append('gray')
    plt.scatter(Xs, Ys, color=colours)
    plt.show()


def reset_game():
    global agents
    global driver
    global agent_distances
    agent_distances = np.zeros((NUM_SHEEP, NUM_SHEEP))
    agents = [sheep(i) for i in range(NUM_SHEEP)]
    driver = shepherd()
    update_distances()
    # visualize()


def play_shepherd_move():
    global agents
    global driver
    global agent_distances
    driver.history.append(np.copy(driver.coords))
    if (shepherd_proximity_check()):
        return ()
    move = furthest_from_gcm()
    driving = False
    if (move == -1):
        driving = True
        move = get_gcm()
    else:
        move = agents[move].coords
    ideal_sheph_position = move + normalize(move) * (
        inter_agent_repulsion_threshold * ((NUM_SHEEP) ** 0.5) if driving else inter_agent_repulsion_threshold)
    displacement = normalize(ideal_sheph_position - driver.coords)
    driver.coords += displacement * shepherd_displacement


def play_sheep_move(i):
    global agents
    global driver
    global agent_distances
    agents[i].history.append(np.copy(agents[i].coords))
    inertia = relative_inertia * normalize(agents[i].headings[-1])
    shepherd_repulsion = relative_shepherd_repulsion * normalize(shepherd_agent_repulsion(i))
    herd_attraction = relative_agent_attraction * normalize(get_lcm(i) - agents[i].coords)
    neighbour_repulsion = relative_agent_repulsion * normalize(inter_agent_repulsion(i))
    noise_factor = relative_angular_noise * get_error()
    new_heading = inertia + herd_attraction + neighbour_repulsion + shepherd_repulsion + noise_factor
    agents[i].headings.append(new_heading)
    displacement = agent_displacement * new_heading
    if (shepherd_repulsion.all() == 0):
        displacement = 0
    agents[i].coords += displacement


def plot_current():
    plt.xlim([0, 300])
    plt.ylim([0, 300])
    df1 = pd.DataFrame(np.asarray(driver.coords).reshape(-1, 2))
    sns.scatterplot(x=df1[0], y=df1[1])
    plt.xlim([-300, 300])
    plt.ylim([-300, 300])
    df2 = pd.DataFrame(np.asarray([i.coords for i in agents]).reshape(-1, 2))
    sns.scatterplot(x=df2[0], y=df2[1])
    plt.savefig('./images/fig_{0}.png'.format(count))
    plt.close()

## Function calls

reset_game()

count = 0
#
for j in range(10000):
    plot_current()
    count+=1
    for i in range(NUM_SHEEP):
        play_sheep_move(i)
    play_shepherd_move()
    update_distances()
    gcm = get_gcm()
    if (np.all(np.abs(gcm) < 20)):
        print("done")
        break
visualize_trajectory()

