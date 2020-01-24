import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
import sys
from read_lmp_dump import read_file


def f(v, T):
    return 4 * np.pi * v**2 * (1 / (2 * np.pi * T))**(3 / 2) * \
        np.exp(-v**2 / (2 * T))


def plot_histogram(v, v_arr, no_tsteps, no_atoms, T=2.5):
    for i in range(0, no_tsteps, no_tsteps // 6):
        plt.hist(v_arr[i, :], bins=40, density=True)
        plt.plot(v_arr[i, :], f(v_arr[i, :], T))
        plt.title('Time: %0.1f' % (i / 100))
        plt.show()

    v = np.sort(v)

    plt.hist(v, bins=500, density=True)
    plt.plot(v, f(v, T))
    plt.show()


def histogram_time_evo(v, v_arr):
    hist_list = []
    hist_last = plt.hist(v_arr[-1, :], bins=40)
    hist_last2 = np.sum(hist_last[0] * hist_last[0])

    for i in range(len(v_arr[:, 0])):
        hist_list.append(plt.hist(v_arr[i, :], bins=40))
        h = plt.hist(v_arr[i, :], bins=40)
        hh = np.sum(h[0] * hist_last[0])
        hist_list.append(hh / hist_last2)

    for i in range():
        plt.plot(hist_list[i])
    plt.show()


def main():
    if len(sys.argv) < 4:
        print('State 1) filename from which to read data 2) number of timesteps and 3) number of atoms')
        print('E.g. 1) dump.velocities 2) 100 3) 4000')
        exit()

    filename = sys.argv[1]
    no_tsteps = int(sys.argv[2])
    no_atoms = int(sys.argv[3])

    df = read_file(filename)

    v = df[['vx', 'vy', 'vz']].to_numpy()
    v = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])

    v_arr = np.zeros((no_tsteps, no_atoms - 1))

    for i in range(no_tsteps):
        v_arr[i, :] = np.sort(v[i * (no_atoms - 1):(i + 1) * (no_atoms - 1)])

    # plot_histogram(v, v_arr, no_tsteps, no_atoms)
    histogram_time_evo(v, v_arr)


if __name__ == '__main__':
    main()
