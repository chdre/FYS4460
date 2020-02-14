import matplotlib.pyplot as plt
import numpy as np
import sys
from read_lmp_dump import read_file


def f(v, T):
    return 4 * np.pi * v**2 * (1 / (2 * np.pi * T))**(3 / 2) * \
        np.exp(-v**2 / (2 * T))


def plot_histogram(v, v_arr, no_tsteps, no_atoms, no_bins, T=2.5):
    for i in range(0, no_tsteps, no_tsteps // 10):
        plt.hist(v_arr[i, :], bins=no_bins, density=True)
        plt.plot(v_arr[i, :], f(v_arr[i, :], T))
        plt.title(f'Time: {0.005 * i:.2f}')
        print(i)
        plt.show()

#     v = np.sort(v)

#     plt.hist(v, bins=no_bins, density=True)
#     plt.plot(v, f(v, T))
#     plt.show()


def histogram_time_evo(v, v_arr, no_bins):
    hist_list = []
    hist_last = np.histogram(v_arr[-1, :], bins=no_bins)[0]
    hist_last2 = np.sum(hist_last * hist_last)

    t_steps = len(v_arr[:, 0])

    for i in range(t_steps):
        h = np.histogram(v_arr[i, :], bins=no_bins)[0]
        hh = np.sum(h * hist_last)
        hist_list.append(hh / hist_last2)

    x = np.linspace(0, t_steps * 0.005, t_steps)

    plt.plot(x, hist_list, label='Hist. values')
    plt.plot(x, np.ones(t_steps), '--', label='Theoretical')
    plt.legend(loc='best')
    plt.xlabel('Time t')
    plt.ylabel('Histogram values')
    plt.show()


def main(filename):
    no_tsteps = 200
    no_atoms = 4000

    df = read_file(filename)

    v = df[['vx', 'vy', 'vz']].to_numpy()
    v = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])

    v_arr = np.zeros((no_tsteps, no_atoms))

    for i in range(no_tsteps):
        v_arr[i, :] = np.sort(v[i * (no_atoms):(i + 1) * no_atoms])

    plot_histogram(v, v_arr, no_tsteps, no_atoms, 40, T=2.5)
    histogram_time_evo(v, v_arr, 40)


if __name__ == '__main__':
    main('dump.task_a')
