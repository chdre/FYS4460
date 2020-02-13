import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import sys
from read_lmp_dump import read_file
from read_lmp_log import read_log

import ovito.io as ov


def total_energy(df):
    Ep = df[['c_Ep']].to_numpy()
    Ek = df[['c_Ek']].to_numpy()

    tot_energy = Ep + Ek

    no_tsteps = 2000
    no_atoms = 4000

    tot_energy_arr = np.zeros(no_tsteps)
    Ep_arr = np.zeros_like(tot_energy_arr)
    Ek_arr = np.zeros_like(tot_energy_arr)

    for i in range(no_tsteps):
        tot_energy_arr[i] = np.sum(
            tot_energy[i * no_atoms: (i + 1) * no_atoms])
        Ep_arr[i] = np.sum(Ep[i * no_atoms: (i + 1) * no_atoms])
        Ek_arr[i] = np.sum(Ek[i * no_atoms: (i + 1) * no_atoms])

    plt.figure()
    plt.plot(tot_energy_arr, label='Ek + Ep')
    plt.title('Timestep: 0.001')
    plt.ylabel('Energy')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def temperature_(df):
    Ek = df[['c_Ek']].to_numpy()

    no_tsteps = 2000
    no_atoms = 4000

    Ek = Ek[200:]

    Ek_arr = np.zeros(no_tsteps - 200)

    for i in range(no_tsteps - 200):
        Ek_arr[i] = np.sum(Ek[i * no_atoms: (i + 1) * no_atoms])

    Ek_arr /= no_atoms

    T = 2 / 3 * Ek_arr / no_atoms

    T_avg = np.sum(T) / len(T)
    print(f'Average temperature: {T_avg}')

    plt.plot(T)

    plt.show()


def temperature(df):
    for i in range(len(df)):
        # Each timestep
        dataframe = df[i]
        T = dataframe[['Temp']].to_numpy()
        dt = dataframe[['Dt']].to_numpy()[0]

        T = T[int(0.3 * len(T)):]    # System i equilibrium

        T_avg = np.sum(T) / len(T)
        print(f'Average temperature: {T_avg}')

        plt.plot(T, label='dt=' + str(dt[0]))
        plt.title('Temperature')
        plt.xlabel('Time t')
        plt.ylabel('Temerature')
        plt.legend()
        plt.show()


def energy(df):
    for i in range(len(df)):
        # Each timestep
        dataframe = df[i]
        E = dataframe[['TotEng']].to_numpy()
        dt = dataframe[['Dt']].to_numpy()[0]

        plt.plot(E, label='dt=' + str(dt[0]))
        plt.xlabel('Timesteps')
        plt.ylabel('Energy')
        plt.legend()
        plt.show()


def pressure(df, plot=False):
    for i in range(len(df)):
        dataframe = df[i]
        P = dataframe[['Press']].to_numpy()
        T = dataframe[['Temp']].to_numpy()

    T, P = zip(*sorted(zip(T, P)))

    if plot:
        plt.plot(T, P, '-o')
        plt.xlabel('Temperature')
        plt.ylabel('Pressure')
        plt.show()


def density(df, plot=False):
    for i in range(len(df)):
        dataframe = df[i]
        P = dataframe[['Press']].to_numpy()[:, 0]
        rho = dataframe[['Density']].to_numpy()[:, 0]
        T = dataframe[['Temp']].to_numpy()[:, 0]

    T, P, rho = zip(*sorted(zip(T, P, rho)))

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(T, rho, P)

        plt.show()


def displacement(df):
    D = []
    msd = []
    T = []

    steps = df[0][['Step']].to_numpy(dtype=np.int)[:, 0]
    t = np.linspace(steps[0], steps[-1] * 0.005, steps.shape[0])

    for temp in range(len(df)):
        dataframe = df[temp]
        T.append(dataframe[['Temp']].to_numpy()[0, 0])
        msd.append(dataframe[['c_msd[4]']].to_numpy()[:, 0])
        D.append(msd[temp] / (6 * t))

    for i in range(len(df)):
        plt.plot(t, msd[i])
        plt.xlabel('Time t')
        plt.ylabel('Mean squared displacement')
    plt.legend([f'T = {temp}' for temp in T])
    plt.show()

    for i in range(len(df)):
        plt.plot(t, D[i])
        plt.xlabel('Time t')
        plt.ylabel('Diffusion constant')
    plt.legend([f'T={temp}' for temp in T])
    plt.show()


def rdf(df):
    kk = 1


def main():
    # df_log = read_log()
    # temperature(df_log)
    # energy(df_log)
    # pressure(df_log, plot=True)
    # density(df_log, plot=True)
    # displacement()
    rdf(df)


if __name__ == '__main__':
    main()
