import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import sys
from read_lmp_dump import read_file
from read_lmp_log import read_log


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
    id = df[['id']].to_numpy()
    x = df[['x']].to_numpy()
    y = df[['y']].to_numpy()
    z = df[['z']].to_numpy()

    r = np.sqrt(x * x + y * y + z * z)

    no_tsteps = 201
    no_atoms = 4000

    id = id.reshape(no_atoms, no_tsteps)
    print(np.sort(id[:, 2])[:20])
    exit()
    r = r.reshape(no_atoms, no_tsteps)
    # x = x.reshape(no_atoms, no_tsteps)
    # y = y.reshape(no_atoms, no_tsteps)
    # z = z.reshape(no_atoms, no_tsteps)

    for i in range(no_tsteps):
        # Sorting by atom number for all timesteps
        id[:, i], r[:, i] = zip(*sorted(zip(id[:, i], r[:, i])))
        print(id[:10, i])


def main():
    # df_log = read_log()
    # temperature(df_log)
    # energy(df_log)
    # pressure(df_log, plot=True)
    # density(df_log, plot=True)
    displacement(read_file('dump.displacement'))


if __name__ == '__main__':
    main()
