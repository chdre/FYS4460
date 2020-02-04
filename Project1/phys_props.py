import matplotlib.pyplot as plt
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


def pressure(df):
    P = np.zeros(len(df))
    T = np.zeros_like(P)

    for i in range(len(df)):
        # Each temperature
        dataframe = df[i]
        P[i] = np.average(dataframe[['Press']].to_numpy())
        T[i] = dataframe[['Temp']].to_numpy()[0]

    plt.plot(T, P, '-o')
    plt.xlabel('Temperature')
    plt.ylabel('Pressure')
    # plt.legend()
    plt.show()


def main():
    df_log = read_log()
    # temperature(df_log)
    # energy(df_log)
    pressure(df_log)


if __name__ == '__main__':
    main()
