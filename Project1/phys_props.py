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

        T_avg = np.sum(T) / len(T)
        print(f'Average temperature: {T_avg}')

        plt.plot(T)
        plt.show()


def main():
    # df = read_file('dump.energy_b_0.001')
    # total_energy(df)
    # temperature(df)

    df_log = read_log()
    temperature(df_log)


if __name__ == '__main__':
    main()
