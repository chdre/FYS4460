import matplotlib.pyplot as plt
import numpy as np
import sys
from read_lmp_dump import read_file


def total_energy(df):
    sigma = 0.3405e-6
    eps = 1.0318e-2

    x = df[['x', 'y', 'z']].to_numpy()
    v = df[['vx', 'vy', 'vz']].to_numpy()

    Ep =


def main():
    filename = 'dump.task_b'

    df = read_file(filename)


if __name__ == '__main__':
    main()
