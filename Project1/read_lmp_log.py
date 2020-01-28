import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def read_log():
    with open('log.lammps', 'r', newline='\n') as f_open:
        data = f_open.read()

    # data = data.split()

    idx_col_titles = []

    for match in re.finditer('Step', data):
        start_idx = match.start()
        idx_col_titles.append(start_idx)

    idx_end_num = []

    for match in re.finditer('Loop', data):
        start_idx = match.start() - 1
        idx_end_num.append(start_idx)

    print(idx_col_titles)


def main():
    read_log()


if __name__ == '__main__':
    main()
