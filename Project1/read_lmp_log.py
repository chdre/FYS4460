import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def read_log():
    with open('log.lammps', 'r', newline='\n') as f_open:
        data = f_open.read().replace('\r', '')

    # data = data.split()

    idx_col_titles_start = [match.start()
                            for match in re.finditer('Step', data)]
    idx_end_num = [match.start() for match in re.finditer('Loop', data)]

    datasets = len(idx_col_titles_start)

    idx_col_titles_end = [
        data.find('\n', idx_col_titles_start[i]) for i in range(datasets)]

    col_titles = data[idx_col_titles_start[0]: idx_col_titles_end[0]].split()

    dataframe_list = []  # List of dataframe for all timesteps

    for k in range(datasets):
        dict_list = []  # List of dataframe for current timestep

        num_data = data[idx_col_titles_end[k] + 1: idx_end_num[k] - 1]
        num_data = re.sub(' +', ' ', num_data)

        lines = num_data.split('\n')

        for i in range(len(lines)):
            elms = lines[i].split()

            data_dict = dict((j, elms[col_titles.index(j)])
                             for j in col_titles)

            dict_list.append(data_dict)

        df = pd.DataFrame(dict_list, columns=col_titles, dtype=np.float)

        dataframe_list.append(df)

    if len(dataframe_list) < 2:
        return dataframe_list[0]

    return dataframe_list


def main():
    read_log()


if __name__ == '__main__':
    main()
