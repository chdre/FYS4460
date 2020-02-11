import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def read_file(filename):
    """
    Stores data and sorts.
    """
    f_open = open(filename, 'r', newline='\n')
    data = f_open.read()
    f_open.close()

    no_atoms_str = 'ITEM: NUMBER OF ATOMS'
    no_atoms = int(data[data.find(no_atoms_str) +
                        len(no_atoms_str):data.find('ITEM: BOX')])

    str_itm_atm = 'ITEM: ATOMS'

    idx_itm_atm = data.find(str_itm_atm) + len(str_itm_atm) + 1
    idx_end_cols = data.find('\n', idx_itm_atm) - 1

    if '[' in data[idx_itm_atm: idx_end_cols]:
        # If Lammps dump file contains [] from printing an array, it must be
        # removed such that finditer can find proper indices. See below.
        data = data.replace('[', '')
        data = data.replace(']', '')
        idx_itm_atm = data.find(str_itm_atm) + len(str_itm_atm) + 1
        idx_end_cols = data.find('\n', idx_itm_atm) - 1

    # Column names of dataframe
    col_titles = data[idx_itm_atm: idx_end_cols].split(' ')

    dict_list = []

    # Creating dictionary with empty list as values
    data_dict = dict.fromkeys(col_titles, [])

    idx_end_num_data = [match.start()
                        for match in re.finditer('ITEM: TIMESTEP', data[1:])]
    idx_start_num_data = [match.end() + 2
                          for match in re.finditer(col_titles[-1], data)]
    idx_end_num_data.append(-1)  # Including end

    for k in range(len(idx_start_num_data)):
        num_data = data[idx_start_num_data[k]: idx_end_num_data[k]]
        lines = num_data.split('\n')

        for i in range(len(lines)):
            elms = lines[i].split(' ')

            data_dict = dict((j, elms[col_titles.index(j)])
                             for j in col_titles)
            dict_list.append(data_dict)

    dataframe = pd.DataFrame(dict_list, columns=col_titles, dtype=np.float)

    return dataframe


def main():
    """
    Takes arguments of filename from command line.
    """
    if len(sys.argv) < 2:
        print('State filename from which to read data')
        exit()

    filename = sys.argv[1]

    read_file(filename)


if __name__ == '__main__':
    main()
