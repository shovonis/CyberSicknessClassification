import os
import numpy as np


def get_data(fileName=None):
    data = []
    try:
        f = open(fileName)
        data = f.read()
        f.close()
    except FileExistsError:
        print("File Not found ")

    return data


def split_data(data):
    lines = data.split('\n')
    header = lines[0].split(",")
    lines = lines[1:]
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        float_data[i, :] = [float(x) for x in line.split(",")[1:]]  # Ignoring first column

    return header, float_data