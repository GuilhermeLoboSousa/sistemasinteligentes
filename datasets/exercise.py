import numpy as np
from src.si.data.dataset import Dataset
from src.io.csv_file import read_csv, write_csv 
from src.io.data_file import *

filename="iris.csv"

dataset=read_csv(filename, sep=",")
dataset