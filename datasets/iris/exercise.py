import numpy as np
import pandas as pd
from src.si.data.dataset import Dataset
from src.io.csv_file import read_csv, write_csv 
from src.io.data_file import *

filename = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\iris\iris.csv"

dataset=read_csv(filename, sep=",")
dataset