from type_def import *
from settings import *
import numpy as np
import pickle
from utils import tools




if __name__ == '__main__':
    record = pickle.load(open('../work/NER/record-2021_10_23_23_25_8', 'rb'))
    # m = parse_record_data(record)
