from tqdm import tqdm
import pickle
import numpy as np
from time import time
from config import Config as cfg 
import random


def validate(search_object, base_speed_file='./base_speed.pickle') -> float:
    '''
    Validates baseline and improved searh
    Return:
            metric : float - score for search
    ''' 
    sample = cfg.samples 

    with open(base_speed_file, 'rb') as f:
        base_speed = pickle.load(f)

    search_fun = search_object.search
    N, C, C_time, T_base = 0, 0, 0, 0
    
    random.seed(42)
    pass_dict = random.sample(search_object.pass_dict.items(), sample)
    for i, tup in enumerate(tqdm(pass_dict, total=sample)):
        idx, passes = tup
        for q  in passes:
            t0 = time()
            c_output = search_fun(query=q) 
            t1 = time()
            T_base += (t1 - t0) 

            C_set = [True for tup in c_output if tup[0] == idx]
            if len(C_set):
                C += 1
                C_time += (t1 - t0)
                print(t1 - t0)
            N += 1

        if i > sample:
            break
    print(C, N, C_time)
    metric = calc_metric(C, N, C_time, base_speed) 
    return metric

def calc_metric(C: int, N: int, C_time: float, base_speed : float) -> float:
    '''
    Calculates comparison metric of the search
    Arguments:
        C : int - corrent search count
        N : int - searches count
        C_time : float - total time spent for correct searches
    
    Return:
        S : int - metric value (the lesser the better)
    '''

    alpha = np.exp(2 - (C / N))
    S = C_time / N  + (1 - C / N) * alpha * base_speed
    return S
