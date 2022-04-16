from telnetlib import SE
from tqdm import tqdm
from utils.util_funcs import validate 
from search.search_baseline import SearchBase
from search.search_solution import SearchSolution 
from time import time


if __name__ == "__main__":

    base_obj = SearchBase()
    base_set_time = 0
    t0 = time()
    base_obj.set_base_from_pickle()
    t1 = time()
    base_obj.cal_base_speed()
    set_base_metric = t1 - t0
    print(f"Base data init metric: {set_base_metric}")

    sol_obj = SearchSolution()
    t0 = time()
    sol_obj.set_base_from_pickle()
    t1 = time()
    set_base_metric = t1 - t0
    metric = validate(sol_obj)
    print(f"Solution search metric: {metric}")
    print(f"Solution data init metric: {set_base_metric}")
