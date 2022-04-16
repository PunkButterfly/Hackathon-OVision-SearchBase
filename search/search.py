import numpy as np 
from typing import List, Tuple
from  abc import ABC, abstractclassmethod

class Base(ABC):
    ''' SearchBase class implements 
    search through the database to find matching
    vector for query vector. It measures
    search speed and assign a score based
    on the search time and correctness of the
    search 
    '''
    @abstractclassmethod
    def __init__(self, data_file='./data/train_data.pickle') -> None:
        '''
        Creates regestration matrix and passes 
        dictionary. Measures baseline speed on
        a given machine
        '''

    @abstractclassmethod
    def set_base_from_pickle(self):
        '''
        Downloads the data, if it does not exist.
        Sets reg_matrix and pass_dict

        reg_matrix : np.array(N, 512)
        pass_dict : dict -> dict[idx] = [np.array[1, 512]]
        '''
        pass  

    @abstractclassmethod
    def search(self, query: np.array) -> List[Tuple]:
        '''
        Baseline search algorithm. 
        Uses simple matrix multiplication
        on normalized feature of face images

        Arguments:
            query : np.array - 1x512
        Return:
            Tuple[List[Tuple], float]
            List[Tuple] - indicies of search, similarity
        '''
        pass 

    @abstractclassmethod
    def cos_sim(self, query: np.array) -> np.array:
        pass

    # def insert(self, feature: np.array) -> None:
    #     ## there no inplace concationation in numpy so far. For inplace
    #     ## concationation operation both array should be contingious in 
    #     ## memory. For now, let us suffice the naive implementation of insertion
    #     pass