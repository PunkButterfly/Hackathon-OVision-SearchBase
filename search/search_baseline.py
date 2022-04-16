import pickle, os, time, gdown
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg 
from .search import Base


class SearchBase(Base):
    ''' SearchBase class implements 
    search through the database to find matching
    vector for query vector. It measures
    search speed and assign a score based
    on the search time and correctness of the
    search 
    '''
    # @profile
    def __init__(self, data_file='./data/train_data.pickle', 
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        '''
        Creates regestration matrix and passes 
        dictionary. Measures baseline speed on
        a given machine
        '''
        print('reading from pickle')
        self.data_file = data_file
        self.data_url = data_url

    # @profile
    def set_base_from_pickle(self):
        '''
        Downloads the data, if it does not exist.
        Sets reg_matrix and pass_dict

        reg_matrix : np.array(N, 512)
        pass_dict : dict -> dict[idx] = [np.array[1, 512]]
        '''
        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data') 
            gdown.download(self.data_url, self.data_file, quiet=False) 

        with open(self.data_file, 'rb') as f:
            data = pickle.load(f) 
    
        self.reg_matrix = [None] * len(data['reg'])
        self.ids = {} 
        for i, key in enumerate(data['reg']):
            self.reg_matrix[i] = data['reg'][key][0][None]
            self.ids[i] = key

        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)        
        self.pass_dict = data['pass']
    
    # @profile
    def cal_base_speed(self, base_speed_path='./base_speed.pickle') -> float:
        '''
        Validates baseline and improved searh
        Return:
                metric : float - score for search
        ''' 

        samples = cfg.samples 
        N, C, C_time, T_base = 0, 0, 0, 0
        for i, tup in enumerate(tqdm(self.pass_dict.items(), total=samples)):

            idx, passes = tup
            for q  in passes:
                t0 = time.time()
                c_output = self.search(query=q) 
                t1 = time.time()
                T_base += (t1 - t0)

                C_set = [True for tup in c_output if tup[0] == idx]
                if len(C_set):
                    C += 1
                    C_time += (t1 - t0) 
                N += 1

            if i > samples:
                break

        base_speed = T_base / N
        print(f"Base Line Speed: {base_speed}")
        print(f"Base Line Accuracy: {C / N * 100}")
        with open(base_speed_path, 'wb') as f:
            pickle.dump(base_speed, f)

    # @profile
    def search(self, query: np.array) -> List[Tuple]:
        '''
        Baseline search algorithm. 
        Uses simple matrix multiplication
        on normalized feature of face images

        Arguments:
            query : np.array - 1x512
        Return:
            List[Tuple] - indicies of search, similarity
        '''
        similarity = self.cos_sim(query) 
        return [(self.ids[i], sim) for i, sim in enumerate(similarity)] 
    

    def insert_base(self, feature: np.array) -> None:

        ## there no inplace concationation in numpy so far. For inplace
        ## concationation operation both array should be contingious in 
        ## memory. For now, let us suffice the naive implementation of insertion
        self.reg_matrix = np.concatenate(self.reg_matrix, feature, axis=0) 

    def cos_sim(self, query: np.array) -> np.array:
        return np.dot(self.reg_matrix, query)
