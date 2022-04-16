import pickle, os, time, gdown
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg
from .search import Base
from annoy import AnnoyIndex


class SearchSolution(Base):
    # check
    def __init__(self, data_file='./data/train_data.pickle',
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        self.data_file = data_file
        self.data_url = data_url
        self.trees = 20
        self.n_count = 100

    def set_base_from_pickle(self):
        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data')
            gdown.download(self.data_url, self.data_file, quiet=False)
        print('reading pickle file..')
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
        print('pickle is read')
        self.reg_matrix = [None] * len(data['reg'])
        for i, key in enumerate(data['reg']):
            self.reg_matrix[i] = data['reg'][key][0][None]

        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)
        self.pass_dict = data['pass']

        self.annoy_model = AnnoyIndex(512, 'angular')
        for i in range(len(self.reg_matrix)):
            self.annoy_model.add_item(i, self.reg_matrix[i])
        print('building model...')
        self.annoy_model.build(self.trees)
        print('model build')

    def search(self, query: np.array) -> List[Tuple]:
        response_indexes = self.annoy_model.get_nns_by_vector(query, self.n_count)

        similarity = self.cos_sim(query, response_indexes)
        return [(response_indexes[i], sim) for i, sim in enumerate(similarity)]

    def cos_sim(self, query: np.array, response_indexes) -> np.array:
        nei_matrix = [self.reg_matrix[i] for i in response_indexes]
        return np.dot(nei_matrix, query)
