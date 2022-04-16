import pickle, os, time, gdown
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg
from .search import Base
import faiss


class SearchSolution(Base):
    # check
    def __init__(self, data_file='./data/train_data.pickle',
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        self.data_file = data_file
        self.data_url = data_url
        self.dim = 512
        self.n_clusters = 1000
        self.top_n = 1

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

        self.reg_matrix = self.reg_matrix.astype('float32')
        self.pass_dict = data['pass']

        print("Building model...")
        self.quantiser = faiss.IndexFlatL2(self.dim)
        self.faiss_model = faiss.IndexIVFFlat(self.quantiser, self.dim,
                                              self.n_clusters, faiss.METRIC_INNER_PRODUCT)

        self.faiss_model.train(self.reg_matrix)
        self.faiss_model.add(self.reg_matrix)
        print("Model build")

    def search(self, query: np.array) -> List[Tuple]:
        D, I = self.faiss_model.search(np.array([query.astype('float32')]), self.top_n)

        # similarity = self.cos_sim(query, response_indexes)
        return [(I[0][i], D[0][i]) for i in range(self.top_n)]

    def cos_sim(self, query: np.array, response_indexes) -> np.array:
        pass
