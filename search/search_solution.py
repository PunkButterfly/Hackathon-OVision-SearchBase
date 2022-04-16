import pickle, os, time, gdown
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg
from .search import Base
from sklearn.cluster import MiniBatchKMeans


class SearchSolution(Base):
    # check
    def __init__(self, data_file='/content/drive/MyDrive/train_data.pickle',
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        self.data_file = data_file
        self.data_url = data_url
        self.n_models = 10
        self.n_clusters = 100

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

        self.k_means_list = []
        self.models_labels_dicts = []

        for k in range(self.n_models):
            k_means = MiniBatchKMeans(n_clusters=self.n_clusters)
            k_means.fit(self.reg_matrix)

            model_labels_dict = {}
            labels = k_means.labels_

            for i, lable in enumerate(labels):
                if lable not in model_labels_dict:
                    model_labels_dict[lable] = []
                model_labels_dict[lable].append(i)

            self.k_means_list.append(k_means)
            self.models_labels_dicts.append(model_labels_dict)
            print(f"Train {k+1} models.")

    def search(self, query: np.array) -> List[Tuple]:
        response_indexes = []
        for i in range(self.n_models):
            curr_cluster = self.k_means_list[i].predict([query])[0]
            response_indexes += self.models_labels_dicts[i][curr_cluster]
        response_indexes = list(set(response_indexes))

        similarity = self.cos_sim(query, response_indexes)
        return [(response_indexes[i], sim) for i, sim in enumerate(similarity)]

    def cos_sim(self, query: np.array, response_indexes) -> np.array:
        nei_matrix = [self.reg_matrix[i] for i in response_indexes]
        return np.dot(nei_matrix, query)
