import numpy as np 
from .search import Base
from typing import List, Tuple

class SearchSolution(Base):

    def __init__(self, data_file='./data/train_data.pickle', 
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        self.data_file = data_file
        self.data_url = data_url
        pass

    def set_base_from_pickle(self):
        pass

    def search(self, query: np.array) -> List[Tuple]:
        pass

    def cos_sim(self, query: np.array) -> np.array:
        pass
