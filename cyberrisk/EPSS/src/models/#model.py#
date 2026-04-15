from abc import ABC, abstractmethod
from src.utils import dynamic_import, ConfigParserEPSS
import pickle
import pandas as pd


class EPSS(ABC):
    def __init__(self, config: ConfigParserEPSS) -> None:
        self.config = config
        need_LM = False
        self.lm = None
        lm_dependent_modules = ["ClassifierCWE", "ClassifierCVSS", "DescriptionEmbedder"]
        selected_modules = set([k for k,v in config.modules.items() if v])
        if len(selected_modules.intersection(set(lm_dependent_modules))) > 0:
            from sentence_transformers import SentenceTransformer
            self.lm = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        self.modules = []
        for m in selected_modules:
            Module = dynamic_import("src.modules", m)
            self.modules.append(Module(self.lm))
        self.nvd_features = config.nvd_features

    @abstractmethod
    def train(self, data: pd.DataFrame):
        ## actually train model
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def evaluate(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def cast_target(self, data: pd.DataFrame):
        ## needs to be different for each model type
        pass

    def save(self,  filepath: str) -> None:
        with open(filepath, 'wb') as o:
            o.write(pickle.dumps(self.__dict__))
            
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.__dict__ = pickle.loads(f.read())

