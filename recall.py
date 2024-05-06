from abc import ABC, abstractmethod
from typing import List, Dict
from search_engine import QASearchEngine


class RecallChannelBase(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def query_recalls(self, query: str, labels: Dict[str, List[str]]):
        pass


class BM25RecallChannel(RecallChannelBase):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.search_engine = QASearchEngine(config["search_engine"])

    def query_recalls(self, query: str, labels: List[List[str]]):
        search_body = {
            "query": query,
            "top_n": self.config["top_n"]
        }
        results = self.search_engine.search(search_body)
        return results


def get_recall_configured_channels(config):
    pass