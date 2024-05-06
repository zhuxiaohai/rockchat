from snownlp.sim.bm25 import BM25
import numpy as np
import jieba
import joblib


class EntityExtractor:
    def __init__(self, entity_path="/data/dataset/kefu/all_entity.json") -> None:
        self.all_entity = joblib.load(entity_path)
        self.entity_inv = {}
        for k, v in self.all_entity.items():
            for u in v:
                self.entity_inv[u] = k
        self.alias_corpus = [jieba.lcut(name) for name in self.entity_inv]
        self.engine = BM25(self.alias_corpus)

    def simple_match(self, query):
        query_result = []
        for name in self.all_entity:
            if (name == query) or (query in name) or (name in query):
                query_result.append(name)
        for alias in sorted(self.entity_inv.keys(), key=lambda k: len(k), reverse=True):
            if (alias == query) or (query in alias) or (alias in query):
                if self.entity_inv[alias] not in query_result:
                    query_result.append(self.entity_inv[alias])
        return query_result

    def query_entity(self, query):
        entities = self.simple_match(query["question"])
        if len(entities) == 0:
            index = np.argsort(-np.array(self.engine.simall(query["keywords"])))[0]
            entities = [self.entity_inv["".join(self.alias_corpus[index])]]
        return entities
