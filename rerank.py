from abc import ABC, abstractmethod
import copy
from utils import ranking_metric
import pandas as pd

show_cols = ["question", "answer"]


class RerankBase(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def rerank(self, results):
        pass


class QAReranker(RerankBase):
    def __init__(self, config):
        super().__init__(config)
        self.rank_key = config["rank_key"]
        self.reranking_scheme = config.get("reranking_scheme", None)
        self.df = pd.read_csv(config["database_path"]).set_index("qa_id")

    def rerank(self, results):
        ranked_results = copy.deepcopy(results)
        ranked_results = sorted(ranked_results, key=lambda x: [x[key] if ascending else -x[key]
                                                               for key, ascending in self.rank_key])
        if not self.reranking_scheme:
            return ranked_results

        topping_indices = []
        for i, item in enumerate(ranked_results):
            if (
                    (item["info"].find("model") >= 0) |
                    (item["info"].find("error") >= 0) |
                    (len(item["info"].split(",")) > 1)
            ) & (item["score"] > self.reranking_scheme["recall_ranking_score_threshold"]):
                item["rerank"] = ranking_metric(item["info"])
                topping_indices.append(i)
        topping_indices = sorted(
            topping_indices,
            key=lambda index: (ranked_results[index]["rerank"],
                               -ranked_results[index]["score"])
        )[:self.reranking_scheme["recall_ranking_top_n"]]
        topping_indices = {index: rerank_order for rerank_order, index
                           in sorted(enumerate(topping_indices), key=lambda x: x[1])}

        reranked_results = []
        for index in topping_indices:
            ranked_results[index].update({"rerank": topping_indices[index]})
            for col in show_cols:
                ranked_results[index].update({col: self.df.loc[ranked_results[index]["id"], col]})
            reranked_results.append(ranked_results[index])
        for index, item in enumerate(ranked_results):
            if index not in topping_indices:
                ranked_results[index].update({"rerank": -1})
                for col in show_cols:
                    ranked_results[index].update({col: self.df.loc[ranked_results[index]["id"], col]})
                reranked_results.append(ranked_results[index])

        return reranked_results
