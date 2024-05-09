from abc import ABC, abstractmethod
import requests
from time import time
import copy
from FlagEmbedding import FlagReranker

import pandas as pd


def send_rank_requests(pairs, num_requests=3, url="http://localhost:8501/rerank/", verbose=False):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    duplication_flag = False
    if len(pairs) == 1:
        duplication_flag = True
        pairs = [pairs[0], pairs[0]]
    json_data = {
        "sentence_pairs": pairs
    }

    if verbose:
        start_time = time()
        print("Sending requests...")

    responses = []
    for i in range(num_requests):
        response = requests.post(url, headers=headers, json=json_data)
        if response.status_code == 200:
            # 直接处理 JSON 响应
            responses.append(response.json())
        else:
            if verbose:
               print(f"Request {i + 1} failed with status code {response.status_code}")

    if verbose:
        end_time = time()
        duration = end_time - start_time
        print(f"All requests have been sent.\nTotal time taken: {duration:.2f} seconds")

    if len(responses) > 0:
        if duplication_flag:
            return [responses[0]["scores"][0]]
        else:
            return responses[0]["scores"]
    else:
        return None


class Ranker(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def compute_ranking_score(self, query, recall):
        pass


class QAScorer(Ranker):
    def __init__(self, config):
        super().__init__(config)
        self.online = config.get("online", False)
        self.model_path = config["model_path"]
        if not self.online:
            self.model = FlagReranker(config["model_path"], use_fp16=True)
        self.query_key = config["query_key"]
        self.item_key = config["item_key"]
        self.df = pd.read_csv(config["database_path"]).set_index("qa_id")

    def get_score(self, pairs):
        if self.online:
            return send_rank_requests(pairs, url=self.model_path)
        else:
            return self.model.compute_score(pairs)

    def compute_ranking_score(self, query, recall):
        query = copy.deepcopy(query)
        recall = copy.deepcopy(recall)
        pairs = []
        for item in recall:
            pairs.append((query[self.query_key], self.df.loc[item["id"], self.item_key]))
        scores = self.get_score(pairs)
        results = []
        for i, item in enumerate(recall):
            item["rank"] = scores[i]
            item.update({"query": query})
            results.append(item)
        return results