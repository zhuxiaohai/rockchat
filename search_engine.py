from abc import ABC, abstractmethod
from collections.abc import Iterable
import heapq
import requests
from time import time
import numpy as np
import pandas as pd
from fastbm25 import fastbm25
from sentence_transformers import SentenceTransformer


def send_embedding_requests(string_list, num_requests=3, url="http://0.0.0.0:8501/embeddings/", verbose=False):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    json_data = {
        "sentences": string_list
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
            break
        else:
            if verbose:
                print(f"Request {i + 1} failed with status code {response.status_code}")

    if verbose:
        end_time = time()
        duration = end_time - start_time
        print(f"All requests have been sent.\nTotal time taken: {duration:.2f} seconds")

    if len(responses) > 0:
        return responses[0]["embeddings"]
    else:
        return None


class FastBM25(fastbm25):
    def __init__(self, corpus):
        super().__init__(corpus)

    def top_k_sentence(self, document, k=1, filter_indices=None):
        assert isinstance(document, Iterable), 'document is not iterable'
        score_overall = {}
        for word in document:
            if word not in self.document_score:
                continue
            for key, value in self.document_score[word].items():
                if filter_indices and (key not in filter_indices):
                    continue
                if key not in score_overall:
                    # print(score_overall)
                    score_overall[key] = value
                else:
                    score_overall[key] += value
        k_keys_sorted = heapq.nlargest(k, score_overall, key=score_overall.__getitem__)
        return [{"index": index, "score": score_overall.get(index, None)}
                for index in k_keys_sorted]


class VectorSim:
    def __init__(self, corpus, model_path, online=False):
        self.corpus = corpus
        self.online = online
        self.model_path = model_path
        if not online:
            self.model = SentenceTransformer(model_path)
        self.embeddings = self.get_embedding(corpus)

    def get_embedding(self, document_list):
        if self.online:
            return send_embedding_requests(document_list, url=self.model_path)
        else:
            return self.model.encode(document_list, normalize_embeddings=True).tolist()

    def top_k_sentence(self, document, k=1, filter_indices=None):
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        embedding = self.get_embedding([document])[0]
        if filter_indices is None:
            filter_indices = range(len(self.embeddings))
        # else:
        #     filter_indices = sorted(list(filter_indices))
        score_overall = []
        for index in filter_indices:
            score = cosine_similarity(embedding, self.embeddings[index])
            score_overall.append({"index": index, "score": score})
        k_keys_sorted = sorted(score_overall, key=lambda x: (-x["score"], x["index"]))[:k]
        return [{"index": item["index"], "score": item["score"]}
                for item in k_keys_sorted]


class PandasSearchEngine(ABC):
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(config["database_path"]).reset_index(drop=True)
        self.build_index(config["index_columns"])
        if config["score_model"]["type"] == "bm25":
            self.tokenizer = config["score_model"]["tokenizer"]
            document_list = [self.tokenizer.cut(doc) for doc in self.df[config["score_model"]["embedding_col"]]]
            self.score_function = config["score_model"]["class"](document_list)
        elif config["score_model"]["type"] == "vector":
            self.score_function = config["score_model"]["class"](
                self.df[config["score_model"]["embedding_col"]].tolist(),
                config["score_model"]["embedding_model_path"],
                config["score_model"].get("online", False),
            )

    def build_index(self, columns):
        self.index = {}
        for col, rename in columns:
            inv_dict = {}
            for i in range(self.df.shape[0]):
                row = self.df[col].iloc[i]
                if pd.isna(row):
                    row = ""
                for entity in row.split(","):
                    if entity in inv_dict:
                        inv_dict[entity].add(i)
                    else:
                        inv_dict[entity] = {i}
            self.index[rename] = inv_dict

    def get_filters(self, labels):
        filters = {}
        for col in labels:
            if not isinstance(labels[col], list):
                continue
            indices = set()
            for entity in labels[col]:
                indices = self.index[col][entity] | indices
            filters[col] = indices
        return filters

    @abstractmethod
    def search(self, body):
        pass


class QASearchEngine(PandasSearchEngine):
    def search(self, body):
        query = body["query"]
        if self.config["score_model"]["type"] == "bm25":
            query = self.tokenizer.cut(query)
        filters = self.get_filters(body["labels"])
        top_n = body["top_n"]

        def pattern1(main_filter):
            results = []
            if len(main_filter) == 0:
                return results
            intersection_indices = main_filter & filters["error"]
            if len(intersection_indices) > 0:
                result = self.score_function.top_k_sentence(query, k=int(top_n/2), filter_indices=intersection_indices)
                results += result
                difference_indices = main_filter - filters["error"]
                result = self.score_function.top_k_sentence(query, k=int(top_n/2), filter_indices=difference_indices)
                results += result
            else:
                result = self.score_function.top_k_sentence(query, k=int(top_n/2), filter_indices=main_filter)
                results += result
                result = self.score_function.top_k_sentence(query, k=int(top_n/2), filter_indices=filters["error"])
                results += result
            difference_indices = filters["cat"] - main_filter
            result = self.score_function.top_k_sentence(query, k=int(top_n/2), filter_indices=difference_indices)
            results += result
            return results

        def pattern2(main_filter):
            results = []
            if len(main_filter) == 0:
                return results
            all_set = set(range(self.df.shape[0]))
            result = self.score_function.top_k_sentence(query, k=int(top_n/2), filter_indices=main_filter)
            results += result
            difference_filter = all_set - main_filter
            result = self.score_function.top_k_sentence(query, k=int(top_n/2), filter_indices=difference_filter)
            results += result
            return results

        def add_info():
            for result in result_list:
                doc_index = result["index"]
                doc_id = self.df[self.config["id_col"]].iloc[doc_index]
                result["id"] = doc_id
                result.pop("index")
                reasons = []
                for col in filters:
                    if doc_index in filters[col]:
                        reasons.append(col)
                result["info"] = "|".join(reasons)

        result_list = []
        if len(filters["model"]) > 0:
            result = pattern1(filters["model"])
            result_list += result
        else:
            result = pattern1(filters["cat"])
            result_list += result
        if len(result_list) == 0:
            if len(filters["error"]) > 0:
                result = pattern2(filters["error"])
                result_list += result
            else:
                result = self.score_function.top_k_sentence(query, k=top_n)
                result_list += result
        add_info()

        return result_list

