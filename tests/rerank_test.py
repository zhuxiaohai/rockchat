import pandas as pd
from rerank import QAReranker
import json

key_map = {"result": "id", "reason": "info", "similarities": "score"}


def test_qa_reranker():
    test = pd.read_csv("data/data_rerank.csv")

    config = {
        "rank_key": [("rank", False)],
        "reranking_scheme": {
            "recall_ranking_score_threshold": 0.75,
            "recall_ranking_top_n": 2,
                             },
        "database_path": "../data/database20240506.csv",
    }

    reranker = QAReranker(config)
    for i in range(test.shape[0]):
        scores = json.loads(test["score"].iloc[i])
        recalled_results = json.loads(test["recall_bge"].iloc[i])
        for j in range(len(recalled_results)):
            for key in key_map:
                recalled_results[j][key_map[key]] = recalled_results[j].pop(key)
            recalled_results[j].update({"rank": scores[j]})

        results = reranker.rerank(recalled_results)
        ground_truth_list = json.loads(test["reranking"].iloc[i])

        try:
            assert len(results) == len(ground_truth_list)
        except:
            print("item number mismatch", i)

        results = [item["id"] for item in results]
        ground_truth_list = [item["result"] for item in ground_truth_list]
        try:
            assert results == ground_truth_list
        except:
            print("item ranking mismatch", i)


test_qa_reranker()