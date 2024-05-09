import pandas as pd
from find_keywords import LabellerByRules
from recall import RecallBySearchEngine
from search_engine import QASearchEngine, VectorSim
import json


def format_info(v):
    if v == "errorcode":
        return "error"
    elif v == "none":
        return ""
    else:
        return v


def test_recall_vector_search():
    test = pd.read_csv("data/data_vector_search.csv")

    config_router = {
        "dim_df_path": "../data/dim_df20240315.csv",
        "model_col": ("model", "model"),
        "cat_col": ("cat_name", "cat"),
        "error_col": ("error", "error"),
    }
    router = LabellerByRules(config_router)

    config = {
        "search_engine": {
            "class": QASearchEngine,
            "database_path": "../data/database20240506.csv",
            "id_col": "qa_id",
            "index_columns": [("model_list", "model"), ("cat_name", "cat"), ("error_list", "error")],
            "score_model": {
                "type": "vector",
                "class": VectorSim,
                "embedding_col": "question",
                "embedding_model_path": "/workspace/data/private/zhuxiaohai/models/bge_finetune_emb"
            },
        },
        "top_n": 10,
    }
    top_n = config.pop("top_n")
    vector_search = RecallBySearchEngine(config)
    for i in range(test.shape[0]):
        query = test["question"].iloc[i]
        query_body = router.extract_keywords(query)
        query_cleaned = query_body["query_cleaned"]
        search_body = {
            "query": query_cleaned,
            "top_n": top_n,
            "labels": query_body["labels"]
        }
        results = vector_search.query_recalls(search_body)
        for j in range(len(results)):
            for key in results[j]:
                if key == "info":
                    results[j][key] = "|".join(sorted(results[j][key].split("|")))
                if key == "score":
                    results[j][key] = round(results[j][key], 2)
        results = sorted(results, key=lambda x: (x["id"], x["info"]))

        ground_truth_list = json.loads(test["recall_list"].iloc[i])
        for j in range(len(ground_truth_list)):
            for key in ground_truth_list[j]:
                if key == "info":
                    ground_truth_list[j][key] = "|".join(
                        sorted([format_info(v) for v in ground_truth_list[j][key].split("|")]))
                if key == "score":
                    ground_truth_list[j][key] = round(ground_truth_list[j][key], 2)
        ground_truth_list = sorted(ground_truth_list, key=lambda x: (x["id"], x["info"]))

        try:
            assert len(results) == len(ground_truth_list)
        except:
            print("recall number mismatch", i)
        for j in range(len(ground_truth_list)):
            try:
                for key in ["id", "score", "info"]:
                    assert results[j][key] == ground_truth_list[j][key]
            except:
                print("recall item mismatch", i)


def test_recall_vector_search_online():
    test = pd.read_csv("data/data_vector_search.csv")

    config_router = {
        "dim_df_path": "../data/dim_df20240315.csv",
        "model_col": ("model", "model"),
        "cat_col": ("cat_name", "cat"),
        "error_col": ("error", "error"),
    }
    router = LabellerByRules(config_router)

    config = {
        "search_engine": {
            "class": QASearchEngine,
            "database_path": "../data/database20240506.csv",
            "id_col": "qa_id",
            "index_columns": [("model_list", "model"), ("cat_name", "cat"), ("error_list", "error")],
            "score_model": {
                "type": "vector",
                "class": VectorSim,
                "online": True,
                "embedding_col": "question",
                "embedding_model_path": "http://0.0.0.0:8501/embeddings/"
            },
        },
    }
    top_n = 10
    vector_search = RecallBySearchEngine(config)
    for i in range(test.shape[0]):
        query = test["question"].iloc[i]
        query_body = router.extract_keywords(query)
        query_cleaned = query_body["query_cleaned"]
        search_body = {
            "query": query_cleaned,
            "top_n": top_n,
            "labels": query_body["labels"]
        }
        results = vector_search.query_recalls(search_body)
        for j in range(len(results)):
            for key in results[j]:
                if key == "info":
                    results[j][key] = "|".join(sorted(results[j][key].split("|")))
                if key == "score":
                    results[j][key] = round(results[j][key], 2)
        results = sorted(results, key=lambda x: (x["id"], x["info"]))

        ground_truth_list = json.loads(test["recall_list"].iloc[i])
        for j in range(len(ground_truth_list)):
            for key in ground_truth_list[j]:
                if key == "info":
                    ground_truth_list[j][key] = "|".join(
                        sorted([format_info(v) for v in ground_truth_list[j][key].split("|")]))
                if key == "score":
                    ground_truth_list[j][key] = round(ground_truth_list[j][key], 2)
        ground_truth_list = sorted(ground_truth_list, key=lambda x: (x["id"], x["info"]))

        try:
            assert len(results) == len(ground_truth_list)
        except:
            print("recall number mismatch", i)
        for j in range(len(ground_truth_list)):
            try:
                for key in ["id", "score", "info"]:
                    assert results[j][key] == ground_truth_list[j][key]
            except:
                print("recall item mismatch", i)


test_recall_vector_search_online()