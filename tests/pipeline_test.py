import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import json
import pandas as pd
from pipeline import QAPineline
from search_engine import QASearchEngine, VectorSim
from find_keywords import LabellerByRules
from recall import get_recall_channels, RecallBySearchEngine
from merge import QAMerge
from rank import QAScorer
from rerank import QAReranker

key_map = {"result": "id"}
one_channel_map = {"vector_search": "raw"}
one_duplication_map = {"vector_search": "duplicated"}

def format_info(v):
    if v == "errorcode":
        return "error"
    elif v == "none":
        return ""
    else:
        return v

def check_results(i, phase, results, ground_truth_list):
    for j in range(len(results)):
        for key in results[j]:
            if key == "info":
                results[j][key] = "|".join(sorted([format_info(v) for v in results[j][key].split("|")]))
            if key == "score":
                results[j][key] = round(results[j][key], 2)
    results = sorted(results, key=lambda x: (x["id"], x["info"]))

    for j in range(len(ground_truth_list)):
        for key in key_map:
            ground_truth_list[j][key_map[key]] = ground_truth_list[j].pop(key)
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
        print(f"{phase} number mismatch", i)
    for j in range(len(ground_truth_list)):
        try:
            for key in ["id", "score", "info"]:
                assert results[j][key] == ground_truth_list[j][key]
        except:
            print(f"{phase} item mismatch", i)


def test_qa_pipeline():
    test = pd.read_csv("data/data_pipeline.csv")

    keywords_config = {
        "class": LabellerByRules,
        "config": {
            "dim_df_path": "../data/dim_df20240315.csv",
            "model_col": ("model", "model"),
            "cat_col": ("cat_name", "cat"),
            "error_col": ("error", "error"),
        }
    }

    recall_config = {
        "vector_search": {
            "class": RecallBySearchEngine,
            "config": {
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
            },
            "top_n": 10,
        }
    }

    merge_config = {
        "class": QAMerge,
        "config": {
            "vector_search": 1,
        }
    }

    rank_config = {
        "class": QAScorer,
        "config": {
            "model_path": "/workspace/data/private/zhuxiaohai/models/bge_finetune_reranker_question_top20",
            "query_key": "query_cleaned",
            "item_key": "question",
            "database_path": "../data/database20240506.csv",
        }
    }

    rerank_config = {
        "class": QAReranker,
        "config": {
            "rank_key": [("rank", False)],
            "reranking_scheme": {
                "recall_ranking_score_threshold": 0.75,
                "recall_ranking_top_n": 2,
                                 },
            "database_path": "../data/database20240506.csv",
        }
    }

    pipeline_config = {
        "router": keywords_config,
        "recall": recall_config,
        "merger": merge_config,
        "ranker": rank_config,
        "reranker": rerank_config,
    }

    qa_pipeline = QAPineline(pipeline_config)

    for i in range(test.shape[0]):

        query = test["raw_question"].iloc[i]
        results = qa_pipeline.run(query)
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


def test_qa_pipeline_online():
    test = pd.read_csv("data/data_pipeline.csv")

    keywords_config = {
        "class": LabellerByRules,
        "config": {
            "dim_df_path": "../data/dim_df20240315.csv",
            "model_col": ("model", "model"),
            "cat_col": ("cat_name", "cat"),
            "error_col": ("error", "error"),
        }
    }

    recall_config = {
        "vector_search": {
            "class": RecallBySearchEngine,
            "config": {
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
            },
            "top_n": 10,
        }
    }

    merge_config = {
        "class": QAMerge,
        "config": {
            "vector_search": 1,
        }
    }

    rank_config = {
        "class": QAScorer,
        "config": {
            "model_path": "http://localhost:8501/rerank/",
            "query_key": "query_cleaned",
            "item_key": "question",
            "online": True,
            "database_path": "../data/database20240506.csv",
        }
    }

    rerank_config = {
        "class": QAReranker,
        "config": {
            "rank_key": [("rank", False)],
            "reranking_scheme": {
                "recall_ranking_score_threshold": 0.75,
                "recall_ranking_top_n": 2,
                                 },
            "database_path": "../data/database20240506.csv",
        }
    }

    pipeline_config = {
        "router": keywords_config,
        "recall": recall_config,
        "merger": merge_config,
        "ranker": rank_config,
        "reranker": rerank_config,
    }

    qa_pipeline = QAPineline(pipeline_config)

    for i in range(test.shape[0]):

        query = test["raw_question"].iloc[i]
        results = qa_pipeline.run(query)
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


test_qa_pipeline_online()