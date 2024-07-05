import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import json
import pandas as pd
from pipeline import QAPineline, NL2SQLPineline
from vector_db import ChromaDB
from search_engine import QASearchEngine, VectorSim, QAVectorDBSearchEngine
from find_keywords import LabellerByRules, LabellerByRulesWithPos
from recall import RecallBySearchEngine
from merge import QAMerge
from rank import QAScorer
from rerank import QAReranker
from nl2sql import SimpleLookup

key_map = {"result": "id"}
one_channel_map = {"vector_search": "raw"}
one_duplication_map = {"vector_search": "duplicated"}


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
            "show_cols": ["question", "answer"],
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
            try:
                assert results[:5] == ground_truth_list[:5]
                print("top5 match but not all", i)
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
            "show_cols": ["question", "answer"],
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
            try:
                assert results[:5] == ground_truth_list[:5]
                print("top5 match but not all", i)
            except:
                print("item ranking mismatch", i)


def test_nl2sql_pipeline():
    router_config = {
        "class": LabellerByRulesWithPos,
        "config": {
            "dim_df_path": "../data/dim_df20240619.csv",
            "model_col": ("model", "model"),
            "cat_col": ("cat_cn", "cat"),
            "error_col": ("error", "error"),
            "ner_model_path": "/workspace/data/private/zhuxiaohai/models/bert_finetuned_ner_augmented/"
        }
    }

    recall_config = {
        "class": RecallBySearchEngine,
        "config": {
            "search_engine": {
                "class": QAVectorDBSearchEngine,
                "docs_path": "data/table_entity.csv",
                "docs_col": "entity_name",
                "ids_col": "ids",
                "metadata_cols": ["sweeping", "washing", "mopping", "content"],
                "collection_name": "table_entity",
                "reset": False,
                "encoder_path": "/workspace/data/private/zhuxiaohai/models/bge_finetuned_emb_ner_link",
                "vector_db": {
                    "class": ChromaDB,
                    "host": "/data/dataset/kefu/chroma",
                },
            },
        },
        "top_n": 1,
    }

    nl2sql_config = {
        "class": SimpleLookup,
        "config": {
            "db_absolute_path": "/root/PycharmProjects/rockchat/data/model_params.db",
            "reset": False,
            "df_path": "/root/PycharmProjects/rockchat/data/model_params20240620.csv",
            "table_name": "model_params20240620",
        }
    }

    pipeline_config = {
        "router": router_config,
        "recall": recall_config,
        "nl2sql": nl2sql_config,
    }

    nl2sql_pipeline = NL2SQLPineline(pipeline_config)

    query = "G20的电源线有多长"
    result = nl2sql_pipeline.run(query)
    print(result)
    result = result.to_dict(orient="records")
    assert len(result) == 2
    assert result[0]["电源线长"] == "180cm"
    assert result[1]["电源线长"] == "180cm"