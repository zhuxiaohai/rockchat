def get_first_available_gpu():
    import subprocess
    # 使用nvidia-smi命令获取当前可用的GPU列表
    result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                            capture_output=True,
                            text=True)
    if result.returncode == 0:
        available_gpus = result.stdout.strip().split('\n')
        # 转换为整数列表
        available_gpus = [int(gpu) for gpu in available_gpus]
        # 返回列表中的第一个GPU
        if available_gpus:
            return available_gpus[0]
    # 如果没有可用的GPU或者命令执行失败，返回None
    return None
first_available_gpu = get_first_available_gpu()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = f'{first_available_gpu}'
import pandas as pd
from rank import QAScorer
import json

key_map = {"result": "id"}


def test_qa_ranker():
    test = pd.read_csv("data/data_rank.csv")
    df = pd.read_csv("../data/database20240506.csv")

    config = {
        "model_path": "/workspace/data/private/zhuxiaohai/models/bge_finetune_reranker_question_top20",
        "query_key": "query_cleaned",
        "item_key": "question",
        "database_path": "../data/database20240506.csv",
    }

    ranker = QAScorer(config)
    for i in range(test.shape[0]):
        query = {"query_cleaned": test["question_cleaned"].iloc[i]}
        recalls = json.loads(test["recall_bge"].iloc[i])
        recall_question = json.loads(test["recall_question"].iloc[i])
        for j in range(len(recalls)):
            assert recall_question[j] == df.loc[df["qa_id"] == recalls[j]["result"], "question"].iloc[0]
        recalls = [{key_map[key]: item[key] for key in key_map} for item in recalls]
        results = ranker.compute_ranking_score(query, recalls)
        ground_truth_list = json.loads(test["score"].iloc[i])

        try:
            assert len(results) == len(ground_truth_list)
        except:
            print("item number mismatch", i)

        results = [item["rank"]for item in results]
        try:
            assert (sorted(range(len(results)), key=lambda k: results[k]) ==
                    sorted(range(len(ground_truth_list)), key=lambda k: ground_truth_list[k]))
        except:
            print("item score mismatch", i)


def test_qa_ranker_online():
    test = pd.read_csv("data/data_rank.csv")
    df = pd.read_csv("../data/database20240506.csv")

    config = {
        "online": True,
        "model_path": "http://localhost:8501/rerank/",
        "query_key": "query_cleaned",
        "item_key": "question",
        "database_path": "../data/database20240506.csv",
    }

    ranker = QAScorer(config)
    for i in range(test.shape[0]):
        query = {"query_cleaned": test["question_cleaned"].iloc[i]}
        recalls = json.loads(test["recall_bge"].iloc[i])
        recall_question = json.loads(test["recall_question"].iloc[i])
        for j in range(len(recalls)):
            assert recall_question[j] == df.loc[df["qa_id"] == recalls[j]["result"], "question"].iloc[0]
        recalls = [{key_map[key]: item[key] for key in key_map} for item in recalls]
        results = ranker.compute_ranking_score(query, recalls)
        ground_truth_list = json.loads(test["score"].iloc[i])

        try:
            assert len(results) == len(ground_truth_list)
        except:
            print("item number mismatch", i)

        results = [item["rank"]for item in results]
        try:
            assert (sorted(range(len(results)), key=lambda k: results[k]) ==
                    sorted(range(len(ground_truth_list)), key=lambda k: ground_truth_list[k]))
        except:
            print("item score mismatch", i)


test_qa_ranker()