import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from search_engine import send_embedding_requests
from rank import send_rank_requests

key_map = {"result": "id"}


def test_embedding_model():
    test = pd.read_csv("data/data_vector_search.csv")
    model_path = "/workspace/data/private/zhuxiaohai/models/bge_finetune_emb"
    num_requests = 3
    model = SentenceTransformer(model_path)
    for i in range(test.shape[0]):
        query = test["question"].iloc[i]
        response = send_embedding_requests([query], num_requests, verbose=True)
        if len(response) == 1:
            response = response[0]
        embedding = model.encode(query, normalize_embeddings=True).tolist()
        assert response == embedding
    response = send_embedding_requests(test["question"].tolist(), num_requests)
    if len(response) == 1:
        response = response[0]
    embedding = model.encode(test["question"].tolist(), normalize_embeddings=True).tolist()
    assert len(response) == len(embedding)
    for i in range(len(response)):
        assert response[i] == embedding[i]


def test_rank_model():
    test = pd.read_csv("data/data_rank.csv")
    df = pd.read_csv("../data/database20240506.csv")
    model_path = "/workspace/data/private/zhuxiaohai/models/bge_finetune_reranker_question_top20"
    num_requests = 3
    model = FlagReranker(model_path, use_fp16=True)
    for i in range(test.shape[0]):
        query = {"query_cleaned": test["question_cleaned"].iloc[i]}
        recall = json.loads(test["recall_bge"].iloc[i])
        recall = [{key_map[key]: item[key] for key in key_map} for item in recall]
        pairs = []
        for item in recall:
            pairs.append((query["query_cleaned"], df.loc[df["qa_id"] == item["id"], "question"].iloc[0]))
        responses = send_rank_requests(pairs, num_requests, verbose=True)
        scores = model.compute_score(pairs)
        try:
            assert len(scores) == len(responses)
        except:
            print("item number mismatch", i)

        try:
            assert (sorted(range(len(scores)), key=lambda k: scores[k]) ==
                    sorted(range(len(responses)), key=lambda k: responses[k]))
        except:
            print("item score mismatch", i)


test_rank_model()