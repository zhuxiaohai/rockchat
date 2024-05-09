import json
import pandas as pd
from merge import QAMerge

key_map = {"reason": "info", "result": "id", "similarities": "score"}
one_channel_map = {"vector_search": "raw"}
one_duplication_map = {"vector_search": "duplicated"}
two_channel_map = {"bm25": "recall_list_bm25", "vector_search": "recall_list_bge"}
two_duplication_map = {"bm25": "recall_bm25", "vector_search": "recall_bge"}


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


def test_merge_one_channel():
    test = pd.read_csv("data/data_merge_one_recall_channel.csv")
    config = {
        "vector_search": 1,
    }

    merger = QAMerge(config)
    for i in range(test.shape[0]):
        for channel_name in config:
            phase = "duplication_" + channel_name
            recalled_results = json.loads(test[one_channel_map[channel_name]].iloc[i])
            results = merger.duplicate_channel(recalled_results)
            ground_truth_list = json.loads(test[one_duplication_map[channel_name]].iloc[i])
            check_results(i, phase, results, ground_truth_list)

        phase = "merging"
        recalled_results = {channel_name: json.loads(test[one_channel_map[channel_name]].iloc[i])
                            for channel_name in config}
        results = merger.merge(recalled_results)
        ground_truth_list = json.loads(test["merged"].iloc[i])
        check_results(i, phase, results, ground_truth_list)


def test_merge_two_channels():
    test = pd.read_csv("data/data_merge_two_recall_channel.csv")
    config = {
        "vector_search": 0.9,
        "bm25": 0.84,
    }

    merger = QAMerge(config)
    for i in range(test.shape[0]):
        for channel_name in config:
            phase = "duplication_" + channel_name
            recalled_results = json.loads(test[two_channel_map[channel_name]].iloc[i])
            results = merger.duplicate_channel(recalled_results)
            ground_truth_list = json.loads(test[two_duplication_map[channel_name]].iloc[i])
            check_results(i, phase, results, ground_truth_list)

        phase = "merging"
        recalled_results = {channel_name: json.loads(test[two_channel_map[channel_name]].iloc[i])
                            for channel_name in config}
        results = merger.merge(recalled_results)
        ground_truth_list = json.loads(test["recall_all"].iloc[i])
        check_results(i, phase, results, ground_truth_list)


test_merge_two_channels()