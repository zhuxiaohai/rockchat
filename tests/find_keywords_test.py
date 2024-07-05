import json
import pandas as pd
from find_keywords import LabellerByRules, LabellerByRulesWithPos


def test_labeller():
    test = pd.read_csv("data/data_labeller_test.csv")
    config = {
        "dim_df_path": "../data/dim_df20240315.csv",
        "model_col": ("model", "model"),
        "cat_col": ("cat_name", "cat"),
        "error_col": ("error", "error"),
    }
    router = LabellerByRules(config)
    for i in range(test.shape[0]):
        query = test["question"].iloc[i]
        results = router.extract_keywords(query)
        ground_truth_dict = json.loads(test["labeller"].iloc[i])
        key = "query_cleaned"
        assert results[key] == ground_truth_dict[key]
        for key in ground_truth_dict.keys():
            if key != "query_cleaned":
                assert set(results["labels"][key]) == set(ground_truth_dict[key])


def test_labeller_wih_pos():
    test = pd.read_csv("data/data_table_qa.csv")
    config = {
        "dim_df_path": "../data/dim_df20240619.csv",
        "model_col": ("model", "model"),
        "cat_col": ("cat_cn", "cat"),
        "error_col": ("error", "error"),
        "ner_model_path": "/workspace/data/private/zhuxiaohai/models/bert_finetuned_ner_augmented/"
    }
    router = LabellerByRulesWithPos(config)
    for i in range(test.shape[0]):
        query = test["question"].iloc[i]
        results_all = router.extract_keywords(query)["labels"]

        results = results_all["model"]
        ground_truth = json.loads(test["model_list"].iloc[i])
        ground_truth = [{"word": item[0], "start": item[1], "end": item[2]} for item in ground_truth]
        results = sorted(results, key=lambda x: x["word"])
        ground_truth = sorted(ground_truth, key=lambda x: x["word"])
        try:
            assert results == ground_truth
        except:
            print("model extraction error", i)

        results = results_all["entities"]
        ground_truth = json.loads(test["keywords"].iloc[i])
        ground_truth = [{"word": item[0], "start": item[1], "end": item[2]} for item in ground_truth]
        results = sorted(results, key=lambda x: x["word"])
        ground_truth = sorted(ground_truth, key=lambda x: x["word"])
        try:
            assert results == ground_truth
        except:
            print("entities extraction error", i)

