import pandas as pd
from find_keywords import LabellerByRules
import json


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


test_labeller()


