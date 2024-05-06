from abc import ABC, abstractmethod
import re
import pandas as pd
from utils import find_non_chinese_substrings, clean_string


def find_model(x, all_model_list):
    x = x.replace("\n", "")
    x = find_non_chinese_substrings(x)
    result = [clean_string(s) for s in x]
    return [model for model in all_model_list if model in result]


def find_cat(x, all_cat_list):
    return [name for name in all_cat_list if name in x]


def find_error_with_reason(a):
    # 第一次匹配“错误xxx”
    pattern1 = r"错误\s*\d+"
    matches1 = re.findall(pattern1, a)

    # 第二次匹配“错误原因xxx”
    pattern2 = r"错误原因\s*\d+"
    matches2 = re.findall(pattern2, a)

    # 合并两次匹配的结果
    matches = matches1 + matches2

    return [name.replace(" ", "").replace("原因", "") for name in matches]


def remove_model_name(x, all_model_list):
    x = x.replace("\n", "")
    candidates = find_non_chinese_substrings(x)
    for name in candidates:
        if clean_string(name) in all_model_list:
            x = x.replace(name, "")
    return x


class KeywordBase(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def extract_keywords(self, query):
        pass


class Labeller(KeywordBase):
    def __init__(self, config):
        super().__init__(config)
        path = config["dim_df_path"]
        self.dim_df = pd.read_csv(path)
        self.all_model_list = self.dim_df.model.tolist()
        self.all_cat_list = self.dim_df.cat_name.unique().tolist()

    def extract_keywords(self, query):
        model_list = find_model(query, self.all_model_list)
        cat_list = find_cat(query, self.all_cat_list)
        cat_list += [
            cat for cat in
            self.dim_df.loc[self.dim_df.model.isin(model_list), 'cat_name'].tolist()
            if cat not in cat_list
        ]
        error_list = find_error_with_reason(query)
        query_cleaned = remove_model_name(query, self.all_model_list)
        return {
            "model_list": model_list,
            "cat_list": cat_list,
            "error_list": error_list,
            "query_cleaned": query_cleaned
        }