from abc import ABC, abstractmethod
import pandas as pd
from transformers import pipeline
from utils import (find_model, find_cat, find_error_with_reason, remove_model_name,
                   find_model_with_pos, find_cat_with_pos, find_error_with_reason_with_pos)


class KeywordBase(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def extract_keywords(self, query):
        pass


class LabellerByRules(KeywordBase):
    def __init__(self, config):
        super().__init__(config)
        self.model_col = config["model_col"]
        self.cat_col = config["cat_col"]
        self.error_col = config["error_col"]
        self.dim_df = pd.read_csv(config["dim_df_path"])
        self.all_model_list = self.dim_df[self.model_col[0]].tolist()
        self.all_cat_list = self.dim_df[self.cat_col[0]].unique().tolist()

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
            "query_cleaned": query_cleaned,
            "query": query,
            "labels": {
                self.model_col[1]: list(set(model_list)),
                self.cat_col[1]: list(set(cat_list)),
                self.error_col[1]: list(set(error_list))
            },
        }


class LabellerByRulesWithPos(KeywordBase):
    def __init__(self, config):
        super().__init__(config)
        self.model_col = config["model_col"]
        self.cat_col = config["cat_col"]
        self.error_col = config["error_col"]
        self.dim_df = pd.read_csv(config["dim_df_path"])
        self.all_model_list = self.dim_df[self.model_col[0]].tolist()
        self.all_cat_list = self.dim_df[self.cat_col[0]].unique().tolist()
        self.cat_name_mapping = self.dim_df.drop_duplicates(
            self.cat_col[0]).set_index(self.cat_col[0]).to_dict()[self.cat_col[1]]
        self.model_name_mapping = self.dim_df.drop_duplicates(
            self.model_col[0]).set_index(self.model_col[0]).to_dict()[self.cat_col[1]]
        self.ner_model = pipeline(
            "token-classification",
            model=config["ner_model_path"],
            aggregation_strategy="simple"
        )

    def extract_keywords(self, query):
        model_list, query_cleaned = find_model_with_pos(query, self.all_model_list)
        cat_list = find_cat_with_pos(query, self.all_cat_list)
        for cat in cat_list:
            cat["word"] = self.cat_name_mapping[cat["word"]]
        cat_list += [
            {"word": self.model_name_mapping[model["word"]],
             "start": model["start"],
             "end": model["end"]} for model in model_list
        ]
        error_list = find_error_with_reason_with_pos(query)
        entities = self.ner_model(query)
        return {
            "query_cleaned": query_cleaned,
            "query": query,
            "labels": {
                self.model_col[1]: model_list,
                self.cat_col[1]: cat_list,
                self.error_col[1]: error_list,
                "entities": [{
                    "word": query[entity["start"]:entity["end"]],
                    "start": entity["start"],
                    "end": entity["end"]
                } for entity in entities],
            },
        }
