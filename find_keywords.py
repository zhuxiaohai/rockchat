from abc import ABC, abstractmethod
import pandas as pd
from utils import find_model, find_cat, find_error_with_reason, remove_model_name


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