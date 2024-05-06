from abc import ABC, abstractmethod
from collections.abc import Iterable
import heapq
import re
import pandas as pd
from utils import find_error_with_reason
from fastbm25 import fastbm25


class FastBM25(fastbm25):
    def __init__(self, corpus):
        super().__init__(corpus)

    def top_k_sentence(self, document, k=1, filter_indices=None):
        assert isinstance(document, Iterable), 'document is not iterable'
        score_overall = {}
        for word in document:
            if word not in self.document_score:
                continue
            for key, value in self.document_score[word].items():
                if filter_indices and (key not in filter_indices):
                    continue
                if key not in score_overall:
                    # print(score_overall)
                    score_overall[key] = value
                else:
                    score_overall[key] += value
        k_keys_sorted = heapq.nlargest(k, score_overall, key=score_overall.__getitem__)
        return [{"doc": self.corpus[item], "index": item, "score": score_overall.get(item, None)}
                for item in k_keys_sorted]

class WordCut:
    def __init__(self, all_model_list=None):
        with open('/data/dataset/kefu/hit_stopwords.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
            con = f.readlines()
            stop_words = set()
            for i in con:
                i = i.replace("\n", "")  # 去掉读取每一行数据的\n
                stop_words.add(i)
        self.stop_words = stop_words
        self.all_model_list = all_model_list

    def cut(self, mytext):
        # jieba.load_userdict('自定义词典.txt')  # 这里你可以添加jieba库识别不了的网络新词，避免将一些新词拆开
        # jieba.initialize()  # 初始化jieba
        # 文本预处理 ：去除一些无用的字符只提取出中文出来
        # new_data = re.findall('[\u4e00-\u9fa5]+', mytext, re.S)
        # new_data = " ".join(new_data)
        # 匹配中英文标点符号，以及全角和半角符号
        pattern = r'[\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65\u2018\u2019\u201c\u201d\u2026\u00a0\u2022\u2013\u2014\u2010\u2027\uFE10-\uFE1F\u3001-\u301E]|[\.,!¡?¿\-—_(){}[\]\'\";:/]'
        # 使用 re.sub 替换掉符合模式的字符为空字符
        new_data = re.sub(pattern, '', mytext)
        new_data = transform_model_name(new_data, self.all_model_list)
        # 文本分词
        seg_list_exact = jieba.lcut(new_data)
        result_list = []
        # 去除停用词并且去除单字
        for word in seg_list_exact:
            if word not in self.stop_words and len(word) > 1:
                result_list.append(word)
        return result_list


class PandasSearchEngine(ABC):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.df = pd.read_csv(config["database_path"]).reset_index(drop=True)
        self.build_index(config["index_columns"])
        if config["score_model"]["type"] == "bm25":
            self.tokenizer = config["score_model"]["tokenizer"]
            document_list = [self.tokenizer.cut(doc) for doc in self.df[config["embedding_col"]]]
            self.score_function = config["score_model"]["class"](document_list)

    def build_index(self, columns):
        self.index = {}
        for col in columns:
            inv_dict = {}
            for i in range(self.df.shape[0]):
                row = self.df[col].iloc[i]
                for entity in row.split(","):
                    if entity in inv_dict:
                        inv_dict[entity].add(i)
                    else:
                        inv_dict[entity] = {i}
            self.index[col] = inv_dict

    def get_filters(self, labels):
        filters = {}
        for col in labels:
            if isinstance(labels[col]) is not dict:
                continue
            indices = set()
            for entity in labels[col]:
                indices = self.index[col][entity] | indices
            filters[col] = indices
        return filters

    @abstractmethod
    def search(self, body):
        pass


class QASearchEngine(PandasSearchEngine):
    def __init__(self, config):
        super().__init__(config)
        self.preprocess()

    def preprocess(self):
        self.df["error_list"] = self.df.question.apply(lambda x: ",".join(find_error_with_reason(x)))

    def search(self, body):
        query = body["query"]
        if self.config["score_model"]["type"] == "bm25":
            query = self.tokenizer.cut(query)
        filters = self.get_filters(body["labels"])
        top_n = body["top_n"]

        def pattern1(main_filter):
            if len(main_filter) == 0:
                return
            results = []
            intersection_indices = main_filter & filters["error_list"]
            if len(intersection_indices) > 0:
                result = self.score_function.top_k_sentence(query, filter_indices=intersection_indices)
                results += result
                difference_indices = main_filter - filters["error_list"]
                result = self.score_function.top_k_sentence(query, filter_indices=difference_indices)
                results += result
            else:
                result = self.score_function.top_k_sentence(query, filter_indices=main_filter)
                results += result
                result = self.score_function.top_k_sentence(query, filter_indices=filters["error_list"])
                results += result
            difference_indices = filters["cat_list"] - main_filter
            result = self.score_function.top_k_sentence(query, filter_indices=difference_indices)
            results += result
            return results

        def pattern2(main_filter):
            if len(main_filter) == 0:
                return
            results = []
            all_set = set(range(self.df.shape[0]))
            result = self.score_function.top_k_sentence(query, filter_indices=main_filter)
            results += result
            difference_filter = all_set - main_filter
            result = self.score_function.top_k_sentence(query, filter_indices=difference_filter)
            results += result
            return results

        def add_info():
            for result in result_list:
                doc_id = result["id"]
                reasons = []
                for col in filters:
                    if doc_id in filters[col]:
                        reasons.append(col)
                result["info"] = "|".join(reasons)

        result_list = []
        if len(filters["model_list"]) > 0:
            result = pattern1(filters["model_list"])
            result_list += result
        else:
            result = pattern1(filters["cat_list"])
            result_list += result
        if len(result_list) == 0:
            if len(filters["error_list"]) > 0:
                result = pattern2(filters["error_list"])
                result_list += result
            else:
                result = self.score_function.top_k_sentence(query, top_n)
                result_list += result
        add_info()

        return result_list

