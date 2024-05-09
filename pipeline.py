from abc import ABC, abstractmethod


class PipelineBase(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def run(self, query):
        pass


class QAPineline(PipelineBase):
    def __init__(self, config):
        super().__init__(config)
        self.router = config["router"]["class"](config["router"]["config"])
        recall_config = config["recall"]
        self.recall_config = recall_config
        self.recall_channels = {channel_name: recall_config[channel_name]["class"](recall_config[channel_name]["config"])
                                for channel_name in recall_config}
        self.merger = config["merger"]["class"](config["merger"]["config"])
        self.ranker = config["ranker"]["class"](config["ranker"]["config"])
        self.reranker = config["reranker"]["class"](config["reranker"]["config"])

    def run(self, query):
        query_body = self.router.extract_keywords(query)
        recalled_results = {}
        for channel_name in self.recall_channels:
            query_cleaned = query_body["query_cleaned"]
            search_body = {
                "query": query_cleaned,
                "top_n": self.recall_config[channel_name]["top_n"],
                "labels": query_body["labels"]
            }
            recalled_results[channel_name] = self.recall_channels[channel_name].query_recalls(search_body)
        merged_results = self.merger.merge(recalled_results)
        scored_results = self.ranker.compute_ranking_score(query_body, merged_results)
        ranked_results = self.reranker.rerank(scored_results)

        return ranked_results
