import copy

class Ranker:
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model
        self.ranking_obj = config["ranking_obj"]

    def compute_ranking_score(self, query, recall):
        pairs = []
        for item in recall:
            pairs.append([query, item[self.ranking_obj]])
        scores = self.model.compute_score(pairs)
        ranked_results = []
        for i, item in enumerate(recall):
            item = copy.deepcopy.copy(item)
            item["ranking_score"] = scores[i]
            ranked_results.append(item)
        ranked_results.sort(key=lambda x: -x["ranking_score"])
        return ranked_results