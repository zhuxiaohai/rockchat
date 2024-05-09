from abc import ABC, abstractmethod
import copy
from utils import ranking_metric


class MergeBase(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def merge(self, recall_channels):
        pass


class QAMerge(MergeBase):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def duplicate_channel(self, results):
        num_result = len(results)
        results_dict = {}
        for j in range(num_result):
            id = results[j]["id"]
            score = results[j]["score"]
            info = results[j]["info"]
            if id in results_dict:
                if (ranking_metric(info) <= ranking_metric(results_dict[id]["info"])
                ) & (score > results_dict[id]["score"]):
                    results_dict.update({id: {
                        "score": score,
                        "info": info,
                    }})
            else:
                results_dict.update({id: {
                    "score": score,
                    "info": info,
                }})
        return [{
            "id": id,
            "score": results_dict[id]["score"],
            "info": results_dict[id]["info"],
        } for id in results_dict]

    def merge(self, recalled_results):
        recall_channels = {channel_name: self.duplicate_channel(recalled_results[channel_name])
                           for channel_name in recalled_results}

        results_dict = {}
        weights_sum = sum(list(self.config.values()))
        for channel_name in self.config:
            weight = self.config[channel_name]
            for item in recall_channels[channel_name]:
                id = item["id"]
                if id in results_dict:
                    if ranking_metric(item["info"]) < ranking_metric(results_dict[id]["info"]):
                        results_dict[id]["info"] = item["info"]
                    results_dict[id]["score"] += weight * item["score"] / weights_sum
                    results_dict[id]["recall"].update({channel_name: copy.deepcopy(item)})
                else:
                    results_dict[id] = {
                        "info": item["info"],
                        "score": weight * item["score"] / weights_sum,
                        "recall": {channel_name: copy.deepcopy(item)},
                                  }
        return [{
            "id": id,
            "score": results_dict[id]["score"],
            "info": results_dict[id]["info"],
            "recall": results_dict[id]["recall"],
        } for id in results_dict]