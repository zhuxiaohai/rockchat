from abc import ABC, abstractmethod


class RecallChannelBase(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def query_recalls(self, query_body):
        pass


class RecallBySearchEngine(RecallChannelBase):
    def __init__(self, config):
        super().__init__(config)
        search_engine_class = config["search_engine"].pop("class")
        self.search_engine = search_engine_class(config["search_engine"])

    def query_recalls(self, query_body):
        results = self.search_engine.search(query_body)
        return results


def get_recall_channels(config):
    return {channel_name: config[channel_name]["class"](config[channel_name]["config"])
            for channel_name in config}