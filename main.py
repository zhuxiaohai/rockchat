from keyword import Labeller
from recall import get_recall_configured_channels
from merge import duplicate_channel, merge_channels
from rank import Ranker
from rerank import rerank


def pipeline(query, labeller_config, recall_config, merge_config, rank_config, rerank_config):
    # get all labels
    router = Labeller(labeller_config)
    labels = router.extract_keywords(query)

    # recall
    recall_channels = get_recall_configured_channels(recall_config)
    recalled_results = {channel_name: worker.query_recalls(query, labels) for channel_name, worker in recall_channels}

    # merge
    duplicated_recall_results = {channel_name: duplicate_channel(recalled_results[channel_name])
                                 for channel_name in recalled_results}
    merged_results = merge_channels(duplicated_recall_results, merge_config)

    # rank
    ranker = Ranker(rank_config)
    ranked_results = ranker.compute_ranking_score(query, merged_results)

    # rerank
    if rerank_config:
        ranked_results = rerank(ranked_results)

    return ranked_results
