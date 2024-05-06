import copy


def rerank(ranked_results):
    topping_indices = []
    for i, item in enumerate(ranked_results):
        if (item["info"].find("model") >= 0) | (item["info"].find("error") >= 0) | (len(item["info"].split(",")) > 1):
            topping_indices.append(i)
    topping = [copy.deepcopy(ranked_results[i]) for i in topping_indices]
    topping.sort(key=lambda x: (x["ranking"], -x["score"]))
    reranked_results = []
    for i in range(len(ranked_results)):
        if i in topping_indices:
            reranked_results.append(topping[topping_indices.index(i)])
        else:
            reranked_results.append(copy.deepcopy(ranked_results[i]))
    return reranked_results
