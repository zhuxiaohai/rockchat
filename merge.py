def ranking_metric(x):
    if (x.find("error") >= 0) and (x.find("model") >= 0):
        return 1
    elif (x.find("error") >= 0) and (x.find("cat") >= 0):
        return 2
    elif (x.find("error") >= 0):
        return 3
    elif (x.find("model") >= 0):
        return 4
    elif (x.find("cat") >= 0):
        return 5
    else:
        return 6


def duplicate_channel(results):
    num_result = len(results)
    results_dict = {}
    for j in range(num_result):
        id = results[j]["id"]
        score = results[j]["score"]
        reason_code = results[j]["info"]
        if id in results_dict:
            if (ranking_metric(reason_code) <= ranking_metric(results_dict[id]["info"])
            ) & (score > results_dict[id]["score"]):
                results_dict.update({id: {
                    "score": score,
                    "info": reason_code,
                    "ranking": ranking_metric(reason_code)
                }})
        else:
            results_dict.update({id: {
                "score": score,
                "info": reason_code,
                "ranking": ranking_metric(reason_code)
            }})
    return [{"id": id,
             "score": results_dict[id]["score"],
             "info": results_dict[id]["info"],
             "ranking": results_dict[id]["ranking"]
             } for id in results_dict]


def merge_channels(recall_channels, weights):
    results_dict = {}
    for channel_name in weights:
        weight = weights[channel_name]
        for item in recall_channels[channel_name]:
            id = item["id"]
            if id in results_dict:
                if ranking_metric(item["info"]) < ranking_metric(results_dict[id]["info"]):
                    results_dict[id]["info"] = item["info"]
                    results_dict[id]["ranking"] = ranking_metric(item["info"])
                results_dict[id]["score"] += weight * item["score"] / sum(weights)
                results_dict[id]["full_reason"] = results_dict[id]["full_reason"]+","+item["info"]+"_"+channel_name
            else:
                results_dict[id] = {
                    "info": item["info"],
                    "score": weight * item["score"] / sum(weights),
                    "full_reason": item["info"]+"_"+channel_name,
                    "ranking": ranking_metric(item["info"])
                              }
    return [{"id": id,
             "score": results_dict[id]["score"],
             "info": results_dict[id]["info"],
             "full_reason": results_dict[id]["full_reason"],
             "ranking": results_dict[id]["ranking"]
             } for id in results_dict]