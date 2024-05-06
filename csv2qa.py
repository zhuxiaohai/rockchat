import os
import json
import pandas as pd
import numpy as np
import random


template_simple_multiple_rows = """
请你根据以下json列表中的key和value，生成1个question和answer的对儿，我要利用它做数据集来微调大模型问答机器人。

生成的question和answer必须满足以下要求：
1 question必须用到“商品型号”这个key对应的value形成限定值，如果json里面有“版本”这个key则必须将它的value拼接在商品型号value后面以形成限定值，
为了语言的丰富多样，你可以从json里面选择一些key对应的value作为辅助值，但应确保整个question通顺简洁。
2 除此之外还要从“{}”这几个key里面挑选{}个作为询问的重要键，比如只挑选1个key电源线长度，又比如挑选多个key电源线长度和包装尺寸。
3 question必须为json列表里的每个json形成1个限定值且为每个json必须挑选1个或多个重要键。
4 question和answer中涉及的所有key、value必须从对应的json中提取并且一一对应，而不是凭空想象。
5 对每个question要生成{}个answer并且尽可能用不同的句式。

请用json输出你的结果，举例如下（如果json列表中有2个json）：
{"question": "[限定值0]和[限定值1]的[重要键0]和[重要键1]分别是多少？",
 "answer": ["[限定值0]的[重要键0]为[重要值0-0]、[重要键1]为[重要值0-1]，[限定值1]的[重要键0]为[重要值1-0]、[重要键1]为[重要值1-1]。",
            "[限定值0]的[重要键0]是[重要值0-0]、[重要键1]是[重要值0-1]，[限定值1]的[重要键0]是[重要值1-0]、[重要键1]是[重要值1-1]。",
            "[限定值0]和[限定值1]的[重要键0]分别是[重要值0-0]和[重要值1-0]、[重要键1]分别是[重要值0-1]和[重要值1-1]。"],
 "prompt": [{"primary_value0": "限定值0", "key0": "重要键0|重要键1", "重要键0": "重要值0-0", "重要键1": "重要值0-1"},
            {"primary_value1": "限定值1", "key1": "重要键0|重要键1", "重要键0": "重要值1-0", "重要键1": "重要值1-1"}],
 "replace": {"限定值0": "G20", "限定值1": "H1", "重要键0": "电源线", "重要键1": "功率",
             "重要值0-0": "1.25m", "重要值0-1": "10W", "重要值1-0": "1.2m", "重要值1-1": "20.5W"}
}
{"question": "[限定值0]的[重要键0-0]、[重要键0-1]和[限定值1]的[重要键1-0]、[重要键1-1]分别是多少？",
 "answer": ["[限定值0]的[重要键0-0]为[重要值0-0-0]、[重要键0-1]为[重要值0-0-1]，[限定值1]的[重要键1-0]为[重要值1-1-0]、[重要键1-1]为[重要值1-1-1]。",
            "[限定值0]的[重要键0-0]是[重要值0-0-0]、[重要键0-1]是[重要值0-0-1]，[限定值1]的[重要键1-0]是[重要值1-1-0]、[重要键1-1]是[重要值1-1-1]。",
            "[限定值0]和[限定值1]的[重要键0-0]和[重要键1-0]分别是[重要值0-0-0]和[重要值1-1-0]、[重要键0-1]和[重要键1-1]分别是[重要值0-0-1]和[重要值1-1-1]。"],
 "prompt": [{"primary_value0": "限定值0", "key0": "重要键0-0|重要键0-1", "重要键0-0": "重要值0-0-0", "重要键0-1": "重要值0-0-1"},
            {"primary_value1": "限定值1", "key1": "重要键1-0|重要键1-1", "重要键1-0": "重要值1-1-0", "重要键1-1": "重要值1-1-1"}],
 "replace": {"限定值0": "G20标准版", "重要键0-0": "电源线", "重要键0-1": "输入功率",
             "限定值1": "H1", "重要键1-0": "电源线", "重要键1-1": "输入功率",
             "重要值0-0-0": "1.25m", "重要值0-0-1": "25w",
             "重要值1-1-0": "1.25m", "重要值1-1-1": "25w"}
}

输入的json列表:
{}
"""


template_statistics = """
{"question": "[筛选键0][比较符0][筛选值0][逻辑运算符][筛选键1][比较符1][筛选值1]的[表0]是哪些？",
 "answer": ["[筛选键0][比较符0][筛选值0][逻辑运算符][筛选键1][比较符1][筛选值1]的[表0]有：
             [限定值0]、[筛选键0][筛选值0-0]、[筛选键1][筛选值0-1]，[限定值1]、[筛选键0][筛选值1-0]、[筛选键1][筛选值1-1]。"],
 "prompt": [{"primary_value0": "限定值0", "key0": "筛选键0|筛选键1", "筛选键0": "筛选值0-0", "筛选键1": "筛选值0-1"},
            {"primary_value1": "限定值1", "key1": "筛选键0|筛选键1", "筛选键0": "筛选值1-0", "筛选键1": "筛选值1-1"}],
 "replace": {"筛选键0": "电源线", "比较符0": "大于", "筛选值0": "1.2m", "逻辑运算符": "且",
             "筛选键1": "输入功率", "比较符1": "等于", "筛选值1": "15w",
             "表0": "扫地机器人", 
             "限定值0": "A20", "筛选值0-0": "1.3m", "筛选值0-1": "15w",
             "限定值1": "A20 Pro", "筛选值1-0": "1.5m", "筛选值1-1": "15w"}
}

{"question": "[筛选键0][比较符0][筛选值0][逻辑运算符][筛选键1][比较符1][筛选值1]的[表0]中，[重要键0][排序词0][排序数0]是哪些？",
 "answer": ["[筛选键0][比较符0][筛选值0][逻辑运算符][筛选键1][比较符1][筛选值1]的[表0]中，
             [重要键0][排序词0][排序数0]是：[限定值0]、[重要键0][重要值0-0]，[限定值1]、[重要键0][重要值1-0]。"],
 "prompt": [{"primary_value0": "限定值0", "key0": "重要键0", "重要键0": "重要值0-0"},
            {"primary_value1": "限定值1", "key1": "重要键0", "重要键0": "重要值1-0"}],
 "replace": {"筛选键0": "电源线", "比较符0": "大于", "筛选值0": "1.2m", "逻辑运算符": "且",
             "筛选键1": "输入功率", "比较符1": "等于", "筛选值1": "15w",
             "表0": "扫地机器人",
             "重要键0": "高度", "排序词0": "最大的", "排序数0": "2个",
             "限定值0": "A20", "重要值0-0": "50cm", "限定值1": "A20 Pro", "重要值1-0": "48cm"}
}

{"question": "[筛选键0][比较符0][筛选值0][逻辑运算符][筛选键1][比较符1][筛选值1]的[表0]中，[重要键0][排序词0]、[重要键1][排序词1]是多少？",
 "answer": ["[筛选键0][比较符0][筛选值0][逻辑运算符][筛选键1][比较符1][筛选值1]的[表0]中，
             [重要键0][排序词0]是[重要值0]、[重要键1][排序词1]是[重要值1]。"],
 "prompt": [{"primary_value0": "重要键0", "key0": "重要键0", "重要键0": "重要值0"},
            {"primary_value1": "重要键1", "key0": "重要键1", "重要键1": "重要值1"}],
 "replace": {"筛选键0": "电源线", "比较符0": "大于", "筛选值0": "1.2m", "逻辑运算符": "且",
             "筛选键1": "输入功率", "比较符1": "等于", "筛选值1": "15w",
             "表0": "扫地机器人",
             "重要键0": "高度", "排序词0": "最大值", "重要值0": "50cm",
             "重要键1": "功率", "排序词1": "最大值", "重要值1": "12wh"}
}


输入的json列表:
{}
"""


# 原始数据处理
def format_model(x):
    model_list = x.split(',')
    model_list = [i.strip().lower().replace(" ", "") for i in model_list]
    new_list = [model_list[0]]
    i = 1
    while i < len(model_list):
        if (i != len(model_list) - 1) and (model_list[i-1] == model_list[i]):
            new_list.append(model_list[i]+model_list[i+1])
            if i < len(model_list) - 1:
                i += 2
            else:
                break
        elif (i != len(model_list) - 1) and (model_list[i-1] != model_list[i]):
            new_list.append(model_list[i])
            i += 1
        elif (model_list[i] == "上下水") or (model_list[i] == "air"):
            for j in range(len(new_list)):
                if model_list[i-1] == new_list[j]:
                    new_list.pop(j)
                    break
            new_list.append(model_list[i-1]+model_list[i])
            i += 1
        else:
            new_list.append(model_list[i])
            break
    return new_list


def look_up_same_keywords_different_models(
        df,
        exclude_cols=["平台ID", "商品id", "商品编码", "商品分类", "商品名字", "版本", "商品型号"],
        query_num=1, keywords_num=1, selected_queries=[], selected_cols=[]
):
    primary_cols = ["商品型号"]
    if "版本" in df.columns:
        primary_cols.append("版本")
    df["primary_key"] = df[primary_cols].apply(lambda x: "".join([x[col] for col in primary_cols if not pd.isna(x[col])]), axis=1)
    query_cols = ["商品型号", "primary_key"]
    valid_columns = [col for col in df.columns if col not in exclude_cols + query_cols]

    if len(selected_queries) == 0:
        selected_queries = []
        col_indices = range(len(query_cols))
        row_indices = range(df.shape[0])
        for i in range(query_num):
            selected_col_indices = random.sample(col_indices, 1)[0]
            selected_row_indices = random.sample(row_indices, 1)[0]
            query = df[query_cols[selected_col_indices]].iloc[selected_row_indices]
            selected_queries.append((query_cols[selected_col_indices], query))
    else:
        query_num = len(selected_queries)

    if len(selected_cols) == 0:
        indices = range(len(valid_columns))
        selected_indices = random.sample(indices, keywords_num)
        selected_cols = [valid_columns[i] for i in selected_indices]
    else:
        keywords_num = len(selected_cols)

    question_template = ("和".join([f"[问询词{i}]" for i in range(query_num)]) + "的" +
                         "和".join([f"[关键词{i}]" for i in range(keywords_num)]) + "是什么？")

    question = "和".join([i[1] for i in selected_queries]) + "的" + "和".join(selected_cols) + "是什么？"
    replace = {k: v for k, v in zip([f"[问询词{i}]" for i in range(query_num)] +
                                    [f"[关键词{i}]" for i in range(keywords_num)],
                                    [i[1] for i in selected_queries] + selected_cols)}

    prompt = []
    for i in range(len(selected_queries)):
        for _, row in df.loc[df[selected_queries[i][0]] == selected_queries[i][1]].iterrows():
            result = {"primary_value": row["primary_key"], "key": "|".join(selected_cols)}
            for j in range(len(selected_cols)):
                result[selected_cols[j]] = row[selected_cols[j]]
            prompt.append(result)
    gen = {"question": question, "prompt": prompt, "replace": replace}

    return gen, question_template


def gen_same_keywords_different_models(output_path, seed=20):
    random.seed(seed)
    exclude_cols = ["平台ID", "商品id", "商品编码", "商品分类", "商品名字", "版本", "商品型号", "primary_key"]
    dir_name = "/data/dataset/kefu"
    result_list = []

    csv_name_list = ["sweeping", "mopping", "washing"]
    for csv_name in csv_name_list:
        csv_path = os.path.join(dir_name, csv_name+".csv")
        df = pd.read_csv(csv_path)
        df["商品型号"] = df["商品型号"].apply(lambda x: format_model(x)[0])
        primary_cols = ["商品型号"]
        if "版本" in df.columns:
            primary_cols.append("版本")
        df["primary_key"] = df[primary_cols].apply(
            lambda x: "".join([x[col] for col in primary_cols if not pd.isna(x[col])]), axis=1)
        selected_queries = [("primary_key", df.iloc[0]["primary_key"])]
        for col in df.columns:
            if col not in exclude_cols:
                selected_cols = [col]
                gen, question_template = look_up_same_keywords_different_models(
                    df,
                    exclude_cols=exclude_cols,
                    selected_queries=selected_queries,
                    selected_cols=selected_cols
                )
                result = {"cat": csv_name, "gen": gen, "template": question_template}
                result_list.append(result)

    csv_name = "knowledge"
    csv_path = os.path.join(dir_name, csv_name+".csv")
    df = pd.read_csv(csv_path)
    df["商品型号"] = df["商品型号"].apply(lambda x: format_model(x)[0])
    query_num_list = [2, 3]
    keywords_num_list = [1, 2, 3]
    for query_num in query_num_list:
        for keywords_num in keywords_num_list:
            gen, question_template = look_up_same_keywords_different_models(
                df,
                exclude_cols=exclude_cols,
                query_num=query_num,
                keywords_num=keywords_num
            )
            result = {"cat": csv_name, "gen": gen, "template": question_template}
            result_list.append(result)


    # 打开一个文件用于写入
    with open(output_path, 'w') as file:
        for result in result_list:
            # 将每个JSON对象转换为字符串并写入文件，每个对象占一行
            file.write(json.dumps(result) + '\n')


def look_up_different_keywords_different_models(
        df,
        exclude_cols=["平台ID", "商品id", "商品编码", "商品分类", "商品名字", "版本", "商品型号"],
        query_num=[1, 1], keywords_num=[1, 1],
        selected_queries=[], selected_cols=[]
):
    primary_cols = ["商品型号"]
    if "版本" in df.columns:
        primary_cols.append("版本")
    df["primary_key"] = df[primary_cols].apply(lambda x: "".join([x[col] for col in primary_cols if not pd.isna(x[col])]), axis=1)
    query_cols = ["商品型号", "primary_key"]
    valid_columns = [col for col in df.columns if col not in exclude_cols + query_cols]

    if len(selected_queries) == 0:
        selected_queries = []
        col_indices = range(len(query_cols))
        row_indices = range(df.shape[0])
        for query_num_segment in query_num:
            selected_queries_segment = []
            for i in range(query_num_segment):
                selected_col_indices = random.sample(col_indices, 1)[0]
                selected_row_indices = random.sample(row_indices, 1)[0]
                query = df[query_cols[selected_col_indices]].iloc[selected_row_indices]
                selected_queries_segment.append((query_cols[selected_col_indices], query))
            selected_queries.append(selected_queries_segment)
    else:
        query_num = [len(segment) for segment in selected_queries]

    if len(selected_cols) == 0:
        indices = range(len(valid_columns))
        selected_indices = [random.sample(indices, keywords_num_segment) for keywords_num_segment in keywords_num]
        selected_cols = [[valid_columns[i] for i in selected_indices_segment]
                         for selected_indices_segment in selected_indices]
    else:
        keywords_num = [len(segment) for segment in selected_cols]

    segment_list = []
    for i, (query_num_segment, keywords_num_segment) in enumerate(zip(query_num, keywords_num)):
        segment = ("、".join([f"[问询词{i}-{j}]" for j in range(query_num_segment)]) + "的" +
                   "、".join([f"[关键词{i}-{j}]" for j in range(keywords_num_segment)]))
        segment_list.append(segment)
    question_template = "和".join(segment_list) + "是什么？"

    segment_list = []
    replace = dict()
    for i, (selected_queries_segment, selected_cols_segment) in enumerate(zip(selected_queries, selected_cols)):
        segment = ("、".join([j[1] for j in selected_queries_segment]) + "的" +
                   "、".join(selected_cols_segment))
        replace |= {k: v for k, v in zip([f"[问询词{i}-{j}]" for j in range(len(selected_queries_segment))] +
                                         [f"[关键词{i}-{j}]" for j in range(len(selected_cols_segment))],
                                         [j[1] for j in selected_queries_segment] + selected_cols_segment)}
        segment_list.append(segment)
    question = "和".join(segment_list) + "是什么？"

    prompt = []
    for selected_queries_segment, selected_cols_segment in zip(selected_queries, selected_cols):
        for i in range(len(selected_queries_segment)):
            for _, row in df.loc[df[selected_queries_segment[i][0]] == selected_queries_segment[i][1]].iterrows():
                result = {"primary_value": row["primary_key"], "key": "|".join(selected_cols_segment)}
                for j in range(len(selected_cols_segment)):
                    result[selected_cols_segment[j]] = row[selected_cols_segment[j]]
                prompt.append(result)
    gen = {"question": question, "prompt": prompt, "replace": replace}

    return gen, question_template


def gen_different_keywords_different_models(output_path, seed=20):
    random.seed(seed)
    exclude_cols = ["平台ID", "商品id", "商品编码", "商品分类", "商品名字", "版本", "商品型号", "primary_key"]
    dir_name = "/data/dataset/kefu"
    result_list = []

    csv_name_list = ["sweeping", "mopping", "washing"]
    for csv_name in csv_name_list:
        csv_path = os.path.join(dir_name, csv_name+".csv")
        df = pd.read_csv(csv_path)
        df["商品型号"] = df["商品型号"].apply(lambda x: format_model(x)[0])
        primary_cols = ["商品型号"]
        if "版本" in df.columns:
            primary_cols.append("版本")
        df["primary_key"] = df[primary_cols].apply(
            lambda x: "".join([x[col] for col in primary_cols if not pd.isna(x[col])]), axis=1)
        selected_queries = [[("primary_key", df.iloc[0]["primary_key"])], [("primary_key", df.iloc[1]["primary_key"])]]
        valid_columns = [col for col in df.columns if col not in exclude_cols]
        num_cols = len(valid_columns)
        for i in range(1, num_cols):
            previous_col, col = valid_columns[i-1], valid_columns[i]
            selected_cols = [[previous_col], [col]]
            gen, question_template = look_up_different_keywords_different_models(
                df,
                exclude_cols=exclude_cols,
                selected_queries=selected_queries,
                selected_cols=selected_cols
            )
            result = {"cat": csv_name, "gen": gen, "template": question_template}
            result_list.append(result)

    csv_name = "knowledge"
    csv_path = os.path.join(dir_name, csv_name+".csv")
    df = pd.read_csv(csv_path)
    df["商品型号"] = df["商品型号"].apply(lambda x: format_model(x)[0])
    query_num_list = [2, 3]
    keywords_num_list = [1, 2, 3]
    for query_num in query_num_list:
        for keywords_num in keywords_num_list:
            gen, question_template = look_up_different_keywords_different_models(
                df,
                exclude_cols=exclude_cols,
                query_num=[query_num, query_num],
                keywords_num=[keywords_num, keywords_num]
            )
            result = {"cat": csv_name, "gen": gen, "template": question_template}
            result_list.append(result)


    # 打开一个文件用于写入
    with open(output_path, 'w') as file:
        for result in result_list:
            # 将每个JSON对象转换为字符串并写入文件，每个对象占一行
            file.write(json.dumps(result) + '\n')

# gen_same_keywords_different_models(output_path="/data/dataset/kefu/gen_same_keywords_for_models.jsonl", seed=20)
gen_different_keywords_different_models(output_path="/data/dataset/kefu/gen_different_keywords_different_models.jsonl", seed=20)