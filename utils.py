import re
import jieba
import Levenshtein


def find_closest_string(a, string_list):
    closest_string = None
    min_distance = float('inf')

    for s in string_list:
        distance = Levenshtein.distance(a, s)
        if distance < min_distance:
            min_distance = distance
            closest_string = s

    return closest_string, min_distance


def find_non_chinese_substrings(s):
    # 正则表达式解释：
    # [^\u4e00-\u9fff\W]+ 匹配非中文字符和非ASCII标点的连续字符
    # 但这样会排除空格，所以我们需要允许空格存在
    # 我们使用(?:[^\u4e00-\u9fff\W]| )+ 来实现这一点，(?:) 是非捕获组，用于匹配模式但不作为捕获结果返回
    # [^\u4e00-\u9fff\W] 匹配非中文且非标点的字符，| 表示或，空格 ' ' 被显式允许
    pattern = r'(?:[^\u4e00-\u9fff\W]| )+'

    # 使用findall方法查找所有匹配项
    matches = re.findall(pattern, s)

    # 过滤掉只包含空格的字符串
    matches = [match for match in matches if not match.isspace()]

    return matches


def find_non_chinese_substrings_with_pos(s):
    pattern = r'(?:[^\u4e00-\u9fff\W]| )+(?:上下水(?:版(?:本)?)?)?'
    matches = re.finditer(pattern, s)

    substrings_with_positions = []
    for match in matches:
        start, end = match.span()
        substring = match.group()

        # 去除左右两边的空格并调整位置
        stripped_substring = substring.strip()
        start += len(substring) - len(substring.lstrip())
        end -= len(substring) - len(substring.rstrip())

        if stripped_substring and not stripped_substring.isspace():
            substrings_with_positions.append((stripped_substring, start, end))

    return substrings_with_positions


def clean_string(s):
    s = s.replace(" ", "").lower()
    return s


def find_error_with_reason(a):
    # 第一次匹配“错误xxx”
    pattern1 = r"错误\s*\d+"
    matches1 = re.findall(pattern1, a)

    # 第二次匹配“错误原因xxx”
    pattern2 = r"错误原因\s*\d+"
    matches2 = re.findall(pattern2, a)

    # 合并两次匹配的结果
    matches = matches1 + matches2

    return [name.replace(" ", "").replace("原因", "") for name in matches]


def find_error_with_reason_with_pos(a):
    # 第一次匹配“错误xxx”
    pattern1 = r"错误\s*\d+"
    matches1 = [(match.group(), match.start(), match.end()) for match in re.finditer(pattern1, a)]

    # 第二次匹配“错误原因xxx”
    pattern2 = r"错误原因\s*\d+"
    matches2 = [(match.group(), match.start(), match.end()) for match in re.finditer(pattern2, a)]

    # 合并两次匹配的结果
    matches = matches1 + matches2

    # 处理匹配结果，去掉空格和“原因”
    results = [{"word": name.replace(" ", "").replace("原因", ""), "start": start, "end": end}
               for name, start, end in matches]

    return results


def find_model(x, all_model_list):
    x = x.replace("\n", "")
    x = find_non_chinese_substrings(x)
    result = [clean_string(s) for s in x]
    return [model for model in all_model_list if model in result]


def find_model_with_pos(x, all_model_list):
    substrings_with_positions = find_non_chinese_substrings_with_pos(x)
    results = []

    for substring, start, end in substrings_with_positions:
        cleaned_substring = clean_string(substring)
        cleaned_substring = cleaned_substring.replace("版本", "").replace("版", "")
        if cleaned_substring in all_model_list:
            results.append({"word": cleaned_substring, "start": start, "end": end})
            x = x.replace(substring, "")

    return results, x


def find_cat(x, all_cat_list):
    return [name for name in all_cat_list if name in x]


def find_cat_with_pos(x, all_cat_list):
    positions = []
    for name in all_cat_list:
        for match in re.finditer(re.escape(name), x):
            start = match.start()
            end = match.end()
            positions.append({"word": name, "start": start, "end": end})
    return positions


def remove_model_name(x, all_model_list):
    x = x.replace("\n", "")
    candidates = find_non_chinese_substrings(x)
    for name in candidates:
        if clean_string(name) in all_model_list:
            x = x.replace(name, "")
    return x


class WordCut:
    def __init__(self):
        with open('/data/dataset/kefu/hit_stopwords.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
            con = f.readlines()
            stop_words = set()
            for i in con:
                i = i.replace("\n", "")  # 去掉读取每一行数据的\n
                stop_words.add(i)
        self.stop_words = stop_words

    def cut(self, text):
        # jieba.load_userdict('自定义词典.txt')  # 这里你可以添加jieba库识别不了的网络新词，避免将一些新词拆开
        # jieba.initialize()  # 初始化jieba
        # 文本预处理 ：去除一些无用的字符只提取出中文出来
        # new_data = re.findall('[\u4e00-\u9fa5]+', mytext, re.S)
        # new_data = " ".join(new_data)
        # 匹配中英文标点符号，以及全角和半角符号
        pattern = r'[\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65\u2018\u2019\u201c\u201d\u2026\u00a0\u2022\u2013\u2014\u2010\u2027\uFE10-\uFE1F\u3001-\u301E]|[\.,!¡?¿\-—_(){}[\]\'\";:/]'
        # 使用 re.sub 替换掉符合模式的字符为空字符
        new_data = re.sub(pattern, '', text)
        # 文本分词
        seg_list_exact = jieba.lcut(new_data)
        result_list = []
        # 去除停用词并且去除单字
        for word in seg_list_exact:
            if word not in self.stop_words and len(word) > 1:
                result_list.append(word)
        return result_list


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