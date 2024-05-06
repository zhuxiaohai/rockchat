import jieba
import re


def extract_models(s, all_model_list):
    # 匹配所有非中文、非标点符号的子串，以及可选的特定字眼（air等）
    pattern = re.compile(r"([a-zA-Z0-9]+(?:\s*(?:air|pro|auto|plus|ultra|ultrae|neo|pure))?)", re.IGNORECASE)

    # 使用finditer()找到所有匹配项和它们的位置
    matches = [(match.group(), match.start(), match.end()) for match in pattern.finditer(s)]

    # 过滤空字符串
    matches = [match for match in matches if match[0].strip()]

    # 替换匹配到的模式
    models = []
    for content, start, end in reversed(matches):
        content = content.replace(" ", "").lower()
        for i, model in enumerate(all_model_list):
            if content == model:
                if model not in models:
                    models.append(model)
                s = s[:start] + f"model_{i}" + s[end:]
                break

    return models, s


def extract_versions(s):
    # 定义一个正则表达式，用于优先匹配更长的版本词
    pattern0 = re.compile(r'(标准|正常|普通|一般)(版本|型号|版|型)', re.IGNORECASE)
    pattern1 = re.compile(r'(上下水)(版本|型号|版|型)', re.IGNORECASE)

    # 使用findall找出所有匹配的版本
    matches0 = [(match.group(), match.start(), match.end()) for match in pattern0.finditer(s)]
    matches1 = [(match.group(), match.start(), match.end()) for match in pattern1.finditer(s)]

    # 过滤空字符串
    matches0 = [match for match in matches0 if match[0].strip()]
    matches1 = [match for match in matches1 if match[0].strip()]

    # 替换匹配到的模式
    versions = []
    if matches0:
        versions.append("标准版")
    for _, start, end in reversed(matches0):
        s = s[:start] + f'version_0' + s[end:]

    if matches1:
        versions.append("上下水版")
    for _, start, end in reversed(matches1):
        s = s[:start] + f'version_1' + s[end:]

    return versions, s


class WordCut:
    def __init__(self, stop_words_path="/data/dataset/kefu/hit_stopwords.txt",
                 entity_path=None):
        self.stop_words = None
        # 可根据需要打开停用词库，然后加上不想显示的词语
        if stop_words_path is not None:
            with open(stop_words_path, encoding="utf-8") as f:
                lines = f.readlines()
                stop_words = set()
                for line in lines:
                    # 去掉读取每一行数据的\n
                    line = line.replace("\n", "")
                    stop_words.add(line)
            self.stop_words = stop_words
        if entity_path is not None:
            # 这里你可以添加jieba库识别不了的网络新词，避免将一些新词拆开
            jieba.load_userdict(entity_path)
            # 初始化jieba
            jieba.initialize()

    def cut(self, text):
        # 文本预处理 ：去除一些无用的字符只提取出中文出来
        # new_data = re.findall('[\u4e00-\u9fa5]+', mytext, re.S)
        # new_data = " ".join(new_data)
        # 匹配中英文标点符号，以及全角和半角符号
        pattern = (r"[\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65\u2018\u2019\u201c\u201d\u2026\u00a0\u2022\u2013\u2014\u2010\u2027\uFE10-\uFE1F\u3001-\u301E]|[\.,!¡?¿\-—_(){}[\]\'\";:/]")
        # 使用 re.sub 替换掉符合模式的字符为空字符
        text = re.sub(pattern, '', text)
        # 文本分词
        token_list = jieba.lcut(text)
        result_list = []
        # 去除停用词并且去除单字
        for word in token_list:
            if word not in self.stop_words and len(word) > 1:
                result_list.append(word)
        return result_list
