import re

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