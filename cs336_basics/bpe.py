import regex
from collections import Counter
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 按照特殊token分割
def cut(input_path, special_tokens):
    a = [regex.escape(s) for s in special_tokens]
    b = "|".join(a)
    result = regex.split(b, input_path)
    return result

# 输入字符串，输出b'字节'
def to_byte_tuple(s):
    data = s.encode("utf-8")
    list_result = [data[i:i+1] for i in range(len(data))]
    return tuple(list_result)

def get_frequency(text):
    c = Counter()
    for m in regex.finditer(PAT, text):
        match = m.group()
        byte_match = to_byte_tuple(match)
        c[byte_match] += 1
    return c

#数 pair
def count_pair(table):
    c = Counter()
    for pre_token, count in table.items(): 
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i+1])
            c[pair] += count
    return c

def merge_one(t, pair):
    new_list = []
    i = 0
    while i < len(t):
        if i + 1 < len(t) and t[i] == pair[0] and t[i+1] == pair[1] :                     # 见下面三点
            new_list.append(t[i] + t[i+1])
            i += 2
        else:
            new_list.append(t[i])
            i += 1
    return tuple(new_list)

def merge(table, pair):
    new_c = Counter()
    for pre_token, count in table.items():
        new_token = merge_one(pre_token, pair)
        new_c[new_token] += count
    return new_c

def train_bpe(input_path, vocab_size, special_tokens):

    # 打开文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 建立字典，塞进去special tokens和256个字节
    vocab = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    offset = len(special_tokens)
    for i in range(256):
        vocab[offset + i] = bytes([i])

    # 建立初始频次表
    table = Counter()
    for seg in cut(text, special_tokens):
        table = table + get_frequency(seg)

    # 主循环 反复挑最高频pair合并
    num_merges = vocab_size - len(vocab)
    merges = []
    for _ in range(num_merges):
        counts = count_pair(table)
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        table = merge(table, best)

    return vocab, merges