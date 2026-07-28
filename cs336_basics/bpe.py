import regex
from collections import Counter, defaultdict
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

# 以上已经实现了这道题的逻辑，以下是加速部分
def build_index(table):
    pair2count = Counter()
    pair2tokens = defaultdict(set)
    for pre_token, count in table.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i+1])            
            pair2count[pair] += count         
            pair2tokens[pair].add(pre_token)      
    return pair2count, pair2tokens

#merge循环加速

# 遍历一个pre_token的所有相邻pair
def pairs_of(t):
    return [(t[i], t[i+1]) for i in range(len(t) - 1)]

def train_bpe_fast(input_path, vocab_size, special_tokens): 
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

    # 建两张索引表
    pair2count, pair2tokens = build_index(table)
    merges = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair2count:
            break
        best = max(pair2count, key=lambda p: (pair2count[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        affected = list(pair2tokens[best])   # 快照，避免遍历时改集合出错
        for old in affected:
            count = table[old]
            new = merge_one(old, best)
            for p in pairs_of(old): 
                pair2count[p] -= count
                pair2tokens[p].discard(old)
                if pair2count[p] == 0:
                    del pair2count[p]
            for p in pairs_of(new):
                pair2count[p] += count
                pair2tokens[p].add(new)


            del table[old]
            table[new] += count

    return vocab, merges









# 快速验证
if __name__ == "__main__":
    # to_byte_tuple
    assert to_byte_tuple("low") == (b'l', b'o', b'w')
    assert to_byte_tuple("牛") == (b'\xe7', b'\x89', b'\x9b')

    # get_frequency
    assert get_frequency("low low low") == {
        (b'l', b'o', b'w'): 1,
        (b' ', b'l', b'o', b'w'): 2,
    }

    # 共用的小语料
    table = {
        to_byte_tuple("low"): 5,
        to_byte_tuple("lower"): 2,
        to_byte_tuple("widest"): 3,
        to_byte_tuple("newest"): 6,
    }

    # count_pair
    assert count_pair(table)[(b'e', b's')] == 9
    assert count_pair(table)[(b's', b't')] == 9

    # merge
    assert merge(table, (b's', b't')) == {
        (b'l', b'o', b'w'): 5,
        (b'l', b'o', b'w', b'e', b'r'): 2,
        (b'w', b'i', b'd', b'e', b'st'): 3,
        (b'n', b'e', b'w', b'e', b'st'): 6,
    }

    # build_index —— 新增，本轮要验的
    pair2count, pair2tokens = build_index(table)
    assert pair2count[(b'e', b's')] == 9
    assert pair2count[(b's', b't')] == 9
    assert pair2tokens[(b'e', b's')] == {to_byte_tuple("widest"), to_byte_tuple("newest")}
    assert pair2tokens[(b'l', b'o')] == {to_byte_tuple("low"), to_byte_tuple("lower")}

    # 对拍：fast 和朴素版必须一致
    v1, m1 = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    v2, m2 = train_bpe_fast("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    assert m1 == m2, f"merges 不一致！第一处分叉：{next(i for i in range(min(len(m1),len(m2))) if m1[i]!=m2[i])}"
    assert v1 == v2, "vocab 不一致"
    print("对拍通过：fast == naive")

    print("all ok")



