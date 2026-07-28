import regex
from collections import Counter, defaultdict
import os
from typing import BinaryIO
from multiprocessing import Pool
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


def process_chunk(path, start, end, special_tokens):
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8", errors="ignore")

    table = Counter()
    for seg in cut(text, special_tokens):
        table = table + get_frequency(seg)

    return table

# 示例代码

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def build_table_parallel(path, special_tokens, num_procs=4):
    with open(path, "rb") as f:
        split_token = special_tokens[0].encode("utf-8")
        boundaries = find_chunk_boundaries(f, num_procs, split_token)


    tasks = [(path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(processes=num_procs) as pool:
        results = pool.starmap(process_chunk, tasks)

    total = Counter()
    for c in results:
        total = total + c
    return total


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



#加了个并行
def train_bpe_fast_parallel(input_path, vocab_size, special_tokens): 
    vocab = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    offset = len(special_tokens)
    for i in range(256):
        vocab[offset + i] = bytes([i])


    # 建立初始频次表（并行）
    table = build_table_parallel(input_path, special_tokens)

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







