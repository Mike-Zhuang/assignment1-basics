from cs336_basics.bpe import train_bpe_fast_parallel
import time
import pickle
import psutil

if __name__ == "__main__":
    train_path = "data/TinyStoriesV2-GPT4-train.txt"

    
    t0 = time.time()
    vocab, merges = train_bpe_fast_parallel(train_path, 10000, ["<|endoftext|>"])
    t1 = time.time()
    print(f"耗时 {t1 - t0:.1f} 秒")

    # 存盘
    with open("model/ts_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("model/ts_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


    # 最长 token（排除 special）
    special_bytes = [s.encode("utf-8") for s in ["<|endoftext|>"]]
    longest = max((v for v in vocab.values() if v not in special_bytes), key=len)
    print(f"最长 token: {longest!r}, 长度 {len(longest)} 字节")

    mem = psutil.Process().memory_info().rss / 1e9    # rss = 实际占用的物理内存，字节
    print(f"内存占用 {mem:.2f} GB")
