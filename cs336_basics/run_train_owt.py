from cs336_basics.bpe import train_bpe_fast_parallel
import time
import pickle
import psutil
import resource

if __name__ == "__main__":
    train_path = "data/owt_train.txt"

    
    t0 = time.time()
    vocab, merges = train_bpe_fast_parallel(train_path, 32000, ["<|endoftext|>"])
    t1 = time.time()
    print(f"耗时 {t1 - t0:.1f} 秒")

    # 存盘
    with open("model/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("model/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


    # 最长 token（排除 special）
    special_bytes = [s.encode("utf-8") for s in ["<|endoftext|>"]]
    longest = max((v for v in vocab.values() if v not in special_bytes), key=len)
    print(f"最长 token: {longest!r}, 长度 {len(longest)} 字节")


    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS 上单位是字节
    print(f"峰值内存 {peak:.2f} GB")

