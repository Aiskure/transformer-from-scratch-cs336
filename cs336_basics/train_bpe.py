import os
from collections import defaultdict, Counter
import regex as re  # type: ignore
import json

def train_bpe(
    input_path: str | os.PathLike,  # 输入语料文件的路径
    vocab_size: int,             # 目标词表大小（基础字节 + 合并 Token + 特殊 Token）
    special_tokens: list[str],   # 需要保留的特殊 Token 列表，例如 ["<|endoftext|>"]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # --- 1. 初始化基础词表 ---

    #给初始词排序
    vocab = {i: bytes([i]) for i in range(256)}

    #我们目标词表大小
    num_merges = vocab_size - 256 - len(special_tokens)

    # --- 2. 读取并预处理语料 ---
    with open(input_path,"r",encoding="utf-8" ) as f:
        data = f.read()#用data来保存

    

    """
    For special_tokens:
    在训练时，必须保证特殊 Token 不参与频率统计。
    代码逻辑：
        切割语料：在开始统计词频之前，利用正则将语料库在特殊 Token 处切开。
        独立统计：只对切分出来的普通文本片段进行 BPE 统计。
        最后加入：训练结束后，强制将特殊 Token 加入词表（通常放在最后），确保它们有 ID。
    """


    if special_tokens:
        #构造正则表达式["<s>", "</s>"] -> "<s>|</s>"
        pattern = "|".join(map(re.escape, special_tokens))  #map(re.escape, special_tokens) → 对每个特殊 token 调用 re.escape()

        parts = re.split(f"({pattern})", data)#表示对data里面所有的特殊token进行pattern模式的切割



        # 过滤掉从 parts 中提取出的特殊 Token 本身，只保留用于 BPE 训练的普通文本片段。
        # text = "Hello World World<|endoftext|>Hello happy happy<|endoftext|>!"
        # train_segments =  ['Hello World World', 'Hello happy happy', '!']
        train_segments = [p for p in parts if p not in special_tokens]#for循环遍历parts，if保留了不在special_tokens中的p
    else:
        # 如果没有特殊 Token，直接将整个语料作为一个训练片段。
        train_segments = [data]




    # --- 3. 预分词（Pre-tokenization）并统计词频 ---
    # 使用 GPT-2 的 BPE 预分词正则表达式。
    # GPT-2 正则表达式的作用是执行“预分词（Pre-tokenization）”。 它的规则是：
    #   (1)不允许跨越类型合并：比如它会把字母和标点符号分开。
    #   (2)保护空格：它通常会把单词前面的空格和单词连在一起，作为一个整体。
    # text = "Hello World test! ..."
    # 分割后 words = ['Hello', ' World', ' test', '!', ' ...']



    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    #re.compile 在 Python 的作用是 把一个正则表达式字符串编译成一个正则对象，方便后续多次使用，提高性能和可读性。


    raw_counters = Counter()  # 用于统计词频的 Counter 对象
    for segment in train_segments:
        words = PAT.findall(segment)#对每个segment进行预分词，得到words列表
        for word in words:
            # 将单词转换为 UTF-8 字节序列，然后组成元组作为 Counter 的键, 统计这个元组出现的频次

            """
            为什么用tuple呢,因为counter的key必须是不可变类型,而list是可变类型
            
            举例：
                raw_counts = {
                    (b'H', b'i'): 50,
                    (b' ', b't', b'h', b'e', b'r', b'e'): 100,
                    (b'!'): 50,
                    (b'\xe4',b'\xbd',b'\xa0',b'\xe5',b'\xa5',b'\xbd'):20, # 你好
                }
            
            """
            
            byte_seq = word.encode("utf-8")
            symbols = tuple(bytes([b]) for b in byte_seq)#没有循环会变成 (b'Hello',)  有循环会变成 (b'H', b'e', b'l', b'l', b'o')

            raw_counters[symbols] += 1  # 统计词频


    # ---寻找最频繁的pairs---
    """"
    经过我们的处理,counters大概是这样样子:
    raw_counts = {
        (b'h', b'e', b'l', b'l', b'o'):100
        (b'h', b'i'):10,
        (b'h', b'i', b'g', b'h'):5
    }
    我们要开始寻找每一对相邻的符号出现了多少次,我们需要遍历每一个key
    """

    word_list = []
    counter_list = []
    for item,freq in raw_counters.items():
        word_list.append(list(item)) # 转换为 list 以便后面修改
        counter_list.append(freq)

    """
    word_list = [
    (b'h', b'e', b'l', b'l', b'o'),
    (b'h', b'i'),
    (b'h', b'i', b'g', b'h')
    ]

    counter_list = [100,10,5]
    word_list[i] 对应 counter_list[i]
    """

    """
    defaultdict(int) 是一个“带默认初始值”的字典。当你访问一个字典中不存在的键时，它不会报错，而是自动为这个键创建一个默认值 0，而在使用普通字典进行计数时，你必须先检查键是否存在，否则会触发 KeyError。
    stats: 存储所有可能的相邻字节对 (pair) 及其全局出现频率。

    """
    stats = defaultdict(int)#结构：{(byte_a, byte_b): frequency}
    
    # indices: 倒排索引。存储 pair -> {包含该 pair 的单词在 words_list 中的下标集合}
    # 这个结构是性能优化的关键，用于快速找到需要更新的单词。
    #在我们需要合并pair时就不需要遍历所有的单词，直接通过indices找到包含该pair的单词下标

    indices = defaultdict(set)

    """
    问题：为什么用set而不是list呢？
    因为set可以自动去重，避免重复添加同一个单词下标，
    比如一个单词中同一个pair出现多次，用list就会重复添加下标，浪费空间且影响性能
    """

    for idx,word in enumerate(word_list):
        
        freq = counter_list[idx]
        for a,b in zip(word, word[1:]):
            pair = (a,b)
            stats[pair] += freq  # 统计全局频率
            indices[pair].add(idx)  # 记录包含该 pair 的单词下标
    """
    stats(pair → 全局频率）
    stats = {   (b'h', b'e'): 100,      (b'e', b'l'): 100,
                (b'l', b'l'): 100,      (b'l', b'o'): 100,
                (b'h', b'i'): 15,       (b'i', b'g'): 5,
                (b'g', b'h'): 5
            }
    indices(pair → 出现该 pair 的 word_list 下标集合）
    indices = { (b'h', b'e'): {0},      (b'e', b'l'): {0},
                (b'l', b'l'): {0},      (b'l', b'o'): {0},
                (b'h', b'i'): {1, 2},   (b'i', b'g'): {2},
                (b'g', b'h'): {2}
            }

    
     """

    merges = []#记录合并
    #.    --- 4. 迭代合并  ---

    for _ in range(num_merges): #每次循环得到一个词
        #如果states为空，说明没有可以合并的pair了
        if not stats:
            break
        """
        key=lambda x: (x[1],x[0]) 的含义是：先比较 x[1]（频率），如果频率相同，再比较 x[0]（字节对本身的字典序）。
        """

        best_pair = max(stats.items(),key=lambda x: (x[1],x[0]))[0]  # 选择出现频率最高的字节对进行合并
        if stats[best_pair] <= 0:
            break#如果最高频次小于等于0，说明没有可以合并的pair了



        #记录合并对
        merges.append(best_pair)

        # 创建新的合并符号
        new_token = best_pair[0] + best_pair[1]  

        influenced_words = list(indices[best_pair])  # 获取所有包含该 pair 的单词下标

        for idx in influenced_words:
            word = word_list[idx] #获取单词
            freq = counter_list[idx]#获取该单词频次

            #对单词进行扫描，找到pair的位置
            i = 0
            while i < len(word) - 1:
                if word[i] ==best_pair[0] and word[i+1] ==best_pair[1]:#找到了要合并的pair
                    """ 此时我们找到了要合并的pair，我们需要进行以下操作：
                        更新单词：将该 pair 替换为新的合并符，我们要从该pair的左边和右边还有中间进行处理
                    """
                    #1.更新左右邻居的频率
                    #左邻居
                    if i>0:
                        left_pair = (word[i-1],word[i])
                        #更新 stats 和 indices，移除旧的左邻居 pair 贡献
                        stats[left_pair] -= freq
                        if stats[left_pair] == 0:
                            del stats[left_pair]#如果不移除，max时会出错
                        
                    #右邻居
                    if i < len(word)-2:
                        right_pair = (word[i+1],word[i+2])

                        stats[right_pair] -=freq

                        if stats[right_pair] ==0:
                            del stats[right_pair]

                    #2.添加新左右邻居的频率
                    word[i] = new_token
                    del word[i+1]

                    if i >0:
                        new_left = (word[i-1],word[i])
                        stats[new_left] +=freq
                        indices[new_left].add(idx) #添加新的倒排索引
                    #- 新的右邻居：(new_token, word[i+1])(注意：word[i+1] 是旧的 word[i+2])
                    if i < len(word) - 1:
                        new_right = (word[i],word[i+1])
                        stats[new_right] +=freq
                        indices[new_right].add(idx)

                    # 合并后，索引 i 指向的是新 Token。
                    # i 不需要移动（i+=1），因为我们刚刚修改了 word[i] 并且删除了 word[i+1]。
                    # 下一轮循环会检查新的 (word[i], word[i+1])，即 (new_token, old_word[i+2])
                    # 这可以处理像 A A A -> X A 这样的情况，正确地更新新的邻居对

                    
                else:i+=1
        #3.清理旧的best_pair
        if best_pair in stats:del stats[best_pair]
        if best_pair in indices:del indices[best_pair]

    
    # --- 5. 组成最终词表 ---
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]
    for i in special_tokens:
        token_byte = i.encode('utf-8')
        vocab[len(vocab)] = token_byte

    return vocab,merges











def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt" # 你的原始文本路径
    vocab_size = 10000 # 作业要求的词表大小
    # input_path = "data/owt_train.txt" 
    # input_path = "data/chinese.txt" 
    # vocab_size = 1000 # 作业要求的词表大小
    
    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")
    
    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    # 保存结果
    save_tokenizer_files(vocab, merges, output_dir)

if __name__ == "__main__":
    main()

    