import regex as re  # 使用 regex 而非内置 re，因为它支持 Unicode 类别（如 \p{L}）
from collections.abc import Iterable


class BPETokenizer:
    """
    字节级 BPE（Byte-Pair Encoding）分词器实现。
    
    该分词器将任意字符串编码为整数 ID 序列，并能将 ID 序列还原。
    它采用字节级处理，确保不会出现未知词（OOV）错误。
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        初始化分词器。
        
        参数:
            vocab: 词汇表，建立整数 ID 到 字节块(bytes) 的映射。
            merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b)，
                   表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序。
            special_tokens: 特殊标记列表（如 <|endoftext|>），这些标记不会被 BPE 规则拆分。
        """
        # 1. 建立双向映射，方便查表
        self.vocab = vocab  # ID -> 字节块
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()} # 字节块 -> ID
        
        # 2. 将合并规则转换为Rank字典。
        # BPE 编码时，必须优先应用在训练阶段较早出现的合并规则。
        # 字典结构为: {(byte_a, byte_b): 顺序索引}
        self.merges = {pair: i for i, pair in enumerate(merges)}    #转换为dict
        
        self.special_tokens = special_tokens or []

        #3.构建特殊token的正则表达式

        if self.special_tokens:
            #[1]
            #必须按照从长到短排序
            #这样正则引擎会优先匹配最长的特殊标记，防止重叠标记(如<|a|><|b|>)被错误拆分
            sorted_special = sorted(self.special_tokens,key=len,reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special)

            self.special_regex = re.compile(special_pattern)

        else:
            self.special_regex = None
        
        #4. GPT2官方预分词表达

        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self,text: str ) -> list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。
        
        该方法的核心逻辑是：
        1. 作为一个“协调者”，它负责处理文本中的“特殊标记（Special Tokens）”和“普通文本”。
        2. 特殊标记（如 <|endoftext|>）被视为原子，直接映射为 ID，不参与 BPE 的拆分和合并。
        3. 普通文本片段则被交给底层逻辑执行预分词和 BPE 算法。
        
        参数:
            text: 需要编码的原始字符串（例如 "Hello<|end|>World"）。
            
        返回:
            list[int]: 编码后的整数 ID 序列。
        """

        #--- 1.边界条件检查 ---
        #如果输入是空字符串或 None，直接返回空列表。
        if not text:

            return []
        



        #--- 2.情况A：不包含special_token ---
        if not self.special_regex:

            return self._encode_text_segment(text)
        


        #--- 3.情况B:包含special_token ---
        tokens = []
        # parts = re.split(f"({self.special_regex.pattern})",text)
        last_post = 0


        #[2]finditer
        #finditer可以定位到每一个符合special_word正则表达式，也就是可以定位到特殊token
        #我们还可以知道match_start和match_end
        for match in self.special_regex.finditer(text):
            pre_text = text[last_post:match.start()]

            if pre_text:
                #[3]extend vs append
                tokens.extend(self._encode_text_segment(pre_text))
                

            #处理当前标记,我们直接加入tokens中
            special_tok = match.group()
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])
            last_post = match.end()

        #收尾文本
        remaining_text = text[last_post:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens
    
    def _encode_text_segment(self,text:str) -> list[int]:
        """
        内部核心函数：对不含special_token的纯文本片段应用BPE合并逻辑
        """
        ids = []
        # parts = re.split(f"{self.special_regex.pattern}",text)

        # 使用 GPT-2 正则进行预分词，将文本拆成单词/标点符号块
        # 例如："Hello world!" -> ["Hello", " world", "!"]

        pre_tokens = self.gpt2_pat.findall(text)

        #遍历所有词
        for p_tok in pre_tokens:
            #第一步：将当前片段转环卫字节序列，并将每一个字节看作一个独立part
            # 例如："Hello" -> [b'H', b'e', b'l', b'l', b'o']
            bytes_parts = [bytes([b])for b in p_tok.encode("utf-8")]


            #第二部：反复执行合并，直到没有符合条件的合并规则为止

            while len(bytes_parts)>=2:
                
                best_pair = None
                min_rank = float('inf')

                for i in range(len(bytes_parts)-1):
                    pair = (bytes_parts[i],bytes_parts[i+1])

                    if pair in self.merges:
                        rank = self.merges[pair]

                        if rank <min_rank:
                            min_rank = rank
                            best_pair = pair
                #如果找不到任何合并规则，则推出当前片段合并
                if best_pair is None:
                    break

                #第三步：执行合并并操作

                new_byte_parts= []
                i = 0

                # [b'H', b'e', b'l', b'l', b'o', b'H', b'e'] -> [b'He', b'l', b'l', b'o', b'He']
                while i < len(bytes_parts):
                    #如果当前两个部分匹配最高优先规则

                    if i < len(bytes_parts) - 1 and (bytes_parts[i],bytes_parts[i+1]) == best_pair:
                        new_byte_parts.append(best_pair[0]+best_pair[1])
                        i+=2 #跳过下一项因为已经合并了

                    else:
                        new_byte_parts.append(bytes_parts[i])
                        i+=1
                bytes_parts = new_byte_parts #更新序列进入下一轮wihle
            
            for part in bytes_parts:
                ids.append(self.byte_to_id[part])
        
        return ids
    
    def decode(self,ids:list[int]) -> str:

        """"
        将ID表转换为文本
        """
        # 1. 根据文本查找字典
        segment = [self.id_to_byte[b] for b in ids ]
        #2.将所有的字节拼接起来
        full_bytes = b"".join(segment)
        
                #返回解码，replace是防止遇到没见过的id
        return full_bytes.decode("utf-8",errors="replace")


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代编码器。
        
        参数:
            iterable: 一个可迭代的字符串对象（例如文件句柄）。
        返回:
            一个生成器，逐个产出编码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        for chunk in iterable:
            # 对每一块文本进行编码，并通过 yield 吐出结果
            yield from self.encode(chunk)
