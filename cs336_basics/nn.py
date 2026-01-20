import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
import math
from einops import rearrange


class Linear(nn.Module):
    """
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        #将device 和dtype传入进去
        factory_kwargs = {'device':device,'dtype':dtype}
        #1. 定义权重W(shape:out_features,in_features)
        #这里不用bias
        self.weight = nn.Parameter(torch.empty((out_features,in_features), **factory_kwargs))
        #2. 初始化权重
        #按照3.4.1要求
        std = (2.0/(in_features + out_features))**0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std =std,a = -3*std,b = 3*std)
    def forward(self,x:Tensor) -> torch.Tensor:
        # return x @ self.weight.T

        #... 表示任意多个 batch 维度，这样更通用
        #解读 "bi,oi->bo"
        # bi: x 的维度 (batch, in_features)
        # oi: weight 的维度 (out_features, in_features)
        # ->bo: 输出维度 (batch, out_features)
        # 相同字母 i 会被求和（这就是乘法+求和）
        return torch.einsum('...i,oi -> ...o', x, self.weight)


class Embedding(nn.Module):
    """
    num_embeddings: int Size of the vocabulary
    embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    
    """

    def __init__(self,num_embeddings:int,   #词汇表大小
                 embedding_dim:int,         #d_model
                 device = None, dtype = None):
        super().__init__()
        #
        #将device 和dtype传入进去
        factory_kwargs = {'device':device,'dtype':dtype}

        #1. 定义权重W(shape:vocab,d_model)
        #我们可以想象它是一本“字典”，每一个id对应一行d_model

        #nn.Parameter是为了告诉Pytorch:这个张量是模型的一部分，需要通过训练来学习的weight
        #nn.empty只分配内存，不做初始化
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim), **factory_kwargs))

        #2. 初始化权重

        nn.init.trunc_normal_(self.weight,mean=0.0,std = 1.0,a=-3.0,b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]#返回[B,S,D]
class RMSNorm(nn.Module):
    """
    d_model: int Hidden dimension of the model
    eps: float = 1e-5 Epsilon value for numerical stability
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device':device,'dtype':dtype}

        # 1.初始化为1
        self.weight = nn.Parameter(torch.ones((d_model),**factory_kwargs))
        self.eps = eps


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape(batch_size, sequence_length, d_model) 
        """
        #x :(B,S,D)
        in_dtype = x.dtype
        #转换为float32防止平方计算时溢出
        x = x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1,keepdim=True)
        RMS = torch.sqrt(ms+self.eps)


        return ((x/RMS) * self.weight).to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # W1: (d_model → d_ff) 用于 Swish 分支
        # W2: (d_model → d_ff) 用于 gate 分支  
        # W3: (d_ff → d_model) 输出投影
        self.d_ff = d_ff
        self.d_model = d_model

        self.w1 = Linear(d_model,d_ff,device,dtype)

        self.w3 = Linear(d_model,d_ff,device,dtype)

        self.w2 = Linear(d_ff,d_model,device,dtype)

        
    def forward(self, x: Tensor) -> Tensor:
        # 1. swish_out = Swish(x @ W1)
        # 2. gate_out = x @ W2
        # 3. return (swish_out * gate_out) @ W3
        # 方法 1：保存中间结果
        hidden = self.w1(x)
        swish_out = hidden * torch.sigmoid(hidden)

        gate_out = self.w3(x)

        return self.w2(swish_out * gate_out)



class RoPE(nn.Module):
    """
        Constructthe RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k

        #1.计算频率 omega_k = theta^(-2k/d)
        #我们只需要计算dk/2个频率，因为旋转是成对的
        #arange(0,d_k,2)产生[0,2,4,...,d_k-2]，对应公式中的2k-2(k从1开始)
        power = torch.arange(0,d_k,2,device=device).float() /d_k
        freq = 1.0/(theta**power)

        #2.创建序列位置[0,1,2,.....,max_seq_len - 1 ]
       #形状(max_seq_len)
        position = torch.arange(max_seq_len,device=device)
        #3.计算所有位置的所有角度,使用外积
        #shape:(max_seq_len,d_k/2)
        freq_matrix = torch.outer(position,freq)
        #预计算cos和sin并作为buffer注册
        self.register_buffer("cos_cached",freq_matrix.cos(),False)
        self.register_buffer("sin_cached",freq_matrix.sin(),False)

  
       
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #1 提取cos/sin
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时，才需要手动插入 Head 维。
        # 对于 test_rope 这种 3D x vs 2D cos 的情况，PyTorch 会自动左侧补 1，无需操作。
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)


         
        #3.运算
        #x的shape:[B,h,s,d_head]，h为head的数量
        #假设x = [x1,x2,x3,x4]
        x_even = x[...,0::2] #取d_head维度的偶数x1 = [x0,x2]
        x_odd = x[...,1::2] #x2 = [x1,x3]
        output = torch.empty_like(x)
        output[...,0::2] = x_even * cos - x_odd * sin
        output[...,1::2] = x_odd * cos + x_even * sin
        return output





def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 1. 为了数值稳定性，减去指定维度上的最大值
    # dim=-1 通常是 Transformer 中的隐藏层或词表维度
    x_max = torch.max(x,dim=dim,keepdim= True).values
    x_scaled = x - x_max
    exp_x = torch.exp(x_scaled)
    sum_x = torch.sum(exp_x,dim=dim,keepdim= True)

    return (exp_x/sum_x)

def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor = None
) -> torch.Tensor:
    
    """
    参数：
        Q;[...,n,d_k]   n为查询序列长度
        K:[...,m,d_k]   m为键值序列长度  
        V:[...,m,d_v]
        mask:[n,m]bool矩阵，Fasle为屏蔽，True为保留
    """
    d_k =Q.size(-1)
    scores = torch.einsum("...nd,...md -> ...nm",Q,K)/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask ==False,float('-inf'))#-inf表示把相应False上的数换成一个极小值

    probs = softmax(scores,-1) #shape[...,n,m]
   
    output = torch.einsum("...nm,...mk->...nk",probs,V)

    return output


class CausalSelfAttention(nn.Module):
    def __init__(self,d_model:int, num_heads:int, max_seq_len=None,theta=None,
                 device = None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0 #校验d_model是不是num_heads的倍数

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
  

        #如果有theta，我们再初始化一下RoPE

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta,self.d_k,max_seq_len,device=device)

        else:
            self.rope = None
    
    def forward(self,x : torch.Tensor, token_positions : torch.Tensor = None) -> torch.Tensor:
        b,s,d = x.shape 

        #1线性投影并且拆分多头
        #将长度为d的特征维拆为(h d_k)，并将h维移动到s之前
        q = rearrange(self.q_proj(x),"... s (h d) -> ... h s d",h = self.num_heads)
        k = rearrange(self.k_proj(x),"... s (h d) -> ... h s d",h = self.num_heads)
        v = rearrange(self.v_proj(x),"... s (h d) -> ... h s d",h = self.num_heads)

        #3.作用旋转位置编码
        if self.rope is not None:
            if token_positions is None:
                #默认从0开始
                #expand处理batch维度

                token_positions = torch.arange(s,device=x.device).expand(b,s)

            #对QK进行旋转
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)
        #mask为下三角
        mask = torch.tril(torch.ones(s,s,device=x.device,dtype=torch.bool))

        #4.attention(B,h,s,d_k)
        attn_out = scaled_dot_product_attention(q,k,v,mask)

        #5.合并多头并输出投影

        out = rearrange(attn_out,"... h s d -> ... s (h d)")

        #输出投影
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self,d_model:int ,num_heads : int, d_ff:int,max_seq_len:int,
                 theta:float,device = None, dtype =None):
        super().__init__()
        self.attn = CausalSelfAttention(
            d_model,num_heads,max_seq_len,theta,device,dtype
        )
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)

        self.ffn = SwiGLU(d_model,d_ff,device=device,dtype=dtype)
    
    def forward(self,x:torch.Tensor,token_position:torch.Tensor=None):

        x = self.attn(self.ln1(x),token_position) + x

        x = self.ffn(self.ln2(x)) + x

        return x
    
class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,   # 词汇表大小
                 max_seq_len: int,  # 最大序列长度
                 d_model: int,      # 模型隐藏层维度
                 num_heads: int,    # 注意力头数
                 d_ff: int,         # FFN 中间层维度
                 num_layers: int,    # Transformer Block 层数
                   
                 theta: float,      # RoPE 的 theta 参数
                 device=None,
                 dtype=None
                 ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        #embedding layer
        self.token_embeddings = Embedding(vocab_size,d_model,device,dtype)

        #block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,max_seq_len,
                                 theta,device,dtype)
                                 for _ in range(num_layers)
                                ])

        #post-norm
        self.ln_final = RMSNorm(d_model,device = device,dtype=dtype)

        self.lm_head = Linear(d_model,vocab_size,device,dtype)
    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:

        b, s = token_ids.shape  # [B,S]

        # 准备位置信息用于RoPE.  [S] ->[1,S] ->[B,S]
        if token_positions is None:
            token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)

        x = self.token_embeddings(token_ids)  # [B,S,d_model]

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)

        x = self.lm_head(x)
        return x



        




