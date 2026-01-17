import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

class Linear(nn.Module):
    """
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
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

    def __init__(self,num_embeddings:int,embedding_dim:int,device = None, dtype = None):
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








