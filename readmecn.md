# CS336 2025春季 Assignment 1：基础部分

想要查看作业的完整详细说明，请阅读作业讲义文件：
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

如果您发现讲义或代码有任何问题，欢迎随时在 GitHub 上提出 issue，或者直接提交 pull request 来修复。

## 环境搭建

### 环境管理
我们使用 `uv` 来管理环境，以确保可重现性、可移植性和使用上的便利性。  
安装 `uv` 的方式请参考：[https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)（强烈推荐），  
或者也可以使用以下命令安装：
```sh
pip install uv
brew install uv
```

我们建议稍微花点时间阅读一下 `uv` 的项目管理方式说明，这里：  
[https://docs.astral.sh/uv/guides/projects/#managing-dependencies](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)  
（看完你不会后悔的！）

现在你可以通过下面这条命令来运行仓库中的任何 python 文件，  
环境会自动解析并激活（非常方便）：
```sh
uv run <python_file_path>
```

### 运行单元测试

```sh
uv run pytest
```

一开始所有测试都应该会失败，并抛出 `NotImplementedError` 异常。  
要让你的实现能被测试调用，请完成文件  
[./tests/adapters.py](./tests/adapters.py)  
里面的函数，把你的代码“接”到测试系统里。

### 下载数据
下载 TinyStories 数据集，以及 OpenWebText 的一个子样本

```sh
mkdir -p data
cd data

# TinyStories 数据（GPT-4 版本）
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText 子样本
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

祝你完成作业顺利～  
这个作业虽然细节很多很磨人，但做完之后对现代大语言模型的底层理解会提升非常多，加油！
