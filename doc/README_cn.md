## SimiKit: 用 Python 实现的图像相似度工具箱

[English](../README.md)  | 中文

## 概述

SimiKit 是一个常用图像相似度算法的工具箱。该项目提供各种工具来帮助开发者快速比对多种图像相似度算法的效果，帮助开发者选择一个最符合自己需求的图像相似度算法。

## 安装
#### 1. 通过pip直接安装   
```shell
pip install simikit
```

#### 2. 通过[uv](https://github.com/astral-sh/uv)安装
- 2.1. [安装uv](https://docs.astral.sh/uv/getting-started/installation/)(如果当前未安装)
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS and Linux
# on Windows:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- 2.2. 克隆存储库
```shell
git clone https://github.com/yie1d/simikit.git  
cd simikit
```
- 2.3. 创建虚拟环境并激活
```shell
uv venv
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
```
- 2.4. 安装依赖
```shell
uv sync
```

## 基础用法

### 1. 提取图像特征
```python
from simikit.features import AHash, Vit, DinoV2

print(DinoV2().encode('./t1.png'))
print(Vit().encode('./t1.png'))
print(AHash().encode('./t1.png'))

```

### 2. 通过多种算法使用比较器
```python
from simikit.api import Comparator
from simikit.features import AHash, DHash
from simikit.metrics import hamming_distance

comparator = Comparator([
    (DHash(16, vertical=True), hamming_distance),
    (AHash(16), hamming_distance),
    (AHash(8), hamming_distance),
])

print(comparator.compare_image(
    './t1.png',
    './t2.png',
))
```

## 支持的算法

- HASH
  - Average hashing
  - Difference hashing
  - Perceptual hashing
  - Wavelet hashing
- Transformer
  - VIT
  - DINOv2

## 贡献

感谢对`simikit`的关注，欢迎各个方面的提交，一起让`simikit`变的更好！

## 未来计划

- 增加更多的图像相似度算法

如果有任何你想要，但目前项目中没有的相似度算法，欢迎在`Issues`中提出来！
