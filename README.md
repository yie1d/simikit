## SimiKit: Image Similarity Toolkit in Python


English | [中文](doc/README_cn.md) 

## Overview

SimiKit is a toolkit for commonly used image similarity algorithms. This project provides various tools to help developers quickly compare the effects of multiple image similarity algorithms, and assist developers in selecting an image similarity algorithm that best meets their needs.

## Installation
#### 1. Install by pip
```shell
pip install simikit
```

#### 2. Install by [uv](https://github.com/astral-sh/uv)
- 2.1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)(if not currently installed)
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS and Linux
# on Windows:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- 2.2. Clone the repository:
```shell
git clone https://github.com/yie1d/simikit.git  
cd simikit
```
- 2.3. Create a new virtual environment and activate it:
```shell
uv venv
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
```
- 2.4. Install dependencies:
```shell
uv sync
```

## Basic Usage

### 1. extract image features
```python
from simikit.features import AHash, Vit, DinoV2

print(DinoV2().encode('./t1.png'))
print(Vit().encode('./t1.png'))
print(AHash().encode('./t1.png'))

```

### 2. use comparator by multiple algorithms
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

## Supported Algorithms

- HASH
  - Average hashing
  - Difference hashing
  - Perceptual hashing
  - Wavelet hashing
- Transformer
  - VIT
  - DINOv2

## Contribution

Thank you for your interest in `simikit`. Submissions in all aspects are welcome. Let's work together to make `simikit` better!

## Future Plans

- Add more image similarity algorithms

If there is any similarity algorithm that you want but is not currently available in the `simikit`, you are welcome to raise it in the `Issues` section!
