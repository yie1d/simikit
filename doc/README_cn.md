## SimiKit: 用 Python 实现的图像相似度工具箱

[English](../README.md)  | 中文

## 概述

SimiKit 是一个常用图像相似度算法的工具箱。该项目提供各种工具来帮助开发者快速比对多种图像相似度算法的效果，帮助开发者选择一个最符合自己需求的图像相似度算法。

## 安装

> pip instal simikit

## 基础用法
```python
from simikit.features.hash import AHash

image_path = r'tests/image.png'
print(AHash().encode(image_path))
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
