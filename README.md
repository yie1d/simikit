## SimiKit: Image Similarity Toolkit in Python


English | [中文](doc/README_cn.md) 

## Overview

SimiKit is a toolkit for commonly used image similarity algorithms. This project provides various tools to help developers quickly compare the effects of multiple image similarity algorithms, and assist developers in selecting an image similarity algorithm that best meets their needs.

## Installation

> pip instal simikit

## Basic Usage
```python
from simikit.features.hash import AHash

image_path = r'tests/image.png'
print(AHash().encode(image_path))
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
