# CSCE-Net:Cross-Scale Context Extracted Hashing for Fine-Grained Image Binary Encoding (ACML 2022)

Official implementation of [CSCE-Net:Cross-Scale Context Extracted Hashing for Fine-Grained Image Binary Encoding](https://arxiv.org/abs/2210.07572#).

The paper has been accepted by **ACML 2022**.
## Introduction

Deep hashing has been widely applied to large-scale image retrieval tasks owing to efficient computation and low storage cost by encoding high-dimensional image data into binary codes. Since binary codes do not contain as much information as float features, the essence of binary encoding is preserving the main context to guarantee retrieval quality. However, the existing hashing methods have great limitations on suppressing redundant background information and accurately encoding from Euclidean space to Hamming space by a simple sign function. In order to solve these problems, a Cross-Scale Context Extracted Hashing Network (CSCE-Net) is proposed in this paper. Firstly, we design a two-branch framework to capture fine-grained local information while maintaining high-level global semantic information. Besides, Attention guided Information Extraction module (AIE) is introduced between two branches, which suppresses areas of low context information cooperated with global sliding windows. Unlike previous methods, our CSCE-Net learns a content-related Dynamic Sign Function (DSF) to replace the original simple sign function. Therefore, the proposed CSCE-Net is context-sensitive and able to perform well on accurate image binary encoding. We further demonstrate that our CSCE-Net is superior to the existing hashing methods, which improves retrieval performance on standard benchmarks.

<div align="center">
  <img width="90%"src="train_patch_fusion/pipeline.png"/>
</div>

## Codes

### Requirements

- NVIDIA GPU, Linux, Python3(tested on 3.6.10)
- Tested with CUDA 10.2, cuDNN 7.1 and PyTorch 1.8.0


### Training

1. Set datapath, model, training parameters in main.py and run 

```
python main.py 
```


## Citation
```bash
@article{xue2022cross,
  title={Cross-Scale Context Extracted Hashing for Fine-Grained Image Binary Encoding},
  author={Xue, Xuetong and Shi, Jiaying and He, Xinxue and Xu, Shenghui and Pan, Zhaoming},
  journal={arXiv preprint arXiv:2210.07572},
  year={2022}
}
```


