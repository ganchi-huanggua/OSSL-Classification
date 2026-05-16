# SECOS: Semantic Capture for Rigorous Classification in Open-World Semi-Supervised Learning
This is the implementation of our CVPR'26 paper ![SECOS](https://arxiv.org/pdf/2604.27596).

Before running the code, please make sure to:
1. Download and prepare the required datasets;
2. Download the pretrained weights of CLIP-ViT-H-14-laion2B-s32B-b79K released by OpenCLIP;
3. Properly configure the paths to the pretrained model and datasets.

```python
python train.py --dataset cifar10
```
## Acknowledgements
Our code framework refers to TRAILER (CVPR'24). Many thanks.
