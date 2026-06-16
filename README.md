# SECOS: Semantic Capture for Rigorous Classification in Open-World Semi-Supervised Learning
This is the implementation of our CVPR'26 paper [SECOS](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_SECOS_Semantic_Capture_for_Rigorous_Classification_in_Open-World_Semi-Supervised_Learning_CVPR_2026_paper.pdf).

Before running the code, please make sure to:
1. Download and prepare the required datasets;
2. Download the pretrained weights of CLIP-ViT-H-14-laion2B-s32B-b79K released by OpenCLIP;
3. Properly configure the paths to the pretrained model and datasets.

```python
python train.py --dataset cifar10
```

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{liu2026secos,
  title={SECOS: Semantic Capture for Rigorous Classification in Open-World Semi-Supervised Learning},
  author={Liu, Hezhao and Yang, Jiacheng and Gao, Junlong and Li, Mengke and Zhang, Yiqun and Gowda, Shreyank N and Lu, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={39627--39636},
  year={2026}
}
```

## Acknowledgements
Our code framework refers to TRAILER (CVPR'24). Many thanks.
