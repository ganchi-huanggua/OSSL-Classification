import os
import torch
import numpy as np
import random
import shutil
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy

class EMA:
    def __init__(self, model, decay=0.999, device=None):
        """
        初始化EMA对象（支持CUDA）
        Args:
            model: 待更新的模型（已移至GPU）
            decay: 衰减系数（0.999/0.9999常用）
            device: 设备（如torch.device('cuda:0')，默认与model相同）
        """
        # 确定设备（默认与原始模型一致）
        self.device = device if device is not None else next(model.parameters()).device
        
        # 复制模型参数作为EMA初始参数，并移至目标设备
        self.ema_model = deepcopy(model).to(self.device)
        
        # 冻结EMA模型参数（不参与梯度更新）
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        
        self.decay = decay
        self.epoch = 0

    def update(self, model, step=None):
        """
        更新EMA模型参数（确保在CUDA上操作）
        Args:
            model: 最新的模型（需在同一GPU上）
            step: 当前训练步数（用于预热）
        """
        # 检查设备是否匹配（避免CPU/GPU混合使用）
        if next(model.parameters()).device != self.device:
            raise RuntimeError(f"EMA设备({self.device})与模型设备({next(model.parameters()).device})不匹配")

        # 预热策略（可选）
        if step is not None and step < 2000:
            current_decay = min(self.decay, (1 + step) / (10 + step))
        else:
            current_decay = self.decay

        # 遍历参数，在CUDA上更新EMA（使用in-place操作提升效率）
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            # 公式：ema_param = decay * ema_param + (1 - decay) * model_param
            ema_param.data.mul_(current_decay).add_(model_param.data, alpha=1 - current_decay)

        # 同步缓冲区（如BN层的running_mean/running_var）
        for ema_buf, model_buf in zip(self.ema_model.buffers(), model.buffers()):
            ema_buf.data.copy_(model_buf.data.to(self.device))  # 确保缓冲区在同一设备

    def apply(self):
        """返回EMA模型（已在CUDA上）"""
        return self.ema_model

    def state_dict(self):
        """保存EMA模型的状态字典（包含CUDA参数）"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """加载EMA模型的状态字典（自动匹配设备）"""
        self.ema_model.load_state_dict(state_dict)

# class WeightEMA(object):
#     def __init__(self, alpha, model, ema_model):
#         self.model = model
#         self.ema_model = ema_model
#         self.alpha = alpha
#         self.params = list(model.state_dict().values())
#         self.ema_params = list(ema_model.state_dict().values())
#         # self.wd = 0.02 * args.lr

#         for param, ema_param in zip(self.params, self.ema_params):
#             ema_param.data.copy_(param.data)

#     def step(self):
#         one_minus_alpha = 1.0 - self.alpha
#         for param, ema_param in zip(self.params, self.ema_params):
#             if ema_param.dtype==torch.float32:
#                 ema_param.mul_(self.alpha)
#                 ema_param.add_(param * one_minus_alpha)

def linear_rampup(current, start ,end):
    """Linear rampup"""
    assert current >= 0 and end >= 0
    if current <= start:
        return 0
    elif current >= end:
        return 1.0
    else:
        return (current-start) / (end-start)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # import pdb; pdb.set_trace()
        try:
            correct_k = correct[:k].view(-1).float().sum(0)
        except:
            correct_k = correct[:k].reshape(-1).float().sum(0)
        try:
            res.append(correct_k.mul_(100.0 / batch_size))
        except:
            res = (torch.tensor(0.0), torch.tensor(0.0))
    return res


def save_checkpoint(state, is_best, save_path, tag='base'):
    filename=f'checkpoint_{tag}.pth.tar'
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, f'model_best_{tag}.pth.tar'))
# load checkpoint
# ckpt = torch.load(path, map_location="cpu")
# model.load_state_dict(ckpt['state_dict'], strict=False)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def sim_matrix(a, b, args, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def interleave(x, size):
    s = list(x.shape)
    # import pdb; pdb.set_trace()
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave_ssl(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def compute_tau(conf: torch.Tensor, alpha: float=0.9) -> float:
    """
    根据 alpha 分位数计算阈值 τ

    Args:
        conf (torch.Tensor): 形状为 [B, C] 的 softmax 置信度
        alpha (float): 分位数阈值，取值范围 [0, 1]，如 0.9 表示取前 90% 的最小值

    Returns:
        float: τ，softmax 最大值的 alpha 分位数
    """
    # 1. 每个样本的最大置信度
    max_conf = torch.max(conf, dim=1).values  # [B]

    # 2. 排序（升序）
    sorted_conf, _ = torch.sort(max_conf)

    # 3. 计算第 alpha 分位上的值
    idx = int(alpha * (len(sorted_conf) - 1))
    tau = sorted_conf[idx].item()

    return tau

def select_intra_candidate_labels(sorted_conf: torch.Tensor, sorted_indices: torch.Tensor, tau: float) -> list[list[int]]:
    """
    根据置信度和 τ，选出每个样本的候选伪标签集合。

    Args:
        conf (torch.Tensor): [B, C] 的 softmax 概率
        tau (float): 全局置信度累计阈值 ∈ [0, 1]

    Returns:
        List[List[int]]: 每个样本对应的候选标签索引列表
    """

    batch_size, num_classes = sorted_conf.shape
    candidate_labels = []

    for i in range(batch_size):
        cumsum = 0.0
        selected = []

        for j in range(num_classes):
            cumsum += sorted_conf[i, j].item()
            selected.append(sorted_indices[i, j].item())
            if cumsum >= tau:
                break

        candidate_labels.append(selected)

    return candidate_labels

def select_inter_candidate_labels(conf: torch.Tensor, beta: float) -> list[list[int]]:
    """
    对每一类设置置信度阈值，并筛选每个样本的候选伪标签集合

    Args:
        conf (torch.Tensor): [B, C] 的 softmax 概率
        beta (float): 类别级别的分位数阈值 ∈ [0, 1]

    Returns:
        List[List[int]]: 每个样本的候选类别列表
    """
    batch_size, num_classes = conf.shape
    conf_thresholds = []

    # 1~3. 为每一个类计算 β 分位置信度阈值 τ_c
    for c in range(num_classes):
        conf_c = conf[:, c]
        sorted_conf_c, _ = torch.sort(conf_c)
        idx = int(beta * (batch_size - 1))
        tau_c = sorted_conf_c[idx].item()
        conf_thresholds.append(tau_c)

    # 4~5. 为每一个样本选出所有满足条件的类别
    candidate_labels = []

    for i in range(batch_size):
        sample_labels = []
        for c in range(num_classes):
            if conf[i, c].item() >= conf_thresholds[c]:
                sample_labels.append(c)
        candidate_labels.append(sample_labels)

    return candidate_labels

def merge_candidate_labels(
    intra_candidates: list[list[int]],
    inter_candidates: list[list[int]]
) -> list[list[int]]:
    """
    合并两个候选伪标签集合，按样本逐一求并集

    Args:
        intra_candidates (List[List[int]]): 每个样本的 intra 伪标签集合
        inter_candidates (List[List[int]]): 每个样本的 inter 伪标签集合

    Returns:
        List[List[int]]: 每个样本最终伪标签集合（intra ∪ inter）
    """
    merged = []
    for intra, inter in zip(intra_candidates, inter_candidates):
        union = list(set(intra) & set(inter))
        # union.sort()  # 可选：排序方便可视化或一致性
        merged.append(union)
    return merged


def convert_to_one_hot(candidate_labels: list[list[int]], num_classes: int) -> torch.Tensor:
    """
    将候选伪标签集合转换为 one-hot 多标签张量。

    Args:
        candidate_labels (List[List[int]]): 每个样本的候选类别索引列表
        num_classes (int): 总类别数

    Returns:
        torch.Tensor: [B, C] 的 multi-hot 向量，1 表示该类是伪标签
    """
    batch_size = len(candidate_labels)
    one_hot = torch.zeros(batch_size, num_classes)

    for i, labels in enumerate(candidate_labels):
        if labels:  # 有伪标签
            one_hot[i, labels] = 1.0

    return one_hot
