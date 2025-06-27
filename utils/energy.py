import torch

def energy(logits: torch.Tensor, T: float, top_k: int) -> torch.Tensor:
    """
    计算无标签样本的能量值（energy score）
    
    Args:
        logits (torch.Tensor): 模型输出的 logits，形状为 [N, C]
        T (float): 温度系数
        top_k (int): 使用前 K 个 known class（按列索引排序）
    
    Returns:
        torch.Tensor: 每个样本的 energy，形状为 [N]
    """
    # 只取前K类的logits
    logits_known = logits[:, :top_k]  # 假设前K列为known class
    logits_scaled = logits_known / T
    exp_logits = torch.exp(logits_scaled)
    sum_exp = torch.sum(exp_logits, dim=1)
    energy = -T * torch.log(sum_exp)
    return energy


def energy_discrepancy(logits: torch.Tensor, T: float, top_k: int) -> torch.Tensor:
    """
    计算 Energy Drop 指标，用于评估无标签样本是否属于 novel class。
    
    Args:
        logits (torch.Tensor): 模型输出的 logits，形状为 [N, C]
        T (float): 温度参数
        top_k (int): 使用前 K 个 known class
    
    Returns:
        torch.Tensor: 每个样本的 energy drop 值，形状为 [N]
    """
    # 只保留前 K 个 known class 的 logits
    logits_known = logits[:, :top_k]
    scaled_logits = logits_known / T

    # e^(f_j / T)
    exp_logits = torch.exp(scaled_logits)  # [N, K]

    # 第一项：T * log(sum(e^f/T))
    logsumexp_all = torch.log(torch.sum(exp_logits, dim=1))  # [N]
    term1 = T * logsumexp_all

    # 第二项：T * log(sum(e^f/T) - max(e^f/T))
    max_exp = torch.max(exp_logits, dim=1).values  # [N]
    sum_minus_max = torch.sum(exp_logits, dim=1) - max_exp + 1e-12  # avoid log(0)
    logsumexp_minus_max = torch.log(sum_minus_max)
    term2 = T * logsumexp_minus_max

    # 最终 ED = term1 - term2
    energy_drop = term1 - term2
    return energy_drop
