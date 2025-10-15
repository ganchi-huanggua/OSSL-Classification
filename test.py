# ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# from torchvision.datasets import CIFAR10
# dataset = CIFAR10(root='/home/lhz/data', train=True, download=True)
# print(dataset.classes)

# a = [1,2 ,3 ]
# b = [1, 2, 3, 4]
# c = [1,2,3]
# z = zip(a, b, c)
# for idx, i in enumerate(z):
#     print(idx, i)

# import torch
# a = torch.randn((10, 3, 32, 32))
# print(a)
# mask = torch.randn(10) > 0.5 
# print(mask)
# print(a[mask].shape)


# d = dict()
# d[1] = 'a'
# d[2] = 'b'
# d[3] = 'c'
# print(1 in d.keys())
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_evaluate(predictions, targets, offset=0):
    # Hungarian matching
    targets = targets - offset
    predictions = predictions - offset
    predictions_np = predictions.numpy()
    num_elems = targets.size(0)

    # only consider the valid predicts. rest are treated as misclassification
    valid_idx = np.where(predictions_np>=0)[0]
    predictions_sel = predictions[valid_idx]
    targets_sel = targets[valid_idx]
    num_classes = torch.unique(targets).numel()
    num_classes_pred = torch.unique(predictions_sel).numel()

    match = _hungarian_match(predictions_sel, targets_sel, preds_k=num_classes_pred, targets_k=num_classes) # match is data dependent
    reordered_preds = torch.zeros(predictions_sel.size(0), dtype=predictions_sel.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions_sel == int(pred_i)] = int(target_i)

    # Gather performance metrics
    reordered_preds = reordered_preds.numpy()
    acc = int((reordered_preds == targets_sel.numpy()).sum()) / float(num_elems) if float(num_elems) else -1 #accuracy is normalized with the total number of samples not only the valid ones
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())
    
    return {'acc': acc*100, 'ari': ari, 'nmi': nmi, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


pred = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
truth = torch.tensor([1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3, 4, 4, 4])
print(hungarian_evaluate(pred, truth))

# mask = np.array([True, True, True, False, False])


# def accuracy(output, target, topk=(1,)):
#     """
#     计算准确率（支持一维 output 和 target）
#     - output: tensor, shape [batch_size]，每个元素是预测的类别索引
#     - target: tensor, shape [batch_size]，每个元素是真实类别索引
#     """
#     with torch.no_grad():
#         batch_size = target.size(0)

#         # 直接比较预测与标签
#         correct = output.eq(target).float().sum(0)

#         # 计算准确率（百分比）
#         acc = correct.mul_(100.0 / batch_size)

#         # 兼容原接口返回形式（列表）
#         return [acc]

# pred = np.array([1, 2, 3, 4, 5])
# truth = np.array([1, 4, 3, 2, 5])
# mask = np.array([True, True, True, False, False])

# print(accuracy(torch.from_numpy(pred[mask]), torch.from_numpy(truth[mask])), 
#       accuracy(torch.from_numpy(pred), torch.from_numpy(truth)), 
#       accuracy(torch.from_numpy(pred[~mask]), torch.from_numpy(truth[~mask])))