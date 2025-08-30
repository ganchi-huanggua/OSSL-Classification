import argparse
import os
import logging
import sys
import random
import time
import pickle
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from models.build_model import build_model
from datasets.datasets import get_dataset
from utils.evaluate_utils import hungarian_evaluate
from utils.losses import *
from utils.utils import *
from utils.energy import energy_discrepancy, energy


def main():
    parser = argparse.ArgumentParser(description='Base Training')
    parser.add_argument('--data-root', default=f'/home/lhz/data', help='directory to store data')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'oxfordflowers', 
                                 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=512, type=int, help='train batchsize, batch_x + batch_u')
    parser.add_argument('--test-batch-size', default=512, type=int, help='test batchsize')
    parser.add_argument('--lr', default=0.0025, type=float, help='learning rate, default 1e-3')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")
    parser.add_argument('--split-id', default='44007', type=str, help='random data split number')
    # parser.add_argument('--ssl-indexes', default='random_splits/cifar100_50_50_split_70058.pkl', type=str, help='path to random data split')
    parser.add_argument('--rho', default='0.3,0.9', type=str, help='pseudo-label filtering ratio')
    parser.add_argument('--warmup', default=0, type=int, help='warmup epoch')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--chosen_neighbors', default=100, type=int, help='number of chosen neighbors for KNN contrastive learning')
    parser.add_argument('--entropy_q', default=0.3, type=float, help='q for entropy loss')
    parser.add_argument('--temparature', default=0.3, type=float, help='temperature for classification loss')
    parser.add_argument('--knn_weight', default=0.2, type=float, help='weight of KNN contrastive loss')
    args = parser.parse_args()
    run_started = datetime.today().strftime('%y-%m-%d_%H%M')
    if args.split_id == "":
        split_id = f'{random.randint(1, 100000)}'
        args.split_id = split_id
        
    args.ssl_indexes = f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'
    args.exp_name = f'dataset_{args.dataset}_lbl_{args.lbl_percent}_novel_{args.novel_percent}_{run_started}_split_id_{args.split_id}'
    args.img_size = 224
    args.out = os.path.join(args.out, args.exp_name)
    os.makedirs(args.out, exist_ok=True)

    best_acc = 0    
    best_acc_trans = 0
    best_acc_novel_trans = 0
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(f'{args.out}/score_logger_base.txt', mode='a+'),     # 写入文件
            logging.StreamHandler(sys.stdout)             # 输出到控制台
        ]
    )
    
    writer = SummaryWriter(logdir=args.out)

    logging.info('************************************************************************\n\n')
    logging.info(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
    logging.info('\n\n************************************************************************\n')
    
    args.n_gpu = torch.cuda.device_count()
    args.dtype = torch.float32
    if args.seed != -1:
        set_seed(args)

    # set dataset specific parameters
    if args.dataset == 'cifar10':
        args.no_class = 10
    elif args.dataset == 'cifar100':
        args.no_class = 100
    elif args.dataset == 'imagenet100':
        args.no_class = 100
    elif args.dataset == 'oxfordflowers':
        args.no_class = 102
    elif args.dataset == 'oxfordpets':
        args.no_class = 37

    args.data_root = os.path.join(args.data_root, args.dataset)
    # os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)

    # Load dataset
    args.no_known = args.no_class - int((args.novel_percent * args.no_class) / 100)
    lbl_dataset, unlbl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all, classname = get_dataset(args)
    args.classname = classname
    
    # Create dataloaders
    unlbl_batchsize = int((float(args.batch_size) * len(unlbl_dataset)) / (len(lbl_dataset) + len(unlbl_dataset)))
    lbl_batchsize = args.batch_size - unlbl_batchsize
    args.iteration = math.ceil((len(lbl_dataset) + len(unlbl_dataset)) / args.batch_size)

    train_sampler = RandomSampler
    lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=False)
    unlbl_loader = DataLoader(unlbl_dataset, sampler=train_sampler(unlbl_dataset), batch_size=unlbl_batchsize, num_workers=args.num_workers, drop_last=False)
    # Transductive setting
    test_loader_known_trans = DataLoader(test_dataset_known, sampler=SequentialSampler(test_dataset_known), batch_size=args.test_batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_novel_trans = DataLoader(test_dataset_novel, sampler=SequentialSampler(test_dataset_novel), batch_size=args.test_batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_all_trans = DataLoader(test_dataset_all, sampler=SequentialSampler(test_dataset_all), batch_size=args.test_batch_size, num_workers=args.num_workers, drop_last=False)
    torch.cuda.set_device(0)
    logging.info(f"using device: {torch.cuda.current_device()}")
    [args.rho_start, args.rho_end] = [float(item) for item in args.rho.split(',')]
    model = build_model(args)
    # ema_model = build_model(args,ema=True)

    logging.info(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    # ema_model = ema_model.cuda()
    # ema_optimizer= WeightEMA(0.95, model, ema_model)
    # sinkhorn = SinkhornKnopp(num_iters_sk=3,epsilon_sk=0.05,imb_factor=1)

    # optimizer
    # if torch.cuda.device_count() > 1:
    #     optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)
    # else:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 设置warmup学习率调度器
    def get_lr_lambda(current_step: int):
        if current_step < args.warmup * args.iteration:
            return float(current_step) / float(max(1, args.warmup * args.iteration))
        return max(
            0.0, float(args.epochs * args.iteration - current_step) / float(max(1, args.epochs * args.iteration - args.warmup * args.iteration))
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.01 * args.lr)
    start_epoch = 0
    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.zero_grad()
    # train_stat = {
    #     'feature_con_bank': torch.zeros(len(unlbl_loader.dataset),128).cuda(),
    #     'feature_all': torch.zeros(len(unlbl_loader.dataset),512).cuda(),
    #     'target_pu_max': -math.inf,
    #     'pseudo_list_all': torch.zeros((len(unlbl_loader.dataset),args.no_class)),
    #     'prob': np.zeros(len(unlbl_loader.dataset)),
    # }


    # training
    selected_samples = dict()
    # 新增：存储每个样本的真实标签（用于后续计算精度）
    sample_true_labels = dict()
    for cls in range(args.no_class):
        selected_samples[cls] = []
        sample_true_labels[cls] = []  # 存储对应样本的真实标签

    # 存储每个类别的所有预测结果 (置信度, 样本索引, 真实标签)
    all_predictions = {cls: [] for cls in range(args.no_class)}

    for batch_idx, data_unlbl in enumerate(unlbl_loader):
        (inputs_u, inputs_u_w, inputs_u_s), targets_u, index_u = data_unlbl 
        inputs_u = inputs_u.cuda()
        inputs_u_w = inputs_u_w.cuda()
        inputs_u_s = inputs_u_s.cuda()

        # 提取真实标签（转为numpy便于存储）
        true_labels = targets_u.numpy()

        model.train()
        with torch.no_grad():
            zs_logits = model(inputs_u, True)
            conf = F.softmax(zs_logits, dim=1)
            max_conf, pseudo_label = torch.max(conf, dim=1)
            max_conf = max_conf.cpu().numpy()
            pseudo_label = pseudo_label.cpu().numpy()
            index_u = index_u.numpy()

            # 按伪标签分类存储（同时记录真实标签）
            for idx, pl, cf, tl in zip(index_u, pseudo_label, max_conf, true_labels):
                all_predictions[pl].append((-cf, idx, tl))  # 增加真实标签tl

    # 筛选topk并计算精度
    correct = 0
    total = 0
    topk_conf = 16
    for cls in range(args.no_class):
        # 按置信度降序排序
        all_predictions[cls].sort()
        # 取前16个样本
        topk_samples = all_predictions[cls][:topk_conf]

        # 提取样本索引和对应的真实标签
        selected_samples[cls] = [item[1] for item in topk_samples]
        sample_true_labels[cls] = [item[2] for item in topk_samples]
        
        # 计算伪标签精度：伪标签（cls）与真实标签匹配的比例
        if len(topk_samples) != 0:
            # 统计真实标签等于当前伪标签类别的数量
            correct += sum(1 for tl in sample_true_labels[cls] if tl == cls)
            total += len(topk_samples)
        
    print(f"伪标签精度: {correct / total:.4f} ({correct}/{total})")

    all_selected_indices = []
    for cls in selected_samples:
        all_selected_indices.extend(selected_samples[cls])
    # 去重（避免同一样本被多次选中）
    all_selected_indices = list(set(all_selected_indices))
    # 排序（可选，使索引顺序一致）
    all_selected_indices.sort()

    # 2. 获取原始数据集（从unlbl_loader中提取）
    # 注意：unlbl_loader.dataset 是原始数据集对象
    original_dataset = unlbl_loader.dataset

    # 3. 创建筛选后的子集数据集
    # Subset会根据索引从原始数据集中提取对应样本
    subset_dataset = Subset(original_dataset, all_selected_indices)
    selected_dataloader = DataLoader(
        subset_dataset,
        batch_size=int(len(subset_dataset) / args.iteration),
        sampler=train_sampler(subset_dataset),
        num_workers=args.num_workers,
        drop_last=False
    )

    for epoch in range(start_epoch, args.epochs):
        #training
        train(args, lbl_loader, unlbl_loader, selected_dataloader, model, optimizer, scheduler, epoch)
        #test
        test_acc_known_trans = test_known(args, test_loader_known_trans, model, epoch)
        # novel_cluster_results_trans = test_cluster(args, test_loader_novel_trans, model, epoch, offset=args.no_known)
        # all_cluster_results_trans = test_cluster(args, test_loader_all_trans, model, epoch)
        # test_acc_trans = all_cluster_results_trans["acc"]
        # test_acc_novel_trans = novel_cluster_results_trans["acc"]
        novel_cluster_results_trans = test_known(args, test_loader_novel_trans, model, epoch)
        all_cluster_results_trans = test_known(args, test_loader_all_trans, model, epoch)
        test_acc_trans = all_cluster_results_trans
        test_acc_novel_trans = novel_cluster_results_trans

        is_best_trans = test_acc_novel_trans > best_acc_novel_trans
        best_acc_trans = max(test_acc_trans, best_acc_trans)
        best_acc_novel_trans = max(test_acc_novel_trans,best_acc_novel_trans)

        logging.info(f'epoch: {epoch + 1}, acc-known-trans: {test_acc_known_trans}')
        logging.info(f'epoch: {epoch + 1}, acc-novel-trans: {test_acc_novel_trans}')
        logging.info(f'epoch: {epoch + 1}, acc-all-trans: {test_acc_trans}, best-acc: {best_acc_trans}, best-acc-novel: {best_acc_novel_trans}')

        model_to_save = model.module if hasattr(model, "module") else model    
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save,
            'acc': test_acc_trans,
            'best_acc': best_acc_trans,
            'optimizer': optimizer.state_dict()
        }, is_best_trans, args.out, tag='base')

    writer.close()

def train(args, lbl_loader, unlbl_loader, selected_dataloader, model, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, unlbl_loader)
    #For normalization of PU classifier 
    msg = ""
    train_start_time = time.time()

    # 初始化存储结构
    for batch_idx, (data_lbl, data_unlbl) in enumerate(train_loader):
        (inputs_l, inputs_l_w, inputs_l_s), targets_l, index_l = data_lbl 
        (inputs_u, inputs_u_w, inputs_u_s), targets_u, index_u = data_unlbl 
        inputs_l = inputs_l.cuda()
        inputs_l_w = inputs_l_w.cuda()
        inputs_l_s = inputs_l_s.cuda()
        inputs_u = inputs_u.cuda()

        inputs_u_w = inputs_u_w.cuda()
        inputs_u_s = inputs_u_s.cuda()
        targets_l = targets_l.cuda()
        targets_u = targets_u.cuda()
        
        batch_l = inputs_l_w.shape[0]
        batch_u = inputs_u_w.shape[0]
        model.train()
        with torch.no_grad():
            zs_logits = model(inputs_u, True)
            conf = F.softmax(zs_logits, dim=1)
            sorted_conf, sorted_indices = torch.sort(conf, dim=1, descending=True)
            tau = compute_tau(sorted_conf, alpha=0.6)
            intra_candidate_labels = select_intra_candidate_labels(sorted_conf, sorted_indices, tau)
            inter_candidate_labels = select_inter_candidate_labels(conf, beta=0.95)
            candidate_labels = merge_candidate_labels(intra_candidate_labels, inter_candidate_labels)
            pseudo_one_hot = convert_to_one_hot(candidate_labels, num_classes=args.no_class).cuda()
            selected_mask = (pseudo_one_hot.sum(dim=1) > 0) & (pseudo_one_hot.sum(dim=1) < 2)  # choosed 159 pseudo-labeled samples

            # correct = 0
            # correct_intra = 0
            # correct_inter = 0
            # correct_filter = 0
            # total = 0
            # total_filter = 0
            # for i in range(len(targets_u)):
            #     if len(candidate_labels[i]) == 0:
            #         continue  # 无伪标签，不参与统计
            #     total += 1
            #     if 0 < len(candidate_labels[i]) < 2:
            #         total_filter += 1
            #         if targets_u[i].item() in candidate_labels[i]:
            #             correct_filter += 1
            #     if targets_u[i].item() in candidate_labels[i]:
            #         correct += 1
                # if targets_u[i].item() in intra_candidate_labels[i]:
                #     correct_intra += 1
                # if targets_u[i].item() in inter_candidate_labels[i]:
                #     correct_inter += 1

            # pseudo_accuracy = correct / total if total > 0 else 0.0
            # print(f"[Batch {batch_idx}] Pseudo-label Accuracy: {pseudo_accuracy:.4f} ({correct}/{total})")
            # pseudo_accuracy = correct_filter / total_filter if total_filter > 0 else 0.0
            # print(f"[Batch {batch_idx}] Pseudo-label Accuracy Filter: {pseudo_accuracy:.4f} ({correct_filter}/{total_filter})")
        
        logits = model(inputs_u_w)
        ce_loss = F.cross_entropy(logits[selected_mask], pseudo_one_hot[selected_mask])
        # ce_loss = F.cross_entropy(logits, targets_l)

        loss = ce_loss
        
        losses.update(loss.item(), batch_l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema_optimizer.step()
        scheduler.step()
            
        if batch_idx == 0 and epoch == 0:
            updated_param = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.any(param.grad != 0):
                        updated_param.append(name)
            logging.info(f"updated params: {updated_param}")

        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_progress:
            msg = "train epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. lr: {lr:.6f}".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                bt=batch_time.avg,
                loss=losses.avg,
                lr=optimizer.param_groups[0]['lr'],
            )
            p_bar.set_description(msg)
            p_bar.update()
            if batch_idx % 5 == 0:
                with open(f'{args.out}/score_logger_base.txt', mode='a+') as f:
                    f.write(msg + '\n')
                    
    total_train_time = time.time() - train_start_time
    with open(f'{args.out}/score_logger_base.txt', mode='a+') as f:
        f.write(f"Total train time: {total_train_time:.3f}s\n")
    if not args.no_progress:
        p_bar.close()
    # return train_stat


def test_known(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)
    msg = ""
    test_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                msg = "test known epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                )
                test_loader.set_description(msg)
        if not args.no_progress:
            test_loader.close()
    total_test_time = time.time() - test_start_time
    with open(f'{args.out}/score_logger_base.txt', mode='a+') as f:
        f.write(f"Total test time: {total_test_time:.3f}s\n")
    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    gt_targets = []
    predictions = []
    model.eval()
    if not args.no_progress:
        test_loader = tqdm(test_loader)
    msg = ""
    test_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            _, max_idx = torch.max(outputs, dim=1)
            predictions.extend(max_idx.cpu().numpy().tolist())
            gt_targets.extend(targets.cpu().numpy().tolist())
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                msg = "test cluster epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                )
                test_loader.set_description(msg)
        if not args.no_progress:
            test_loader.close()
    total_test_time = time.time() - test_start_time
    with open(f'{args.out}/score_logger_base.txt', mode='a+') as f:
        f.write(f"Total test time: {total_test_time:.3f}s\n")
    predictions = np.array(predictions)
    gt_targets = np.array(gt_targets)

    predictions = torch.from_numpy(predictions)
    gt_targets = torch.from_numpy(gt_targets)
    eval_output = hungarian_evaluate(predictions, gt_targets, offset)
    return eval_output


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
