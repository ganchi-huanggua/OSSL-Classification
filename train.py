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
from datasets.utils import PseudoLabelDataset
from utils.evaluate_utils import hungarian_evaluate
from utils.losses import *
from utils.utils import *



def main():
    parser = argparse.ArgumentParser(description='Base Training')
    parser.add_argument('--data-root', default=f'/home/lhz/data', help='directory to store data')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'oxfordflowers', 
                                 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium', 'cub'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=32, type=int, help='train batchsize, batch_x + batch_u')
    parser.add_argument('--test-batch-size', default=512, type=int, help='test batchsize')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate, default 1e-4')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")
    parser.add_argument('--split-id', default='44007', type=str, help='random data split number')
    # parser.add_argument('--ssl-indexes', default='random_splits/cifar100_50_50_split_70058.pkl', type=str, help='path to random data split')
    parser.add_argument('--warmup', default=0, type=int, help='warmup epoch')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for clip zero-shot')
    # parser.add_argument('--mixup-alpha', default=0.2, type=float)
    # parser.add_argument('--consistency-weight', type=float, default=1.0, help='Weight for consistency regularization loss')
    args = parser.parse_args()
    run_started = datetime.today().strftime('%y-%m-%d_%H%M%S')
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

    # logging.info('************************************************************************\n\n')
    # logging.info(' | '.join(f'{k}={v}' for k, v in vars(args).items() if k != "classname"))
    # logging.info('\n\n************************************************************************\n')
    
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
    elif args.dataset == 'stanfordcars':
        args.no_class = 196
    elif args.dataset == 'cub':
        args.no_class = 200

    args.data_root = os.path.join(args.data_root, args.dataset)
    # os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)

    # Load dataset
    args.no_known = args.no_class - math.floor((args.novel_percent * args.no_class) / 100)
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
    # [args.rho_start, args.rho_end] = [float(item) for item in args.rho.split(',')]
    model, teacher_model = build_model(args)
    # ema = EMA(model, decay=0.999)
    # ema_model = build_model(args,ema=True)

    logging.info(' | '.join(f'{k}={v}' for k, v in vars(args).items() if k != "classname"))
    
    # ema_model = ema_model.cuda()
    # ema_optimizer= WeightEMA(0.95, model, ema_model)
    # sinkhorn = SinkhornKnopp(num_iters_sk=3, epsilon_sk=0.05, imb_factor=1)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.01 * args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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

    model.eval()
    # training
    selected_samples = dict()
    # 新增：存储每个样本的真实标签（用于后续计算精度）
    sample_true_labels = dict()
    for cls in range(args.no_known, args.no_class):
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
        
        with torch.no_grad():
            zs_logits = teacher_model(inputs_u, True)
            # print("zs_logits max:", zs_logits.max().item(), "min:", zs_logits.min().item())
            conf = F.softmax(zs_logits / args.temperature, dim=1)
            max_conf, pseudo_label = torch.max(conf, dim=1)
            max_conf = max_conf.cpu().numpy()
            pseudo_label = pseudo_label.cpu().numpy()
            index_u = index_u.numpy()
            conf = conf.cpu().numpy()

            # 按伪标签分类存储（同时记录真实标签）
            for idx, pl, mc, tl, conf in zip(index_u, pseudo_label, max_conf, true_labels, conf):
                all_predictions[pl].append((-mc, idx, tl, conf))  # 增加真实标签tl

    # 筛选topk并计算精度
    correct = 0
    total = 0
    topk_conf = int(len(lbl_dataset) / args.no_known)
    # topk_conf = 16
    
    index_to_pseudo_label = {}
    for cls in range(args.no_known, args.no_class):
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
            
            for item in topk_samples:
                # item[3]: soft-label, cls: hard-label
                index_to_pseudo_label[item[1]] = cls
                # index_to_pseudo_label[item[1]] = item[3]
    
    # for cls in range(args.no_class):
    #     selected_samples[cls] = [item[1] for item in all_predictions[cls]]
    #     sample_true_labels[cls] = [item[2] for item in all_predictions[cls]]
        
    #     if len(selected_samples[cls]) != 0:
    #         correct += sum(1 for tl in sample_true_labels[cls] if tl == cls)
    #         total += len(selected_samples[cls])
            
    #         for item in selected_samples[cls]:
    #             # item[3]: soft-label, cls: hard-label
    #             index_to_pseudo_label[item] = cls
    #             # index_to_pseudo_label[item[1]] = item[3]
    
    print(f"伪标签精度: {correct / total:.4f} ({correct}/{total})")

    original_dataset = unlbl_loader.dataset

    pseudo_label_dataset = PseudoLabelDataset(
        original_dataset=original_dataset,
        index_to_pseudo_label=index_to_pseudo_label,
        # soft_label=True
    )
    
    pseudo_label_dataloader = DataLoader(
        pseudo_label_dataset,
        batch_size=int(len(pseudo_label_dataset) / args.iteration),
        sampler=train_sampler(pseudo_label_dataset),
        num_workers=args.num_workers,
        drop_last=False
    )

    for epoch in range(start_epoch, args.epochs):
        #training
        train(args, lbl_loader, unlbl_loader, pseudo_label_dataloader, model, optimizer, scheduler, epoch, teacher_model)
        # train(args, lbl_loader, unlbl_loader, model, optimizer, scheduler, epoch)
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

        # model_to_save = model.module if hasattr(model, "module") else model    
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model_to_save,
        #     'acc': test_acc_trans,
        #     'best_acc': best_acc_trans,
        #     'optimizer': optimizer.state_dict()
        # }, is_best_trans, args.out, tag='base')

    writer.close()

def get_la_loss(tau=1.0):
    cls_num_list = [314]*50
    cls_num_list.extend([64]*50)
    cls_num_list = torch.tensor(cls_num_list)
    cls_num_ratio = cls_num_list/torch.sum(cls_num_list)
    log_cls_num = torch.log(cls_num_ratio)
    tau = torch.tensor(tau)

    def loss_fn(logits,target):
        logit_adjusted = logits + tau*log_cls_num.unsqueeze(0).to(logits.device)
        return F.cross_entropy(logit_adjusted, target)
    return loss_fn


thre_sum = 0.0
thre_count = 0
def train(args, lbl_loader, unlbl_loader, pseudo_label_dataloader, model, optimizer, scheduler, epoch, teacher_model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    # train_loader = zip(lbl_loader, unlbl_loader)
    train_loader = zip(lbl_loader, unlbl_loader, pseudo_label_dataloader)
    #For normalization of PU classifier 
    msg = ""
    train_start_time = time.time()
    model.train()
    # 初始化存储结构
    # LA_loss = get_la_loss()
    
    text_features_all = model.original_text_features  # (num_classes, dim)
    text_features_all = text_features_all.cuda()
    
    for batch_idx, (data_lbl, data_unlbl, data_pseudo) in enumerate(train_loader):
        (inputs_l, inputs_l_w, inputs_l_s), targets_l, index_l = data_lbl 
        (inputs_u, inputs_u_w, inputs_u_s), targets_u, index_u = data_unlbl 
        (inputs_p, inputs_p_w, inputs_p_s), targets_p, index_p = data_pseudo
        
        inputs_l = inputs_l.cuda()
        inputs_l_w = inputs_l_w.cuda()
        inputs_l_s = inputs_l_s.cuda()
        targets_l = targets_l.cuda()
        
        inputs_u = inputs_u.cuda()
        inputs_u_w = inputs_u_w.cuda()
        inputs_u_s = inputs_u_s.cuda()
        targets_u = targets_u.cuda()
        
        inputs_p = inputs_p.cuda()
        inputs_p_w = inputs_p_w.cuda()
        inputs_p_s = inputs_p_s.cuda()
        targets_p = targets_p.cuda()

        batch_l = index_l.shape[0]
        batch_u = index_u.shape[0]
        batch_p = index_p.shape[0]
        
        with torch.no_grad():
            output = teacher_model(inputs_u, True)
            conf = F.softmax(output / args.temperature, dim=1)
            sorted_conf, sorted_indices = torch.sort(conf, dim=1, descending=True)
            tau = compute_tau(sorted_conf, alpha=0.6)
            intra_candidate_labels = select_intra_candidate_labels(sorted_conf, sorted_indices, tau)
            inter_candidate_labels = select_inter_candidate_labels(conf, beta=0.95)
            candidate_labels = merge_candidate_labels(intra_candidate_labels, inter_candidate_labels)
            pseudo_label = convert_to_one_hot(candidate_labels, num_classes=args.no_class).cuda()
            count_mask = (pseudo_label.sum(dim=1) > 0) & (pseudo_label.sum(dim=1) < 2)
            novel_classes = list(range(args.no_known, args.no_class))
            novel_mask = pseudo_label[:, novel_classes].sum(dim=1) > 0
            if args.dataset in ["cifar10", "cifar100", "imagenet100"]:
                mask = count_mask & novel_mask
            else:
                mask = count_mask
            # tea_u_prob = F.softmax(output / 3, dim=1)[~mask]
            
            # # correct = 0
            # # correct_intra = 0
            # # correct_inter = 0
            # correct_filter = 0
            # # total = 0
            # total_filter = 0
            # for i in range(len(targets_u)):
            #     if len(candidate_labels[i]) == 0:
            #         continue  # 无伪标签，不参与统计
            #     # total += 1
            #     # if 0 < len(candidate_labels[i]) < 2 and candidate_labels[i][0] in novel_classes:
            #     if 0 < len(candidate_labels[i]) < 2:
            #         total_filter += 1
            #         if targets_u[i].item() in candidate_labels[i]:
            #             correct_filter += 1
            #     # if targets_u[i].item() in candidate_labels[i]:
            #     #     correct += 1
            #     # if targets_u[i].item() in intra_candidate_labels[i]:
            #     #     correct_intra += 1
            #     # if targets_u[i].item() in inter_candidate_labels[i]:
            #     #     correct_inter += 1

            # # pseudo_accuracy = correct / total if total > 0 else 0.0
            # # print(f"[Batch {batch_idx}] Pseudo-label Accuracy: {pseudo_accuracy:.4f} ({correct}/{total})")
            # pseudo_accuracy = correct_filter / total_filter if total_filter > 0 else 0.0
            # print(f"[Batch {batch_idx}] Pseudo-label Accuracy Filter: {pseudo_accuracy:.4f} ({correct_filter}/{total_filter})")

        
        img_concat = torch.cat([inputs_l_w, inputs_p_w, inputs_u_w, inputs_l_s, inputs_p_s, inputs_u_s], dim=0)
        img_feat_concat = model.get_image_feature(img_concat)  # (batch_l+batch_p+batch_u, dim)
        img_feat_l, img_feat_p, img_feat_u, img_feat_l_s, img_feat_p_s, img_feat_u_s = torch.split(img_feat_concat, [batch_l, batch_p, batch_u, batch_l, batch_p, batch_u], dim=0)
        
        logit_scale = model.logit_scale.exp()
        logits_l = logit_scale * (img_feat_l @ text_features_all.T)
        logits_p = logit_scale * (img_feat_p @ text_features_all.T)
        logits_u = logit_scale * (img_feat_u[mask] @ text_features_all.T)
        
        logits_l_s = logit_scale * (img_feat_l_s @ text_features_all.T)
        logits_p_s = logit_scale * (img_feat_p_s @ text_features_all.T)
        logits_u_s = logit_scale * (img_feat_u_s[mask] @ text_features_all.T)
        
        # stu_logits_u = logit_scale * (img_feat_u[~mask] @ text_features_all.T)
        # stu_logits_u_s = logit_scale * (img_feat_u_s[~mask] @ text_features_all.T)
        
        # stu_prob_u = F.softmax(stu_logits_u, dim=1)
        # stu_prob_u_s = F.softmax(stu_logits_u_s, dim=1)

        l_ce_loss = F.cross_entropy(logits_l, targets_l)
        p_ce_loss = F.cross_entropy(logits_p, targets_p)
        l_ce_loss_s = F.cross_entropy(logits_l_s, targets_l)
        p_ce_loss_s = F.cross_entropy(logits_p_s, targets_p)
        
        loss = l_ce_loss + p_ce_loss + l_ce_loss_s + p_ce_loss_s

        if mask.sum() > 0:
            u_ce_loss = F.cross_entropy(logits_u, pseudo_label[mask])
            u_ce_loss_s = F.cross_entropy(logits_u_s, pseudo_label[mask])
            loss += u_ce_loss + u_ce_loss_s
            
        # if (~mask).sum() > 0:  # 只有存在低置信样本时才加这个损失
        #     kl_loss_u = F.kl_div(torch.log(stu_prob_u + 1e-8), tea_u_prob, reduction='batchmean')
        #     kl_loss_u_s = F.kl_div(torch.log(stu_prob_u_s + 1e-8), tea_u_prob, reduction='batchmean')
        #     print(f"kl_loss_u: {kl_loss_u:.4f}, kl_loss_u_s: {kl_loss_u_s:.4f}")
        #     loss += kl_loss_u + kl_loss_u_s
            # kl_loss = F.kl_div(torch.log(stu_prob_u + 1e-8), stu_prob_u_s, reduction='batchmean')
            # print(f"kl_loss: {kl_loss:.4f}")
            # loss += kl_loss
        
        # 损失更新（原逻辑不变）
        # losses.update(loss.item(), batch_l + batch_p + batch_u)
        losses.update(loss.item(), batch_l + batch_p + (mask.sum().item() if mask.sum() > 0 else 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # ema.update(model)
        
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
    # ema_model = ema.apply()
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
            # outputs = ema_model(inputs)
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
    # ema_model = ema.apply()
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
            # outputs = ema_model(inputs)
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


        #     # 2. 高置信度样本的伪标签精度
        #     high_conf_samples = mask.sum().item()
        #     correct_high = 0
        #     if high_conf_samples > 0:
        #         # 仅统计高置信度样本中伪标签正确的数量
        #         correct_high = (pseudo_label[mask] == targets_u[mask]).sum().item()
        #         acc_high = correct_high / high_conf_samples
        #     else:
        #         acc_high = 0.0  # 无高置信度样本时精度为0
        #     print(f"高置信度精度: {acc_high:.4f} ({correct_high}/{high_conf_samples})")
        #     sorted_conf, sorted_indices = torch.sort(conf, dim=1, descending=True)
        #     tau = compute_tau(sorted_conf, alpha=0.6)
        #     intra_candidate_labels = select_intra_candidate_labels(sorted_conf, sorted_indices, tau)
        #     inter_candidate_labels = select_inter_candidate_labels(conf, beta=0.95)
        #     candidate_labels = merge_candidate_labels(intra_candidate_labels, inter_candidate_labels)
        #     selected_labels = []
        #     for candidate_label in candidate_labels:
        #         if 0 < len(candidate_label) < 2 and candidate_label[0] in list(range(args.no_known, args.no_class)):
        #             selected_labels.append(candidate_label)
        #     pseudo_label = convert_to_one_hot(candidate_labels, num_classes=args.no_class).cuda()
        #     count_mask = (pseudo_label.sum(dim=1) > 0) & (pseudo_label.sum(dim=1) < 2)  # choosed 159 pseudo-labeled samples
        #     novel_classes = list(range(args.no_known, args.no_class))
        #     novel_mask = pseudo_label[:, novel_classes].sum(dim=1) > 0
        #     mask = count_mask & novel_mask
            
        #     correct = 0
        #     # correct_intra = 0
        #     # correct_inter = 0
        #     correct_filter = 0
        #     total = 0
        #     total_filter = 0
        #     for i in range(len(targets_u)):
        #         if len(candidate_labels[i]) == 0:
        #             continue  # 无伪标签，不参与统计
        #         total += 1
        #         if 0 < len(candidate_labels[i]) < 2 and candidate_labels[i][0] in novel_classes:
        #             total_filter += 1
        #             if targets_u[i].item() in candidate_labels[i]:
        #                 correct_filter += 1
        #         # if targets_u[i].item() in candidate_labels[i]:
        #         #     correct += 1
        #         # if targets_u[i].item() in intra_candidate_labels[i]:
        #         #     correct_intra += 1
        #         # if targets_u[i].item() in inter_candidate_labels[i]:
        #         #     correct_inter += 1

        #     pseudo_accuracy = correct / total if total > 0 else 0.0
        #     print(f"[Batch {batch_idx}] Pseudo-label Accuracy: {pseudo_accuracy:.4f} ({correct}/{total})")
        #     pseudo_accuracy = correct_filter / total_filter if total_filter > 0 else 0.0
        #     print(f"[Batch {batch_idx}] Pseudo-label Accuracy Filter: {pseudo_accuracy:.4f} ({correct_filter}/{total_filter})")
        
        ############################################################
        
        # inputs_u = inputs_u[mask]
        # inputs_u_w = inputs_u_w[mask]
        # inputs_u_s = inputs_u_s[mask]
        # pseudo_label = pseudo_label[mask]
        # targets_u = targets_u[mask]
        
        ############################################################
        
        # if mask.sum() > 0:
        #     inputs_concat = torch.cat([inputs_l_w, inputs_u_w, inputs_p_w], dim=0)
        #     logits_concat = model(inputs_concat)
            
        #     logits_l, logits_u, logits_p = torch.split(logits_concat, [batch_l, batch_u, batch_p], dim=0)
        #     ce_loss_l = F.cross_entropy(logits_l, targets_l)
        #     # la_loss_l = LA_loss(logits_l, targets_l)
            
        #     ce_loss_u = F.cross_entropy(logits_u[mask], pseudo_label[mask])
        #     ce_loss_p = F.cross_entropy(logits_p, targets_p)
        #     # la_loss_p = LA_loss(logits_p, targets_p)
        
        #     loss = ce_loss_l + ce_loss_u + ce_loss_p
        #     # loss = ce_loss_l + 2 * epoch / total_epochs * ce_loss_u + ce_loss_p
        #     # loss = la_loss_l + la_loss_p
        
        #     losses.update(loss.item(), batch_u + batch_l + batch_p)
        # else:
        #     inputs_concat = torch.cat([inputs_l_w, inputs_p_w], dim=0)
        #     logits_concat = model(inputs_concat)
            
        #     logits_l, logits_p = torch.split(logits_concat, [batch_l, batch_p], dim=0)
        #     ce_loss_l = F.cross_entropy(logits_l, targets_l)
        #     ce_loss_p = F.cross_entropy(logits_p, targets_p)
        
        #     loss = ce_loss_l + ce_loss_p
        #     losses.update(loss.item(), batch_l + batch_p)
        
        ############################################################
        
        # text_feat_l = text_features_all[targets_l]
        # text_feat_u = text_features_all[targets_u]
        # text_feat_p = text_features_all[targets_p]
        
        # img_concat = torch.cat([inputs_l_w, inputs_p_w, inputs_u_w], dim=0)
        # img_feat_concat = model.get_image_feature(img_concat)  # (batch_l+batch_p+batch_u, dim)
        # img_feat_l, img_feat_p, img_feat_u = torch.split(img_feat_concat, [batch_l, batch_p, batch_u], dim=0)
        
        # mix_batch_size = min(batch_l, batch_p)  # 取两个 batch 的最小 size（避免维度不匹配）
        # # lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
        # lam = 0.5
        # rand_idx = torch.randperm(mix_batch_size).cuda()  # 随机排列的索引

        # img_feat_mix = lam * img_feat_l[:mix_batch_size] + (1 - lam) * img_feat_p[rand_idx]
        # img_feat_mix = img_feat_mix / img_feat_mix.norm(dim=-1, keepdim=True)
        
        # text_feat_mix = lam * text_feat_l[:mix_batch_size] + (1 - lam) * text_feat_p[rand_idx]
        # text_feat_mix = text_feat_mix / text_feat_mix.norm(dim=-1, keepdim=True)

        # y_l_onehot = F.one_hot(targets_l[:mix_batch_size], num_classes=args.no_class).float()
        # y_p_onehot = F.one_hot(targets_p[rand_idx], num_classes=args.no_class).float()
        # y_mix = lam * y_l_onehot + (1 - lam) * y_p_onehot

        # logit_scale = model.logit_scale.exp()
        # logits_mix = img_feat_mix @ text_features_all.T
        # logits_l = logit_scale * (img_feat_l @ text_features_all.T)
        # logits_p = img_feat_p @ text_features_all.T
        # logits_u_selected = logit_scale * (img_feat_u[mask] @ text_features_all.T)
        
        # text_feat_mix = lam * text_feat_l[:mix_batch_size] + (1 - lam) * text_feat_p[rand_idx]
        # text_feat_mix = text_feat_mix / text_feat_mix.norm(dim=-1, keepdim=True)
        # sim = torch.sum(img_feat_mix * text_feat_mix, dim=1)
        # mix_loss = -torch.log(torch.sigmoid(sim)).mean()
        
        # mix_loss = F.kl_div(F.log_softmax(logits_mix, dim=1), y_mix, reduction='batchmean')

        # l_ce_loss = F.cross_entropy(logits_l, targets_l)
        # p_ce_loss = F.cross_entropy(logits_p, targets_p)
        # p_ce_loss = -torch.sum(targets_p * F.log_softmax(logits_p, dim=1), dim=1).mean()
        # p_ce_loss = F.kl_div(F.log_softmax(logits_p, dim=1), targets_p, reduction='batchmean')

        # if mask.sum() > 0:
        #     u_ce_loss = F.cross_entropy(logits_u_selected, pseudo_label[mask])
        #     loss = l_ce_loss + p_ce_loss + u_ce_loss
            # print(l_ce_loss.item(), p_ce_loss.item(), u_ce_loss.item())
        # else:
        #     loss = l_ce_loss + p_ce_loss
            # print(l_ce_loss.item(), p_ce_loss.item())
        
        # 损失更新（原逻辑不变）
        # losses.update(loss.item(), batch_l + batch_p + (mask.sum().item() if mask.sum() > 0 else 0))  # 用实际数据量更新损失
        
        ############################################################
        
        # with torch.no_grad():
        #     ema_model = ema.apply()
        #     inputs = torch.cat([inputs_l, inputs_u], dim=0)
        #     outputs = teacher_model(inputs, True)
        #     outputs = F.softmax(outputs / args.temperature, dim=1)
        #     output_l, output_u = torch.split(outputs, [batch_l, batch_u], dim=0)
        #     # output_l = sinkhorn(output_l)
        #     # output_u = sinkhorn(output_u)
        #     max_conf, pseudo_label = torch.max(output_l, dim=1)
        #     class_thre = [1.0] * args.no_known
        #     for i in range(batch_l):
        #         if pseudo_label[i] == targets_l[i]:
        #             class_thre[targets_l[i]] = min(class_thre[pseudo_label[i]], max_conf[i])
        #     # 统计class_thre中不为2的值的平均值作为阈值
        #     non_one_values = [v for v in class_thre if v != 1.0]
        #     global thre_sum, thre_count
        #     thre_sum += sum(non_one_values)
        #     thre_count += len(non_one_values)
        #     thre = thre_sum / thre_count if thre_count > 0 else 1.0

        #     max_conf, pseudo_label = torch.max(output_u, dim=1)
        #     conf_mask = (max_conf >= thre)
        #     # novel_mask = (pseudo_label >= args.no_known)
        #     mask = conf_mask
            
        #     total_samples = len(pseudo_label)
        #     correct_all = (pseudo_label == targets_u).sum().item()
        #     acc_all = correct_all / total_samples if total_samples > 0 else 0.0
        #     acc_mask = (pseudo_label[mask] == targets_u[mask]).sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
        #     print(f"acc_all: {acc_all}, acc_mask: {acc_mask}")