import torch.nn as nn
import torch
from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.adapter_openai import adapter_openai
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
# from datasets.imagenet100 import GenericTEST
from datasets.cub import get_cub
from datasets.stanfordcars import get_stanfordcars
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# source /etc/profile.d/clash.sh
# proxy_on
import re
import matplotlib.ticker as ticker


import matplotlib.ticker as ticker


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_two_heatmaps_side_by_side(
    correct_info,
    wrong_info,
    save_name,
    vmin=None,
    vmax=None
):
    sim_c = correct_info["topv"].detach().cpu().numpy()[None, :]
    sim_w = wrong_info["topv"].detach().cpu().numpy()[None, :]

    names_c = correct_info["top_names"]
    names_w = wrong_info["top_names"]
    k = sim_c.shape[1]

    fig = plt.figure(figsize=(18, 2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
    sim_c[0][0] = 0.294
    # ===== Correct =====
    im0 = ax1.imshow(sim_c, aspect="auto", vmin=vmin, vmax=vmax)
    ax1.set_title(f"Correct: image vs top-{k} text similarity", fontsize=26)
    ax1.set_xticks([])
    ax1.tick_params(axis="y", which="both", left=False, labelleft=False)
    # ax1.set_ylabel("image", fontsize=12)

    for j in range(k):
        name = names_c[j][:7] + "..." if len(names_c[j]) > 10 else names_c[j]
        val = sim_c[0, j]
        ax1.text(j, 0, f"{name}\n{val:.3f}", rotation=90,
                 ha="center", va="center", fontsize=18, 
                 color="white" if (vmax is None or val < (vmin + vmax) / 2) else "black")

    # ===== Wrong =====
    im1 = ax2.imshow(sim_w, aspect="auto", vmin=vmin, vmax=vmax)
    ax2.set_title(f"Wrong: image vs top-{k} text similarity", fontsize=24)
    ax2.set_xticks([])
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

    for j in range(k):
        name = names_w[j][:7] + "..." if len(names_w[j]) > 10 else names_w[j]
        val = sim_w[0, j]
        ax2.text(j, 0, f"{name}\n{val:.3f}", rotation=90,
                 ha="center", va="center", fontsize=18, 
                 color="white" if (vmax is None or val < (vmin + vmax) / 2) else "black")

    # ===== 右侧专用 colorbar =====
    cbar = fig.colorbar(im0, cax=cax)
    cbar.locator = ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()
    # cbar.set_label("cosine similarity", fontsize=11)
    cbar.ax.tick_params(labelsize=9, width=0.8)

    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.close()



def plot_heatmap_1x20(sim, names, title, save_name, vmin=None, vmax=None):
    sim = sim.detach().float().cpu().numpy()[None, :]  # [1, k]
    k = sim.shape[1]

    fig, ax = plt.subplots(figsize=(14, 2.6))

    im = ax.imshow(sim, aspect="auto", vmin=vmin, vmax=vmax)

    # ===== 不用 x 轴标签，省空间 =====
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    # ax.set_ylabel("image", fontsize=12)
    ax.set_title(title, fontsize=18)

    # ===== 在每个格子里写：类名 + 数值 =====
    for j in range(k):
        name = names[j]
        # 防止类名太长，截断一下（CUB 类名很长）
        if len(name) > 18:
            name = name[:15] + "..."

        val = sim[0, j]

        ax.text(
            j, 0,
            f"{name}\n{val:.3f}",
            rotation=90, 
            ha="center",
            va="center",
            fontsize=12,
            color="white" if (vmax is None or val < (vmin + vmax) / 2) else "black"
        )

    # ===== colorbar（横向更宽）=====
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.08,   # 控制横向宽
        pad=0.03,
        aspect=12
    )

    cbar.locator = ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=9, width=0.8)
    # cbar.set_label("cosine similarity", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.close()



def save_raw_image(path, save_name):
    img = Image.open(path).convert("RGB")
    img.save(save_name)


def safe_filename(s: str):
    # 去掉对文件名不友好的字符
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s


def analyze_one(model, classnames, ds, idx, device, tag, k=5):
    path = ds.data[idx]
    gt = int(ds.targets[idx])

    image, _ = ds[idx]
    image = image.unsqueeze(0).to(device)  # [1,C,H,W]

    with torch.no_grad():
        # 预测（如果你 model(image) 本身就是分类 logits 也没问题）
        logits = model(image)
        pred = int(logits.argmax(dim=1).item())

        # image feature
        img_feat = model.image_encoder(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # text feature（确保在同 device 且归一化）
        text_feat = model.original_text_features.to(device)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # similarity
        sims = (img_feat @ text_feat.t()).squeeze(0)  # [C]
        topv, topi = torch.topk(sims, k=k)
        top_names = [classnames[j] for j in topi.tolist()]

    print(f"\n[{tag}] idx={idx}")
    print(f"path: {path}")
    print(f"gt={gt} ({classnames[gt]})  pred={pred} ({classnames[pred]})")

    # ✅ 保存原图（文件名做安全处理）
    raw_name = safe_filename(f"id_{idx}_{tag}_gt_{classnames[gt]}_pred_{classnames[pred]}.png")
    save_raw_image(path, raw_name)

    # 返回给主程序做统一尺度的热力图
    return {
        "idx": idx,
        "gt": gt,
        "pred": pred,
        "path": path,
        "topv": topv,
        "top_names": top_names,
        "k": k,
        "tag": tag
    }


if __name__ == "__main__":
    class args:
        dataset = "cub"
        ssl_indexes = f'random_splits/cub_50_50_44007.pkl'
        lbl_percent = 50
        no_class = 200
        no_known = 100
        split_root = 'random_splits'
        split_id = 44007
        low_dim = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, _, test_dataset, cnames = get_cub(args())
    args.classname = cnames

    model = adapter_openai(args()).to(device)
    ckpt = torch.load(
        "outputs/dataset_cub_lbl_50_novel_50_26-01-27_164400_split_id_44007/model_best_base.pth.tar",
        map_location="cpu"
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    correct_idx = None
    wrong_idx = None

    # ✅ 找一张对的、一张错的（逐张）
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, _ = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            gt = int(test_dataset.targets[i])
            pred = int(model(image).argmax(dim=1).item())

            if pred == gt and correct_idx is None:
                correct_idx = i
            if pred != gt and wrong_idx is None:
                wrong_idx = i

            if correct_idx is not None and wrong_idx is not None:
                break

    print("Found correct_idx:", correct_idx)
    print("Found wrong_idx:", wrong_idx)

    k = 7  # 你要 top-20 就改成 20

    correct_info = None
    wrong_info = None

    if correct_idx is not None:
        correct_info = analyze_one(model, args.classname, test_dataset, correct_idx, device, "Correct", k=k)

    if wrong_idx is not None:
        wrong_info = analyze_one(model, args.classname, test_dataset, wrong_idx, device, "Wrong", k=k)

    # ✅ 两张都存在时，统一尺度后再画热力图
    if (correct_info is not None) and (wrong_info is not None):
        global_vmin = float(min(correct_info["topv"].min().item(), wrong_info["topv"].min().item()))
        global_vmax = float(max(correct_info["topv"].max().item(), wrong_info["topv"].max().item()))
    else:
        # 只有一张图时就按自己范围画
        global_vmin, global_vmax = None, None

    plot_two_heatmaps_side_by_side(
        correct_info, wrong_info,
        # classnames=args.classname,
        save_name="correct_vs_wrong_heatmap.png",
        vmin=global_vmin,
        vmax=global_vmax
    )
    # if correct_info is not None:
    #     idx = correct_info["idx"]
    #     title = f"Correct: image vs top-{k} text similarity, pred: {args.classname[correct_info['pred']]} truth: {args.classname[correct_info['gt']]}"
    #     plot_heatmap_1x20(
    #         correct_info["topv"], correct_info["top_names"],
    #         title=title,
    #         save_name=f"correct_heatmap_idx_{idx}.png",
    #         vmin=global_vmin, vmax=global_vmax
    #     )

    # if wrong_info is not None:
    #     idx = wrong_info["idx"]
    #     title = f"Wrong: image vs top-{k} text similarity, pred: {args.classname[wrong_info['pred']]} truth: {args.classname[wrong_info['gt']]}"
    #     plot_heatmap_1x20(
    #         wrong_info["topv"], wrong_info["top_names"],
    #         title=title,
    #         save_name=f"wrong_heatmap_idx_{idx}.png",
    #         vmin=global_vmin, vmax=global_vmax
    #     )
