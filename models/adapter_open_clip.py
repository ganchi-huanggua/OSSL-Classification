import math
import logging
import torch
import torch.nn as nn
import open_clip  # 关键：使用open_clip框架
from datasets.cub import get_cub
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Hugging Face 离线模式
os.environ["OPENCLIP_OFFLINE"] = "1"      # open_clip 自定义环境变量

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "oxfordflowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "stanfordcars": "a photo of a {}, a type of car.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "cifar100": "a photo of a {}.",
    "cifar10": "a photo of a {}.",
    "imagenet100": "a photo of a {}.",
    "cub": "a photo of a {}, a type of bird.",
}

def adapter_open_clip(args):
    """
    加载open_clip的ViT-H-14模型（替代原OpenAI CLIP加载逻辑）
    args需包含：low_dim（Adapter瓶颈维度）、classname（类别名列表）
    """
    # 1. 加载open_clip预训练模型（ViT-H-14 + laion2B-s32B-b79K）
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-H-14",
        pretrained="laion2B-s32B-b79K",
        cache_dir="./pretrained",
        device="cuda:0"
    )
    clip_model.eval()  # 初始设为评估模式，后续会解冻Adapter参数
    device = torch.device("cuda:0")
    clip_model = clip_model.to(device, dtype=torch.float32)  # 与原代码保持float精度
    low_dim = 10
    # 2. 构建带Adapter的模型
    model = CLIPAdapter(
        classnames=args.classname,
        clip_model=clip_model,
        low_dim=low_dim,
        device=device
    )

    # 3. 冻结原模型参数，只训练Adapter相关参数
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in ["adapter", "adapter_proj"]):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # 4. 预计算文本特征（与原逻辑一致）
    model.get_text_feature(args.dataset, args.classname)
    return model


class AdaptFormer(nn.Module):
    """核心Adapter模块：LayerNorm + 降维 + GELU + 升维 + 残差缩放"""
    def __init__(self, in_dim, bottle_dim=10, dtype=torch.float32, device="cuda:0"):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype, device=device)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype, device=device)
        self.relu = nn.GELU()
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype, device=device)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype, device=device))  # 残差缩放因子

        # 初始化：与原代码保持一致（保证训练稳定性）
        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x * self.scale  # 残差连接（原逻辑：Adapter输出加在MLP之后）


class AdapterResidualBlock(nn.Module):
    """适配open_clip的Transformer块：加入Adapter残差"""
    def __init__(self, block, low_dim, dtype=torch.float32, device="cuda:0"):
        super().__init__()
        # 1. 复用open_clip Transformer块的原有组件
        self.ln_1 = block.ln_1
        self.attn = block.attn  # 关键：open_clip用self_attn，替代原attn
        self.ls_1 = block.ls_1
        
        self.ln_2 = block.ln_2
        self.mlp = block.mlp
        self.ls_2 = block.ls_2
        # 2. 加入Adapter（与原逻辑一致：在MLP后加残差）
        self.ln_1_kv = block.ln_1_kv if hasattr(block, 'ln_1_kv') else None
        self.ln_2_kv = block.ln_2_kv if hasattr(block, 'ln_2_kv') else None
        
        self.adapter = AdaptFormer(
            in_dim=block.attn.embed_dim,  # ViT-H-14的embed_dim=1024
            bottle_dim=low_dim,
            dtype=dtype,
            device=device
        )

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x,
            need_weights=False,
            attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ):
        """open_clip Transformer块前向：加入Adapter"""
        # 原有逻辑：注意力残差 + MLP残差
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        
        x = q_x + self.ls_1(self.attention(self.ln_1(q_x), k_x, v_x, attn_mask=attn_mask))
        # 新增：MLP后加入Adapter残差
        x = x + self.ls_2(self.mlp(self.ln_2(x))) + self.adapter(x)
        return x


class AdapterTransformer(nn.Module):
    """适配open_clip的Transformer层：用带Adapter的块替换原有块"""
    def __init__(self, transformer, low_dim, dtype=torch.float32, device="cuda:0"):
        super().__init__()
        # 关键：open_clip用layers存储块，替代原resblocks；逐个包装Adapter
        self.width = transformer.width
        self.layers = transformer.layers
        self.batch_first = transformer.batch_first
        self.resblocks = nn.ModuleList([
            AdapterResidualBlock(
                block=block,
                low_dim=low_dim,
                dtype=dtype,
                device=device
            ) for block in transformer.resblocks
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        for block in self.resblocks:
            x = block(x, attn_mask=attn_mask)  # 每个块都传attn_mask
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # LND -> NLD
        return x


class ImageEncoder(nn.Module):
    def __init__(self, vision_transformer, low_dim, dtype=torch.float32, device="cuda:0"):
        super().__init__()
        self.output_tokens = vision_transformer.output_tokens
        self.grid_size = vision_transformer.grid_size
        self.final_ln_after_pool = vision_transformer.final_ln_after_pool
        self.output_dim = vision_transformer.output_dim
        
        self.conv1 = vision_transformer.conv1
        self.class_embedding = vision_transformer.class_embedding
        
        # 2. 补充：pos_embed 同步到模型设备（避免后续设备不匹配）
        self.pos_embed = vision_transformer.positional_embedding.to(device)
        self.patch_dropout = vision_transformer.patch_dropout
        self.ln_pre = vision_transformer.ln_pre

        self.transformer = AdapterTransformer(
            transformer=vision_transformer.transformer,
            low_dim=low_dim,
            dtype=dtype,
            device=device
        )
        self.attn_pool_type = vision_transformer.attn_pool_type if hasattr(vision_transformer, 'attn_pool_type') else None
        self.pool_type = vision_transformer.pool_type
        self.attn_pool = vision_transformer.attn_pool
        self.attn_pool_contrastive = vision_transformer.attn_pool_contrastive if hasattr(vision_transformer, 'attn_pool_contrastive') else None
        self.ln_post = vision_transformer.ln_post

        proj_weight = vision_transformer.proj  # 原proj是Parameter
        proj_shape = proj_weight.shape  # 例如：torch.Size([512, 768])
        
        # 创建与原proj同形状的adapter_proj（用Parameter）
        self.adapter_proj = nn.Parameter(torch.empty(proj_shape, device=device))
        
        # 复制初始权重（与原proj完全相同）
        with torch.no_grad():
            self.adapter_proj.copy_(proj_weight)
    
    # 前向传播方法完全不变，无需修改
    def forward(self, x: torch.Tensor, normalize=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # class_emb = self.class_embedding.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
        # x = torch.cat([class_emb, x], dim=1)
        def _expand_token(token, batch_size: int):
            return token.view(1, 1, -1).expand(batch_size, -1, -1)
        
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.pos_embed.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        
        # x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)
        
        def _global_pool(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.pool_type == 'avg':
                pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
            elif self.pool_type == 'tok':
                pooled, tokens = x[:, 0], x[:, 1:]
            else:
                pooled = tokens = x

            return pooled, tokens
        
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = _global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = _global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = _global_pool(x)
        
        # x = self.ln_post(x[:, 0, :])
        
        if self.adapter_proj is not None:
            x = pooled @ self.adapter_proj
        else:
            x = pooled
            
        return F.normalize(x, dim=-1) if normalize else x

class CLIPAdapter(nn.Module):
    """完整的带Adapter的open_clip模型：视觉Encoder+文本Encoder+逻辑计算"""
    def __init__(self, classnames, clip_model, low_dim, device):
        super().__init__()
        self.clip_model = clip_model  # open-clip 3.2.0 CLIP 实例
        self.dtype = next(clip_model.parameters()).dtype  # 3.2.0 无 dtype 属性，从参数取
        self.logit_scale = clip_model.logit_scale  # 温度系数（固定或可训练）
        self.original_text_features = None  # 预计算文本特征
        self.device = device
        # 1. 视觉编码器：带 Adapter
        self.image_encoder = ImageEncoder(
            vision_transformer=clip_model.visual,
            low_dim=low_dim,
            dtype=self.dtype,
            device=self.device
        )
        self.logit_scale = clip_model.logit_scale
        
    def forward(self, image, clip_zs=False):
        """前向传播：支持带Adapter的推理/原CLIP零样本推理"""
        if not clip_zs:
            # 1. 带Adapter的视觉特征提取
            image_features = self.image_encoder(image.type(self.dtype), False)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化

            # 2. 计算分类逻辑（与原逻辑一致）
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ self.original_text_features.T

        else:
            # 3. 原CLIP零样本推理（对比用）
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ self.original_text_features.T
        return logits
        
    def get_text_feature(self, dataset_name, classnames):
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        if dataset_name == "imagenet100":
            # 保留原有的模板逻辑（需确保 CUSTOM_TEMPLATES 已定义）
            temp = CUSTOM_TEMPLATES[dataset_name]
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            
            # 关键修改1：OpenCLIP 用 open_clip.tokenize 替代 clip.tokenize
            prompts = torch.cat([tokenizer(p) for p in prompts])
            
            # 关键修改2：设备配置（从 .cuda() 改为模型设备，更灵活）
            prompts = prompts.to(self.device)

            # 关键修改3：OpenCLIP 文本编码（复用模型组件，与原 encode_text 逻辑一致）
            with torch.no_grad():
                # OpenCLIP 文本编码流程（还原原 encode_text 内部逻辑）
                original_text_features = self.clip_model.encode_text(prompts)
                original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)

            logging.info(f"classnames={classnames}")

        # 2. 处理自定义类-句子映射（classnames 为字典：{类名: [句子列表]}）
        else:
            all_sentence_features = []  # 存储所有句子的特征
            class_mapping = []          # 存储每个句子对应的“类索引”

            # 遍历每个类的所有句子（原逻辑完全保留）
            for class_idx, (class_name, sentences) in enumerate(classnames.items()):               
                # 3.2 编码句子特征（OpenCLIP 流程，还原原 encode_text 逻辑）
                sentences_token = tokenizer(sentences).to(self.device)
                with torch.no_grad():
                    sentence_features = self.clip_model.encode_text(sentences_token)
                    sentence_features = sentence_features / sentence_features.norm(dim=-1, keepdim=True)
                
                # 3.3 记录特征和类映射（原逻辑完全保留）
                all_sentence_features.append(sentence_features)
                class_mapping.extend([class_idx] * len(sentences))  # 每个句子对应同一个类索引

            # 合并所有句子特征（原逻辑完全保留）
            all_sentence_features = torch.cat(all_sentence_features, dim=0)
            class_mapping = torch.tensor(class_mapping, device=self.device)  # 形状：[总句子数]

            # 4. 按类聚合（平均池化，原逻辑完全保留）
            num_classes = len(classnames)
            feature_dim = all_sentence_features.shape[1]
            class_features = torch.zeros((num_classes, feature_dim), device=self.device)

            for class_idx in range(num_classes):
                class_mask = (class_mapping == class_idx)  # 筛选当前类的句子
                class_sentence_features = all_sentence_features[class_mask]
                
                if class_sentence_features.shape[0] > 0:
                    class_features[class_idx] = class_sentence_features.mean(dim=0)
                    class_features[class_idx] = class_features[class_idx] / class_features[class_idx].norm(dim=-1, keepdim=True)
            
            original_text_features = class_features
            logging.info(f"classnames={list(classnames.keys())}")

        # 存储最终文本特征（与原逻辑一致）
        self.original_text_features = original_text_features
        
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)  
cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

if __name__ == "__main__":
    # test_dataset = CIFAR10(root='/home/lhz/data', train=False, download=True, transform=transforms.Compose([
    #         transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 保持结构更好
    #         # transforms.Resize(224),
    #         transforms.CenterCrop(224),  # 实际没必要crop，但写上更标准
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    #     ]))
    class args:
        dataset = "cub"
        ssl_indexes = f'random_splits/cub_50_50_44007.pkl'
        lbl_percent = 50 
        no_class = 200
        no_known = 100
        split_root = 'random_splits'
        split_id = 44007
        
        # classname = test_dataset.classes
        # classname = ['robin', 'water_ouzel', 
        # 'box_turtle', 'sea_snake', 'diamondback', 'sidewinder', 'scorpion', 'goose', 'tusker', 'American_coot', 'oystercatcher', 
        # 'albatross', 'toy_terrier', 'bluetick', 'Staffordshire_bullterrier', 'Border_terrier', 'Norfolk_terrier', 'cairn', 'giant_schnauzer', 
        # 'Scotch_terrier', 'flat-coated_retriever', 'Irish_setter', 'schipperke', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Doberman', 
        # 'dalmatian', 'coyote', 'Arctic_fox', 'grey_fox', 'cougar', 'leopard', 'American_black_bear', 'ringlet', 'wood_rabbit', 'guinea_pig', 
        # 'guenon', 'proboscis_monkey', 'analog_clock', 'ashcan', 'bicycle-built-for-two', 'broom', 'bucket', 'computer_keyboard', 'cowboy_hat', 
        # 'crash_helmet', 'dam', 'dumbbell', 'electric_guitar', 'envelope', 'file', 'gown', 'hand_blower', 'hatchet', 'honeycomb', 'knee_pad', 
        # 'lawn_mower', 'maillot', 'manhole_cover', 'maze', 'microphone', 'mitten', 'neck_brace', 'obelisk', 'oboe', 'organ', 'pickelhaube', 
        # 'picket_fence', 'plane', 'planetarium', 'pop_bottle', 'printer', 'purse', 'recreational_vehicle', 'shoe_shop', 'shower_curtain', 
        # 'sleeping_bag', 'steel_arch_bridge', 'stole', 'stretcher', 'stupa', 'table_lamp', 'thresher', 'tobacco_shop', 'totem_pole', 'trimaran', 
        # 'unicycle', 'upright', 'vending_machine', 'washer', 'Windsor_tie', 'wing', 'wreck', 'guacamole', 'trifle', 'bagel', 'mashed_potato', 
        # 'banana', 'rapeseed']

    _, _, _, _, test_dataset, cnames = get_cub(args())
    args.classname = cnames
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    model = adapter_open_clip(args())
    # model, ppp = clip.load("ViT-B/16", device="cuda:5")
    # 设置默认pipipip设备为cuda:0
    # torch.set_default_device("cuda:4")
    print(model)
    model.eval()
    ground_truth = []
    pred_label = []
    with torch.no_grad():
        # all_text = [f"a photo of a {c}." for c in test_dataset.classes]
        # text = tokenizer(all_text).to(device)
        # text_features = model.encode_text(text)
        # print(text_features)
        for idx, (image, label) in enumerate(test_loader):
            image = image.cuda()
            # image_embeds = model.encode_image(image)
            # image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            # logits = image_embeds @ text_features.T
            # preds = logits.argmax(dim=-1)
            # logits = model(image, text)[0]
            logits = model(image, True)
            preds = logits.argmax(dim=1)
            pred_label.append(preds)
            ground_truth.append(label)
            print(idx)
    
    ground_truth = torch.cat(ground_truth, dim=0).cpu()
    pred_label = torch.cat(pred_label, dim=0).cpu()
    acc = (pred_label == ground_truth).float().mean()
    print(f"Accuracy: {acc}")
