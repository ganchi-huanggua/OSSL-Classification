import torch.nn as nn
import torch
from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import math
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import os
# from datasets.imagenet100 import GenericTEST
from datasets.cub import get_cub
from datasets.stanfordcars import get_stanfordcars
import logging
# source /etc/profile.d/clash.sh
# proxy_on

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

_tokenizer = _Tokenizer()

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="./pretrained")
    logging.info("Pretrained clip model parameters will be saved in {}".format(model_path))
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model
   
def adapter_openai(args):
    backbone_name = "ViT-B/16"
    classnames = args.classname
    low_dim = args.low_dim
    clip_model = load_clip_to_cpu(backbone_name)
    # clip_model, _ = clip.load("ViT-B/16", device="cuda:5")
    clip_model.float()
    model = CLIPAdapter(classnames, clip_model, low_dim)
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    # for name, param in model.named_parameters():
    #     # 判断参数名是否不在需要训练的参数列表中
    #     if not any(name.startswith(x) for x in ['prototypes', 'concept_prototypes', "proj", "proj2", "prompt_learner"]):
    #         param.requires_grad_(False)
    # print("learnable parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    model.cuda()
    model.get_text_feature(args.dataset, classnames)
    # print(model.original_text_features)
    return model


class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim=10, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.GELU()
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x


class AdapterResidualBlock(nn.Module):
    def __init__(self, block, low_dim):
        super().__init__()
        self.attn = block.attn
        self.ln_1 = block.ln_1
        self.mlp = block.mlp
        self.ln_2 = block.ln_2
        self.attn_mask = block.attn_mask
        
        self.adapter = AdaptFormer(self.attn.embed_dim, low_dim)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) + self.adapter(x)
        return x

class AdapterTransformer(nn.Module):
    def __init__(self, transformer, low_dim):
        super().__init__()
        self.width = transformer.width
        self.layers = transformer.layers
        self.resblocks = nn.Sequential(*[
            AdapterResidualBlock(block, low_dim) 
            for block in transformer.resblocks
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class ImageEncoder(nn.Module):
    def __init__(self, vision_transformer, low_dim):
        super().__init__()
        self.input_resolution = vision_transformer.input_resolution
        self.output_dim = vision_transformer.output_dim
        self.conv1 = vision_transformer.conv1

        self.class_embedding = vision_transformer.class_embedding
        self.positional_embedding = vision_transformer.positional_embedding
        self.ln_pre = vision_transformer.ln_pre

        self.transformer = AdapterTransformer(vision_transformer.transformer, low_dim)

        self.ln_post = vision_transformer.ln_post
        
        proj_weight = vision_transformer.proj  # 原proj是Parameter
        proj_shape = proj_weight.shape  # 例如：torch.Size([512, 768])
        
        # 创建与原proj同形状的adapter_proj（用Parameter）
        self.adapter_proj = nn.Parameter(torch.empty(proj_shape, device=proj_weight.device))
        
        # 复制初始权重（与原proj完全相同）
        with torch.no_grad():
            self.adapter_proj.copy_(proj_weight)
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.adapter_proj is not None:
            x = x @ self.adapter_proj

        return x
    
class CLIPAdapter(nn.Module):
    def __init__(self, classnames, clip_model, low_dim):
        super().__init__()
        self.clip_model = clip_model
        
        self.text_encoder = clip_model.transformer
        self.image_encoder = ImageEncoder(clip_model.visual, low_dim)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, clip_zs=False, return_feat=False):
        if clip_zs == False:
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ self.original_text_features.t()
            # logits = image_features @ text_features.t()
            if return_feat:
                return logits, image_features
            else:
                return logits
        else:
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # logit_scale = self.clip_model.logit_scale.exp()
            # logits = logit_scale * image_features @ self.original_text_features.t()
            logits = image_features @ self.original_text_features.T
            if return_feat:
                return logits, image_features
            else:
                return logits
            
    def get_image_feature(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def get_text_feature(self, dataset_name, classnames):
        if dataset_name == "imagenet100":
            temp = CUSTOM_TEMPLATES[dataset_name]
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.cuda()
            with torch.no_grad():
                original_text_features = self.clip_model.encode_text(prompts)
                original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)
            logging.info(f"classnames={classnames}")

        else:
            all_sentence_features = []  # 存储所有句子的特征
            class_mapping = []          # 存储每个句子对应的“类索引”（如0代表"pink primrose"，1代表"red rose"）

            for class_idx, (class_name, sentences) in enumerate(classnames.items()):
                # 3.1 Tokenize当前类的所有句子
                sentence_tokens = clip.tokenize(sentences).cuda()  # 形状：[句子数, 77]
                
                # 3.2 编码句子特征
                with torch.no_grad():
                    sentence_features = self.clip_model.encode_text(sentence_tokens)  # 形状：[句子数, 512]
                    sentence_features = sentence_features / sentence_features.norm(dim=-1, keepdim=True)  # 归一化
                
                # 3.3 记录特征和类映射
                all_sentence_features.append(sentence_features)
                class_mapping.extend([class_idx] * len(sentences))  # 每个句子对应同一个类索引

            # 合并所有句子特征（形状：[总句子数, 512]）
            all_sentence_features = torch.cat(all_sentence_features, dim=0)
            class_mapping = torch.tensor(class_mapping, device=sentence_tokens.device)  # 形状：[总句子数]

            # ----------------------
            # 4. 步骤2：按类聚合（平均池化）
            num_classes = len(classnames)
            feature_dim = all_sentence_features.shape[1]  # CLIP特征维度（如512）
            class_features = torch.zeros((num_classes, feature_dim), device=sentence_tokens.device)  # 存储类级特征

            # 遍历每个类，计算该类所有句子特征的平均值
            for class_idx in range(num_classes):
                # 筛选出当前类的所有句子特征
                class_mask = (class_mapping == class_idx)  # 布尔掩码：当前类的句子为True
                class_sentence_features = all_sentence_features[class_mask]  # 形状：[该类句子数, 512]
                
                # 计算平均值（避免除以0，若类无句子则为0向量）
                if class_sentence_features.shape[0] > 0:
                    class_features[class_idx] = class_sentence_features.mean(dim=0)
                    # 可选：再次归一化（确保类特征范数为1）
                    class_features[class_idx] = class_features[class_idx] / class_features[class_idx].norm(dim=-1, keepdim=True)
            original_text_features = class_features
            logging.info(f"classnames={list(classnames.keys())}")
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
        dataset = "stanfordcars"
        ssl_indexes = f'random_splits/stanfordcars_50_50_44007.pkl'
        lbl_percent = 50 
        no_class = 196
        no_known = 98
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

    _, _, _, _, test_dataset, cnames = get_stanfordcars(args())
    args.classname = cnames
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    model = adapter_openai(args())
    # model, ppp = clip.load("ViT-B/16", device="cuda:5")
    # 设置默认设备为cuda:0
    # torch.set_default_device("cuda:4")
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
