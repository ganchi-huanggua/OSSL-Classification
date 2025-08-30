import torch.nn as nn
import torch
from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from datasets.imagenet100 import GenericTEST
import logging
# source /etc/profile.d/clash.sh
# proxy_on

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "oxfordflowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
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
    "imagenet100": "a photo of a {}."
}

_tokenizer = _Tokenizer()

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="../pretrained")
    logging.info("Pretrained clip model parameters will be saved in {}".format(model_path))
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def coop(args):
    backbone_name = "ViT-B/16"
    classnames = args.classname
    ctx_init = ""
    n_ctx = 32
    ctp = "end"
    clip_model = load_clip_to_cpu(backbone_name)
    # clip_model, _ = clip.load("ViT-B/16", device="cuda:5")
    clip_model.float()
    model = CoOp(classnames, clip_model, n_ctx, ctx_init, ctp)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding  # [77, 512]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, concat_prompt=None):
        # prompts: [3, 77, 512]
        # tokenized_prompts: [3, 77]
        x = prompts + self.positional_embedding.type(self.dtype)
        if concat_prompt is not None:
            x = torch.cat(
                [x, concat_prompt.to(x.device) + torch.zeros(x.shape[0], 12, x.shape[-1],
                                                           dtype=x.dtype,
                                                           device=x.device)], dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # [3, 77, 512]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x  # [3, 512]


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx, ctx_init, ctp, cfg_imsize=224, csc=False):
        super().__init__()
        n_cls = len(classnames)
        # number of context vectors, default 16, if ctx_init is not empty, this param will fail
        # n_ctx = 16
        # ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # should be 512
        clip_imsize = clip_model.visual.input_resolution
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))  # 4, number of words = number of context vectors
            # A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
            # for example: "a photo of a", prompt shape: [1, 77], LongTensor
            # 77 indices used to describe a string, 49406 and 49407 represent "<|startoftext|>" and "<|endoftext|>", respectively
            prompt = clip.tokenize(ctx_init)  # tensor([[49406, 320, 1125, 539, 320, 49407,     0,     ...     ]])
            with torch.no_grad():
                # self.token_embedding = nn.Embedding(vocab_size: 16e6, transformer_width: 512)  bpe_simple_vocab_16e6
                embedding = clip_model.token_embedding(prompt).type(dtype)  # [1, 77, 512]  torch.float16
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # [4, 512], removing "<|startoftext|>" and "<|endoftext|>"
            prompt_prefix = ctx_init  # 'a photo of a'

        else:
            # random initialization
            if csc:
                logging.info("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)  # [397, 16, 512]
            else:
                logging.info("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # [16, 512]
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)  # 'X X X X X X X X X X X X X X X X'

        logging.info(f'Initial context: "{prompt_prefix}"')
        logging.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, random initialized or tokenized "a photo of a"

        classnames = [name.replace("_", " ") for name in classnames]  # [cat, dog, mocking bird]
        # logging.info(_tokenizer.encode(classnames[2]))  # [37870, 3329]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]  # [1, 1, 2]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # "a photo of a dog." or "X X X dog."

        # After clip.tokenize, an index will be added before and after the original text to represent SOS and EOS
        # for example: A sentence with 7 words will get a total length of 77 indices after clip.tokenized, of which only
        # the first 9 indices are not 0, and the first and ninth represent SOS and EOS, and only the second to the
        # eighth indices correspond to the seven words of the original sentence.
        # but _tokenizer.encode won't
        # So the clip can only handle sentences with 75 words at most.
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # [3, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [3, 77, 512]  torch.float16

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.LongTensor
        self.name_lens = name_lens
        self.class_token_position = ctp
        # self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx  # to be optimized, random initialized or tokenized "a photo of a"
        if ctx.dim() == 2:  # [4, 512] or [397, 16, 512] or [16, 512]
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [x, 512] => [397, x, 512]

        # original sentence "<|SOS|> a photo of a <|CLS|>. <|EOS|>" or "<|SOS|> X X X <|CLS|>. <|EOS|>"
        # <|CLS|> refers to clip.tokenize tokenized and clip_model.token_embedding embedded classnames
        prefix = self.token_prefix  # "<|SOS|>"
        suffix = self.token_suffix  # "<|CLS|>. <|EOS|>"
        # X X X seems to be placeholder for ctx (to be optimized)

        # The relative position of prefix and suffix remains unchanged. Only change the position of <|CLS|> and ctx.
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )  # "<|SOS|> a photo of a <|CLS|>. <|EOS|>"

        elif self.class_token_position == "middle":  # Divide ctx into two halves and place them on either side of <|CLS|>
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )  # "<|SOS|> a photo <|CLS|> of a. <|EOS|>"
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )  # "<|SOS|> <|CLS|> a photo of a. <|EOS|>"
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # clip.tokenize tokenized and clip_model.token_embedding embedded "<|SOS|> a photo of a <|CLS|>. <|EOS|>"
        # "a photo of a" is to be optimized, clip.tokenize tokenized and clip_model.token_embedding embedded
        # and "a photo of a" maybe randomly initialized by "X X X"
        return prompts  # [3, 77, 512]


class CoOp(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", ctp="end"):
        super().__init__()
        self.clip_model = clip_model
        
        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx, ctx_init, ctp)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  # [3, 77], tokenized "a photo of a dog/cat"
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, clip_zs=False):
        if clip_zs == False:
            image_features = self.image_encoder(image.type(self.dtype))

            prompts = self.prompt_learner()  # [3, 77, 512]
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            # logits = image_features @ text_features.t()

            return logits
        else:
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ self.original_text_features.t()
            # logits = image_features @ self.original_text_features.T
            return logits
    
    def get_text_feature(self, dataset_name, classnames):
        temp = CUSTOM_TEMPLATES[dataset_name]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        # prompts = [temp.format(c) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.cuda()

        with torch.no_grad():
            original_text_features = self.clip_model.encode_text(prompts)
            original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)

        self.original_text_features = original_text_features
        
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)  
cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

if __name__ == "__main__":
    test_dataset = CIFAR10(root='/home/lhz/data', train=False, download=True, transform=transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 保持结构更好
            # transforms.Resize(224),
            transforms.CenterCrop(224),  # 实际没必要crop，但写上更标准
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ]))
    # test_dataset = GenericTEST(os.path.join("/home/lhz/data/imagenet100", 'val'), no_class=100, transform=transforms.Compose([
    #         transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 保持结构更好
    #         # transforms.Resize(224),
    #         transforms.CenterCrop(224),  # 实际没必要crop，但写上更标准
    #         transforms.ToTensor(),
    #         transforms.Normalize(imgnet_mean, imgnet_std)
    #     ]))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    class args:
        dataset = "cifar10"
        classname = test_dataset.classes
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
    model = coop(args())
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
