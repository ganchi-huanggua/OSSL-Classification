import torch
import open_clip
from models.CoOp import load_clip_to_cpu
def print_module_structure(module, prefix=''):
    """递归打印模块结构和属性"""
    # 打印当前模块名称和类型
    print(f"{prefix}[{module.__class__.__name__}]")
    
    # 打印模块的所有属性（过滤掉私有属性）
    attrs = [attr for attr in dir(module) if not attr.startswith('_')]
    for attr in attrs:
        try:
            value = getattr(module, attr)
            # 只打印基础类型和模块类型，避免冗长输出
            if isinstance(value, (int, float, str, bool, torch.nn.Module)):
                print(f"{prefix}  .{attr}: {type(value).__name__}")
        except Exception:
            continue
    
    # 递归打印子模块
    for name, child in module.named_children():
        print_module_structure(child, prefix + '  |-')

# 加载 open-clip-torch==3.2.0 的 ViT-H-14 模型
# clip_model, _, _ = open_clip.create_model_and_transforms(
#     model_name="ViT-H-14",
#     pretrained="laion2B-s32B-b79K",
#     cache_dir="./pretrained",
#     device="cpu"

clip_model = load_clip_to_cpu("ViT-B/16")
# 打印视觉编码器的 Transformer 层结构
print("="*50)
print("Vision Transformer 层级结构：")
print("="*50)
# print_module_structure(clip_model.visual.transformer)

print(clip_model.visual.transformer.resblocks)