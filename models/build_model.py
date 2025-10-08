import torch
from .CoOp import coop
from .adapter_openai import adapter_openai
from .adapter_open_clip import adapter_open_clip
import logging
def build_model(args, ema=False):

    # model = adapter_open_clip(args)
    model = adapter_openai(args)
    teacher_model = adapter_open_clip(args)
    # model = coop(args)

    # use dataparallel if there's multiple gpus
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model, teacher_model
