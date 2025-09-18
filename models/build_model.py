import torch
from .CoOp import coop
from .adapter import adapter

import logging
def build_model(args, ema=False):

    model = adapter(args)
    # model = coop(args)

    # use dataparallel if there's multiple gpus
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model
