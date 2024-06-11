import torch

# build densenet121
import torch.nn as nn
import torchvision.models as models


def _cfg_to_densenet(args):
    return {
        "num_classes": args.n_classes
    }
    
def densenet121(num_classes=1000):
    model = models.densenet121(pretrained=False, num_classes=num_classes)
    return model

