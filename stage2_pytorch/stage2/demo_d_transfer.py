"""★Demo D：迁移学习——冻结 vs 差异化学习率，体会 backbone_lr_scale=0.1。
这个文件教你'迁移学习'：拿一个已经学过看图的预训练模型，只换最后的分类头，在小数据集上微调——体验'骨干小步、新头大步'的效果。


对应主仓库：fresh 模式加载 pretrained.pth.tar（563 键骨干）+ 头部随机初始化
+ AdamW 两组旋钮（骨干 lr×0.1）——"转学生底子只精修不重学"的用意。

本模块可测试的核心（离线）：
  split_transfer_groups()：预训练骨干组 ×scale + 新头组 ×1.0；
  freeze_backbone()：冻结的另一种做法（requires_grad=False，梯度不产生）。
真实练习（作业）：--dataset 取 torchvision Flowers102 或自备猫狗小数据集，
观察"冻结快但上限低 / 差异化慢热但上限高"。

运行：python -m stage2.demo_d_transfer
"""
import torch
from torch import nn


class PretrainedLikeBackbone(nn.Module):
    """扮演 ResNet18.backbone：名字带 backbone. 前缀。"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU())
        self.head = nn.Linear(64, 2)          # 新头：随机初始化

    def forward(self, x):
        return self.head(self.backbone(x))


def load_pretrained_backbone(model: nn.Module) -> int:
    """模拟主仓库 load_pretrained_encoder：只装骨干、头部保持随机。
    返回命中键数。"""
    donor = PretrainedLikeBackbone()
    donor_keys = {k: v for k, v in donor.state_dict().items()
                  if k.startswith("backbone.")}
    target = model.state_dict()
    hit = 0
    with torch.no_grad():
        for k, v in donor_keys.items():
            if k in target and target[k].shape == v.shape:
                target[k].copy_(v)
                hit += 1
    model.load_state_dict(target)
    return hit


def split_transfer_groups(model: nn.Module, lr=1e-3, backbone_lr_scale=0.1):
    """差异化学习率：转学生组小步、新头组大步（主仓库同款结构）。"""
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    return [{"params": bb, "lr": lr * backbone_lr_scale, "name": "backbone"},
            {"params": hd, "lr": lr, "name": "head"}]


def freeze_backbone(model: nn.Module) -> nn.Module:
    """冻结式：骨干参数完全不更新（对应'冻结与否两种方式'的前者）。"""
    for n, p in model.named_parameters():
        if n.startswith("backbone"):
            p.requires_grad_(False)
    return model


def train_two_ways(model_ctor, loader, epochs=5, lr=1e-3):
    """同一数据下对比两种迁移方式的准确率（合成数据即可复现结论）。"""
    results = {}
    for mode in ("freeze", "differential"):
        torch.manual_seed(0)
        model = model_ctor()
        load_pretrained_backbone(model)
        if mode == "freeze":
            freeze_backbone(model)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=lr)
        else:
            opt = torch.optim.AdamW(
                split_transfer_groups(model, lr=lr, backbone_lr_scale=0.1))
        model.train()
        for _ in range(epochs):
            for x, y in loader:
                loss = nn.functional.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        hit = tot = 0
        with torch.no_grad():
            for x, y in loader:
                hit += int((model(x).argmax(1) == y).sum()); tot += len(y)
        results[mode] = hit / tot
    return results


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    # 合金数据：x 32 维，标签由骨干已学到的线性映射决定（预训练优势成立）
    W = torch.randn(32, 64)
    x = torch.randn(256, 32)
    y = (x @ W).relu().sum(1) > 0
    loader = DataLoader(TensorDataset(x, y.long()), batch_size=32)

    donor = PretrainedLikeBackbone()
    with torch.no_grad():
        donor.backbone[1].weight.copy_(W.T)      # "预训练"真的学过这个映射
    torch.save({"weight": donor.state_dict()}, "./_pretrained_like.pth")

    class M(PretrainedLikeBackbone):
        def __init__(self):
            super().__init__()
            sd = torch.load("./_pretrained_like.pth", weights_only=False)["weight"]
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU())
            self.backbone[1].load_state_dict({k.split('.')[-1]: v for k, v in sd.items()
                                              if 'backbone.1' in k})

    print(train_two_ways(M, loader, epochs=5))
