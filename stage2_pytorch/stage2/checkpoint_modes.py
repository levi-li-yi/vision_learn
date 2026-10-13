"""课时 6：断点五件套与 fresh/resume/finetune——base_utils/utils.py 的复刻。
这个文件教你训练中断后怎么恢复：断点文件里存了什么(五件套)、三种重启方式(fresh/resume/finetune)各恢复哪些东西。


对应主仓库：
  - save_trainer_state / load_trainer_state（五件套：weight/optimizer/
    scheduler_last_epoch/scaler/step，任一失败 warning 降级不崩）；
  - 三模式语义：fresh 删旧断点重来；resume 全恢复；finetune 只穿权重。
断点的本质：一个 dict 存进 .pth 文件——没有魔法，只有字典。

运行：python -m stage2.checkpoint_modes
"""
import copy
import tempfile
from pathlib import Path

import torch
from torch import nn


def save_five(model, optimizer, scheduler, scaler, step, path):
    """五件套打包（DDP 解包 .module 的动作在主仓库，这里单卡略）。"""
    torch.save({
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_last_epoch": scheduler.last_epoch,
        "scaler": scaler.state_dict(),
        "step": step,
    }, path)


def load_five(path, model, optimizer=None, scheduler=None, scaler=None):
    """恢复五件套；optimizer/scheduler/scaler 任一失败只返回标志不抛异常
    （对应主仓库 warning 降级）。返回恢复到的 step。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["weight"], strict=False)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass                                   # 降级：全新优化器
    if scheduler is not None:
        try:
            scheduler.last_epoch = ckpt.get("scheduler_last_epoch", 0)
        except Exception:
            pass
    if scaler is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
    return ckpt.get("step", 0)


def apply_mode(mode: str, has_old_state: bool) -> str:
    """三模式对断点的处置（主仓库 train_entry+trainer 的分支逻辑）。"""
    if mode == "fresh":
        return "delete_old_state"                 # 只删断点，不删部署权重
    if mode in ("resume", "finetune"):
        return "restore_five" if has_old_state else "warn_from_scratch"
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    torch.manual_seed(0)
    m = nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters())
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    sc = torch.amp.GradScaler(enabled=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "trainer_state.pth.tar"
        save_five(m, opt, sch, sc, step=1234, path=p)
        m2 = nn.Linear(4, 2)
        step = load_five(p, m2, opt=None, scheduler=None, scaler=None)
        same = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(),
                                                    m2.state_dict().values()))
        print(f"恢复 step={step}, 权重逐位一致={same}")
    print("三模式处置:", {mm: apply_mode(mm, True) for mm in ("fresh", "resume", "finetune")})
