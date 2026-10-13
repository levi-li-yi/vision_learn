"""课时 3：完整训练循环 + 梯度累积——trainer.train() 的可测试骨架。
这个文件教你写一个完整的训练循环——就是'喂图、猜答案、算错多少、调旋钮'这四步的代码版，还教'显存不够时怎么分小批攒着当一大批用'。


对应主仓库 trainer.py 114-157 行逐句：
  前向 -> loss_masks -> 总分÷accumulation_steps -> scale 反传 ->
  (累积最后一步) unscale -> 裁剪 -> step -> sched.step -> zero_grad。
梯度累积的数学：N 小批各自 backward 梯度会**相加**，故每小批 loss 先 ÷N，
总效果恰等价于一大批——本模块用测试钉死这个等价性。

运行：python -m stage2.train_loop
"""
import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, accumulation_steps=1,
                    max_norm=1.0, use_scaler=None):
    """单 epoch 训练。use_scaler 传 GradScaler 或 None（bf16 场景近似恒等）。

    与主仓库一致的三个细节：
      1) loss ÷ accumulation_steps；
      2) 只有累积的最后一步才 step/zero（中间步只 backward）；
      3) step 前先 unscale 再裁剪 max_norm。
    """
    model.train()
    total, nbatch = 0.0, 0
    for i, (x, y) in enumerate(loader, 1):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss = loss / accumulation_steps
        loss.backward()
        if i % accumulation_steps == 0:
            if use_scaler is not None:
                use_scaler.unscale_(optimizer)
            if max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if use_scaler is not None:
                use_scaler.step(optimizer)
                use_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        total += loss.item() * accumulation_steps
        nbatch += 1
    return total / max(1, nbatch)


@torch.no_grad()
def accuracy(model, loader):
    """
    算模型在 loader 上的分类准确率（评估用，不算梯度）。
    """
    model.eval()
    hit = tot = 0
    for x, y in loader:
        hit += int((model(x).argmax(1) == y).sum())
        tot += len(y)
    return hit / max(1, tot)


def grads_close(m1: nn.Module, m2: nn.Module, tol=1e-5) -> bool:
    """两个模型同名参数梯度是否近似相等（累积等价性验证用）。"""
    for (k1, p1), (k2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if k1 != k2 or p1.grad is None or p2.grad is None:
            return False
        if not torch.allclose(p1.grad, p2.grad, atol=tol):
            return False
    return True


if __name__ == "__main__":
    from .nn_module_dataset_loader import BrightCornerDataset, collate_stack, TinyCNN
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    ds = BrightCornerDataset(512)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_stack)
    model, opt = TinyCNN(), torch.optim.AdamW(model.parameters(), lr=3e-3)
    for ep in range(1, 4):
        loss = train_one_epoch(model, loader, opt)
        print(f"epoch {ep}: loss={loss:.4f} acc={accuracy(model, loader):.2%}")
