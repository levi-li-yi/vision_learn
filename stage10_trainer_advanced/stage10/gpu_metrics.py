"""课时 3：GPU 侧指标累加与考试暖机 —— trainer.py::pop_metrics + main.py 暖机段。

  1. 训练指标 GPU 侧累加：loss / IoU 全程驻留 device（`self._m_bce += ...`），
     只在打日志时 pop_metrics 一次 `.tolist()` 同步回主机——每步一次 GPU→CPU
     同步会把流水线打穿（每次同步 ~ms 级，每步 4 个指标就是 4 次 stalls）；
  2. batch IoU 也在 GPU 上算：preds/gt/inter/union 全 tensor 化，`torch.where`
     只对含缺陷样本计 IoU（union=0 的除零用 clamp_min(1) 兜底）；
  3. 考试算型暖机：开训前用考试的极端规格（b16 与尾批 b8）各跑一遍 bf16 前向，
     固化算子计划——不然考试第一张大图的内核编译长尾会顶到"单图限时 10s"红线。

对应主仓库：src/train/trainer.py::train（metrics 段）/ pop_metrics、
src/train/main.py::_worker（暖机段："[暖机] 考试算型 (bf16 b16/b8) 已固化"）。

运行：python -m stage10.gpu_metrics
"""
import time

import torch
import torch.nn as nn


class GPUMetricAccumulator:
    """四指标 GPU 侧累加器（focal/dice/iou_sum/iou_cnt），pop 时一次同步。"""

    def __init__(self, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dev = torch.device(device)
        self.sync_count = 0                     # 观测用：主机同步次数
        self._m = torch.zeros(4, device=self._dev)

    @torch.no_grad()
    def update(self, focal: torch.Tensor, dice: torch.Tensor,
               probs: torch.Tensor, masks: torch.Tensor):
        """focal/dice 为标量；probs (B,H,W) sigmoid 概率；masks 值域 {0,1,255}。"""
        valid = (masks != 255).float()
        preds = (probs > 0.5).float() * valid
        gt = (masks == 1).float() * valid
        inter = (preds * gt).sum((1, 2))
        union = preds.sum((1, 2)) + gt.sum((1, 2)) - inter
        ok = (gt.sum((1, 2)) > 0) & (union > 0)
        iou_sum = torch.where(ok, inter / union.clamp_min(1.0),
                              torch.zeros_like(inter)).sum()
        row = torch.stack([focal.detach(), dice.detach(), iou_sum, ok.float().sum()])
        self._m += row.to(self._dev)

    def pop_metrics(self, n_steps: int) -> dict:
        """取回并清零（一次同步；headroom 上这就是唯一一次）。"""
        self.sync_count += 1
        f, d, iou_sum, iou_cnt = self._m.tolist()
        self._m.zero_()
        n = max(1, n_steps)
        return {"focal": f / n, "dice": d / n,
                "iou": iou_sum / iou_cnt if iou_cnt else 0.0}


def warmup_infer_shapes(model: nn.Module, shapes=((16, 3, 64, 64), (8, 3, 64, 64))):
    """考试算型暖机：用极端规格（整批 b16 + 尾批 b8）各跑一遍前向固化算子计划。"""
    model.eval()
    was = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        for shape in shapes:
            model(torch.randn(*shape, device=next(model.parameters()).device))
    finally:
        torch.set_grad_enabled(was)
    return len(shapes)


class TinySeg(nn.Module):
    """迷你分割网（1×1 conv 链），替身 EfficientViT-SAM 前向。"""

    def __init__(self, cin=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinySeg()
    acc = GPUMetricAccumulator(device="cpu")    # 学习环境 CPU；真工程驻留 cuda

    # 指标累加：100 步只允许 1 次主机同步
    for _ in range(100):
        logits = model(torch.randn(8, 3, 64, 64))
        masks = torch.randint(0, 2, (8, 64, 64)) * 255
        masks[:, 20:28, 20:28] = 1              # 每张埋一个缺陷
        probs = torch.sigmoid(logits.squeeze(1))
        acc.update(torch.tensor(0.5), torch.tensor(0.3), probs, masks)
    m = acc.pop_metrics(100)
    print(f"100 步指标: {m} | 主机同步次数 = {acc.sync_count} (每步同步的反例会是 400 次)")

    # 暖机前后首推理耗时
    t0 = time.perf_counter()
    model(torch.randn(16, 3, 64, 64))
    cold = time.perf_counter() - t0
    warmup_infer_shapes(model, ((16, 3, 64, 64), (8, 3, 64, 64)))
    t0 = time.perf_counter()
    model(torch.randn(16, 3, 64, 64))
    warm = time.perf_counter() - t0
    print(f"暖机实验: 首次前向 {cold * 1000:.2f}ms → 暖机后 {warm * 1000:.2f}ms "
          f"(真工程 bf16 大图的算子编译长尾更夸张，会顶到单图 10s 限时)")
