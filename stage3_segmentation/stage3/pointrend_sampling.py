"""课时 3：PointRend 不确定性采样——loss_utils.loss_masks 的抽题官复刻。

对应主仓库（detectron2 三函数的语义）：
  1) 撒 oversample_ratio(3.0) 倍候选点（均匀随机）；
  2) 在候选点上算"不确定性" = -|logit|（概率越接近 0.5 越拿不准）；
  3) 取 importance(0.75) 比例最不确定的 + 其余纯随机；
  4) 共 num_points = 256x256 = 65,536 个（约占 1024² 全图的 6%）。

"在省什么"的答案：百万像素逐点算损失太贵，且绝大多数是"一眼正常"
的简单题——把算力集中到模型最犹豫的边界带。

运行：python -m stage3.pointrend_sampling
"""
import torch


def uncertainty_neg_abs_logit(logits: torch.Tensor) -> torch.Tensor:
    """主仓库 calculate_uncertain：logit 离 0 越近(概率≈0.5)值越大。"""
    return -torch.abs(logits)


def sample_uncertain_points(logits: torch.Tensor, num_points: int = 256 * 256,
                            oversample_ratio: float = 3.0,
                            importance: float = 0.75, seed: int = 0) -> torch.Tensor:
    """复刻 get_uncertain_point_coords_with_randomness 的两段式抽题。

    logits: [B,1,H,W]；返回 [B, num_points, 2] 的 (y,x) 坐标。
    """
    b = logits.shape[0]
    g = torch.Generator().manual_seed(seed)
    n_over = int(num_points * oversample_ratio)
    n_imp = int(num_points * importance)

    coords_all = []
    for i in range(b):
        flat = logits[i].flatten()                         # [H*W]
        # 第 1 段：均匀撒 n_over 个候选
        cand = torch.randperm(flat.numel(), generator=g)[:n_over]
        # 第 2 段：候选里挑最不确定的 n_imp 个
        unc = -flat[cand].abs()
        imp_idx = cand[unc.topk(n_imp).indices]
        # 第 3 段：其余名额纯随机补（防"视野被当前水平锁死"）
        rest_pool = cand[~torch.isin(cand, imp_idx)]
        rest_idx = rest_pool[torch.randperm(len(rest_pool), generator=g)
                             [:num_points - n_imp]]
        chosen = torch.cat([imp_idx, rest_idx])
        ys, xs = chosen // logits.shape[3], chosen % logits.shape[3]
        coords_all.append(torch.stack([ys, xs], -1))
    return torch.stack(coords_all, 0)


def coverage_stats(logits, coords) -> dict:
    """被抽中的点落在"边界带"(不确定性高)的比例 vs 纯随机基线。"""
    flat_unc = -logits.flatten().abs()
    unc_of_chosen = flat_unc[coords[..., 0] * logits.shape[-1]
                             + coords[..., 1]]
    rand = torch.randperm(flat_unc.numel())[:coords.numel() // 2]
    return {"chosen_mean_unc": unc_of_chosen.mean().item(),
            "random_mean_unc": flat_unc[rand].mean().item()}


if __name__ == "__main__":
    torch.manual_seed(0)
    # 合成 logit：大部分区域很确定(|logit|>4)，少数边界带不确定(|logit|<0.5)
    logits = (torch.rand(1, 1, 128, 128) - 0.5) * 10
    logits[:, :, 50:60, :] = (torch.rand(1, 1, 10, 128) - 0.5) * 0.5  # 边界带
    coords = sample_uncertain_points(logits, num_points=1024, seed=0)
    stats = coverage_stats(logits, coords)
    print("抽题数:", coords.shape, "(应 [1, 1024, 2])")
    print(f"被抽中点平均不确定性 = {stats['chosen_mean_unc']:.3f}")
    print(f"纯随机基线           = {stats['random_mean_unc']:.3f}")
    print("-> 抽题官显著偏向不确定区域（边界带），这就是火力集中")
