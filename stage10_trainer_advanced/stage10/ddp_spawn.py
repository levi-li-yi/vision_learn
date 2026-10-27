"""课时 4：DDP 最小可用工程（gloo CPU 双进程）—— main.py::_worker / run_training 复刻。

真工程多卡训练的全部工程要点（本课在 Windows CPU 上用 gloo 后端真实跑通）：

  · mp.spawn 拉起 world_size 个进程 + env:// rendezvous（MASTER_ADDR/PORT，
    端口自动探测空闲）；
  · init_process_group(backend="nccl"/"gloo", timeout=60min)——60 分钟看门狗：
    rank0 独占跑整图考试时其余 rank 要干等，默认 30min 超时会误杀；
  · SyncBatchNorm.convert_sync_batchnorm（GPU 环境）：把 BN 换成跨卡同步版
    （running stats 全卡一致；仅支持 CUDA 模块，本课 CPU 学习环境自动跳过）；
  · find_unused_parameters：冻结段/LoRA 部分前向不触碰全部参数，需要时打开；
  · 标量广播用张量 dist.broadcast：torch 2.5.1 的 broadcast_object_list 在
    NCCL 上有 SymInt bug——工程上宁可用张量直传；
  · 梯度 all_reduce(AVG)（LoRA 训练侧手动 all_reduce 同款）保证各 rank 参数一致。

对应主仓库：src/train/main.py::run_training（mp.spawn 段）/_worker（init_process_group、
broadcast、考试段）/ _dist_timeout、src/train/trainer.py（SyncBN 转换、DDP 包装）。

运行：python -m stage10.ddp_spawn          # 双进程真实 DDP（gloo/CPU）
"""
import os
import socket

import torch
import torch.multiprocessing as mp
import torch.nn as nn


def find_free_port() -> int:
    """探测空闲端口（rendezvous 用；真工程 _free_lora_port 同款）。"""
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ToyNet(nn.Module):
    """玩具任务网：拟合 y = x 的均值（梯度跨卡 all_reduce 后各 rank 参数应一致）。"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)            # 转 SyncBN 的目标

    def forward(self, x):
        return self.bn(self.fc(x))


def ddp_worker(rank: int, world_size: int, port: int, out_queue=None):
    """单个 rank 的训练 worker：init → SyncBN → DDP → 训练 → 广播 → all_reduce。"""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="gloo",                        # Windows/CPU 用 gloo；真工程 NCCL
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.manual_seed(42)                      # 各 rank 同初始化
    model = ToyNet()
    if torch.cuda.is_available():
        # SyncBN 仅支持 GPU 模块（真工程在 cuda 上必转；CPU/gloo 学习环境保持普通 BN）
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=False)
    device = torch.device("cpu")
    if rank == 0:
        torch.manual_seed(7)                   # rank0 单独换种子 → 数据分片不同
    xs = torch.randn(32, 4)
    ys = xs.mean(dim=1, keepdim=True).expand(-1, 4)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        out = model(xs.to(device))
        loss = ((out - ys) ** 2).mean()
        opt.zero_grad()
        loss.backward()                        # DDP 钩子自动 all_reduce 梯度
        opt.step()
        losses.append(loss.item())

    # 标量用张量广播（规避 torch 2.5.1 broadcast_object_list 的 NCCL SymInt bug）
    t = torch.tensor([losses[-1]])
    dist.broadcast(t, src=0)
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([losses[-1]]))

    # 各 rank 参数一致性验证
    w = next(model.module.fc.parameters()).detach().clone()
    ws = [torch.zeros_like(w) for _ in range(world_size)]
    dist.all_gather(ws, w)
    all_same = all(torch.allclose(ws[0], x) for x in ws[1:])

    if rank == 0 and out_queue is not None:
        out_queue.put({
            "first_loss": losses[0], "last_loss": losses[-1],
            "broadcast": float(t.item()), "all_same": all_same,
        })
    dist.destroy_process_group()


def run_ddp(world_size: int = 2) -> dict:
    """mp.spawn 拉起多进程跑完返回 rank0 结果（pytest 安全：模块级 worker 函数）。"""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = find_free_port()
    mp.start_processes(ddp_worker, args=(world_size, port, q),
                       nprocs=world_size, start_method="spawn")
    return q.get(timeout=300)


if __name__ == "__main__":
    res = run_ddp(world_size=2)
    print(f"双进程 DDP (gloo/CPU): loss {res['first_loss']:.4f} → {res['last_loss']:.4f}")
    print(f"rank0 广播标量 = {res['broadcast']:.4f} (张量直传, 各 rank 收到同一值)")
    print(f"各 rank 参数完全一致 = {res['all_same']} (梯度 all_reduce 的效果)")
    print("真工程差异: NCCL 后端 + cuda rank + 60min 看门狗 (rank0 考试时其余 rank 干等) "
          "+ find_unused_parameters=True (冻结段不碰全部参数)")
