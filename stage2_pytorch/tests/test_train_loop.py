import torch
from torch.utils.data import DataLoader

from stage2.nn_module_dataset_loader import (
    BrightCornerDataset, TinyCNN, collate_stack,
)
from stage2.train_loop import accuracy, grads_close, train_one_epoch


def make_loader(bs=32, n=256, seed=0):
    return DataLoader(BrightCornerDataset(n, seed=seed), batch_size=bs,
                      shuffle=True, collate_fn=collate_stack)


class TestDatasetContract:
    def test_len_and_item_shapes(self):
        ds = BrightCornerDataset(64)
        assert len(ds) == 64
        img, lab = ds[0]
        assert img.shape == (1, 16, 16)
        assert lab in (0, 1)

    def test_collate_batching(self):
        loader = make_loader(bs=8)
        x, y = next(iter(loader))
        assert x.shape == (8, 1, 16, 16) and y.shape == (8,)


class TestTrainLoop:
    def test_loss_decreases_and_acc_high(self):
        torch.manual_seed(0)
        model = TinyCNN()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        loader = make_loader(256)
        first = train_one_epoch(model, loader, opt)
        for _ in range(4):
            last = train_one_epoch(model, loader, opt)
        assert last < first                       # 在学
        assert accuracy(model, make_loader(128, seed=9)) > 0.95

    def test_gradient_accumulation_equivalence(self):
        """accum 份小批的梯度之和 == 一大批的梯度（主仓库 loss÷N 的合法性）。"""
        data = BrightCornerDataset(64, seed=3)
        xs = torch.stack([data[i][0] for i in range(64)])
        ys = torch.tensor([data[i][1] for i in range(64)])
        crit = torch.nn.functional.cross_entropy

        torch.manual_seed(0); m_big = TinyCNN()
        torch.manual_seed(0); m_acc = TinyCNN()

        crit(m_big(xs), ys).backward()                       # 一大批一步

        for s in range(4):                                   # 4 小批，loss/4
            chunk = slice(s * 16, (s + 1) * 16)
            loss = crit(m_acc(xs[chunk]), ys[chunk]) / 4.0
            loss.backward()

        assert grads_close(m_big, m_acc, tol=1e-6)
