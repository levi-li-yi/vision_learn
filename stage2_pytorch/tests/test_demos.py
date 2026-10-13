"""Demo C（smoke 骨架）与 Demo D（迁移分组/冻结）的离线测试。"""
import pytest
import torch
from torch import nn

from stage2.demo_c_cifar import SmallResNet, run_training
from stage2.demo_d_transfer import (
    PretrainedLikeBackbone, freeze_backbone, load_pretrained_backbone,
    split_transfer_groups,
)


class TestDemoC:
    def test_model_shapes(self):
        net = SmallResNet()
        out = net(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10)

    def test_smoke_run_writes_ckpt_and_resumes(self, tmp_path, capsys):
        ckpt = tmp_path / "demo_c.pth"
        acc1 = run_training(epochs=1, smoke=True, ckpt=str(ckpt))
        assert ckpt.exists()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert state["step"] > 0
        acc2 = run_training(epochs=2, smoke=True, ckpt=str(ckpt), resume=True)
        out = capsys.readouterr().out
        assert "[resume]" in out                          # 真的续上了
        assert 0.0 <= acc2 <= 1.0


class TestDemoD:
    def test_pretrained_only_hits_backbone(self):
        m = PretrainedLikeBackbone()
        before = m.head.weight.clone()
        hit = load_pretrained_backbone(m)
        assert hit > 0
        assert torch.equal(before, m.head.weight)         # 头部保持随机

    def test_transfer_groups_mirror_repo(self):
        groups = split_transfer_groups(PretrainedLikeBackbone(),
                                       lr=4e-4, backbone_lr_scale=0.1)
        assert groups[0]["lr"] == pytest.approx(4e-5)     # = lr*0.1（主仓库默认）
        assert groups[1]["lr"] == pytest.approx(4e-4)

    def test_freeze_backbone(self):
        m = freeze_backbone(PretrainedLikeBackbone())
        bb_frozen = all(not p.requires_grad
                        for n, p in m.named_parameters() if n.startswith("backbone"))
        head_live = all(p.requires_grad
                        for n, p in m.named_parameters() if not n.startswith("backbone"))
        assert bb_frozen and head_live
