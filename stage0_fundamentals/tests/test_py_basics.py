"""py_basics 的测试：CSV 解析 / GrayImage 类 / 装饰器缓存 / dataclass 配置。"""

from stage0.py_basics import (
    GrayImage, TrainingArgs, apply_whitelist, memoize, parse_csv, parse_csv_int
)


class TestParseCsv:
    """对照仓库 train_entry.py 的 _parse_csv 语义：None=不过滤，空串=全过滤。"""

    def test_none_returns_empty(self):
        assert parse_csv(None) == []

    def test_basic_split_and_strip(self):
        assert parse_csv("砂眼, 崩边") == ["砂眼", "崩边"]

    def test_empty_and_whitespace_items_dropped(self):
        assert parse_csv(" a , ,b ") == ["a", "b"]
        assert parse_csv("") == []

    def test_int_variant(self):
        assert parse_csv_int("1, 2,3") == [1, 2, 3]


class TestGrayImage:
    def test_shape_and_mean(self):
        img = GrayImage([[10, 200], [30, 240]])
        assert img.shape == (2, 2)
        assert abs(img.mean_brightness - 120.0) < 1e-9

    def test_binarize(self):
        img = GrayImage([[10, 200], [30, 240]])
        assert img.binarize(100).tolist() == [[0, 1], [0, 1]]

    def test_rejects_non_2d(self):
        import numpy as np
        import pytest
        with pytest.raises(ValueError):
            GrayImage(np.zeros((2, 2, 3)))

    def test_rejects_out_of_range(self):
        import pytest
        with pytest.raises(ValueError):
            GrayImage([[0, 300]])


class TestMemoize:
    def test_caches_by_argument(self):
        @memoize
        def double(x):
            return x * 2

        assert double(3) == 6
        assert double(3) == 6
        assert double(4) == 8
        # 缓存键是参数元组（args 元组天然可哈希），3 只算了一次
        assert double.cache == {(3,): 6, (4,): 8}

    def test_memoized_result_unchanged(self):
        from stage0.py_basics import slow_square
        assert slow_square(12) == 144


class TestDataclassConfig:
    def test_defaults_match_repo(self):
        """默认值与 vision-ai-training 仓库 TrainingArgs 保持一致（学习锚点）。"""
        args = TrainingArgs()
        assert args.epochs == 25
        assert args.lr == 4e-4
        assert args.train_mode == "fresh"
        assert args.defect_whitelist == "砂眼, 崩边, 缺肉"

    def test_override_and_whitelist_pipeline(self):
        args = TrainingArgs(train_mode="finetune", defect_whitelist="崩边")
        assert args.train_mode == "finetune"
        assert apply_whitelist(args) == ["崩边"]

    def test_mutable_default_not_shared(self):
        """extras 用 field(default_factory=list)：两个实例的列表必须独立。"""
        a, b = TrainingArgs(), TrainingArgs()
        a.extras.append("x")
        assert b.extras == []
