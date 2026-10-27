"""stage11 全模块测试：指纹与执行计划 / 三层渲染缓存 / 子进程管理。"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from stage11.fingerprint_plan import (
    ALIGN_VER, FP_FILENAME, archive_rotate, bmp_meta_fp,
    build_startup_execution_plan, fingerprint_tree, load_saved_fingerprints,
    sha256_fp, step_fp, update_saved_fingerprints,
)
from stage11.proc_manage import (
    SHUTDOWN_FILE, STATUS_FILE, WORKER_SNIPPET, is_worker, kill_tree,
    poll_until_done, request_shutdown, spawn_child, write_status,
)
from stage11.render_cache import Debouncer as Deb, Renderer


class TestFingerprint:
    def test_bmp_meta_fp_design(self, tmp_path):
        # 元数据档：同名同大小不同内容 → 指纹相同（这是设计：bmp 读不起内容）
        d1, d2 = tmp_path / "p1", tmp_path / "p2"
        d1.mkdir(), d2.mkdir()
        f1, f2 = d1 / "a.bmp", d2 / "a.bmp"
        f1.write_bytes(b"x" * 100)
        f2.write_bytes(b"y" * 100)                       # 内容不同、大小相同
        if int(os.path.getmtime(f1)) == int(os.path.getmtime(f2)):
            assert bmp_meta_fp(str(f1)) == bmp_meta_fp(str(f2))   # 元数据档的盲区
        f3 = d1 / "b.bmp"
        f3.write_bytes(b"x" * 101)
        assert bmp_meta_fp(str(f1)) != bmp_meta_fp(str(f3))        # 大小变化必变

    def test_sha256_catches_content_change(self, tmp_path):
        fa = tmp_path / "z.json"
        fa.write_text('{"a": 1}')
        fb = tmp_path / "y.json"
        fb.write_text('{"a": 2}')                        # 同大小不同内容
        assert sha256_fp(str(fa)) != sha256_fp(str(fb))

    def test_step_fp_version_sensitive(self):
        inp = {"a.bmp": "deadbeef"}
        assert step_fp(1, inp) != step_fp(2, inp)        # 算法版本一变指纹必变
        assert step_fp(1, inp) == step_fp(1, dict(inp))  # 与 dict 序无关

    def test_plan_lifecycle(self, tmp_path):
        bmp = tmp_path / "a.bmp"
        bmp.write_bytes(b"x" * 100)
        inputs = {"align": fingerprint_tree([str(bmp)])}

        p1 = build_startup_execution_plan(inputs, saved={}, products_done={})
        assert p1["plan"]["align"] == "full"             # 首跑全量

        update_saved_fingerprints(str(tmp_path), "align", p1["cur_fps"]["align"])
        saved = load_saved_fingerprints(tmp_path)
        p2 = build_startup_execution_plan(inputs, saved, products_done={"align": 12})
        assert p2["plan"]["align"] == "skip"             # 指纹同且产出>0

        p3 = build_startup_execution_plan(inputs, saved, products_done={"align": 0})
        assert p3["plan"]["align"] == "incremental"      # 指纹同但产出 0 → 只补差额

        time.sleep(0.02)
        bmp.write_bytes(b"x" * 200)                      # 输入变化
        inputs2 = {"align": fingerprint_tree([str(bmp)])}
        p4 = build_startup_execution_plan(inputs2, saved, products_done={"align": 12})
        assert p4["plan"]["align"] == "full"

    def test_atomic_update_resume_safety(self, tmp_path):
        update_saved_fingerprints(str(tmp_path), "align", "fp1")
        # 模拟后续步骤崩掉——align 指纹已安全落盘
        saved = load_saved_fingerprints(tmp_path)
        assert saved == {"align": "fp1"}
        assert not os.path.isfile(os.path.join(tmp_path, FP_FILENAME + ".tmp"))
        with open(os.path.join(tmp_path, FP_FILENAME), encoding="utf-8") as f:
            assert json.load(f)["align"] == "fp1"        # 文件内容完整可解析

    def test_version_bump_invalidates(self, tmp_path):
        bmp = tmp_path / "a.bmp"
        bmp.write_bytes(b"x" * 100)
        inputs = {"align": fingerprint_tree([str(bmp)])}
        p1 = build_startup_execution_plan(inputs, {}, {"align": 9})
        update_saved_fingerprints(str(tmp_path), "align", p1["cur_fps"]["align"])
        p2 = build_startup_execution_plan(inputs, load_saved_fingerprints(tmp_path),
                                          {"align": 9}, versions={"align": ALIGN_VER + 1})
        assert p2["plan"]["align"] == "full"             # 版本升级全量失效

    def test_archive_rotate_keeps_three(self, tmp_path):
        arch = tmp_path / "_archive"
        arch.mkdir()
        for i in range(4):
            (arch / f"2026010{i + 1}_000000").mkdir()
        sess = tmp_path / "web"
        sess.mkdir()
        removed = archive_rotate(str(sess), str(arch), keep=3)
        assert not sess.exists()                          # 会话目录入档
        assert len(os.listdir(arch)) == 3                 # 只留 3 份
        assert len(removed) == 2


class TestRenderCache:
    def test_threshold_only_re_renders(self):
        r = Renderer()
        mt = {"ep1": 1.0}
        r.render("ep1", "ng1", 0.5, mt)
        r.render("ep1", "ng1", 0.3, mt)
        r.render("ep1", "ng1", 0.7, mt)
        assert r.counters["predict"] == 1                 # 只推理一次
        assert r.counters["render"] == 3                  # 阈值每次重画

    def test_heat_lru_cap(self):
        r = Renderer()
        mt = {"ep1": 1.0}
        for i in range(4):
            r.render("ep1", f"ng{i}", 0.5, mt)
        assert len(r._heats) == Renderer.HEAT_N
        assert ("ep1", "ng0") not in r._heats             # 最旧被挤出

    def test_model_mtime_cascade_invalidation(self):
        r = Renderer()
        mt = {"ep2": 1.0}
        r.render("ep2", "ngA", 0.5, mt)
        assert r.counters["predict"] == 1
        mt["ep2"] = 2.0                                   # 权重被新一轮覆盖
        r.render("ep2", "ngA", 0.5, mt)
        assert r.counters["model_load"] == 2              # mtime 失效重载
        assert r.counters["predict"] == 2                 # 旧热力图作废重推

    def test_model_lru_cap(self):
        r = Renderer()
        mt = {f"ep{i}": 1.0 for i in (1, 2, 3)}
        for k in ("ep1", "ep2", "ep3"):
            r.render(k, "ngX", 0.5, mt)
        assert len(r._models) == Renderer.MODEL_N

    def test_heatval_hits_cache(self):
        r = Renderer()
        mt = {"ep1": 1.0}
        r.render("ep1", "ng1", 0.5, mt)
        before = r.counters["predict"]
        val = r.heat_val("ep1", "ng1", 25, 25, mt)
        assert 0.0 <= val <= 1.0
        assert r.counters["predict"] == before            # 悬停查值零新推理

    def test_vbase_lru(self):
        r = Renderer()
        for i in range(7):
            r.vbase(f"img{i}")
        assert len(r._vbase) == Renderer.VBASE_N

    def test_debouncer(self):
        d = Deb(120)
        t0 = 1000.0
        fires = [d.should_run(t0 + i * 0.03) for i in range(10)]  # 270ms 内 10 次
        assert fires[0] and sum(fires) == 3               # 0ms/120ms/240ms 三次


class TestProcManage:
    def make_worker(self, tmp_path):
        script = tmp_path / "_worker.py"
        script.write_text(WORKER_SNIPPET, encoding="utf-8")
        return str(script)

    def test_write_status_atomic_roundtrip(self, tmp_path):
        p = tmp_path / STATUS_FILE
        write_status(str(p), {"step": 3, "state": "running"})
        with open(p, encoding="utf-8") as f:
            assert json.load(f)["step"] == 3
        assert not os.path.isfile(str(p) + ".tmp")

    def test_is_worker_env(self):
        old = os.environ.get("TRAIN_WORKER")
        try:
            os.environ["TRAIN_WORKER"] = "1"
            assert is_worker()
            del os.environ["TRAIN_WORKER"]
            assert not is_worker()
        finally:
            if old is not None:
                os.environ["TRAIN_WORKER"] = old

    def test_spawn_and_natural_finish(self, tmp_path):
        script = self.make_worker(tmp_path)
        status = str(tmp_path / STATUS_FILE)
        proc = spawn_child(sys.executable, script, str(tmp_path), args=[status])
        st, code = poll_until_done(status, proc, timeout=30)
        assert st.get("state") == "done" and st.get("step") == 20
        assert code == 0 or proc.poll() is not None

    def test_worker_respects_shutdown_flag(self, tmp_path):
        # 先立旗标再 spawn：worker 第一轮就自觉退出（无竞态）
        script = self.make_worker(tmp_path)
        status = str(tmp_path / STATUS_FILE)
        (tmp_path / SHUTDOWN_FILE).write_text("now")
        proc = spawn_child(sys.executable, script, str(tmp_path), args=[status])
        st, _ = poll_until_done(status, proc, timeout=30)
        assert st.get("state") == "stopped_by_web"

    def test_kill_tree_stubsborn_child(self, tmp_path):
        # 杀一棵"赖着不走"的进程树（sleep 60 的孙进程由 /T 连带）
        code = "import subprocess,sys,time;\n" \
               "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);\n" \
               "time.sleep(60)"
        proc = subprocess.Popen([sys.executable, "-c", code])
        time.sleep(0.5)                                   # 等孙进程出生
        kill_tree(proc)
        assert proc.poll() is not None                    # 进程树被终止

    def test_request_shutdown_kills_child(self, tmp_path):
        script = self.make_worker(tmp_path)
        status = str(tmp_path / STATUS_FILE)
        proc = spawn_child(sys.executable, script, str(tmp_path), args=[status])
        time.sleep(0.2)
        request_shutdown(str(tmp_path), proc)
        assert proc.poll() is not None
        assert not (tmp_path / SHUTDOWN_FILE).exists()    # 旗标已清理
