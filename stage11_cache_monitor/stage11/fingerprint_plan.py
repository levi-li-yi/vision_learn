"""课时 1：指纹缓存与开局全局执行计划 —— prep_plan.py 复刻。

训练从 0→1 会生成四类中间缓存（逆向对齐 / 真实缺陷小图 / LoRA 权重 / 合成缺陷
小图），再次运行时要"只重算有变化的"。真工程的设计：

  1. 指纹分两档：
     · 大文件（.bmp）用"名字 + 字节大小 + mtime"元数据指纹——零 IO 读内容；
     · 小文件（配置/标注 json）用 SHA256 内容哈希——元数据不可靠（内容变了大小
       mtime 可能没变），小文件读得起；
  2. 每阶段绑定算法版本常量（ALIGN_VER/REAL_PATCH_VER/LORA_VER/SYNTH_VER）——
     算法逻辑升级后只改版本号，全部缓存自动失效重算，不用手动清缓存；
  3. **开局一次性全局执行计划**：所有步骤的 skip/增量/全量判定集中在一处完成，
     而不是各子步骤零散探测（探测本身也是成本，且中途状态易漂）；
  4. 原子更新：每步完成即 tmp+os.replace 更新 prep_fingerprints.json——
     第二步崩了第一步的指纹已落盘，断点续跑不重算已完成的步骤。

对应主仓库：src/train/prep_plan.py（常量 / _sha256_file / _bmp_meta /
build_startup_execution_plan / update_saved_fingerprints）。

运行：python -m stage11.fingerprint_plan
"""
import hashlib
import json
import os
import time

ALIGN_VER = 1        # 对应 prep_plan.ALIGN_VER
REAL_PATCH_VER = 1   # 对应 REAL_PATCH_VER
LORA_VER = 1         # 对应 LORA_VER
SYNTH_VER = 1        # 对应 SYNTH_VER
FP_FILENAME = "prep_fingerprints.json"


def bmp_meta_fp(path: str) -> str:
    """大文件元数据指纹：名字 + 字节大小 + mtime（不读内容）。"""
    st = os.stat(path)
    raw = f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()


def sha256_fp(path: str) -> str:
    """小文件内容哈希（配置/标注——元数据不可靠，读得起）。"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fingerprint_tree(paths: list) -> dict:
    """一组输入文件的指纹树：bmp 走元数据档，json 走 SHA256 档。"""
    tree = {}
    for p in paths:
        tree[os.path.basename(p)] = (bmp_meta_fp if p.endswith(".bmp") else sha256_fp)(p)
    return tree


def step_fp(ver: int, inputs_fp: dict) -> str:
    """步骤指纹 = 算法版本 + 输入指纹树的稳定序列化哈希。"""
    raw = json.dumps({"ver": ver, "inputs": inputs_fp}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def build_startup_execution_plan(inputs_by_step: dict, saved: dict,
                                 products_done: dict, versions: dict = None) -> dict:
    """开局一次性全局执行计划：每步三态 skip / incremental / full。

    inputs_by_step: {步骤名: 输入指纹 dict}；saved: 上次落盘的步骤指纹；
    products_done: {步骤名: 已产出数量}（合成配额增量补齐判据）。
    incremental 专指"指纹匹配但产出数量不足"——只补差额，不推倒重来。
    versions: 覆盖算法版本号（演示/测试用；python -m 双模块陷阱的干净解法）。
    """
    vers = {"align": ALIGN_VER, "real_patches": REAL_PATCH_VER,
            "lora": LORA_VER, "synth": SYNTH_VER}
    if versions:
        vers.update(versions)
    plan, cur_fps = {}, {}
    for step, inp in inputs_by_step.items():
        ver = vers.get(step, 1)
        fp = step_fp(ver, inp)
        cur_fps[step] = fp
        if saved.get(step) == fp and products_done.get(step, 0) > 0:
            plan[step] = "skip"
        elif saved.get(step) == fp:                    # 指纹同但产出为 0 → 补齐
            plan[step] = "incremental"
        else:
            plan[step] = "full"
    return {"plan": plan, "cur_fps": cur_fps}


def update_saved_fingerprints(root: str, section: str, val) -> None:
    """每步完成即原子更新指纹文件（tmp + os.replace），断点续跑安全。"""
    fp_path = os.path.join(root, FP_FILENAME)
    saved = {}
    if os.path.isfile(fp_path):
        with open(fp_path, encoding="utf-8") as f:
            saved = json.load(f)
    saved[section] = val
    tmp_path = fp_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=1)
    os.replace(tmp_path, fp_path)


def load_saved_fingerprints(root: str) -> dict:
    fp_path = os.path.join(root, FP_FILENAME)
    if not os.path.isfile(fp_path):
        return {}
    with open(fp_path, encoding="utf-8") as f:
        return json.load(f)


def archive_rotate(session_dir: str, archive_root: str, keep: int = 3) -> list:
    """单次会话运行态归档轮换：搬进 _archive/ 只留最近 keep 份（.tmp 分家治理）。"""
    if not os.path.isdir(session_dir):
        return []
    os.makedirs(archive_root, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(archive_root, stamp)
    os.replace(session_dir, dst)
    olds = sorted(d for d in os.listdir(archive_root)
                  if os.path.isdir(os.path.join(archive_root, d)))
    removed = []
    for d in olds[:-keep] if len(olds) > keep else []:
        import shutil
        shutil.rmtree(os.path.join(archive_root, d))
        removed.append(d)
    return removed


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        # 造素材：两张 bmp + 一份配置 json
        bmp1 = os.path.join(root, "a.bmp")
        bmp2 = os.path.join(root, "b.bmp")
        cfg = os.path.join(root, "zones.json")
        for p, data in ((bmp1, b"x" * 1000), (bmp2, b"y" * 2000)):
            with open(p, "wb") as f:
                f.write(data)
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('{"regions": []}')

        inputs = {"align": fingerprint_tree([bmp1, bmp2, cfg]),
                  "lora": fingerprint_tree([cfg])}
        done = {"align": 12, "lora": 0}

        # 第一次：全部 full（无历史指纹）
        plan = build_startup_execution_plan(inputs, saved={}, products_done=done)
        print(f"首跑计划: {plan['plan']}")

        # align 完成 → 指纹落盘
        update_saved_fingerprints(root, "align", plan["cur_fps"]["align"])
        # 模拟第二步（lora）崩掉——align 的指纹已保存
        print(f"中途崩溃后已保存指纹: {list(load_saved_fingerprints(root))} (断点续跑不重算 align)")

        # 第二次：align skip（指纹同且产出>0）；lora 仍 full
        plan2 = build_startup_execution_plan(inputs, load_saved_fingerprints(root), done)
        print(f"二跑计划: {plan2['plan']}")

        # 配置内容变化（大小没变——元数据指纹会漏，SHA256 抓住）
        time.sleep(0.02)
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('{"regions": [{"name": "new"}]}')
        inputs2 = {"align": fingerprint_tree([bmp1, bmp2, cfg]), "lora": inputs["lora"]}
        plan3 = build_startup_execution_plan(inputs2, load_saved_fingerprints(root), done)
        print(f"配置内容变化后: {plan3['plan']} (json 用 SHA256, 哪怕大小没变也失效)")

        # 算法版本升级 → 全量失效（versions 参数覆盖，避免 python -m 双模块陷阱）
        plan4 = build_startup_execution_plan(inputs, load_saved_fingerprints(root), done,
                                             versions={"align": ALIGN_VER + 1})
        print(f"ALIGN_VER 1→2 后: {plan4['plan']} (版本号是最粗暴也最可靠的失效开关)")

        # 归档轮换
        sess = os.path.join(root, "web")
        os.makedirs(sess)
        arch = os.path.join(root, "_archive")
        removed = []
        for i in range(5):
            d = os.path.join(arch, f"2026010{i}_000000")
            os.makedirs(d)
        os.makedirs(sess, exist_ok=True)
        removed = archive_rotate(sess, arch, keep=3)
        print(f"归档轮换: 本次会话入档 _archive/2026.., 淘汰旧档 {removed} (只留 3 份)")
