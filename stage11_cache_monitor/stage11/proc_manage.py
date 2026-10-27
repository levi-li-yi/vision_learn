"""课时 3：训练子进程管理与网页远程关停 —— web.py::MonitorServer / main.py 双角色复刻。

监控 web 与训练进程的父子关系（真工程的"launcher 双角色"）：

  · 同一个 train 程序两种角色：父进程（TRAIN_WORKER 未设置）起 web 服务器并
    spawn 训练子进程；子进程（TRAIN_WORKER=1）只跑训练、往 .tmp/web/ 写状态——
    一个入口两种人格，跨进程靠**状态文件**交换（web 进程绝不 import torch 重件）；
  · 子进程管理：subprocess.Popen 拉起；关停要**杀进程树**——子进程可能还带着
    DataLoader worker 孙进程，只杀儿子会留孤儿（Windows: taskkill /T /F；
    跨平台真工程用 psutil 进程树 terminate→kill）；
  · 网页远程关停：POST /api/shutdown → 延迟关 httpd（先把响应发回去）→
    杀子进程 → os._exit 立即退出（不再等非守护线程）；
  · 状态文件轮询：父进程轮询子进程 exitcode 与状态文件，崩溃即报告原因。

对应主仓库：src/train/web.py::MonitorServer（spawn_child / kill_child /
request_shutdown / do_GET shutdown 分支）、src/train/main.py（TRAIN_WORKER
双角色判断段）。

运行：python -m stage11.proc_manage
"""
import json
import os
import subprocess
import sys
import time

TRAIN_WORKER_ENV = "TRAIN_WORKER"
STATUS_FILE = "worker_status.json"
SHUTDOWN_FILE = "shutdown.flag"


def is_worker() -> bool:
    """双角色判别：环境变量 TRAIN_WORKER=1 即训练 worker 人格。"""
    return os.environ.get(TRAIN_WORKER_ENV) == "1"


def worker_main(status_path: str) -> None:
    """训练 worker 人格：干一小段"训练"，写状态文件，监听关停旗标。

    （替身真工程 _worker：训练循环 + 每轮写 .tmp/web/state.json）
    """
    for step in range(1, 21):
        if os.path.isfile(os.path.join(os.path.dirname(status_path), SHUTDOWN_FILE)):
            write_status(status_path, {"step": step, "state": "stopped_by_web"})
            return
        time.sleep(0.05)
        write_status(status_path, {"step": step, "state": "running",
                                   "loss": 1.0 / step})
    write_status(status_path, {"step": 20, "state": "done", "loss": 1.0 / 20})


def write_status(status_path: str, payload: dict,
                 retries: int = 60, delay: float = 0.01) -> None:
    """原子写状态文件（tmp+replace）：父进程永远读到完整 JSON。

    Windows 上 os.replace 会被"恰好打开着目标文件的读者"顶出 PermissionError
    （监控页轮询与训练写盘的天然竞态）——重试兜底，真工程 runtime_state 同思路。
    """
    tmp = status_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    for _ in range(retries):
        try:
            os.replace(tmp, status_path)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(tmp, status_path)                        # 仍失败则如实抛出


def spawn_child(python_exe: str, script_path: str, workdir: str, args: list = None):
    """Popen 拉起训练子进程（注入 TRAIN_WORKER=1；输出落 workdir 日志文件备查）。"""
    env = {**os.environ, TRAIN_WORKER_ENV: "1"}
    out_log = open(os.path.join(workdir, "worker_out.log"), "w", encoding="utf-8")
    err_log = open(os.path.join(workdir, "worker_err.log"), "w", encoding="utf-8")
    cmd = [python_exe, script_path] + [str(a) for a in (args or [])]
    return subprocess.Popen(
        cmd, cwd=workdir, env=env, stdout=out_log, stderr=err_log,
    )


def kill_tree(proc: subprocess.Popen) -> None:
    """杀进程树（含 DataLoader worker 孙进程）——只杀儿子会留孤儿。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def request_shutdown(workdir: str, proc: subprocess.Popen, delay: float = 0.0,
                     grace: float = 0.5):
    """网页远程关停：立旗标 → 优雅窗口等 worker 自觉退出 → 超时杀整棵进程树。

    真工程另起线程先回 HTTP 响应再 shutdown httpd，最后 os._exit 立即退出。
    """
    flag = os.path.join(workdir, SHUTDOWN_FILE)
    with open(flag, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if delay:
        time.sleep(delay)
    deadline = time.time() + grace
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.05)                                 # 优雅退出窗口
    kill_tree(proc)                                      # 顽固不化才动刀
    try:
        os.remove(flag)
    except OSError:
        pass


def read_status(status_path: str, retries: int = 60, delay: float = 0.01) -> dict:
    """容错读状态文件：换文件/写一半的瞬间短暂 PermissionError → 重试。"""
    for _ in range(retries):
        try:
            with open(status_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(delay)
    return {}


def poll_until_done(status_path: str, proc: subprocess.Popen, timeout: float = 30.0):
    """父进程轮询：返回 (最终状态 dict, 子进程退出码)。崩溃时给出原因。"""
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        if os.path.isfile(status_path):
            got = read_status(status_path, retries=2, delay=0.005)
            if got:
                status = got
        if status.get("state") in ("done", "stopped_by_web"):
            break
        if proc.poll() is not None:
            status["state"] = f"crashed(exitcode={proc.returncode})"
            break
        time.sleep(0.05)
    return status, proc.poll()


WORKER_SNIPPET = f'''
import sys, os
sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
from stage11.proc_manage import is_worker, worker_main
assert is_worker()
worker_main(sys.argv[1])
'''


if __name__ == "__main__":
    workdir = os.path.abspath("stage11_proc_demo")
    os.makedirs(workdir, exist_ok=True)
    status_path = os.path.join(workdir, STATUS_FILE)
    if os.path.isfile(status_path):
        os.remove(status_path)                            # 清掉上次的旧状态

    if is_worker():                                       # 子进程人格
        worker_main(status_path)
        sys.exit(0)

    # 父进程人格：spawn → 中途快照 → 远程关停 → 再 spawn → 等自然结束
    script = os.path.join(workdir, "_worker.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(WORKER_SNIPPET)

    proc = spawn_child(sys.executable, script, workdir, args=[status_path])
    time.sleep(0.3)                                       # 让 worker 跑几步
    mid = read_status(status_path)
    print(f"中途快照: step={mid.get('step')} state={mid.get('state')} "
          f"loss={mid.get('loss'):.3f}")

    request_shutdown(workdir, proc)                       # 网页点了"停止训练"
    st, code = poll_until_done(status_path, proc, timeout=5)
    print(f"远程关停后: state={st.get('state')} 退出码={code}")

    os.remove(status_path)                                # 清旧状态，观测新一轮
    proc2 = spawn_child(sys.executable, script, workdir, args=[status_path])
    st, code = poll_until_done(status_path, proc2, timeout=15)
    print(f"自然跑完: state={st.get('state')} 最终 step={st.get('step')} "
          f"loss={st.get('loss') or 0:.3f}")
    print("真工程对照: web 父进程零 torch 依赖, 子进程写 .tmp/web/state.json, "
          "kill_child 杀整棵进程树 (taskkill /T /F 或 psutil)")
