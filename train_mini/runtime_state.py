#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""runtime_state.py —— 训练侧监控状态写入(跨进程文件交换,真工程同构位)。

训练进程把事件线与逐轮指标原子写入 .tmp/{产品}/web/state.json,
web.py 常驻进程按需读取渲染(进程间零耦合,谁先谁后都行)。
与真工程 runtime_state.py 同角色;从简差异:单 state 文件,无会话归档/ng.json。
"""

from __future__ import annotations

import json
import os
import time

WEB_DIRNAME = "web"    # .tmp/{产品}/web/(与真工程同路径口径)
EVENTS_KEEP = 400      # 事件环形上限(逐轮日志足够)


def state_path(product: str, tmp_root: str = ".tmp") -> str:
    return os.path.join(tmp_root, product, WEB_DIRNAME, "state.json")


class Monitor:
    """训练监控状态累积器:main.py 各事件点调用,内部原子落盘。"""

    def __init__(self, product: str, config: dict):
        self.path = state_path(product)
        self.state = {
            "product": product,
            "status": "running",     # running | done
            "start_ts": time.time(),
            "update_ts": time.time(),
            "config": config,
            "events": [],            # [{t,msg}] 事件线(与控制台 log 同文案)
            "history": [],           # 逐轮 {epoch,train_loss,val_loss[,inter,hit,hit_total,alarm,quality]}
            "best": None,
            "stop_reason": "",
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.save()

    def event(self, msg: str) -> None:
        self.state["events"].append({"t": time.strftime("%H:%M:%S"), "msg": msg})
        del self.state["events"][:-EVENTS_KEEP]
        self.save()

    def epoch(self, rec: dict) -> None:
        self.state["history"].append(rec)
        self.save()

    def best(self, name: str, quality: float) -> None:
        self.state["best"] = {"name": name, "quality": quality}
        self.save()

    def finish(self, stop_reason: str) -> None:
        self.state["status"] = "done"
        self.state["stop_reason"] = stop_reason
        self.save()

    def save(self) -> None:
        """原子落盘(写临时文件后 os.replace,web 侧永不读到半截 json)。"""
        self.state["update_ts"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)
        os.replace(tmp, self.path)
