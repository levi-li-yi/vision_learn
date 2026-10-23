#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""web.py —— 监控 web(零依赖轻量版,真工程 web.py 的同构缩影)。

标准库 http.server 常驻进程:按需读 .tmp/{产品}/web/state.json 渲染
(逐轮 loss/quality 曲线 + 事件线 + best/早停状态,2s 自动刷新)。
训练与监控进程零耦合(文件交换),先后启动、只起一个都行;训练结束后页面仍可回看。
从简差异:无 NG 热力图按需推理、无常驻整图渲染、无训练子进程管理。
无鉴权,默认仅 127.0.0.1;内网观看加 --host 0.0.0.0。

用法:
  python web.py --product HT12 [--port 8301] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from runtime_state import state_path


def _load_web_port() -> int:
    """读 config/app_config.jsonc 的 web_port。

    内联极简 JSONC 解析(与 io_utils.load_jsonc 同规则),不 import config——
    那会连带 io_utils→cv2,破坏本文件"仅标准库"的依赖承诺(无 torch/cv2 的机器也能看监控)。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "config", "app_config.jsonc")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read()
        return int(json.loads(re.sub(r"//[^\n]*", "", text))["web_port"])
    except (OSError, ValueError, KeyError):
        return 8301

PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>train_mini 监控</title>
<style>
  :root{
    --bg:#0f1419; --panel:#161d26; --border:#232d3a;
    --text:#d8e1ea; --muted:#7d8b99;
    --run:#3fb950; --done:#58a6ff; --warn:#d29922; --wait:#5a6b7a;
  }
  *{box-sizing:border-box}
  body{margin:0;padding:16px;background:var(--bg);color:var(--text);
       font:14px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}
  header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:12px}
  h1{font-size:18px;margin:0}
  .sub{color:var(--muted);font-size:12px}
  .badge{padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600;
         background:var(--wait);color:#fff}
  .badge.run{background:var(--run)} .badge.done{background:var(--done)}
  .badge.warn{background:var(--warn)} .badge.pulse{animation:p 1.2s infinite}
  @keyframes p{50%{opacity:.55}}
  .cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;
        padding:8px 14px;min-width:130px}
  .card .k{color:var(--muted);font-size:12px}
  .card .v{font-size:17px;font-weight:600;font-family:Consolas,monospace;
           word-break:break-all}
  .charts{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;
         padding:10px 12px;flex:1;min-width:340px}
  .panel h2{font-size:13px;margin:0 0 4px;color:var(--muted);font-weight:600}
  .legend{font-size:12px;color:var(--muted);margin:0 0 2px 4px}
  .legend i{display:inline-block;width:14px;height:3px;border-radius:2px;
            margin:0 4px 2px 0;vertical-align:middle}
  svg{width:100%;height:auto;display:block}
  #events{max-height:320px;overflow-y:auto;font:12px/1.7 Consolas,monospace}
  #events div{padding:0 6px;border-bottom:1px solid #1b232e;white-space:pre-wrap}
  #events .t{color:var(--muted)}
  .e-best{color:#e3b341} .e-exam{color:#5fd7c0} .e-end{color:#58a6ff;font-weight:700}
</style>
</head>
<body>
<header>
  <h1>train_mini 监控 <span id="product"></span></h1>
  <span class="badge" id="badge">等待数据</span>
  <span class="sub" id="stopinfo"></span>
  <span class="sub" style="margin-left:auto" id="elapsed"></span>
</header>
<div class="sub" id="cfgline" style="margin-bottom:10px"></div>
<div class="cards">
  <div class="card"><div class="k">轮次</div><div class="v" id="c-ep">-</div></div>
  <div class="card"><div class="k">best 模型</div><div class="v" id="c-best">-</div></div>
  <div class="card"><div class="k">best quality</div><div class="v" id="c-bq">-</div></div>
  <div class="card"><div class="k">最近考试 检出/总缺陷</div><div class="v" id="c-hit">-</div></div>
  <div class="card"><div class="k">最近考试 交集</div><div class="v" id="c-inter">-</div></div>
</div>
<div class="charts">
  <div class="panel"><h2>损失曲线(逐轮)</h2><div class="legend" id="lg-loss"></div>
    <svg id="chart-loss" viewBox="0 0 560 210"></svg></div>
  <div class="panel"><h2>考卷 quality_score(逐轮)</h2><div class="legend" id="lg-q"></div>
    <svg id="chart-q" viewBox="0 0 560 210"></svg></div>
</div>
<div class="panel"><h2>事件线(与训练控制台同文案)</h2><div id="events"></div></div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtDur = s => { s = Math.max(0, s|0);
  const h = s/3600|0, m = (s%3600)/60|0, ss = s%60, p = n => String(n).padStart(2,'0');
  return (h ? h + ':' : '') + p(m) + ':' + p(ss); };

function setBadge(cls, txt){ const b = $('badge'); b.className = 'badge ' + cls;
  b.textContent = txt; }

/* 折线图:series = [{label,color,dash,xs:[],ys:[]}](各系列可缺轮,按 xs 对齐) */
function chart(id, series, fmt){
  const svg = $(id), W=560, H=210, L=48, R=16, T=12, B=24, iw=W-L-R, ih=H-T-B;
  let xs=[], pts=[];
  series.forEach(s => { xs = xs.concat(s.xs); pts = pts.concat(s.ys); });
  if(!pts.length){ svg.innerHTML =
    '<text x="'+W/2+'" y="'+H/2+'" fill="#5a6b7a" text-anchor="middle">等待数据…</text>';
    return; }
  let x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...pts), y1=Math.max(...pts);
  if(x0===x1){ x0-=0.5; x1+=0.5; }
  if(y0===y1){ y0-=Math.abs(y0)*0.1+0.01; y1+=Math.abs(y1)*0.1+0.01; }
  const pad=(y1-y0)*0.08; y0-=pad; y1+=pad;
  const X=x=>L+(x-x0)/(x1-x0)*iw, Y=y=>T+ih-(y-y0)/(y1-y0)*ih;
  let g='';
  for(let i=0;i<=4;i++){ const yv=y0+(y1-y0)*i/4, yy=Y(yv);
    g+='<line x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'" stroke="#1f2937"/>'+
       '<text x="'+(L-6)+'" y="'+(yy+4)+'" fill="#5a6b7a" font-size="10" text-anchor="end">'
       +fmt(yv)+'</text>'; }
  for(let i=0;i<=4;i++){ const xv=x0+(x1-x0)*i/4, xx=X(xv);
    g+='<text x="'+xx+'" y="'+(H-8)+'" fill="#5a6b7a" font-size="10" text-anchor="middle">'
       +(xv%1?xv.toFixed(1):xv)+'</text>'; }
  let lines='';
  series.forEach(s=>{
    if(!s.ys.length) return;
    const d = s.xs.map((x,i)=>(i?'L':'M')+X(x).toFixed(1)+' '+Y(s.ys[i]).toFixed(1)).join('');
    lines+='<path d="'+d+'" fill="none" stroke="'+s.color+'" stroke-width="1.8"'+
           (s.dash?' stroke-dasharray="5 4"':'')+'/>';
    const lx=s.xs[s.xs.length-1], ly=s.ys[s.ys.length-1];
    lines+='<circle cx="'+X(lx)+'" cy="'+Y(ly)+'" r="2.6" fill="'+s.color+'"/>';
    lines+='<text x="'+Math.min(X(lx)+5,W-R-8)+'" y="'+(Y(ly)-6)+'" fill="'+s.color+
           '" font-size="10">'+fmt(ly)+'</text>';
    s.xs.forEach((x,i)=>{ lines+='<circle cx="'+X(x)+'" cy="'+Y(s.ys[i])+
      '" r="1.8" fill="'+s.color+'" opacity="0.85"/>'; });
  });
  svg.innerHTML=g+lines;
}

function legend(id, series){
  $(id).innerHTML = series.map(s =>
    '<i style="background:'+s.color+'"></i>'+esc(s.label)).join('&ensp;');
}

function render(d){
  if(!d || d.exists === false){ setBadge('', '等待数据');
    $('stopinfo').textContent = '尚未读到训练状态(.tmp/{产品}/web/state.json)';
    return; }
  $('product').textContent = '— ' + d.product;
  const cfg = d.config || {}, h = d.history || [];
  $('cfgline').textContent = 'tile '+(cfg.tile_size??'-')+' · batch '+
    (cfg.batch_size??'-')+' · epochs '+(cfg.epochs??'-')+' · '+
    (cfg.device??'-')+' · '+(cfg.amp??'-')+' · seed '+(cfg.seed??'-');
  $('c-ep').textContent = h.length + ' / ' + (cfg.epochs ?? '-');

  const stale = Date.now()/1000 - d.update_ts > 120;
  if(d.status === 'done'){ setBadge('done', '已结束'); }
  else if(stale){ setBadge('warn', '疑似中断'); }
  else{ setBadge('run pulse', '运行中'); }
  $('stopinfo').textContent = d.stop_reason ? '停止原因: ' + d.stop_reason : '';

  const b = d.best;
  $('c-best').textContent = b ? b.name : '暂无';
  $('c-bq').textContent = b ? b.quality : '-';
  const last = h[h.length-1] || {};
  $('c-hit').textContent = (last.hit ?? '-') + ' / ' + (last.hit_total ?? '-');
  $('c-inter').textContent = last.inter ?? '-';

  const f3 = v => Number(v).toFixed(3);
  const sTrain = {label:'train_loss', color:'#4fc3f7',
    xs:h.filter(r=>r.train_loss!=null).map(r=>r.epoch),
    ys:h.filter(r=>r.train_loss!=null).map(r=>r.train_loss)};
  const sVal = {label:'val_loss', color:'#f78166', dash:1,
    xs:h.filter(r=>r.val_loss!=null).map(r=>r.epoch),
    ys:h.filter(r=>r.val_loss!=null).map(r=>r.val_loss)};
  const hq = h.filter(r=>r.quality!=null);
  chart('chart-loss', [sTrain, sVal], f3);
  chart('chart-q', [{label:'quality_score', color:'#9ae6b4',
    xs:hq.map(r=>r.epoch), ys:hq.map(r=>r.quality)}], f3);
  legend('lg-loss', [sTrain, sVal]);
  legend('lg-q', [{label:'quality_score', color:'#9ae6b4'}]);

  const box = $('events'), stick = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.innerHTML = (d.events||[]).map(e=>{
    let cls = '';
    if(e.msg.startsWith('5.★')) cls = 'e-best';
    else if(e.msg.includes('5.考试')) cls = 'e-exam';
    else if(e.msg.startsWith('训练结束')) cls = 'e-end';
    return '<div class="'+cls+'"><span class="t">['+esc(e.t)+']</span> '+esc(e.msg)+'</div>';
  }).join('');
  if(stick) box.scrollTop = box.scrollHeight;
  $('elapsed').textContent = '已运行 ' + fmtDur(Date.now()/1000 - d.start_ts) +
    ' · 更新于 ' + new Date(d.update_ts*1000).toLocaleTimeString();
}

async function tick(){
  try{ render(await (await fetch('/api/state')).json()); }
  catch(e){ setBadge('warn', '连接失败'); }
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    state_file: str = ""

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/state":
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._send(200, "application/json; charset=utf-8",
                               f.read().encode("utf-8"))
            except (OSError, ValueError):   # 尚未创建 / 意外半截(理论不会,原子写)
                self._send(200, "application/json; charset=utf-8",
                           b'{"exists": false}')
        else:
            self._send(404, "text/plain; charset=utf-8", b"404")

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):   # 静默逐请求访问日志
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    port = args.port or _load_web_port()
    Handler.state_file = state_path(args.product)
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    print(f"train_mini 监控: http://{args.host}:{port}  (状态文件 {Handler.state_file}",
          flush=True)
    print("Ctrl+C 退出;训练侧另起终端 python main.py --product " + args.product, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
