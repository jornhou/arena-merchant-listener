# -*- coding: utf-8 -*-
"""战地行商 · 拾取音实时鉴定（本机监听版）

原理：用 WASAPI 回环录下扬声器正在播放的声音（等于录游戏的输出），
      检测到一次「拾取音」事件后，取 1.4s 波形与 17 段原声做归一化互相关，
      取最相似的一组，弹窗显示组名、价位带与该组藏品上限。

用法：
    python listen.py               正常监听（游戏开着就行，不用切窗口）
    python listen.py --list        列出可用的回环设备
    python listen.py --device 耳机  指定设备（名称片段）
    python listen.py --selftest    自检：依次播放 4 段原声，看能否认回来
"""
import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000          # 采集与素材采样率
TSR = 16000         # 比对用降采样率
STRIDE = SR // TSR
NFFT = 65536
WIN_SEC = 1.45      # 每次事件取多长的窗口
PREROLL = 0.06      # 提前一点，避免切掉起音
DEVICE = None       # --device 指定的设备名片段
SELFTEST_LOG = None # 自检时记录识别序列
READY = threading.Event()   # 采集流就绪
RECORD_PATH = None  # 调试：把采集到的声音存成 wav

# 音色指纹：16k 采样下 2048 点 FFT（频率分辨率 7.8Hz），24 个对数频带
NBAND = 24
W_NCC = 0.85        # 波形相关权重（离线核对：≥0.85 全对，再低就会把 04/05 认混）
W_TONE = 0.15       # 音色余弦权重
BEDGE = np.round(np.geomspace(80.0, 7000.0, NBAND + 1) / (TSR / 2048)).astype(int)


# ---------------------------------------------------------------- 素材
def load_refs(snd_dir, groups):
    refs = []
    for g in groups:
        path = os.path.join(snd_dir, "snd%02d.wav" % g["id"])
        if not os.path.exists(path):
            raise SystemExit(
                "缺少声音样本：%s\n"
                "本仓库不附带游戏原声（版权原因）。请按 docs/准备声音样本.md "
                "把自己录制的 %d 段 wav 放进 snd\\ 后再运行。" % (path, len(groups))
            )
        with wave.open(path, "rb") as w:
            n, ch, sw, fr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(n)
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if fr != SR:
            raise SystemExit("素材采样率不是 48k：%s" % path)
        t = preprocess(x)
        refs.append({
            "id": g["id"], "name": g["name"], "info": g,
            "len": len(t),
            "fft": np.fft.rfft(t, NFFT),
            "energy": float(np.sum(t * t)),
            "band": bandvec(t),
        })
    return refs


def preprocess(x):
    """一阶高通 + 降采样到 16k，去掉低频轰鸣，让比对更稳。"""
    x = np.asarray(x, dtype=np.float32)
    d = np.empty_like(x)
    d[0] = x[0]
    d[1:] = x[1:] - 0.965 * x[:-1]
    n = len(d) // STRIDE
    d = d[: n * STRIDE].reshape(n, STRIDE)
    y = d.mean(axis=1)
    y = y - y.mean()
    return y.astype(np.float32)


def bandvec(seg):
    """24 个对数频带的能量形状（去掉整体响度，只留音色），用于第二道鉴别。"""
    seg = np.asarray(seg, dtype=np.float32)
    if len(seg) < 64:
        return np.zeros(NBAND)
    n = 2048
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)) ** 2
    v = np.zeros(NBAND)
    for b in range(NBAND):
        lo, hi = BEDGE[b], BEDGE[b + 1]
        v[b] = spec[lo:hi].sum()
    v = np.log(v + 1e-12)
    v -= v.mean()
    nrm = np.linalg.norm(v) + 1e-9
    return v / nrm


def match(refs, win):
    """返回 [(score, ref, lag16), ...] 降序；lag16 是素材在窗口里的起点（16k 采样点）。

    score = 0.72 * 波形相关 + 0.28 * 音色余弦，避免"波形像但音色不像"的错认。
    """
    w = preprocess(win)
    if len(w) < 64:
        return []
    n = len(w)
    W = np.fft.rfft(w, NFFT)
    cs = np.concatenate(([0.0], np.cumsum(w * w, dtype=np.float64)))
    out = []
    for r in refs:
        L = r["len"]
        if L >= n:
            t = np.fft.irfft(np.conj(r["fft"]) * W, NFFT)[:n]
            cc = np.array([float(np.sum(t))])
            es = np.array([cs[n]])
        else:
            m = n - L + 1
            cc = np.fft.irfft(W * np.conj(r["fft"]), NFFT)[:m]
            es = cs[L:] - cs[:m]
        den = np.sqrt(np.maximum(es, 1e-12) * max(r["energy"], 1e-12))
        ratio = cc / den
        k = int(np.argmax(ratio))
        out.append((float(ratio[k]), r, k))
    out.sort(key=lambda p: -p[0])
    # 前 4 名再过一道音色比对
    ranked = []
    for i, (ncc, r, k) in enumerate(out):
        if i < 4:
            seg = w[k:k + r["len"]]
            cos = float(np.dot(bandvec(seg), r["band"]))
            ncc = W_NCC * ncc + W_TONE * (cos + 1.0) * 0.5
        ranked.append((ncc, r, k))
    ranked.sort(key=lambda p: -p[0])
    return ranked


# ---------------------------------------------------------------- 采集
class Capture:
    def __init__(self, device_name=None, dev=None, pa=None):
        import pyaudiowpatch as pyaudio
        self.p = pyaudio
        self.own_pa = pa is None
        self.pa = pa if pa is not None else pyaudio.PyAudio()
        self.dev = dev if dev is not None else self._pick(device_name)
        self.rate = int(self.dev["defaultSampleRate"])
        self.buf = np.zeros(0, dtype=np.float32)
        self.blocks = []            # 按块存，避免每读一次就整段拼接（会越读越慢、越读越旧）
        self.counter = 0
        self.t_start = time.time()  # 用来核对"读到哪了"和真实时间差多少
        self.noise = 0.004
        self.last_t = time.time()
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=min(2, int(self.dev["maxInputChannels"])),
            rate=self.rate,
            input=True,
            input_device_index=self.dev["index"],
            frames_per_buffer=2048,
        )
        # 关键：给同一个扬声器喂静音，让回环一直有数据流，否则设备一空闲 read 就永久卡住
        self._keepalive()
        self.silent_since = time.time()
        self.alive = True
        threading.Thread(target=self._pump, daemon=True).start()

    def _reopen(self):
        """设备闲置太久后，老的回环句柄就听不到新声音了，只能重开一次。"""
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass
        for _ in range(6):
            try:
                self.stream = self.pa.open(
                    format=self.p.paFloat32,
                    channels=min(2, int(self.dev["maxInputChannels"])),
                    rate=self.rate,
                    input=True,
                    input_device_index=self.dev["index"],
                    frames_per_buffer=2048,
                )
                self.silent_since = time.time()
                if os.environ.get("ZD_DEBUG"):
                    print("  (重开采集流成功)", flush=True)
                return True
            except Exception as e:
                if os.environ.get("ZD_DEBUG"):
                    print("  (重开失败 %s)" % e, flush=True)
                time.sleep(0.2)
        self.silent_since = time.time()
        return False

    def _pump(self):
        """后台按设备的真实节奏连续读，保证音频不断、不丢、不落后。"""
        ch = min(2, int(self.dev["maxInputChannels"]))
        while getattr(self, "alive", False):
            # 一直没声音超过 1.5 秒：把流重开一下，老句柄会开始只吐静音
            # 只在长时间（20 秒以上）全程无声音时才重开，避免打断正常拾取
            if self.silent_since is not None and time.time() - self.silent_since > 20.0:
                self._reopen()
            try:
                raw = self.stream.read(2400, exception_on_overflow=False)
            except Exception:
                time.sleep(0.15)
                continue
            a = np.frombuffer(raw, dtype=np.float32)
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            self.buf = np.concatenate((self.buf, a))[-int(self.rate * 3.2):]
            self.counter += len(a)
            if float(np.sqrt(np.mean(np.square(a)))) > 1e-3:
                self.silent_since = None
            elif self.silent_since is None:
                self.silent_since = time.time()

    def _keepalive(self):
        """往这个设备对应的扬声器持续写静音，回环才不会因为"没人放声音"而卡死。"""
        try:
            base = self.dev["name"].replace("[Loopback]", "").strip()
            out = None
            for i in range(self.pa.get_device_count()):
                d = self.pa.get_device_info_by_index(i)
                if (int(d["maxOutputChannels"]) > 0 and "Loopback" not in d["name"]
                        and d["name"].strip() == base):
                    out = d
                    break
            if out is None:
                return
            st = self.pa.open(format=self.p.paFloat32, channels=2, rate=self.rate,
                              output=True, output_device_index=out["index"],
                              frames_per_buffer=1024)
        except Exception:
            self.keep = None
            return
        self.keep = st
        self.keep_alive = True
        # 静音保活：让回环一直有数据流，否则设备一空闲 read 会卡住
        keep_sig = np.zeros(1024 * 2, dtype=np.float32).tobytes()

        def feed():
            while getattr(self, "keep_alive", False):
                try:
                    st.write(keep_sig)     # 阻塞写，节奏由设备自己定，不会写太快
                except Exception:
                    return

        threading.Thread(target=feed, daemon=True).start()

    def _pick(self, want):
        cands = list(self.pa.get_loopback_device_info_generator())
        if not cands:
            raise SystemExit("没有找到回环录制设备（WASAPI loopback 不可用）")
        if want:
            for d in cands:
                if want in d["name"]:
                    return d
            print("没找到含「%s」的设备，改用默认输出对应的回环。" % want)
        try:
            return self.pa.get_default_wasapi_loopback()
        except Exception:
            d = self.pa.get_device_info_by_index(
                self.p.get_host_api_info_by_type(self.p.paWASAPI)["defaultOutputDevice"])
            return d

    def read(self, frames):
        raw = self.stream.read(frames, exception_on_overflow=False)
        a = np.frombuffer(raw, dtype=np.float32)
        ch = min(2, int(self.dev["maxInputChannels"]))
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        self.buf = np.concatenate((self.buf, a))[-int(self.rate * 3.2):]
        self.counter += len(a)
        return a

    def pick(self, scan, floor=0.0008):
        """音频由后台线程连续读，这里只取最近一段窗口。"""
        if len(self.buf) < scan:
            return None
        return self.buf[-scan:], self.counter - scan, self.counter, self.dev["name"]

    def window(self, onset_counter, sec):
        start = max(0, onset_counter - self.counter + len(self.buf))
        n = int(self.rate * sec)
        seg = self.buf[start:start + n]
        return seg

    def close(self):
        self.keep_alive = False
        self.alive = False
        try:
            time.sleep(0.25)
        except Exception:
            pass
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass
        try:
            if getattr(self, "keep", None) is not None:
                self.keep.stop_stream()
                self.keep.close()
        except Exception:
            pass
        if self.own_pa:
            try:
                self.pa.terminate()
            except Exception:
                pass


class AutoCapture:
    """采集入口：优先用上次记住的设备，其次系统默认输出。

    本机有多个输出设备（Realtek / XIBERIA / G24H1 / FxSound / YY AI），
    游戏走哪个不确定，所以用 选设备.bat 指定一次，之后就记住了。
    """

    SAVE = "记住的设备.txt"
    passive = False

    def __init__(self, device_name=None):
        self.cap = None
        if not self._open(device_name):
            self.cap = Capture(None)
        print("监听设备：%s（%d Hz）" % (self.cap.dev["name"], self.cap.rate), flush=True)

    def _saved(self):
        try:
            p = os.path.join(HERE, self.SAVE)
            if os.path.exists(p):
                return open(p, encoding="utf-8").read().strip() or None
        except Exception:
            pass
        return None

    def _open(self, want):
        for cand in (want, self._saved()):
            if not cand:
                continue
            try:
                self.cap = Capture(cand)
                print("使用指定设备：%s" % self.cap.dev["name"], flush=True)
                return True
            except SystemExit:
                continue
        return False

    @property
    def dev(self):
        return self.cap.dev

    @property
    def rate(self):
        return self.cap.rate

    def pick(self, scan, floor=0.0008):
        return self.cap.pick(scan)

    def close(self):
        self.cap.close()


def rms(a):
    return float(np.sqrt(np.mean(a * a) + 1e-12))


# ---------------------------------------------------------------- 界面
UI_CFG = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                      "战地行商听声辨物", "ui.json")


def load_shape_filter():
    """上次点的格型（宽,高）；没选过就是 None＝全部。"""
    try:
        with open(UI_CFG, encoding="utf-8") as f:
            v = json.load(f).get("shape")
        if isinstance(v, list) and len(v) == 2:
            return (int(v[0]), int(v[1]))
    except Exception:
        pass
    return None


def save_shape_filter(v):
    try:
        os.makedirs(os.path.dirname(UI_CFG), exist_ok=True)
        with open(UI_CFG, "w", encoding="utf-8") as f:
            json.dump({"shape": list(v) if v else None}, f, ensure_ascii=False)
    except Exception:
        pass


class Overlay:
    """置顶小窗，显示最近一次鉴定结果；可点格型按钮只看横6格 / 竖6格。"""

    def __init__(self, groups):
        import tkinter as tk
        self.tk = tk
        self.q = queue.Queue()
        self.root = tk.Tk()
        self.root.title("战地行商 · 听声辨物")
        self.root.geometry("452x120+40+40")
        self.root.configure(bg="#0e1411")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.dead = False

        # 空态：只一句提示
        self.idle = tk.Label(self.root, text="等待拾取音…", bg="#0e1411", fg="#8d968f",
                             font=("Microsoft YaHei", 11), anchor="w", justify="left")
        self.idle.place(x=16, y=16)
        # 底部小字：正在听哪个设备（听错设备时点「选输出设备.bat」换一个）
        self.devhint = tk.Label(self.root, text="", bg="#0e1411", fg="#5f6a63",
                                font=("Microsoft YaHei", 8), anchor="w", justify="left")
        self.devhint.place(x=16, y=100)
        # 内容：只有大金，按占格分行
        self.t4 = tk.Label(self.root, text="", bg="#0e1411", fg="#e9e4d6",
                           font=("Microsoft YaHei", 10), anchor="nw", justify="left", wraplength=422)
        # 格型选择条：点一下只看某种格子，点「全部」回来
        self.chip_frame = tk.Frame(self.root, bg="#0e1411")
        self.chip_widgets = []
        # 概率
        self.t3 = tk.Label(self.root, text="", bg="#0e1411", fg="#c9a24a",
                           font=("Consolas", 15, "bold"), anchor="w", justify="left")
        self.bar = tk.Frame(self.root, bg="#e3a93a", height=3, width=0)
        self.btn = type("Holder", (object,), {})()
        self.btn.clear = tk.Button(self.root, text="清空", command=self._clear, bg="#1a231d",
                                   fg="#8d968f", bd=0, font=("Microsoft YaHei", 9), cursor="hand2")
        self.btn.clear.place(x=406, y=60)

        # 最近一次结果 + 当前选的格型
        self.last_info = None
        self.last_score = 0.0
        self.last_weak = False
        self.shape_filter = load_shape_filter()
        self._layout()
        self.root.after(80, self._poll)

    def _quit(self):
        self.dead = True
        try:
            self.root.destroy()
        finally:
            os._exit(0)       # 后台还在读音频，直接收摊，别留进程

    def set_dev(self, name):
        try:
            self.devhint.configure(text="正在听：" + name)
        except Exception:
            pass

    def _clear(self):
        self.last_info = None
        self.last_score = 0.0
        self.t4.configure(text="")
        self.t4.place_forget()
        self.t3.configure(text="")
        self.t3.place_forget()
        self.bar.configure(width=0)
        self.bar.place_forget()
        self.chip_frame.place_forget()
        self.idle.place(x=16, y=16)
        self._layout()

    def push(self, res, weak=False):
        self.q.put((res, weak))

    # ------------------------------------------------------------------ 格型条
    def _pick(self, key):
        """点某个格型：只看它；再点「全部」回到全显示。选择会记住。"""
        self.shape_filter = key
        save_shape_filter(key)
        self._render()

    def _build_chips(self, specs):
        tk = self.tk
        for w in self.chip_widgets:
            w.destroy()
        self.chip_widgets = []
        if not specs:
            return
        for i, s in enumerate(specs):
            text = "全部" if s is None else shape_label(s[0], s[1])
            on = (s == self.shape_filter)
            lb = tk.Label(self.chip_frame, text=text,
                          bg="#c9a24a" if on else "#1a231d",
                          fg="#0e1411" if on else "#8d968f",
                          font=("Microsoft YaHei", 9), padx=8, pady=2, cursor="hand2")
            lb.grid(row=i // 4, column=i % 4, padx=(0, 6), pady=(0, 4), sticky="ew")
            lb.bind("<Button-1>", lambda e, k=s: self._pick(k))
            self.chip_widgets.append(lb)
        for c in range(4):
            self.chip_frame.grid_columnconfigure(c, weight=1)

    def _render(self):
        """按当前选的格型重排小窗：只有大金 + 概率 + 格型条。"""
        g = self.last_info
        if g is None:
            return
        rows = big_groups(g)
        specs = [(r["gw"], r["gh"]) for r in rows]
        chips = [None, (3, 2), (2, 3)]      # 全部 / 横6格 / 竖6格 常驻，其余按本组补
        for s in specs:
            if s not in chips:
                chips.append(s)
        if self.shape_filter is not None and self.shape_filter not in chips:
            chips.append(self.shape_filter)
        self._build_chips(chips)

        txt = big_lines(g, only=self.shape_filter)
        dim = self.last_weak
        if self.shape_filter is not None and self.shape_filter not in specs:
            dim = True                     # 这一组没有这种格子，灰字说清楚
        self.t4.configure(text=txt, fg="#a8b3a8" if dim else "#e9e4d6")
        pct = round(self.last_score * 100)
        self.t3.configure(text=("概率 %d%%（偏低）" % pct) if self.last_weak else ("概率 %d%%" % pct),
                          fg="#c9a24a" if not self.last_weak else "#7d857e")
        self.bar.configure(width=int(416 * min(1.0, max(0.0, self.last_score))),
                           bg="#e3a93a" if not self.last_weak else "#6b736c")
        self._layout()

    def _layout(self):
        """内容 + 概率 + 格型条，窗口按内容高度收缩。"""
        if self.last_info is None:
            total = 62
            self.root.geometry("452x%d+%d+%d" % (total, self.root.winfo_x(), self.root.winfo_y()))
            self.devhint.place(x=16, y=100)
            self.chip_frame.place_forget()
            return
        self.t4.place(x=16, y=16)
        self.root.update_idletasks()
        h4 = max(16, self.t4.winfo_reqheight())
        y3 = 16 + h4 + 8
        ybar = y3 + 34
        ychip = ybar + 10
        chip_h = 0
        if self.chip_widgets:
            self.chip_frame.place(x=16, y=ychip, width=422)
            self.root.update_idletasks()
            chip_h = self.chip_frame.winfo_reqheight()
        ydev = (ychip + chip_h + 8) if chip_h else (ybar + 32)
        total = ydev + 18
        self.root.geometry("452x%d+%d+%d" % (total, self.root.winfo_x(), self.root.winfo_y()))
        self.t3.place(x=16, y=y3)
        self.bar.place(x=16, y=ybar)
        self.btn.clear.place(x=406, y=y3 + 4)
        self.devhint.place(x=16, y=ydev)

    def _poll(self):
        try:
            while True:
                res, weak = self.q.get_nowait()
                self.last_info = res[0][1]["info"]
                self.last_score = res[0][0]
                self.last_weak = weak
                # 只有大金（按占格分行）+ 概率，其余一律不显示
                self.idle.place_forget()
                self._render()
        except queue.Empty:
            pass
        if not self.dead:
            self.root.after(80, self._poll)

    def run(self):
        self.root.mainloop()


def shape_label(gw, gh):
    """宽×高 → 听得懂的说法（6 格的分横着、竖着两种说法）。"""
    if gw == 1 and gh == 1:
        return "单格"
    if gw == 1:
        return "竖%d格" % gh
    if gh == 1:
        return "横%d格" % gw
    if gw == 2 and gh == 2:
        return "2×2方块"
    if gw * gh == 6:
        return "横6格" if gw > gh else "竖6格"
    return "%d×%d格" % (gw, gh)


def shape_note(gw, gh):
    """横6格＝3×2、竖6格＝2×3，写成小字跟在后面。"""
    if gw * gh == 6 and gw > 1 and gh > 1:
        return "%d×%d" % (gw, gh)
    return ""


def shape_line(g):
    """这一组里的格子分布，例如「单格8 · 竖2格7 · 2×2方块7」。"""
    sh = g.get("shapes") or []
    if not sh:
        return ""
    parts = []
    for k, n in sh:
        try:
            gw, gh = [int(x) for x in k.split("×")]
        except Exception:
            continue
        parts.append("%s%d" % (shape_label(gw, gh), n))
    return "格子：" + "·".join(parts)


def item_line(g, limit=6):
    """把这一组里最值钱的几件物品连占格一起列出来。"""
    items = g.get("items") or []
    if not items:
        return ""
    head = "、".join("%s %s %s" % (i["name"], shape_label(i.get("gw", 1), i.get("gh", 1)),
                              wan(i["price"])) for i in items[:limit])
    if len(items) > limit:
        head += " 等 %d 件" % len(items)
    return head


BIG_MIN_RARITY = 5   # 5=金、6=红，这两种算"大金"


def big_groups(g):
    """把这一组的大金按格子归堆，返回 [(格式, [物品...]), ...]，按价钱降序。"""
    big = [i for i in (g.get("items") or []) if i.get("r", 1) >= BIG_MIN_RARITY]
    bucket = {}
    for it in big:
        key = (shape_label(it.get("gw", 1), it.get("gh", 1)), it.get("gw", 1), it.get("gh", 1))
        bucket.setdefault(key, []).append(it)
    rows = []
    for (label, gw, gh), arr in bucket.items():
        arr.sort(key=lambda x: -x["price"])
        rows.append({"label": label, "gw": gw, "gh": gh, "items": arr})
    rows.sort(key=lambda r: -r["items"][0]["price"])
    return rows


def big_lines(g, max_rows=9, per_row=3, only=None):
    """每个格子一行，只列大金。only=(宽,高) 时只留这一种格型。"""
    rows = big_groups(g)
    if only is not None:
        rows = [r for r in rows if (r["gw"], r["gh"]) == tuple(only)]
        if not rows:
            return "这一组没有 %s" % shape_label(only[0], only[1])
    if not rows:
        return item_line(g, 5)
    out = []
    for r in rows[:max_rows]:
        cells = " · ".join("%s %s" % (i["name"], wan(i["price"])) for i in r["items"][:per_row])
        more = " 等%d件" % len(r["items"]) if len(r["items"]) > per_row else ""
        out.append("%s｜%s%s" % (r["label"], cells, more))
    if len(rows) > max_rows:
        rest = sum(len(r["items"]) for r in rows[max_rows:])
        out.append("…还有 %d 种格子 %d 件大金" % (len(rows) - max_rows, rest))
    return "\n".join(out)


def big_summary(g):
    rows = big_groups(g)
    if not rows:
        return "本组没有大金，只有小件"
    total = sum(len(r["items"]) for r in rows)
    return "大金 %d 件 · 按占格分 %d 行" % (total, len(rows))


def wan(v):
    v = float(v)
    if v >= 1e8:
        return "%.2f亿" % (v / 1e8)
    if v >= 1e4:
        return "%.1f万" % (v / 1e4) if v < 1e6 else "%d万" % round(v / 1e4)
    return "%d" % round(v)


# ---------------------------------------------------------------- 主循环
FRAME_SEC = 0.02
FLOOR = 0.0025      # 触发阈值下限（--sensitivity 可调）
MIN_SCORE = 0.45    # 够这个分才当主结果
MIN_MARGIN = 0.10   # 与次选的最小差距
WEAK_SCORE = 0.26   # 低可信提示线
VERBOSE = False
TICK = 0.12         # --light 时放宽到 0.24，CPU 减半



def listen(refs, overlay=None, seconds=None, cap=None):
    # 音频流必须在主线程里开，放到子线程里开会听不到别的程序（比如游戏）放出来的声音
    if cap is None:
        cap = Capture(DEVICE) if DEVICE else AutoCapture()
    rate = cap.rate
    if overlay:
        overlay.set_dev(cap.dev["name"])
    tick = TICK                # 每 0.12s 对最近 1.2s 音频做一次全量比对
    scan = int(rate * 1.2)
    t0 = time.time()
    due = 0.0
    last_peak = {}
    print("比对窗口 %.1fs，间隔 %.2fs" % (scan / rate, tick), flush=True)
    rec = [] if RECORD_PATH else None
    try:
        while True:
            if seconds and time.time() - t0 > seconds:
                break
            READY.set()
            got = cap.pick(scan)          # 拿最新窗口（Capture 顺便读一小块，AutoCapture 由后台线程喂）
            now_t = time.time()
            if now_t - due < tick:
                time.sleep(0.03)      # 别空转，空转会把后台读音频的线程饿死
                continue
            due = now_t
            if got is None:
                continue
            win, base_abs, now_c, dev_name = got
            if rec is not None:
                rec.append(win.copy())
            res = match(refs, win)
            if os.environ.get("ZD_DEBUG"):
                lag = (time.time() - getattr(cap, "t_start", now_t)) - now_c / rate
                print("tick c=%d lag=%.1f rms=%.4f peak=%.3f top=%s" % (
                    now_c, lag, float(np.sqrt(np.mean(win * win))), float(abs(win).max()),
                    ("%02d %.2f" % (res[0][1]["id"], res[0][0]) if res else "-")), flush=True)
            if not res:
                continue
            s0 = res[0][0]
            s1 = res[1][0] if len(res) > 1 else 0.0
            ref = res[0][1]
            peak_t = (base_abs + res[0][2] * STRIDE) / rate
            now = now_c / rate
            if VERBOSE and dev_name != getattr(cap, "last_name", None):
                cap.last_name = dev_name
                print("（现在的声音来自：%s）" % dev_name, flush=True)
            lp = last_peak.get(dev_name, -9.0)
            if s0 >= MIN_SCORE and (s0 - s1) >= MIN_MARGIN:
                # 同一次声音会被反复扫到，只报一次（按峰值时刻去重）
                if peak_t - lp > 0.55 and now - peak_t < 1.3:
                    last_peak[dev_name] = peak_t
                    report(res, overlay, weak=False)
            elif s0 >= WEAK_SCORE and (s0 - s1) >= 0.06:
                if peak_t - lp > 0.55 and now - peak_t < 1.0:
                    last_peak[dev_name] = peak_t
                    report(res, overlay, weak=True)
            elif VERBOSE and s0 >= 0.12:
                print("  (略过 %.1fs) 最高 %02d %.2f 差距 %.2f" % (
                    peak_t, ref["id"], s0, s0 - s1), flush=True)
    finally:
        cap.close()
        if rec:
            pcm = (np.clip(np.concatenate(rec), -1, 1) * 32767).astype("<i2")
            with wave.open(RECORD_PATH, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
                w.writeframes(pcm.tobytes())
            print("已存录音 %s" % RECORD_PATH, flush=True)


def report(res, overlay, weak=False):
    best = res[0][1]
    scores = [res[0][0], res[1][0] if len(res) > 1 else 0.0]
    g = best["info"]
    line = "%02d %-16s 相似 %.2f 差距 %.2f | %d件 %s-%s 中位%s%s" % (
        g["id"], g["name"], scores[0], scores[0] - scores[1], g["n"],
        wan(g["min"]), wan(g["max"]), wan(g["med"]),
        (" 红货%d" % g["red"]) if g["red"] else "")
    if weak:
        line = "低可信 · " + line
    print(time.strftime("%H:%M:%S"), line, flush=True)
    print("         └ " + big_summary(g), flush=True)
    for ln in big_lines(g, max_rows=12).split("\n"):
        print("         │ " + ln, flush=True)
    if SELFTEST_LOG is not None:
        SELFTEST_LOG.append((g["id"], scores[0], scores[0] - scores[1], weak))
    if overlay:
        overlay.push(res, weak)


def check_offline(refs):
    """离线核对：把 17 段原声塞进带噪声的窗口，看能不能认回自己。"""
    rng = np.random.default_rng(20261114)
    ok_all, fails = [], []
    for n_try in range(3):
        for r in refs:
            with wave.open(os.path.join(HERE, "snd", "snd%02d.wav" % r["id"]), "rb") as w:
                ch = w.getnchannels()
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            if ch > 1:
                raw = raw.reshape(-1, ch).mean(axis=1)
            win_len = int(SR * 1.2)
            gain = float(rng.uniform(0.4, 2.0))
            sig = raw * gain
            off = int(SR * rng.uniform(0.08, 0.30))
            win = np.zeros(win_len, dtype=np.float32)
            seg = sig[: max(0, min(len(sig), win_len - off))]
            win[off:off + len(seg)] = seg
            rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
            noise = rng.normal(0, rms / (10 ** (18 / 20.0)), win_len).astype(np.float32)
            win = win + noise
            res = match(refs, win)
            got = res[0][1]["id"]
            if got == r["id"]:
                ok_all.append((r["id"], res[0][0]))
            else:
                fails.append((r["id"], got, round(res[0][0], 2)))
    hits = len(ok_all)
    total = len(refs) * 3
    print("离线核对（17 段×3 次，带噪声/不同音量/不同位置，SNR≈18dB）：命中 %d/%d" % (hits, total), flush=True)
    if fails:
        print("  错判：%s" % fails, flush=True)
    low = [i for i, s in ok_all if s < 0.5]
    if low:
        print("  其中分数偏低（<0.5）的：%s" % sorted(set(low)), flush=True)
    return hits == total


def main():
    global DEVICE, FLOOR, VERBOSE, TICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="回环设备名片段，如 耳机 / XIBERIA")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-all", action="store_true", dest="selftest_all",
                    help="把 17 段已知原声全部播一遍，逐段核对识别对不对")
    ap.add_argument("--sensitivity", default="normal", choices=["low", "normal", "high"],
                    help="节流：low 只认很响的，high 更灵敏（游戏音量小时用）")
    ap.add_argument("--light", action="store_true",
                    help="省 CPU 模式：比对间隔放宽到 0.24s，识别略慢一点点")
    ap.add_argument("--check", action="store_true",
                    help="离线核对识别：17 段原声加噪声/改音量，看是否都认回自己")
    ap.add_argument("--boost", action="store_true",
                    help="自检时用放大音量的副本（环境噪声大时用）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--record", default=None, help="调试：把采集的声音存成 wav")
    args = ap.parse_args()
    DEVICE = args.device
    VERBOSE = args.verbose
    if args.light:
        TICK = 0.24
    global RECORD_PATH
    RECORD_PATH = args.record
    FLOOR = {"low": 0.012, "normal": 0.0025, "high": 0.0015}[args.sensitivity]

    groups = json.load(open(os.path.join(HERE, "groups.json"), encoding="utf-8"))
    refs = load_refs(os.path.join(HERE, "snd"), groups)

    if args.list:
        import pyaudiowpatch as pyaudio
        pa = pyaudio.PyAudio()
        try:
            for d in pa.get_loopback_device_info_generator():
                print("%3d  %s  %d Hz  %d ch" % (d["index"], d["name"], d["defaultSampleRate"],
                                                 d["maxInputChannels"]))
        finally:
            pa.terminate()
        return

    if args.check:
        check_offline(refs)
        return

    if args.selftest or args.selftest_all:
        global SELFTEST_LOG
        picks = list(range(1, 18)) if args.selftest_all else [1, 6, 12, 17]
        SELFTEST_LOG = []
        READY.clear()
        print("自检：将依次播放 %s 号原声，看是否认回同一个编号。" % picks, flush=True)
        cap0 = Capture(DEVICE) if DEVICE else AutoCapture()      # 主线程开流
        th = threading.Thread(target=listen, args=(refs, None, len(picks) * 2.05 + 5.0),
                              kwargs={"cap": cap0}, daemon=True)
        th.start()
        if not READY.wait(6.0):
            print("采集流没起来，检查回环设备。", flush=True)
            return
        time.sleep(2.4)
        threading.Thread(target=_play_seq,
                         args=(picks, args.boost), daemon=True).start()
        th.join(timeout=len(picks) * 2.05 + 9.0)
        # 按播放分段核对：每段里取分最高的那条当本段答案
        segs, cur = [], None
        for e in SELFTEST_LOG:
            if e[0] == "gap":
                segs.append([])
                cur = segs[-1]
            elif cur is not None:
                cur.append(e)
        ids, reals = [], []
        for k, seg in enumerate(segs[:len(picks)]):
            if seg:
                best = max(seg, key=lambda x: x[1])
                ids.append(best[0])
                reals.append((best[0] - picks[k] == 0, best[1], best[3]))
            else:
                ids.append(0)
                reals.append((False, 0.0, False))
        hit = sum(1 for ok, _, _ in reals if ok)
        miss = [picks[k] for k, (ok, s, w) in enumerate(reals) if s < 0.26 or ids[k] == 0]
        wrong = [(picks[k], ids[k]) for k, (ok, s, w) in enumerate(reals) if ids[k] != picks[k]]
        weakhit = [picks[k] for k, (ok, s, w) in enumerate(reals) if ok and w]
        print("—— 识别序列 %s；期望 %s；命中 %d/%d" % (ids, picks, hit, len(picks), ), flush=True)
        print("—— 错判 %s；没出结果 %s%s" % (
            wrong or "无", miss or "无",
            ("；其中低可信命中 %s" % weakhit) if weakhit else ""), flush=True)
        if th.is_alive():      # 读线程在等声音，不陪它等了
            os._exit(0)
        SELFTEST_LOG = None
        return

    ov = Overlay(groups)
    cap0 = Capture(DEVICE) if DEVICE else AutoCapture()      # 主线程开流
    th = threading.Thread(target=listen, args=(refs, ov), kwargs={"cap": cap0}, daemon=True)
    th.start()
    ov.run()


def _play_seq(picks, boost=False):
    """自检播放：用系统默认回放播素材，尽量贴近游戏出声的链路。"""
    import winsound
    d = os.path.join(HERE, "snd_boost" if boost else "snd")
    time.sleep(0.8)
    for i in picks:
        print("  play %02d" % i, flush=True)
        if SELFTEST_LOG is not None:
            SELFTEST_LOG.append(("gap", -1.0, 0.0, False))
        winsound.PlaySound(os.path.join(d, "snd%02d.wav" % i),
                           winsound.SND_FILENAME)
        time.sleep(1.2)


def _setup_frozen_log():
    """打包成无控制台 exe 后没有 stdout，把输出重定向到本地日志，免得 print 直接崩。"""
    if not getattr(sys, "frozen", False):
        return None
    base = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                        "战地行商听声辨物")
    try:
        os.makedirs(base, exist_ok=True)
        f = open(os.path.join(base, "run.log"), "w", encoding="utf-8", buffering=1)
        sys.stdout = f
        sys.stderr = f
        return base
    except Exception:
        return None


if __name__ == "__main__":
    FROZEN_DIR = _setup_frozen_log()
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        txt = traceback.format_exc()
        log_dir = FROZEN_DIR or os.path.dirname(
            os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
        try:
            with open(os.path.join(log_dir, "出错日志.txt"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S\n") + txt + "\n")
        except Exception:
            pass
        try:
            import tkinter.messagebox as mb
            mb.showerror("战地行商 · 听声辨物", "程序出错了：\n\n" + txt[-1200:])
        except Exception:
            pass
        raise
