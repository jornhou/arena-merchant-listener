# -*- coding: utf-8 -*-
"""选输出设备：列出本机所有回环设备，选一个记下来，之后启动就听它。"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(HERE, "记住的设备.txt")


def main():
    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        print("没装 pyaudiowpatch，先跑 install 依赖。")
        return 1
    pa = pyaudio.PyAudio()
    devs = list(pa.get_loopback_device_info_generator())
    default = None
    try:
        default = pa.get_default_wasapi_loopback()["index"]
    except Exception:
        pass
    print("本机的输出设备（游戏声音在哪个设备播，就听哪个）：")
    for d in devs:
        mark = "  ← 系统默认" if d["index"] == default else ""
        print("  [%d] %s%s" % (d["index"], d["name"], mark))
    cur = ""
    if os.path.exists(SAVE):
        cur = io.open(SAVE, encoding="utf-8").read().strip()
        if cur:
            print("\n现在记住的是：%s" % cur)
    print("\n输入序号后回车（直接回车 = 用系统默认）：")
    try:
        s = input().strip()
    except EOFError:
        return 1
    if not s:
        if os.path.exists(SAVE):
            os.remove(SAVE)
        print("已改回：跟着系统默认输出")
        return 0
    try:
        idx = int(s)
    except ValueError:
        print("请输入数字序号")
        return 1
    d = next((x for x in devs if x["index"] == idx), None)
    if d is None:
        print("没有这个序号")
        return 1
    io.open(SAVE, "w", encoding="utf-8").write(d["name"])
    print("已记住：%s\n下次点「战地行商·听声辨物」就听这个设备。" % d["name"])
    pa.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
