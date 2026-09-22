#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
community_split_merge_simple_v1.py

応募用の簡潔な模式アニメーション。
目的は一つだけ：
「語彙のまとまり（Community）は時間とともに分離・合流する」
ことを直感的に示す。

x候補、W番号、C番号、分析手順は表示しない。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SEED = 20260920
FPS = 60
DURATION = 20.0
FRAMES = int(FPS * DURATION)

rng = np.random.default_rng(SEED)

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(left=0.025, right=0.975, top=0.94, bottom=0.07)

XMIN, XMAX = -5.8, 5.8
YMIN, YMAX = -3.05, 3.05

# 背景の語彙場 + 追跡対象となるまとまり
N_BG = 125
N_GROUP = 55

bg_base = np.column_stack([
    rng.uniform(-5.3, 5.3, N_BG),
    rng.uniform(-2.55, 2.55, N_BG)
])

bg_ampx = rng.uniform(.10, .42, N_BG)
bg_ampy = rng.uniform(.10, .38, N_BG)
bg_fx = rng.uniform(.18, .50, N_BG)
bg_fy = rng.uniform(.18, .48, N_BG)
bg_px = rng.uniform(0, 2*np.pi, N_BG)
bg_py = rng.uniform(0, 2*np.pi, N_BG)

# 中央に最初から「まとまり」として見える点群を作る
local = rng.normal(loc=(0, 0), scale=(0.72, 0.48), size=(N_GROUP, 2))
local[:,0] = np.clip(local[:,0], -1.45, 1.45)
local[:,1] = np.clip(local[:,1], -0.95, 0.95)

half = N_GROUP // 2
left_ids = np.arange(0, half)
right_ids = np.arange(half, N_GROUP)

g_ampx = rng.uniform(.08, .28, N_GROUP)
g_ampy = rng.uniform(.08, .25, N_GROUP)
g_fx = rng.uniform(.22, .48, N_GROUP)
g_fy = rng.uniform(.20, .46, N_GROUP)
g_px = rng.uniform(0, 2*np.pi, N_GROUP)
g_py = rng.uniform(0, 2*np.pi, N_GROUP)

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3 - 2*x)

def background_positions(t):
    x = bg_base[:,0] + bg_ampx*np.sin(bg_fx*t + bg_px)
    y = bg_base[:,1] + bg_ampy*np.cos(bg_fy*t + bg_py)
    return np.column_stack([x,y])

def group_positions(t):
    # 個々の玉の自然な揺らぎ
    x = local[:,0] + g_ampx*np.sin(g_fx*t + g_px)
    y = local[:,1] + g_ampy*np.cos(g_fy*t + g_py)

    # 0-4秒: 一つのまとまり
    # 4-8秒: 分離
    # 8-11秒: 二群
    # 11-16秒: 合流
    # 16-20秒: 再び一群として動く
    if t < 4:
        sep = 0.0
    elif t < 8:
        sep = 1.72 * smoothstep((t-4)/4)
    elif t < 11:
        sep = 1.72
    elif t < 16:
        sep = 1.72 * (1.0 - smoothstep((t-11)/5))
    else:
        sep = 0.0

    x[left_ids] -= sep
    x[right_ids] += sep

    # 二群が完全な鏡像に見えないよう、ごく小さな縦方向差をつける
    y[left_ids] += 0.10 * (sep/1.72 if sep else 0)
    y[right_ids] -= 0.08 * (sep/1.72 if sep else 0)

    return np.column_stack([x,y])

def phase_text(t):
    if t < 4:
        return "一つのまとまり＝C1"
    if t < 8:
        return "分離"
    if t < 11:
        return "二つのまとまり＝C2とC3"
    if t < 16:
        return "合流"
    return "一つのまとまり＝C4"

def update(frame):
    t = frame / FPS
    B = background_positions(t)
    G = group_positions(t)

    ax.clear()
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0, 2.78,
        "語彙のまとまり（Community）は時間とともに変化する",
        ha="center", va="center", fontsize=20, weight="bold"
    )

    # 背景も動く。対象群だけを不自然に孤立させない。
    ax.scatter(B[:,0], B[:,1], s=28, alpha=.48, zorder=2)

    # 追跡対象のまとまり。
    # 色は一色のまま。C番号や境界線も使わない。
    ax.scatter(G[:,0], G[:,1], s=48, alpha=.88, zorder=4)

    label = phase_text(t)
    ax.text(
        0, -2.72, label,
        ha="center", va="center",
        fontsize=18, weight="bold"
    )

ani = FuncAnimation(
    fig, update,
    frames=FRAMES,
    interval=1000/FPS,
    blit=False,
    repeat=True
)

plt.show()

# ffmpeg導入済みならMP4直接保存も可能:
# plt.show() をコメントアウトし、以下を有効化
#
# ani.save(
#     "community_split_merge_simple_v1.mp4",
#     writer="ffmpeg",
#     fps=FPS,
#     dpi=120,
#     bitrate=5000
# )
