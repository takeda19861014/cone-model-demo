"""
円錐モデル + 推論モデル 統合可視化
v4-single-spiral: 各コーン1本の螺旋 + 同一軌道の双方向移動

Based on cone_spiral.ggb (2026-09-07)

依存:
    pip install numpy matplotlib

実行:
    python cone_model_current_v3.py

操作:
    - routeAB      : 0〜8
                     0〜4: inward   P4 → O
                     4〜8: outward  O → P4
                     往路・復路は同一の螺旋軌道

    - Elev         : 上下の視点角度
    - Azim         : 水平の視点角度
    - Zoom         : 表示範囲

    - Sphere α     : 球面の透明度
    - Cone α       : 円錐面の透明度
    - Spiral α     : 単一螺旋・実移動経路の透明度
    - Reasoning α  : 推論モデル全体の透明度

    α = 0 で非表示
    α = 1 で完全表示

    マウスドラッグでも視点を回転可能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

SQRT2 = np.sqrt(2.0)
PI = np.pi


# =============================================================================
# 基本点・パラメータ
# =============================================================================

O = np.array([0.0, 0.0, 0.0])

t1 = 1.0 / (2.0 * SQRT2)
t2 = 1.0 / 2.0
t3 = 1.0 / SQRT2
t4 = 1.0

C1 = np.array([ 0.5,  0.5, 0.0])
C2 = np.array([-0.5,  0.5, 0.0])
C3 = np.array([-0.5, -0.5, 0.0])
C4 = np.array([ 0.5, -0.5, 0.0])

P1 = np.array([
    0.0,
    1.0 / (2.0 * SQRT2),
    0.0
])

P2 = np.array([
    -0.5,
    0.0,
    0.0
])

P3 = np.array([
    0.0,
    -1.0 / SQRT2,
    0.0
])

P4 = np.array([
    1.0,
    0.0,
    0.0
])

omega = 2.0 * PI


# =============================================================================
# 単一螺旋の共通位相
#
# 各コーンには1本だけ螺旋 S が存在する。
# O→P4 と P4→O は、その同一螺旋を逆向きに移動する。
#
# octave則:
#   Theta(t) = 2*pi*log2(t/t1)
#
#   C1, C3 : theta = +Theta(t)
#   C2, C4 : theta = -Theta(t) + pi
#
# t=0 は螺旋式の極限点 O。
# 描画時は極小正数で打ち切り、Oを明示的に接続する。
# =============================================================================

T_MIN = t1 / (SQRT2 ** 18)


def octave_theta(t):
    t = np.asarray(t, dtype=float)

    if np.any(t <= 0.0):
        raise ValueError(
            "octave_theta は t > 0 でのみ定義されます。"
        )

    return (
        2.0
        * PI
        * np.log2(t / t1)
    )


def spiral_theta(which, t):
    """
    各コーンに1本だけ存在する螺旋の位相。

    C1, C3 : +Theta
    C2, C4 : -Theta + pi
    """

    base = octave_theta(t)

    if which in (1, 3):
        return base

    elif which in (2, 4):
        return -base + PI

    raise ValueError("which must be 1..4")


# =============================================================================
# 円錐面上の座標写像
# =============================================================================

def cone_point(which, t, theta):

    t = np.asarray(t)

    if which == 1:

        x = t/2 * (1 - np.cos(theta))
        y = t/2 * (1 + np.cos(theta))

    elif which == 2:

        x = -t/2 * (1 + np.cos(theta))
        y =  t/2 * (1 - np.cos(theta))

    elif which == 3:

        x =  t/2 * (np.cos(theta) - 1)
        y = -t/2 * (1 + np.cos(theta))

    elif which == 4:

        x =  t/2 * (1 + np.cos(theta))
        y =  t/2 * (np.cos(theta) - 1)

    else:
        raise ValueError("which must be 1..4")

    z = (
        t / SQRT2
        * np.sin(theta)
    )

    return np.stack(
        [x, y, z],
        axis=-1
    )

# =============================================================================
# 単一螺旋生成
# =============================================================================

def spiral_point(which, t):
    """
    単一螺旋 S 上の点。

    t=0 は数学的には極限点 O。
    """

    if np.isscalar(t):
        if float(t) <= 0.0:
            return O.copy()

        return cone_point(
            which,
            float(t),
            spiral_theta(
                which,
                float(t)
            )
        )

    t = np.asarray(
        t,
        dtype=float
    )

    result = np.empty(
        t.shape + (3,),
        dtype=float
    )

    mask = t > 0.0

    if np.any(mask):
        result[mask] = cone_point(
            which,
            t[mask],
            spiral_theta(
                which,
                t[mask]
            )
        )

    if np.any(~mask):
        result[~mask] = O

    return result


def spiral(
    which,
    n=900
):
    """
    各コーン上の完全な1本の螺旋。

    O近傍は T_MIN まで描画。
    """

    t = np.geomspace(
        T_MIN,
        1.0,
        n
    )

    return spiral_point(
        which,
        t
    )


# =============================================================================
# routeAB
# =============================================================================

def u_b1_e(s):
    return t1 * s


def u_b2_e(s):
    return (
        t1
        + (t2-t1)*(s-1)
    )


def u_b3_e(s):
    return (
        t2
        + (t3-t2)*(s-2)
    )


def u_b4_e(s):
    return (
        t3
        + (1-t3)*(s-3)
    )


def u_b4_in_ab(s):
    return (
        1
        - (1-t3)*(s-4)
    )


def u_b3_in_ab(s):
    return (
        t3
        - (t3-t2)*(s-5)
    )


def u_b2_in_ab(s):
    return (
        t2
        - (t2-t1)*(s-6)
    )


def u_b1_in_ab(s):
    return (
        t1
        - t1*(s-7)
    )


# =============================================================================
# 単一の大域経路
#
# O -> P1 -> P2 -> P3 -> P4
# P4 -> P3 -> P2 -> P1 -> O
#
# 往路と復路は同じ幾何学的軌道。
# =============================================================================

def _path_O_to_P4(s):
    """O -> P1 -> P2 -> P3 -> P4 の幾何軌道 (s: 0〜4)。"""

    if not (
        0.0 <= s <= 4.0
    ):
        return None

    if np.isclose(s, 0.0):
        return O.copy()

    if s <= 1:
        t = u_b1_e(s)
        w = 1

    elif s <= 2:
        t = u_b2_e(s)
        w = 2

    elif s <= 3:
        t = u_b3_e(s)
        w = 3

    else:
        t = u_b4_e(s)
        w = 4

    return spiral_point(
        w,
        t
    )


# 順序入れ替え版:
#   route 0〜4 : inward   P4 → O
#   route 4〜8 : outward  O → P4

def inward_position(route):

    if not (
        0.0 <= route <= 4.0
    ):
        return None

    return _path_O_to_P4(
        4.0 - route
    )


def outward_position(route):

    if not (
        4.0 < route <= 8.0
    ):
        return None

    return _path_O_to_P4(
        route - 4.0
    )


# 旧名称をUI互換のため保持
def a_out_position(route):
    return outward_position(route)


def b_in_position(route):
    return inward_position(route)


# =============================================================================
# 実際の語彙軌道から螺旋線を生成
#
# aOut : O  → P4
# bIn  : P4 → O
#
# 玉の位置関数そのものを使うため、
# 表示線と玉の軌道が必ず一致する。
# =============================================================================

def make_route_curve(position_func, start, end, n=1600):

    routes = np.linspace(
        start,
        end,
        n
    )

    points = []

    for r in routes:

        p = position_func(float(r))

        if p is not None:
            points.append(
                np.asarray(p, dtype=float)
            )

    return np.array(points)


def make_aout_curve(n=1600):
    # outward: route 4〜8 (O → P4)
    return make_route_curve(
        outward_position,
        4.0 + 1e-8,
        8.0,
        n
    )


def make_bin_curve(n=1600):
    # inward: route 0〜4 (P4 → O)
    return make_route_curve(
        inward_position,
        0.0,
        4.0,
        n
    )


# =============================================================================
# 推論モデル
# =============================================================================

RI = np.array([
    -1.0,
    0.0,
    0.0
])

RII = np.array([
    -1/SQRT2,
    -1/SQRT2,
    0.0
])

RIII = np.array([
    0.0,
    -1.0,
    0.0
])

RIV = np.array([
    1/SQRT2,
    -1/SQRT2,
    0.0
])

RV = np.array([
    1.0,
    0.0,
    0.0
])

RVI = np.array([
    1/SQRT2,
    1/SQRT2,
    0.0
])

RVII = np.array([
    0.0,
    1.0,
    0.0
])

RVIII = np.array([
    -1/SQRT2,
    1/SQRT2,
    0.0
])


R_CENTERS = [
    RI,
    RII,
    RIII,
    RIV,
    RV,
    RVI,
    RVII,
    RVIII
]


sideR = (
    2.0
    * (SQRT2 - 1.0)
)

dR = (
    sideR
    / SQRT2
)


def midpoint(a, b):
    return (a+b)/2.0


# =============================================================================
# 各推論単位
# =============================================================================

def reasoning_unit(
    index,
    center
):

    # -------------------------------------------------------------------------
    # 正方形
    # -------------------------------------------------------------------------

    if index % 2 == 1:

        A = center + np.array([
            -sideR/2,
             sideR/2,
             0.0
        ])

        B = center + np.array([
             sideR/2,
             sideR/2,
             0.0
        ])

        C = center + np.array([
             sideR/2,
            -sideR/2,
             0.0
        ])

        D = center + np.array([
            -sideR/2,
            -sideR/2,
             0.0
        ])

    else:

        A = center + np.array([
            0.0,
            dR,
            0.0
        ])

        B = center + np.array([
            -dR,
            0.0,
            0.0
        ])

        C = center + np.array([
            0.0,
            -dR,
            0.0
        ])

        D = center + np.array([
            dR,
            0.0,
            0.0
        ])

    # -------------------------------------------------------------------------
    # 各辺中点
    # -------------------------------------------------------------------------

    if index == 1:

        mids = [
            midpoint(A,B),
            midpoint(D,A),
            midpoint(B,C),
            midpoint(C,D)
        ]

    elif index == 2:

        mids = [
            midpoint(A,B),
            midpoint(B,C),
            midpoint(A,D),
            midpoint(C,D)
        ]

    elif index == 3:

        mids = [
            midpoint(A,D),
            midpoint(D,C),
            midpoint(A,B),
            midpoint(B,C)
        ]

    elif index == 4:

        mids = [
            midpoint(B,C),
            midpoint(C,D),
            midpoint(A,B),
            midpoint(A,D)
        ]

    elif index == 5:

        mids = [
            midpoint(D,C),
            midpoint(C,B),
            midpoint(D,A),
            midpoint(A,B)
        ]

    elif index == 6:

        mids = [
            midpoint(C,D),
            midpoint(D,A),
            midpoint(C,B),
            midpoint(B,A)
        ]

    elif index == 7:

        mids = [
            midpoint(B,C),
            midpoint(B,A),
            midpoint(C,D),
            midpoint(D,A)
        ]

    elif index == 8:

        mids = [
            midpoint(A,D),
            midpoint(D,C),
            midpoint(A,B),
            midpoint(B,C)
        ]

    else:
        raise ValueError(index)

    return (
        np.array([A,B,C,D]),
        np.array(mids)
    )


REASON_UNITS = [
    reasoning_unit(
        i+1,
        c
    )
    for i, c
    in enumerate(R_CENTERS)
]


# =============================================================================
# 描画補助
# =============================================================================

def draw_polyline(
    ax,
    pts,
    **kwargs
):

    return ax.plot(
        pts[:,0],
        pts[:,1],
        pts[:,2],
        **kwargs
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    fig = plt.figure(
        figsize=(11, 10)
    )

    # 下側に8本のスライダーを配置するため
    # 3D表示領域を少し上へ移動
    ax = fig.add_axes(
        [0.05, 0.36, 0.9, 0.59],
        projection='3d'
    )


    # =========================================================================
    # 球面
    # =========================================================================

    uu = np.linspace(
        0,
        2*PI,
        72
    )

    vv = np.linspace(
        0,
        PI,
        36
    )

    xs = np.outer(
        np.cos(uu),
        np.sin(vv)
    )

    ys = np.outer(
        np.sin(uu),
        np.sin(vv)
    )

    zs = np.outer(
        np.ones_like(uu),
        np.cos(vv)
    )

    sphere_surf = ax.plot_surface(
        xs,
        ys,
        zs,
        alpha=0.35,
        linewidth=0,
        color='#7fb3ff'
    )


    # =========================================================================
    # 球の大円
    # =========================================================================

    cc = np.linspace(
        0,
        2*PI,
        500
    )

    ax.plot(
        np.cos(cc),
        np.sin(cc),
        np.zeros_like(cc),
        linewidth=1.4,
        color='#333333'
    )


    # =========================================================================
    # 4円錐面
    # =========================================================================

    tt = np.linspace(
        0,
        1,
        42
    )

    ph = np.linspace(
        0,
        2*PI,
        72
    )

    T, PH = np.meshgrid(
        tt,
        ph
    )

    cone_colors = [
        '#ff6b6b',
        '#4dabf7',
        '#51cf66',
        '#ffa94d'
    ]

    cone_surfs = []

    for w in range(1, 5):

        surf = cone_point(
            w,
            T,
            PH
        )

        s = ax.plot_surface(
            surf[...,0],
            surf[...,1],
            surf[...,2],
            alpha=0.35,
            linewidth=0,
            color=cone_colors[w-1]
        )

        cone_surfs.append(s)


    # =========================================================================
    # 各コーンに1本だけある完全螺旋
    #
    # 実移動軌道 O <-> P4 は後で太線表示する。
    # ここでは各コーン全体の1本の螺旋を細線表示する。
    # =========================================================================

    full_spiral_artists = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    for w in range(1, 5):

        curve = spiral(
            w,
            n=1800
        )

        line, = ax.plot(
            curve[:, 0],
            curve[:, 1],
            curve[:, 2],
            linewidth=0.65,
            color='#666666',
            alpha=0.85
        )

        full_spiral_artists[w].append(
            line
        )


# =========================================================================
# 実際の outward / inward 軌道を表示
# =========================================================================

    spiral_artists = []

    a_curve = make_aout_curve(
        n=2000
    )

    b_curve = make_bin_curve(
        n=2000
    )


    # -------------------------------------------------------------------------
    # outward
    # O → P1 → P2 → P3 → P4
    #
    # B1 : まず下へ
    # B2 : 上
    # B3 : 下
    # B4 : 上
    # -------------------------------------------------------------------------

    line_a, = ax.plot(
        a_curve[:,0],
        a_curve[:,1],
        a_curve[:,2],
        linewidth=2.8,
        color='red'
    )

    spiral_artists.append(
        line_a
    )


    # -------------------------------------------------------------------------
    # inward
    # P4 → P3 → P2 → P1 → O
    #
    # B4 : 下
    # B3 : 上
    # B2 : 下
    # B1 : P1からまず上へ
    # -------------------------------------------------------------------------

    line_b, = ax.plot(
        b_curve[:,0],
        b_curve[:,1],
        b_curve[:,2],
        linewidth=2.5,
        linestyle='-',
        color='blue'
    )

    spiral_artists.append(
        line_b
    )


    # =========================================================================
    # O / P1 / P2 / P3 / P4
    # =========================================================================

    key_points = [
        ('O', O),
        ('P1', P1),
        ('P2', P2),
        ('P3', P3),
        ('P4', P4)
    ]

    for name, p in key_points:

        ax.scatter(
            *p,
            s=36,
            color='black'
        )

        ax.text(
            *(
                p
                + np.array([
                    0.02,
                    0.02,
                    0.02
                ])
            ),
            name,
            fontsize=10
        )


    # =========================================================================
    # 推論モデル
    # =========================================================================

    reasoning_artists = []

    romans = [
        'I',
        'II',
        'III',
        'IV',
        'V',
        'VI',
        'VII',
        'VIII'
    ]

    for (
        label,
        center,
        (verts, mids)
    ) in zip(
        romans,
        R_CENTERS,
        REASON_UNITS
    ):

        # 正方形
        closed = np.vstack([
            verts,
            verts[0]
        ])

        line_square, = ax.plot(
            closed[:,0],
            closed[:,1],
            closed[:,2],
            linewidth=1.0,
            color='#888888'
        )

        # ジグザグ
        line_zig, = ax.plot(
            mids[:,0],
            mids[:,1],
            mids[:,2],
            linewidth=2.2,
            color='#222222'
        )

        # I〜VIII
        text_label = ax.text(
            *(
                center
                + np.array([
                    0.0,
                    0.0,
                    0.025
                ])
            ),
            label,
            fontsize=10
        )

        # 各ジグザグの始点と終点を結ぶ線
        # → 正八角形の一辺
        line_edge, = ax.plot(
            [
                mids[0,0],
                mids[3,0]
            ],
            [
                mids[0,1],
                mids[3,1]
            ],
            [
                0.0,
                0.0
            ],
            linewidth=1.6,
            color='#222222'
        )

        reasoning_artists.extend([
            line_square,
            line_zig,
            line_edge,
            text_label
        ])


    # =========================================================================
    # 推論モデル中心の正八角形
    # 補助線
    # =========================================================================

    rc = np.array(
        R_CENTERS
        + [R_CENTERS[0]]
    )

    reason_ring, = ax.plot(
        rc[:,0],
        rc[:,1],
        rc[:,2],
        linewidth=1.0,
        linestyle=':',
        color='#555555'
    )

    reasoning_artists.append(
        reason_ring
    )


    # =========================================================================
    # routeAB 語彙点
    # =========================================================================

    # 順序入れ替え後は route=0 で outward が None になるため、
    # 初期表示位置は O を使う(直後の update_route で正しく更新される)。
    a0 = O

    a_artist = ax.scatter(
        *a0,
        s=110,
        marker='o',
        color='red',
        zorder=10
    )

    b_artist = ax.scatter(
        [],
        [],
        [],
        s=110,
        marker='o',
        color='blue',
        zorder=10
    )

    status = ax.text2D(
        0.02,
        0.98,
        '',
        transform=ax.transAxes
    )


    # =========================================================================
    # 軸
    # =========================================================================

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


     # =========================================================================
    # スライダー配置
    # 全体を下へ移動し、右側にボタン用スペースを確保
    # =========================================================================

    ax_route = fig.add_axes([
        0.20, 0.220, 0.48, 0.018
    ])

    ax_elev = fig.add_axes([
        0.20, 0.190, 0.48, 0.018
    ])

    ax_azim = fig.add_axes([
        0.20, 0.160, 0.48, 0.018
    ])

    ax_zoom = fig.add_axes([
        0.20, 0.130, 0.48, 0.018
    ])

    ax_alpha_sphere = fig.add_axes([
        0.20, 0.100, 0.48, 0.018
    ])

    ax_alpha_cone = fig.add_axes([
        0.20, 0.070, 0.48, 0.018
    ])

    ax_alpha_spiral = fig.add_axes([
        0.20, 0.040, 0.48, 0.018
    ])

    # =========================================================================
    # Slider
    # =========================================================================

    s_route = Slider(
        ax_route,
        'routeAB',
        0.0,
        8.0,
        valinit=0.0,
        valstep=0.01
    )

    s_elev = Slider(
        ax_elev,
        'Elev',
        -90,
        90,
        valinit=28,
        valstep=1
    )

    s_azim = Slider(
        ax_azim,
        'Azim',
        0,
        360,
        valinit=305,
        valstep=1
    )

    s_zoom = Slider(
        ax_zoom,
        'Zoom',
        0.6,
        2.0,
        valinit=1.2,
        valstep=0.01
    )

    s_alpha_sphere = Slider(
        ax_alpha_sphere,
        'Sphere α',
        0.0,
        1.0,
        valinit=0.35,
        valstep=0.01
    )

    s_alpha_cone = Slider(
        ax_alpha_cone,
        'Cone α',
        0.0,
        1.0,
        valinit=0.35,
        valstep=0.01
    )

    s_alpha_spiral = Slider(
        ax_alpha_spiral,
        'Spiral α',
        0.0,
        1.0,
        valinit=1.0,
        valstep=0.01
    )

    # =========================================================================
    # 表示 ON / OFF ボタン
    # =========================================================================

    # 初期状態
    reasoning_visible = {'state': False}

    unused_visible = {
        1: False,
        2: False,
        3: False,
        4: False
    }

    # -------------------------------------------------------------------------
    # Reasoning
    # -------------------------------------------------------------------------

        # =========================================================================
    # 表示 ON / OFF ボタン
    # =========================================================================

    reasoning_visible = {'state': False}

    unused_visible = {
        1: False,
        2: False,
        3: False,
        4: False
    }

    # =========================================================================
    # 表示 ON / OFF ボタン
    # =========================================================================

    reasoning_visible = {'state': False}

    unused_visible = {
        1: False,
        2: False,
        3: False,
        4: False
    }


    # -------------------------------------------------------------------------
    # Reasoning
    # -------------------------------------------------------------------------

    ax_reason_button = fig.add_axes([
        0.78, 0.200, 0.11, 0.030
    ])

    reason_button = Button(
        ax_reason_button,
        'Reason OFF'
    )

    for artist in reasoning_artists:
        artist.set_visible(False)

    def toggle_reasoning(event):

        reasoning_visible['state'] = (
            not reasoning_visible['state']
        )

        visible = reasoning_visible['state']

        for artist in reasoning_artists:
            artist.set_visible(visible)

        reason_button.label.set_text(
            'Reason ON'
            if visible
            else 'Reason OFF'
        )

        fig.canvas.draw_idle()

    reason_button.on_clicked(
        toggle_reasoning
    )


    # -------------------------------------------------------------------------
    # Full Spiral C1～C4
    # -------------------------------------------------------------------------

    unused_buttons = {}

    def make_unused_toggle(which):

        def toggle(event):

            unused_visible[which] = (
                not unused_visible[which]
            )

            visible = unused_visible[which]

            for artist in full_spiral_artists[which]:
                artist.set_visible(visible)

            unused_buttons[which].label.set_text(
                f'C{which} ON'
                if visible
                else f'C{which} OFF'
            )

            fig.canvas.draw_idle()

        return toggle


    button_y = {
        1: 0.160,
        2: 0.125,
        3: 0.090,
        4: 0.055
    }

    for which in range(1, 5):

        ax_button_unused = fig.add_axes([
            0.78,
            button_y[which],
            0.11,
            0.028
        ])

        button = Button(
            ax_button_unused,
            f'C{which} OFF'
        )

        unused_buttons[which] = button

        for artist in full_spiral_artists[which]:
            artist.set_visible(False)

        button.on_clicked(
            make_unused_toggle(which)
        )
    # =========================================================================
    # scatter位置更新
    # =========================================================================

    def set_scatter_xyz(
        artist,
        p
    ):

        if p is None:

            artist._offsets3d = (
                [],
                [],
                []
            )

        else:

            artist._offsets3d = (
                [float(p[0])],
                [float(p[1])],
                [float(p[2])]
            )


    # =========================================================================
    # 視点
    # =========================================================================

    def apply_view():

        ax.view_init(
            elev=s_elev.val,
            azim=s_azim.val
        )

        # 軸の範囲は固定
        lim = 1.65
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim*0.67, lim*0.67)

        z = s_zoom.val

        # 全体の大きさ(倍率) ← 「寄ってる感」の主成分
        ax.set_box_aspect((1, 1, 0.72), zoom=z)

        # 近さの遠近感 ← zoomが大きいほど焦点距離を短くして深み強調
        ax.set_proj_type('persp', focal_length=1.0 / z)


    # =========================================================================
    # 透明度
    # =========================================================================

    def apply_alpha():

        # 球
        sphere_surf.set_alpha(
            s_alpha_sphere.val
        )

        # 円錐面
        for s in cone_surfs:

            s.set_alpha(
                s_alpha_cone.val
            )

        # 螺旋
        for artist in spiral_artists:

            artist.set_alpha(
                s_alpha_spiral.val
            )

     # =========================================================================
    # routeAB更新
    # =========================================================================

    def update_route(val):

        r = float(s_route.val)

        pa = a_out_position(r)
        pb = b_in_position(r)

        set_scatter_xyz(a_artist, pa)
        set_scatter_xyz(b_artist, pb)

        if r < 4:
            status.set_text(
                f'inward: P4 → O   routeAB={r:.2f}'
            )

        elif np.isclose(r, 4.0):
            status.set_text(
                'O: 同一螺旋上で移動方向を反転'
            )

        else:
            status.set_text(
                f'outward: O → P4   routeAB={r:.2f}'
            )

        fig.canvas.draw_idle()


    # =========================================================================
    # 視点更新
    # =========================================================================

    def update_view(val):
        apply_view()
        fig.canvas.draw_idle()


    # =========================================================================
    # 透明度更新
    # =========================================================================

    def update_alpha(val):
        apply_alpha()
        fig.canvas.draw_idle()


    # =========================================================================
    # イベント登録
    # =========================================================================

    s_route.on_changed(update_route)
    s_elev.on_changed(update_view)
    s_azim.on_changed(update_view)
    s_zoom.on_changed(update_view)

    s_alpha_sphere.on_changed(update_alpha)
    s_alpha_cone.on_changed(update_alpha)
    s_alpha_spiral.on_changed(update_alpha)

    # =========================================================================
    # 初期状態
    # =========================================================================

    apply_view()
    apply_alpha()
    update_route(0.0)


    # =========================================================================
    # routeAB アニメーション
    # =========================================================================

    animation_speed = 0.006

    def animate(_frame):

        r = float(s_route.val) + animation_speed

        if r > 8.0:
            r = 0.0

        s_route.set_val(r)

    ax_button = fig.add_axes([0.02, 0.93, 0.10, 0.04])
    play_button = Button(ax_button, 'Pause')

    is_playing = {'state': True}

    def toggle_play(event):
        if is_playing['state']:
            ani.event_source.stop()
            play_button.label.set_text('Play')
        else:
            ani.event_source.start()
            play_button.label.set_text('Pause')
        is_playing['state'] = not is_playing['state']

    play_button.on_clicked(toggle_play)

    ani = FuncAnimation(
        fig,
        animate,
        interval=16,
        cache_frame_data=False
    )

    plt.show()


if __name__ == '__main__':
    main()