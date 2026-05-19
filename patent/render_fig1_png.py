from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("fig1_hybrid_mu_mimo_bicm_flow.png")
W, H = 2400, 3400
S = 2


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(66, True)
FONT_BOX_TITLE = font(48, True)
FONT_TEXT = font(42)
FONT_SMALL = font(38)


def scaled_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(int(v * S) for v in box)


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    fnt: ImageFont.FreeTypeFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), value, font=fnt)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), value, fill="black", font=fnt)


def rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], dashed: bool = False) -> None:
    if not dashed:
        draw.rectangle(box, outline="black", width=5)
        return
    x1, y1, x2, y2 = box
    dash, gap = 20, 14
    for x in range(x1, x2, dash + gap):
        draw.line((x, y1, min(x + dash, x2), y1), fill="black", width=5)
        draw.line((x, y2, min(x + dash, x2), y2), fill="black", width=5)
    for y in range(y1, y2, dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2)), fill="black", width=5)
        draw.line((x2, y, x2, min(y + dash, y2)), fill="black", width=5)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill="black", width=5)
    if abs(x2 - x1) < abs(y2 - y1):
        sign = 1 if y2 >= y1 else -1
        points = [(x2, y2), (x2 - 14, y2 - 24 * sign), (x2 + 14, y2 - 24 * sign)]
    else:
        sign = 1 if x2 >= x1 else -1
        points = [(x2, y2), (x2 - 24 * sign, y2 - 14), (x2 - 24 * sign, y2 + 14)]
    draw.polygon(points, fill="black")


def main() -> None:
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)

    text_center(draw, (1200, 90), "图1  多用户 MIMO-BICM 混合预编码选择方法流程图", FONT_TITLE)

    blocks = [
        ((520, 220, 1880, 404), ["S1 获取系统参数与多用户信道", "H₁,...,H_K；N_t、N_r、N_RF、d、M、SNR"]),
        ((420, 504, 1980, 774), ["S2 发射端 CSI 获取", "由离线信道样本估计均值与全协方差矩阵", "结合重复单位矩阵导频执行全协方差 LMMSE 估计"]),
        ((420, 874, 1980, 1114), ["S3 构建共享模拟预编码矩阵 F_RF", "对各用户 CSI 做 SVD，提取右奇异向量相位", "形成满足恒模约束的共享 RF 预编码"]),
        ((420, 1214, 1980, 1424), ["S4 计算各用户等效信道", "G_k = H_k F_RF，k = 1,...,K"]),
        ((420, 1524, 1980, 1780), ["S5 构建块对角化零空间与降维等效信道", "堆叠其他用户等效信道，求零空间基 N_k", "得到降维等效信道 G̃_k = G_k N_k"]),
        ((160, 1880, 2240, 2320), ["S6 生成结构化混合预编码候选"]),
        ((420, 2420, 1980, 2720), ["S7 接收端性能评价", "线性候选：并行检测或 SIC 下的 BICM-GMI / BER", "THP 候选：对角等化、模折叠、周期复制星座 LLR", "不同候选使用相同符号样本与噪声样本"]),
        ((420, 2820, 1980, 3044), ["S8 选择目标混合预编码矩阵", "依据和速率、误码率、泄漏功率或用户公平性排序"]),
        ((520, 3144, 1880, 3304), ["输出目标 F_RF 与 F_BB", "用于多用户 MIMO-BICM 下行传输"]),
    ]

    for box, lines in blocks:
        rect(draw, box)
        if len(lines) == 1:
            text_center(draw, ((box[0] + box[2]) // 2, box[1] + 76), lines[0], FONT_BOX_TITLE)
        else:
            text_center(draw, ((box[0] + box[2]) // 2, box[1] + 70), lines[0], FONT_BOX_TITLE)
            for idx, line in enumerate(lines[1:]):
                text_center(draw, ((box[0] + box[2]) // 2, box[1] + 138 + idx * 64), line, FONT_TEXT if idx < 2 else FONT_SMALL)

    candidate_boxes = [
        ((250, 2030, 670, 2198), ["SVD", "数字预编码"]),
        ((730, 2030, 1150, 2198), ["GMD", "数字预编码"]),
        ((1210, 2030, 1630, 2198), ["UCD", "数字预编码"]),
        ((1690, 2030, 2110, 2198), ["GMD+THP", "模递推候选"]),
    ]
    for box, lines in candidate_boxes:
        rect(draw, box)
        text_center(draw, ((box[0] + box[2]) // 2, box[1] + 70), lines[0], FONT_TEXT)
        text_center(draw, ((box[0] + box[2]) // 2, box[1] + 126), lines[1], FONT_SMALL)
    text_center(draw, (1200, 2276), "拼接各用户数字块并对 F_RF F_BB 执行总功率归一化", FONT_TEXT)

    arrows = [
        ((1200, 404), (1200, 504)),
        ((1200, 774), (1200, 874)),
        ((1200, 1114), (1200, 1214)),
        ((1200, 1424), (1200, 1524)),
        ((1200, 1780), (1200, 1880)),
        ((1200, 2320), (1200, 2420)),
        ((1200, 2720), (1200, 2820)),
        ((1200, 3044), (1200, 3144)),
    ]
    for start, end in arrows:
        arrow(draw, start, end)

    optional = (70, 2420, 330, 2720)
    rect(draw, optional, dashed=True)
    text_center(draw, (200, 2530), "可选", FONT_SMALL)
    text_center(draw, (200, 2590), "联合软优化", FONT_SMALL)
    text_center(draw, (200, 2650), "初始化", FONT_SMALL)
    arrow(draw, (330, 2570), (420, 2570))

    image.save(OUT, dpi=(300, 300))


if __name__ == "__main__":
    main()
