"""
黑白漫画滤镜 —— 照片 → 线稿风

    内容图 → whiten 提亮压平（底色）→ 可选网点
    内容图 → Canny / 自适应阈值 → 墨线
            ↓ 正片叠底
    结果（灰阶线稿风）

不需要神经网络，不需要风格图。纯图像处理。

===== 为什么这里没有风格迁移 =====

早期版本用了完整的 VGG19 Gram 风格迁移（300 轮 LBFGS 优化），实测它对成品
【没有可测量的贡献】，所以整个环节已被移除。证据：

1. 换 5 种风格源（含纯噪声、纯灰块两个对照组）跑同一内容图：
   真实风格之间差 3.12，真实 vs 对照组差 3.07，比值 1.02
   换个风格图只差 3/255 约 1.2%，和换成纯噪声图效果相当

2. 直接把风格损失关掉（style_weight = 0）：
   关掉风格 vs 完全不优化    差 0.03    <- 没有区别
   完整流程 vs 完全不优化    差 5.08
   完整流程 vs 关掉风格      差 5.11

3. 试过 6 种参数组合（放松白化 / 换深层内容层 / 降内容权重 / 不做后处理），
   判定指标"真风格差异 除以 噪声差异"最好只有 1.19，多数在 1.0 以下

原因：风格迁移输出的差异（离内容图 46.78）会被 whiten 压平成一整片留白，
而墨线 100% 来自内容图。所以成品 = 内容图 + 边缘检测 + 一条色调曲线。

移除后：耗时从约 7 秒降到 0.1 秒以内；成品差异 2.2%（肉眼不可辨），
且底色比带优化的版本更干净——优化残留的竖向涂抹痕迹一并消失。

想要真正的风格迁移，用同项目的另一个仓库 neural_transfer 里的
style_transfer_general.py（那里换风格图差 68.9，是货真价实的迁移）。

用法：
  1) 图片放进 assets/，或设 CONTENT_IMAGE 环境变量
  2) python manga_filter.py
  3) 结果在 <内容图名><run_tag>/ 下：结果.png / 对比.png / 输入.png
"""

import os
import time

import numpy as np
from PIL import Image, ImageChops, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    import cv2                      # 提墨线用（Canny / 自适应阈值）
except ImportError:                 # 没装就退回 PIL 的边缘滤波
    cv2 = None

# 对比图有中文标题，matplotlib 默认字体不含 CJK
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

start_time = time.time()

# ============================================================================
# 配置
# ============================================================================
# 路径约定（便于分享/上 GitHub，全用相对路径）：
#   相对路径 -> 以本脚本所在目录为基准；填 "assets/xxx.jpg" 即可
#   绝对路径 -> 原样使用
#   也可用环境变量覆盖：CONTENT_IMAGE
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = _HERE


def resolve_path(p):
    p = os.path.expanduser(os.path.expandvars(p))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_HERE, p))


CONTENT_PATH = resolve_path(os.environ.get("CONTENT_IMAGE", "assets/content.jpg"))

# 输出尺寸。None = 保持原图分辨率；设成整数（如 1024）会等比缩放到该边长内
out_size = None

# ---- 输出色彩模式 ----------------------------------------------------------
#   "gray"    = 灰阶（默认，线稿风）
#   "content" = 保留原图颜色，只把墨线压上去（彩色线稿）
color_mode = "gray"

# ---- 墨线提取 --------------------------------------------------------------
#   "adaptive" = 自适应阈值（默认）。密度阈值法，会把平滑渐变整片转成网点，
#                线条密集、自带网点质感（实测黑像素占比约 45%）。照片类内容用这个。
#   "canny"    = 只抓真实边缘，线条稀疏干净（实测黑像素占比约 2.9%）。
#                想要"少而准"的线稿、或者内容本身就是线稿时用。
line_method = "adaptive"
line_median = 5          # canny 前的中值滤波核（奇数）；0 = 不滤波，越大越干净
line_lo = 60             # Canny 低阈值
line_hi = 160            # Canny 高阈值
line_dilate = 1          # 线条加粗程度，0 = 不加粗
line_thickness = None    # 想按图宽自动定线宽就填比例，如 0.0015；None = 用 line_dilate
# adaptive 模式的参数（line_method="adaptive" 时才用）
line_block = 15
line_c = 8
line_alpha = 1.0         # 墨线强度 0~1

# ---- 线条连续性：闭运算接断口 ----------------------------------------------
# 阈值会把弱笔画切断，线就碎成一段段。闭运算（先膨胀再腐蚀）能补上小间隙，
# 而且几乎不改变线条粗细。
# 实测（同一张图）：
#   关闭      连通块 6906  线条占比 37.32%
#   line_close=3   2098               40.47%   <- 块数降到 1/3.3，密度只涨 3%
#   line_close=5    723               46.89%
#   line_close=7    262               56.31%   <- 开始把纹理并成大块
line_close = 3           # 闭运算核大小（奇数）；0 = 关闭。推荐 3，越大接得越多也越糊
# 接断口之前先去掉小于这个面积的孤立黑点（碎点接上去只会更乱）；0 = 不去
line_despeckle = 0       # 推荐 15~25；0 = 不去碎点

# ---- 底色：提亮 + 压平对比 + 抬暗部下限 -------------------------------------
# 漫画底色 = 大面积留白，所以要把照片的明暗压成几个淡灰级别。
#
# 【暗部丢细节】就是这个环节造成的：暗部会被抬到接近白，看起来"空了"。
# 实测（夜间插画，暗区占 71.7%，暗区<80 均值）：
#   gamma2.6 gain1.35 flatten0.55 floor0.72   暗区均值 157.6   <- 太亮，暗部发白
#   gamma1.6 gain1.15 flatten0.30 floor0.50   暗区均值 122.8   <- 现默认，暗部有层次
#   gamma1.3 gain1.05 flatten0.15 floor0.35   暗区均值 107.7   <- 更实但偏灰
#   完全不做 whiten                            暗区均值  35.2   <- 太黑，画面闷
# 换一个方向（逆着调得太亮）就把暗部压回来；但暗部局部对比会略降约 20%。
# 三张不同类型的图都验证过，规律一致。
base_whiten = True
whiten_gamma = 1.6       # >1 提亮中间调，越大越白（越大暗部越空）
whiten_gain = 1.15       # 整体增亮系数
base_flatten = 0.30      # 0~1，把对比度压向平均灰。越大越平、暗部越白
base_floor = 0.50        # 最暗不少于这个亮度（0~1）。越大暗部越白

# ---- 网点（screentone）：漫画灰调靠网点纸，不是连续灰阶 ----------------------
use_screentone = False
tone_size = 6            # 网点格子边长（像素）。越小球越密，4~8 比较像漫画
tone_angle = 45          # 网点排列角度，45° 是印刷惯例，摩尔纹最少
tone_strength = 0.75     # 网点与灰底混合比例，1 = 全靠网点

# run_tag 会拼进输出文件夹名，便于同一张图跑多组参数
run_tag = ""


# ============================================================================
# 图像处理
# ============================================================================
def load_content(path, size=None):
    """读图。size 不为 None 时等比缩放到长边不超过 size。返回 PIL 图。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            s = size / max(im.size)
            if s < 1:
                im = im.resize((max(1, round(im.width * s)),
                                max(1, round(im.height * s))), Image.LANCZOS)
    return im


def extract_lineart(pil_gray, method="canny", median=5, lo=60, hi=160,
                    dilate=1, block=15, c=8):
    """提墨线，返回 L 模式图：白底黑线。

    canny    —— 只抓真实边缘。照片里的平滑渐变（海面、天空）完全不受影响。
    adaptive —— 自适应阈值。比的是"比邻域暗多少"，会把水面渐变整片转成网点，
                只适合本身就是线稿的输入。
    """
    from PIL import ImageFilter
    if cv2 is None:
        smooth = pil_gray.filter(ImageFilter.MedianFilter(max(3, median or 3)))
        out = ImageOps.invert(smooth.filter(ImageFilter.FIND_EDGES)).point(
            lambda v: 0 if v < 235 else 255)
        return out.filter(ImageFilter.MinFilter(3)) if dilate else out

    arr = np.asarray(pil_gray, dtype=np.uint8)
    if method == "adaptive":
        b = max(3, block | 1)
        b = min(b, (min(arr.shape) // 2) * 2 + 1)
        lines = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, b, c)
        out = Image.fromarray(lines, "L")
    else:
        src = arr
        if median and median >= 3:
            src = cv2.medianBlur(arr, int(median) | 1)   # 核必须是正奇数
        edges = cv2.Canny(src, int(lo), int(hi), L2gradient=True)
        out = Image.fromarray(255 - edges, "L")          # 边缘=黑线，底=白

    if dilate:
        out = out.filter(ImageFilter.MinFilter(2 * int(dilate) + 1))
    return out


def despeckle_lines(lines, min_area=15):
    """去掉小于 min_area 的孤立黑点（水面碎噪点）。这些点接上去只会让画面更乱。"""
    if min_area <= 0:
        return lines
    inv = (np.asarray(lines) < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n <= 1:
        return lines
    keep = np.zeros_like(inv)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return Image.fromarray(np.where(keep, 0, 255).astype(np.uint8), "L")


def close_line_gaps(lines, ksize=3):
    """闭运算接断口：先膨胀再腐蚀，补上小间隙而几乎不加粗线条。

    极性注意：这里的线是黑色的（0），底色是白的（255），
    形态学要在"线=255"的图上做，所以先反相再还原。
    """
    if not ksize or ksize < 3:
        return lines
    arr = 255 - np.asarray(lines, dtype=np.uint8)          # 线 -> 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize | 1, ksize | 1))
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, ker)
    return Image.fromarray(255 - arr, "L")                 # 还原成 白底黑线


def whiten(pil_gray, gamma=2.6, gain=1.35, flatten=0.55, floor=0.72):
    """提亮 + 压平对比 + 抬暗部下限，把照片变成漫画那种干净底色。"""
    arr = np.asarray(pil_gray, dtype=np.float32) / 255.0
    arr = np.clip(arr * gain, 0, 1) ** (1.0 / gamma)
    if flatten > 0:
        arr = arr * (1.0 - flatten) + arr.mean() * flatten
    if floor > 0:
        arr = floor + arr * (1.0 - floor)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"), mode="L")


def screentone(pil_gray, size=6, angle=45.0, strength=0.75):
    """把连续灰阶转成漫画网点：规则排布的圆点，点径随亮度变化。

    关键：必须先把色调拉满。whiten 之后图会变得很平，直接算的话点径几乎恒为 0，
    网点根本不出现（实测黑像素恒在 4.6%）。
    """
    arr = np.asarray(pil_gray, dtype=np.float32) / 255.0
    h, w = arr.shape
    size = max(2, int(size))
    size = size + 1 if size % 2 else size

    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo > 1e-3:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = np.deg2rad(angle)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)

    du = np.abs((u % size) - size / 2.0) / (size / 2.0)
    dv = np.abs((v % size) - size / 2.0) / (size / 2.0)
    dist = np.clip(np.sqrt(du ** 2 + dv ** 2), 0, 1)

    r = np.sqrt(np.clip(1.0 - arr, 0, 1)) * 0.85
    toned = 1.0 - (dist <= r).astype(np.float32)

    out = toned if strength >= 1.0 else toned * strength + arr * (1.0 - strength)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype("uint8"), mode="L")


def apply_color_mode(comic_pil, content_rgb, mode="gray"):
    """给线稿结果上色。

    gray    : 灰阶（默认）
    content : 底色保留原图颜色，墨线作为压暗层叠上去
    """
    if mode != "content":
        return comic_pil.convert("L").convert("RGB")
    if content_rgb.size != comic_pil.size:
        content_rgb = content_rgb.resize(comic_pil.size, Image.LANCZOS)
    return ImageChops.multiply(content_rgb.convert("RGB"),
                               comic_pil.convert("L").convert("RGB"))


# ============================================================================
def main():
    if not os.path.exists(CONTENT_PATH):
        raise SystemExit(
            "找不到内容图：{}\n"
            "把图片放到 assets/content.jpg，或用环境变量 CONTENT_IMAGE 指定路径。".format(CONTENT_PATH))

    content_rgb = load_content(CONTENT_PATH, out_size)
    print("输入图    : {}  {}".format(CONTENT_PATH, content_rgb.size))

    # 墨线：从内容图提
    dilate = line_dilate
    if line_thickness:
        dilate = max(0, int(round(content_rgb.width * line_thickness)) - 1)
    lines = extract_lineart(content_rgb.convert("L"), method=line_method,
                            median=line_median, lo=line_lo, hi=line_hi,
                            dilate=dilate, block=line_block, c=line_c)

    # 线条连续性增强：先去掉孤立碎点，再用闭运算把断口接起来
    raw_ratio = (np.asarray(lines) < 128).mean() * 100
    if line_despeckle:
        lines = despeckle_lines(lines, line_despeckle)
    if line_close:
        if cv2 is None:
            print("[warn] 没装 opencv，无法做线条连续性增强（line_close 被忽略）")
        else:
            lines = close_line_gaps(lines, line_close)
    dark_ratio = (np.asarray(lines) < 128).mean() * 100
    print("墨线      : {} 方式，黑像素占比 {:.2f}% -> 连续性增强后 {:.2f}%"
          .format(line_method, raw_ratio, dark_ratio))

    # 底色：直接用内容图的明暗，压平提亮
    base = content_rgb.convert("L")
    if base_whiten:
        base = whiten(base, gamma=whiten_gamma, gain=whiten_gain,
                      flatten=base_flatten, floor=base_floor)
    if use_screentone:
        base = screentone(base, size=tone_size, angle=tone_angle,
                          strength=tone_strength)

    # 正片叠底：底 × 线
    flat = base.convert("RGB")
    if line_alpha >= 1.0:
        comic = ImageChops.multiply(flat, lines.convert("RGB"))
    else:
        comic = Image.blend(
            flat, ImageChops.multiply(flat, lines.convert("RGB")), line_alpha)

    result = apply_color_mode(comic, content_rgb, color_mode)

    # 输出：一个文件夹三张图
    stem = os.path.splitext(os.path.basename(CONTENT_PATH))[0]
    out_dir = os.path.join(OUT_DIR, "{}{}".format(stem, run_tag))
    os.makedirs(out_dir, exist_ok=True)

    result_path = os.path.join(out_dir, "结果.png")
    compare_path = os.path.join(out_dir, "对比.png")
    input_path = os.path.join(out_dir, "输入.png")
    result.save(result_path)
    content_rgb.save(input_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (im, title) in zip(axes, [(content_rgb, "输入"),
                                      (lines.convert("RGB"), "墨线"),
                                      (result, "结果")]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(compare_path, dpi=110)
    plt.close(fig)

    print("输出文件夹: {}".format(out_dir))
    print("  结果.png  {}".format(result.size))
    print("  对比.png  输入 / 墨线 / 结果")
    print("  输入.png  {}".format(content_rgb.size))
    print("运行时间：{:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
