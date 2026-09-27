"""
黑白漫画滤镜 —— 照片 → 线稿风

流程：

    VGG19 Gram 风格迁移（可选环节，默认几乎不影响成品，见下）
        ↓  whiten       提亮 + 压平对比，把底色变成干净留白
        ↓  screentone   可选：45° 规则网点（漫画灰调靠网点纸，不是连续灰阶）
    内容原图 → Canny / 自适应阈值 → 墨线
        ↓  正片叠底
    结果（灰阶线稿风）

为什么墨线不交给风格迁移：Gram 矩阵只统计通道间相关性，没有"边缘"这个概念。
试过各种权重和层组合，它只能产出纹理碎块，画不出连续线条，所以墨线用边缘检测
从内容图直接提取。

===== 重要实测结论：这不是风格迁移，是线稿滤镜 =====

用 5 种风格源（含纯噪声、纯灰块两个对照组）跑同一内容图：

    真实风格图之间   平均差 3.12
    真实 vs 对照组   平均差 3.07      比值 1.02

换个风格图只差 3/255 ≈ 1.2%，而且和"换成纯噪声图"效果相当。
更硬的证据是把风格损失完全关掉（style_weight=0）：

    关掉风格损失 vs 完全不优化    平均差 0.03   <- 没有区别
    完整流程       vs 完全不优化  平均差 5.08
    完整流程       vs 关掉风格    平均差 5.11

也就是说：整个 VGG19 + Gram + 300 轮 LBFGS 优化环节对最终成品【没有可测量的贡献】。
风格损失确实被压掉了 99.2%（优化在工作），但那点差异全被 whiten 抹平了，
成品基本只由内容图决定：

    成品 ≈ 内容图 + 边缘检测 + 一条色调曲线

所以本脚本定位为【线稿滤镜】。想要真正的风格迁移（换风格图能看出效果），
用隔壁的 neural_transfer/style_transfer_general.py —— 那里换风格图差 68.9。

用法：
  1) 图片放进 assets/（或设 CONTENT_IMAGE / STYLE_IMAGE 环境变量）
  2) python manga_filter.py
  3) 结果在 <内容图名><run_tag>/ 下：结果.png / 对比.png / 输入.png
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageStat
import matplotlib
matplotlib.use("Agg")  # 无窗口后端，避免 plt.show() 阻塞/闪退
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 对比图上有中文标题，matplotlib 默认的 DejaVu Sans 不含 CJK 字形，
# 不设字体的话中文会变成一串方框并刷一屏 UserWarning。
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

try:
    import cv2                      # 提墨线用（自适应阈值）
except ImportError:                 # 没装就退回 PIL 的边缘滤波
    cv2 = None

start_time = time.time()
# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
# 路径约定（为便于分享/上 GitHub，全用相对路径）：
#   相对路径 -> 以本脚本所在目录为基准；填 "assets/xxx.jpg" 即可
#   绝对路径 -> 原样使用，方便本地指到别处
#   也可以用环境变量覆盖：STYLE_IMAGE / CONTENT_IMAGE
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = _HERE


def resolve_path(p):
    p = os.path.expanduser(os.path.expandvars(p))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_HERE, p))


STYLE_PATH = resolve_path(os.environ.get("STYLE_IMAGE", "assets/style.jpg"))
CONTENT_PATH = resolve_path(os.environ.get("CONTENT_IMAGE", r"assets/content.jpg"))

# 送进 VGG19 的方形尺寸；None = 保持内容图原始分辨率（显存/时间开销大很多）
# 笔触尺度 ≈ net_size / 64（conv5 特征图边长）。512 时每个笔触≈8px，显得粗钝；
# 256 时≈4px，线条细一半，更接近漫画的墨线。想更细继续降，代价是细节变少。
net_size = 256
# 最终输出尺寸；None = 保持网络尺寸输出
final_size = "original"

# ---- 风格图取景：把分格边框/对话框排除掉 --------------------------------
# 漫画页上的粗黑分格线会被 Gram 学成"风格"，再抹到内容图上，表现为
# 水面上那些竖向涂抹条带。中值滤波实测无效（Gram 无差别掉 60%，墨量却没变），
# 所以这里改成直接从风格图里裁一块干净的纹理区。
# 值 = (左, 上, 右, 下)，用 0~1 的【比例】表示；None = 不裁
style_crop = None        # 例：(0.05, 0.05, 0.95, 0.60) 取上面 55% 的区域

# ---- 自动抹掉风格图里的分格边框 / 对话框边框 -----------------------------
# 这些边框是又长又直的细黑线，Gram 会把它们当风格学走。
# 中值滤波实测无效（无差别削弱风格 60%），所以改成：检测长直线 -> 用周围纹理修补。
# 换风格图不用手调，这就是给"跑其他图片"用的。
style_delayout = True
delayout_min_len = 0.25  # 直线最短长度（占画面比例），0.25 = 超过 1/4 边长才算边框
delayout_thick = 5       # 最大线宽（像素，按 net_size=256 计），防止把粗笔画也吃掉
delayout_dilate = 2      # 检测结果外扩像素，保证把抗锯齿的边一并盖住

# 训练轮次 = LBFGS 的 step 次数。注意 closure 内部会自增 run 计数，
# 而一次 step 内部要调 closure 5~20 次（线搜索），所以真实前向次数 ≈ num_steps 的 5~20 倍。
num_steps = 300 if torch.cuda.is_available() else 60
style_weight = 1e5
content_weight = 1

# TV 正则权重：压制那些"沟壑噪点"、逼出平涂色块，是漫画感的关键。
# 太大会糊成一片，太小压不住噪点。纯纹理风格可以调低。
tv_weight = 1e-2

# ---- 输出色彩模式 ----------------------------------------------------------
# 漫画版固定用 gray：风格图是黑白漫画时 Gram 损失本来就把颜色推向灰阶，
# 留着一层半死不活的彩反而更脏，不如干脆干净的灰阶。
#   "gray"    = 强制灰阶        <- 漫画版用这个
#   "color"   = 网络输出原样的颜色（偏灰）
#   "chroma"  = 亮度用漫画化结果、色度取内容原图
#   "content" = 底色用内容原图颜色，墨线只作叠加层（要彩色漫画才用）
color_mode = "gray"

# 彩色模式的曝光：YCbCr 里彩色色度偏离 128，Y 太高会让 RGB 通道溢出截断成白。
# 白化后中间调能到 200+，颜色会被冲淡（实测头发从深蓝 (29,112,163) 变成近白 (202,206,229)）。
# 1.0 = 不动，推荐 0.75~0.9
chroma_exposure = 0.85

# 色度增益：PIL 的 YCbCr 往返会把色度压掉约一半（实测 Cb 的 std 13.3 -> 7.2）。
# 这个系数把色度偏离 128 的量放大回来。1.0 = 不放大，推荐 1.5~2.2
chroma_saturation = 1.8

# ---- 漫画化：轮廓叠加 ------------------------------------------------------
# Gram 矩阵只能匹配纹理统计，画不出「墨线」。所以线条不指望风格迁移，
# 而是从原图直接提边缘，再压到风格化结果上。
line_overlay = True
line_method = "adaptive"   # "adaptive" = 密度阈值（复现 output_manga_lines 那版）
# canny 参数：先中值滤波压掉水面细纹，再做边缘检测
line_median = 5          # 中值滤波核（奇数）；0 = 不滤波。越大越干净
line_lo = 60             # Canny 低阈值
line_hi = 160            # Canny 高阈值
line_dilate = 1          # 线条加粗程度，0 = 不加粗
line_thickness = None    # 想按图宽自动定线宽就填比例，如 0.0015；None = 用 line_dilate
# adaptive 模式的参数（line_method="adaptive" 时才用）
line_block = 15
line_c = 8
line_alpha = 1.0         # 墨线强度 0~1

# 风格化结果通常又脏又暗，先提亮成高调，漫画底色才干净。
# 漫画是大面积留白 + 少量墨线，所以这里要压得狠一点：
# 先把中间调往白里推，再压低残留对比，让底层接近"淡灰平涂"。
base_whiten = True
whiten_gamma = 2.6       # >1 提亮中间调，越大越白
whiten_gain = 1.35       # 整体增亮系数
base_flatten = 0.55      # 0~1，把底层对比度压向平均灰。越小越平（0 = 变成纯色块）
base_floor = 0.72        # 底层最暗不少于这个亮度（0~1），保证不会黑得压死线条

# ---- 网点（screentone）：漫画灰调靠网点纸，不是连续灰阶 --------------------
# 真实网点 = 规则排布的圆点，点的直径随亮度变化。
# 比"密度阈值的碎噪点"干净得多，而且一眼就像印刷品。
use_screentone = False   # lines 那版的"网点"是自适应阈值自己产生的，不用再叠
tone_size = 6            # 网点格子边长（像素）。越小球越密，4~8 比较像漫画
tone_angle = 45          # 网点排列角度，45° 是印刷惯例，摩尔纹最少
tone_strength = 0.75     # 网点与灰底混合比例，1 = 全靠网点

# 每多少轮存一张中间快照到输出目录/snapshots/；0 = 不存。
# 实测本流程各步之间几乎没有差别（成品基本由内容图决定），快照只是白占空间
# （12 张约 37 MB），所以默认关闭。
snapshot_every = 0

# 本轮输出的文件名后缀，避免覆盖以前的成果；留空则用原名
run_tag = "_lines_v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 内容图原始分辨率，to_pil 还原时要用（main 里赋值）
content_original_size = None
# 内容图缩放到正方形后四周的 padding，出图时要裁掉（main 里赋值）
content_crop_box = None


# ----------------------------------------------------------------------------
# 载入与还原
# ----------------------------------------------------------------------------
def image_loader(image_name, size, crop=None, delayout=False):
    """读图 -> 等比缩放到 size 内 -> pad 成方形。返回 (张量, 原始尺寸, 裁剪框)。

    不用 Resize((size,size)) 硬拉方形：那样 4:3 的图会被压扁，出图再拉回去
    变形就是永久的（上一版就踩了这个坑）。这里改成等比缩放 + 黑边补齐。
    """
    with Image.open(image_name) as im:
        im = im.convert("RGB")
        original_size = im.size  # (W, H)
        crop_box = None
        if crop is not None:
            cw, chh = im.size
            left, top, right, bottom = crop
            im = im.crop((int(cw * left), int(chh * top),
                          int(cw * right), int(chh * bottom)))
        if size is not None:
            # 等比缩放，让【长边】正好 = size（不用 Resize(size)：那是缩短边，
            # 长边会超出去；也不用 Resize(size, max_size=size)：torchvision 要求
            # max_size 严格大于 size，会直接报错）
            scale = size / max(im.width, im.height)
            im = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.LANCZOS)
            pad_w = size - im.width
            pad_h = size - im.height
            assert pad_w >= 0 and pad_h >= 0, \
                "Resize 后尺寸 {} 超过了 size={}".format(im.size, size)
            padding = (pad_w // 2, pad_h // 2,
                       pad_w - pad_w // 2, pad_h - pad_h // 2)
            im = transforms.functional.pad(im, padding, fill=0)
            # 出图时裁回内容图的真实比例：pad 后是 size×size，
            # 有效内容区在左上/居中，尺寸 = size - pad
            crop_box = (padding[0], padding[1],
                        size - padding[2],
                        size - padding[3])
        if delayout:
            # 抹掉分格边框，否则 Gram 会把它们当风格抹到内容图上（水面竖向条带）
            im = remove_panel_borders(im, min_len=delayout_min_len,
                                      thick=delayout_thick, dilate=delayout_dilate)
        tensor = transforms.ToTensor()(im).unsqueeze(0)  # [1,3,H,W]
    return tensor.to(device, torch.float), original_size, crop_box


def remove_panel_borders(pil_rgb, min_len=0.25, thick=5, dilate=2):
    """抹掉漫画页的分格边框/对话框边框：检测又长又直的细黑线，用周围纹理修补。

    实测对比（同一张风格图，net_size=256）：
        中值滤波 5   -> 无差别模糊全图，Gram 掉 36%（风格被洗淡），黑像素几乎不变
        本方法       -> 只改 5.28% 像素（就是检测到的边框），Gram 只掉 4.3%
    所以边框要"精准摘除"，不能靠模糊。
    """
    if cv2 is None:
        return pil_rgb
    arr = np.asarray(pil_rgb.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    dark = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    hk = max(3, int(w * min_len)) | 1        # 横线最小长度
    vk = max(3, int(h * min_len)) | 1        # 竖线最小长度
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    mask = cv2.bitwise_or(horiz, vert)

    # 去掉过粗的块状黑（漫画的平涂不该被当边框吃掉）
    k2 = thick | 1
    thick_block = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2)))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(thick_block))

    if dilate:
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))

    rgb = np.asarray(pil_rgb).copy()
    out = cv2.inpaint(rgb[:, :, ::-1], mask, 4, cv2.INPAINT_TELEA)[:, :, ::-1]
    return Image.fromarray(out)


def apply_color_mode(stylized_pil, color_src_pil, mode="gray",
                     exposure=chroma_exposure, saturation=chroma_saturation,
                     lineart_pil=None):
    """给漫画化结果上色。必须在叠线【之后】调用，且色度源要提前留好副本。

    为什么不能在叠线前调用：叠线要 base = out_pil.convert("L")，
    色度在这一步被永久丢弃，事后再调什么模式都救不回来
    （实测：网络输出饱和度 87.0，叠完线变 0.0，调 color 仍旧 0.0）。

    gray   : 灰阶，三通道趋同（风格图是黑白漫画时的正解）
    color  : 色度取【网络风格化输出】
    chroma : 色度取【内容原图】

    彩色模式有两个坑（都踩过，所以这里用 numpy 精确算）：
      1. 白化后 Y 到了 200+，原始色度直接复用会让 RGB 溢出。
         实测头发 YCbCr (93,167,82) 的深蓝，Y 拉到 209 后
         Cb 167->88、Cr 82->146，蓝色直接翻成暖黄。
      2. 固定倍数放大色度同样会溢出，且是单边截断 —— 那正是色相翻转的来源。
    所以这里按【每个像素的最大可行色度】反推缩放系数，保证 RGB 恒在 [0,255]。
    """
    if mode == "gray" or mode not in ("color", "chroma", "content"):
        return stylized_pil.convert("L").convert("RGB")

    src = color_src_pil.convert("RGB")
    if src.size != stylized_pil.size:
        src = src.resize(stylized_pil.size, Image.LANCZOS)
    rgb = np.asarray(src, dtype=np.float32) / 255.0   # 原图就是彩色的，直接用
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    if mode == "content":
        # 底色保持原图颜色；墨迹只压暗"有线的地方"。
        # ink 要从【墨线层】算，不能从风格化底色算：底色本身偏暗（实测中位 0.42），
        # 拿它当蒙版会把整张图压暗。墨线层是白底黑线，乘上去才是"彩照 + 描边"。
        lum_src = lineart_pil if lineart_pil is not None else stylized_pil
        lum = np.asarray(lum_src.convert("L"), dtype=np.float32) / 255.0
        lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
        ink = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)
        ink = ink ** max(saturation, 1e-3)          # 墨迹压暗强度
        if exposure != 1.0:
            ink = 1.0 - (1.0 - ink) * exposure      # 整体墨迹浓度
        return Image.fromarray(
            (np.clip(np.stack([R, G, B], 2) * ink[:, :, None], 0, 1) * 255
             ).round().astype("uint8"), "RGB")

    # color / chroma：色度取指定来源，亮度换成漫画化结果
    Ysim = np.asarray(stylized_pil.convert("L"), dtype=np.float32) / 255.0
    if exposure != 1.0:
        Ysim = np.clip(Ysim * exposure, 0, 1)
    YCbCr = np.asarray(src.convert("YCbCr"), dtype=np.float32) / 255.0
    Cb = (YCbCr[:, :, 1] - 0.5) * 2.0
    Cr = (YCbCr[:, :, 2] - 0.5) * 2.0
    kc, kg, kb = 1.402, 0.714, 1.772

    # 按每个像素的最大可行色度反推缩放系数，保证 RGB 恒在 [0,255]。
    # 不做这一步的话，高 Y 配原始色度会单边溢出——那正是色相翻转的来源
    # （实测头发 YCbCr (93,167,82) 的深蓝，Y 拉到 209 后 Cb 167->88、Cr 82->146，蓝翻成暖黄）。
    with np.errstate(divide="ignore", invalid="ignore"):
        lim_cb = np.where(Cb > 0, (1.0 - Ysim) / kb, np.where(Cb < 0, Ysim / kb, np.inf))
        lim_cr = np.where(Cr > 0, (1.0 - Ysim) / kc, np.where(Cr < 0, Ysim / kc, np.inf))
        mg = np.abs(kg * Cb) + np.abs(kg * Cr)
        lim_g = np.where((np.abs(Cb) + np.abs(Cr)) > 0,
                         Ysim / np.maximum(mg, 1e-6), np.inf)
    lim = np.minimum(np.minimum(lim_cb, lim_cr), lim_g)
    scale = np.minimum(saturation * lim, 1.0)
    scale = np.nan_to_num(scale, nan=0.0, posinf=1.0)
    Cb = np.clip(Cb * scale, -1.0, 1.0)
    Cr = np.clip(Cr * scale, -1.0, 1.0)

    R = Ysim + kc * Cr
    G = Ysim - 0.344 * Cb - kg * Cr
    B = Ysim + kb * Cb
    return Image.fromarray(
        (np.clip(np.stack([R, G, B], 2), 0, 1) * 255).round().astype("uint8"), "RGB")


def tensor_to_pil(tensor, crop_box=None):
    """把 [1,3,H,W] 的 0~1 张量还原成 PIL 图：先裁掉 padding，再缩回目标尺寸。"""
    with torch.no_grad():
        out = tensor.detach().squeeze(0).clamp(0, 1).cpu()
        pil = transforms.ToPILImage()(out)
        if crop_box is not None:
            pil = pil.crop(tuple(int(v) for v in crop_box))
        if final_size is not None:
            target = (
                content_original_size if final_size == "original"
                else (int(final_size), int(final_size))
            )
            if (pil.width, pil.height) != tuple(target):
                pil = pil.resize(tuple(target), Image.LANCZOS)
    return pil


# ----------------------------------------------------------------------------
# 损失模块（逻辑与原版一致）
# ----------------------------------------------------------------------------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # register_buffer + 延迟搬移：不会因为 set_default_device 造成设备错配
        self.register_buffer("mean", mean.detach().clone().view(-1, 1, 1))
        self.register_buffer("std", std.detach().clone().view(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean.to(img.device)) / self.std.to(img.device)


content_layers_default = ["conv_4"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)  # in-place 会破坏 loss 节点
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 砍掉最后一个 loss 层之后的所有层
    last = 0
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            last = j
            break

    return model[: last + 1], style_losses, content_losses


def total_variation(x):
    """总变差：相邻像素差的绝对值和。惩罚它 = 压制噪点、逼出平涂色块。

    漫画是平涂 + 墨线，靠这个才能从"浮雕沟壑"里挣脱出来。
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() if x.size(2) > 1 else 0
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() if x.size(3) > 1 else 0
    return dh + dw


def extract_lineart(pil_gray, method="canny", median=5, lo=60, hi=160,
                    dilate=1, block=15, c=8):
    """从原图提墨线，返回 L 模式图：白底黑线。

    两种方法：
      canny    —— 只抓真实边缘。照片里的平滑渐变（海面、天空）完全不受影响。
                  实测黑像素占比约 3.8%，干净且保留结构。推荐。
      adaptive —— 自适应阈值。比的是"比邻域暗多少"，所以会把水面渐变
                  整片转成网点（实测占比 49.7%，糊成一团）。只适合本身
                  就是线稿的输入。
    """
    from PIL import ImageFilter
    if cv2 is None:
        # 没装 opencv 的退路：中值滤波 + FIND_EDGES
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
            k = int(median) | 1          # 核必须是正奇数
            src = cv2.medianBlur(arr, k)  # 压掉水面细纹，只留主要轮廓
        edges = cv2.Canny(src, int(lo), int(hi), L2gradient=True)
        out = Image.fromarray(255 - edges, "L")   # 边缘=黑线，底=白

    if dilate:
        out = out.filter(ImageFilter.MinFilter(2 * int(dilate) + 1))
    return out


def whiten(pil_gray, gamma=2.6, gain=1.35, flatten=0.45, floor=0.62):
    """提亮 + 压平对比 + 抬暗部下限，把风格化结果变成漫画那种干净底色。

    漫画底色 = 大面积淡灰/留白，所以三步都要做：
      gain/gamma 把中间调推向白，flatten 压掉残留纹理对比，
      floor 保证最暗处不会黑到把墨线淹掉。
    """
    arr = np.asarray(pil_gray, dtype=np.float32) / 255.0
    arr = np.clip(arr * gain, 0, 1) ** (1.0 / gamma)     # 提亮
    if flatten > 0:
        arr = arr * (1.0 - flatten) + arr.mean() * flatten  # 向平均灰收
    if floor > 0:
        arr = floor + arr * (1.0 - floor)                # 抬暗部下限
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"), mode="L")


def screentone(pil_gray, size=6, angle=45.0, strength=0.75):
    """把连续灰阶转成漫画网点：规则排布的圆点，点径随亮度变化。

    这是"真网点"，比自适应阈值那种碎噪点干净得多，也更像印刷品。
    strength=1 全靠网点；<1 则与灰底混合，保留一点连续灰调。

    关键：必须先把色调拉满。whiten 之后图会变得很平（实测标准差 73.9 -> 11.7），
    直接算的话点径几乎恒为 0，网点根本不出现（实测黑像素恒在 4.6%）。
    """
    arr = np.asarray(pil_gray, dtype=np.float32) / 255.0
    h, w = arr.shape
    size = max(2, int(size))
    size = size + 1 if size % 2 else size        # 取奇数，保证点阵对称

    # --- 色调拉伸：把 2%~98% 分位拉到 0~1，给网点留出跨度 ---
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo > 1e-3:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = np.deg2rad(angle)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)

    # 到最近格心的距离 -> 0(格心) ~ 1(格边)
    du = np.abs((u % size) - size / 2.0) / (size / 2.0)
    dv = np.abs((v % size) - size / 2.0) / (size / 2.0)
    dist = np.sqrt(du ** 2 + dv ** 2)
    dist = np.clip(dist, 0, 1)

    # 越暗 -> 点的半径越大。用椭圆形阈值：格心处最容易变成黑点
    r = np.sqrt(np.clip(1.0 - arr, 0, 1)) * 0.85      # 0 ~ 0.85
    dots = (dist <= r).astype(np.float32)             # 1 = 黑点
    toned = 1.0 - dots                                # 白底黑点

    if strength >= 1.0:
        out = toned
    else:
        out = toned * strength + arr * (1.0 - strength)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype("uint8"), mode="L")


def get_input_optimizer(input_img):
    return optim.LBFGS([input_img])


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=num_steps,
                       style_weight=style_weight, content_weight=content_weight,
                       tv_weight=tv_weight,
                       snapshot_every=snapshot_every, snapshot_dir=None):
    print("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    if snapshot_dir and snapshot_every:
        os.makedirs(snapshot_dir, exist_ok=True)

    print("Optimizing.. target closure calls = {}".format(num_steps))
    run = [0]
    total_loss = [float("inf")]      # 最近一次 closure 的总 loss（含 TV）
    best = {"loss": None, "state": None, "run": 0}
    while run[0] < num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            # TV 只参与优化，不算进打印的 Style/Content，保持日志可比
            tv_score = total_variation(input_img) * tv_weight if tv_weight else None
            loss = style_score + content_score + (tv_score if tv_score is not None else 0)
            loss.backward()
            total_loss[0] = float(loss.detach())

            run[0] += 1
            if run[0] % 50 == 0 or run[0] == num_steps:
                if tv_score is not None:
                    print("run {}/{}  Style {:.4f}  Content {:.4f}  TV {:.4f}".format(
                        run[0], num_steps, style_score.item(),
                        content_score.item(), tv_score.item()), flush=True)
                else:
                    print("run {}/{}  Style {:.4f}  Content {:.4f}".format(
                        run[0], num_steps, style_score.item(),
                        content_score.item()), flush=True)

            # 中间快照：训练越久越需要，方便挑一张满意的而不是只能吃最后一张
            if snapshot_dir and snapshot_every and run[0] % snapshot_every == 0:
                with torch.no_grad():
                    img = tensor_to_pil(input_img, content_crop_box)
                    img.save(os.path.join(
                        snapshot_dir, "step_{:05d}.png".format(run[0])))

            return loss

        optimizer.step(closure)

        # ---- 早停：LBFGS 在这套损失下会间歇性失控（见过 Style 从 14 爆到 1720），
        #      一旦比历史最好值恶化 5 倍以上，就回滚到最好的那一步并停下。
        with torch.no_grad():
            cur = total_loss[0]
            if best["loss"] is None or cur < best["loss"]:
                best["loss"] = float(cur)
                best["state"] = input_img.detach().clone()
                best["run"] = run[0]
            elif run[0] > 20 and cur > best["loss"] * 5:
                print("[早停] run {} 的 loss {:.2f} 比最优 {:.2f}(run {}) 恶化 5 倍以上，"
                      "回滚到最优状态".format(run[0], cur, best["loss"], best["run"]),
                      flush=True)
                input_img.data.copy_(best["state"])
                break

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


# ----------------------------------------------------------------------------
def main():
    global content_original_size, content_crop_box

    start_time = time.time()

    if not torch.cuda.is_available():
        print("[warn] 未检测到 CUDA，回落到 CPU；这一版会明显更慢，把 net_size 调小再跑。")

    style_img, style_original_size, _ = image_loader(
        STYLE_PATH, net_size, crop=style_crop, delayout=style_delayout)
    content_img, content_original_size, content_crop_box = image_loader(CONTENT_PATH, net_size)

    print("style   original={} -> tensor={}".format(style_original_size, tuple(style_img.shape)))
    print("content original={} -> tensor={}  crop_box={}".format(
        content_original_size, tuple(content_img.shape), content_crop_box))
    print("net_size={}  style_weight={:g}  tv_weight={:g}  num_steps={}".format(
        net_size, style_weight, tv_weight, num_steps))

    # 两张图都是「等比缩放 + pad 成方形」，形状必须一致
    if style_img.shape != content_img.shape:
        raise RuntimeError(
            "style/content 形状不一致: {} vs {}".format(
                tuple(style_img.shape), tuple(content_img.shape)))

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    cnn.requires_grad_(False)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    input_img = content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,
                                snapshot_dir=os.path.join(OUT_DIR, "snapshots{}".format(run_tag)))

    out_pil = tensor_to_pil(output, content_crop_box)
    net_pil = out_pil.copy()          # 网络风格化输出（带颜色），color 模式要从它取色度

    # 内容原图（缩放到输出尺寸），后面取色度 / 提墨线都要用
    content_rgb = Image.open(CONTENT_PATH).convert("RGB").resize(
        out_pil.size, Image.LANCZOS)

    # ---- 漫画化：先提亮成高调底色，再把原图墨线压上去 --------------------
    # 注意：这一段全程在【灰阶】空间做（线条、网点、白化都只作用于明暗），
    #      颜色留到最后一步再合回来。否则 base.convert("L") 会丢掉色度，
    #      color_mode 就白设了——上一版就是这么错的。
    lines_pil = None
    if line_overlay:
        # 墨线从【内容原图】提，不是从风格化结果提——风格化已经把结构磨没了。
        # 线宽可以按图像宽度自动定，避免大图线条细得像头发丝
        dilate = line_dilate
        if line_thickness:
            dilate = max(0, int(round(out_pil.width * line_thickness)) - 1)
        lines_pil = extract_lineart(content_rgb.convert("L"),
                                    method=line_method, median=line_median,
                                    lo=line_lo, hi=line_hi, dilate=dilate,
                                    block=line_block, c=line_c)

        base = out_pil.convert("L")
        if base_whiten:
            base = whiten(base, gamma=whiten_gamma, gain=whiten_gain,
                          flatten=base_flatten, floor=base_floor)
        if use_screentone:
            base = screentone(base, size=tone_size, angle=tone_angle,
                              strength=tone_strength)
        base_rgb = base.convert("RGB")
        # 正片叠底：底 * 线，线是黑的就压黑，底保持不变
        if line_alpha >= 1.0:
            out_pil = ImageChops.multiply(base_rgb, lines_pil.convert("RGB"))
        else:
            out_pil = Image.blend(
                base_rgb, ImageChops.multiply(base_rgb, lines_pil.convert("RGB")),
                line_alpha)

    # ---- 色彩：必须放在叠线【之后】 ------------------------------------------
    # 放在叠线之前会被 base.convert("L") 抹掉（实测输出饱和度 0.0，网络明明吐出 87.0）
    before_sat = ImageStat.Stat(out_pil.convert("HSV")).mean[1]
    out_pil = apply_color_mode(out_pil, net_pil, color_mode, lineart_pil=lines_pil)
    after_sat = ImageStat.Stat(out_pil.convert("HSV")).mean[1]
    print("色彩模式  : {}   饱和度 {:.1f} -> {:.1f}".format(
        color_mode, before_sat, after_sat))

    # ---- 输出：一次训练一个文件夹，里面固定三张图 ------------------------------
    #   结果.png   风格化成品
    #   对比.png   风格 / 内容 / 结果 三张并排
    #   输入.png   内容原图（同尺寸，方便逐像素对照）
    # 文件夹名 = 内容图名 + run_tag，所以不同图片自动分到不同文件夹。
    stem = os.path.splitext(os.path.basename(CONTENT_PATH))[0]
    out_dir = os.path.join(OUT_DIR, "{}{}".format(stem, run_tag))
    os.makedirs(out_dir, exist_ok=True)

    result_path = os.path.join(out_dir, "结果.png")
    compare_path = os.path.join(out_dir, "对比.png")
    input_path = os.path.join(out_dir, "输入.png")

    out_pil.save(result_path)
    content_rgb.save(input_path)

    # 风格图也缩到同一尺寸，三张并排才好比
    style_show = Image.open(STYLE_PATH).convert("RGB").resize(
        out_pil.size, Image.LANCZOS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    panels = [(style_show, "风格"), (content_rgb, "内容"), (out_pil, "结果")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(compare_path, dpi=110)
    plt.close(fig)

    print("输出文件夹: {}".format(out_dir))
    print("  结果    : {}  {}".format(out_pil.size, result_path))
    print("  对比    : {}".format(compare_path))
    print("  输入    : {}  {}".format(content_rgb.size, input_path))
    end_time = time.time()
    print('运行时间：{}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
