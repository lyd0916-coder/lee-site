"""평촌제 2D 모션그래픽 쇼츠 렌더러 (1080x1920, 30fps)

사용법: python3 평촌제_쇼츠_렌더.py <사진폴더> <글꼴폴더> <출력.mp4>
사진폴더에는 블로그 원본 사진(파일명 앞 8자리 ID)이 있어야 합니다.
"""
import math
import os
import random
import subprocess
import sys
import wave
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.signal import butter, sosfilt

PHOTO_DIR, FONT_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H, FPS = 1080, 1920, 30
SR = 44100

BLUE = (28, 132, 228)
SKY = (90, 185, 255)
YELLOW = (255, 214, 64)
ORANGE = (255, 146, 43)
NAVY = (14, 28, 70)
WHITE = (255, 255, 255)
PINK = (255, 110, 150)

FONT_XB = f"{FONT_DIR}/NanumGothicExtraBold.ttf"
FONT_B = f"{FONT_DIR}/NanumGothicBold.ttf"

# 학생 얼굴 가림 영역 (원본 4000px 기준, EXIF 회전 적용 후)
FACE_BOXES = {
    "d02fef9e": [(1380, 1200, 1520, 1380), (1975, 1080, 2120, 1280), (2220, 1080, 2380, 1300), (110, 1920, 230, 2020)],
    "544b11d4": [(1110, 1150, 1290, 1370), (1890, 1060, 2090, 1250), (1800, 1230, 1990, 1450),
                 (240, 1300, 410, 1400), (620, 1320, 760, 1450), (740, 1560, 850, 1660)],
    "f7a6362a": [(1720, 1720, 1920, 1920), (0, 1680, 90, 1820), (3300, 1515, 3420, 1640), (3720, 1545, 3840, 1680)],
}

# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def out_cubic(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3

def in_out(p):
    p = clamp(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2

def out_back(p, s=1.9):
    p = clamp(p)
    return 1 + (s + 1) * (p - 1) ** 3 + s * (p - 1) ** 2

# ---------------------------------------------------------------- photos
@lru_cache(None)
def photo(pid):
    im = ImageOps.exif_transpose(Image.open(f"{PHOTO_DIR}/{pid}-image.jpg")).convert("RGB")
    for (x0, y0, x1, y1) in FACE_BOXES.get(pid, []):
        pad = int((x1 - x0) * 0.25)
        box = (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
        region = im.crop(box)
        small = region.resize((max(1, region.width // 18), max(1, region.height // 18)), Image.BILINEAR)
        blurred = small.resize(region.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
        mask = Image.new("L", region.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, region.width - 1, region.height - 1), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(pad / 3 + 1))
        region.paste(blurred, (0, 0), mask)
        im.paste(region, box[:2])
    return im

def cover(im, w, h, focus=(0.5, 0.5)):
    s = max(w / im.width, h / im.height)
    r = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
    x = int((r.width - w) * focus[0])
    y = int((r.height - h) * focus[1])
    return r.crop((x, y, x + w, y + h))

@lru_cache(None)
def halftone():
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    step = 34
    for y in range(0, H + step, step):
        for x in range(0, W + step, step):
            k = clamp(1 - math.hypot(x - W, y) / 1100)
            k2 = clamp(1 - math.hypot(x, y - H) / 900)
            r = 11 * max(k, k2)
            if r > 1:
                d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 38))
    return layer

@lru_cache(None)
def background(pid):
    if pid is None:
        base = Image.new("RGB", (W, H), BLUE)
        grad = Image.linear_gradient("L").resize((W, H))
        base = Image.composite(Image.new("RGB", (W, H), (20, 90, 200)), base, grad)
    else:
        base = cover(photo(pid), W // 4, H // 4).filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BILINEAR)
        base = Image.blend(base, Image.new("RGB", (W, H), NAVY), 0.6)
    out = base.convert("RGBA")
    out.alpha_composite(halftone())
    return out

@lru_cache(None)
def card_source(pid, cw, ch, focus):
    return cover(photo(pid), int(cw * 1.12), int(ch * 1.12), focus)

@lru_cache(None)
def round_mask(w, h, r):
    m = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=255)
    return m.resize((w, h), Image.LANCZOS)

@lru_cache(None)
def shadow(w, h):
    pad = 60
    s = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(s).rounded_rectangle((pad, pad + 18, pad + w, pad + h + 18), 36, fill=(0, 0, 0, 150))
    return s.filter(ImageFilter.GaussianBlur(22))

def card(pid, cw, ch, zoom, focus=(0.5, 0.5), border=14):
    src = card_source(pid, cw, ch, focus)
    vw, vh = src.width / zoom, src.height / zoom
    x0 = (src.width - vw) / 2
    y0 = (src.height - vh) / 2
    img = src.resize((cw, ch), Image.BILINEAR, box=(x0, y0, x0 + vw, y0 + vh))
    out = Image.new("RGBA", (cw + border * 2, ch + border * 2), (0, 0, 0, 0))
    white = Image.new("RGBA", out.size, WHITE + (255,))
    out.paste(white, (0, 0), round_mask(out.width, out.height, 34))
    out.paste(img, (border, border), round_mask(cw, ch, 24))
    return out

# ---------------------------------------------------------------- drawing helpers
@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)

@lru_cache(None)
def text_img(text, size, fill=WHITE, stroke=0, stroke_fill=NAVY, path=FONT_XB):
    f = font(path, size)
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    im = Image.new("RGBA", (r - l + 8, b - t + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4 - l, 4 - t), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im

def paste_center(canvas, im, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    if alpha <= 0.01 or scale <= 0.01:
        return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if rot:
        im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1.0:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

def slam(canvas, im, cx, cy, t, t0, dur=0.32, rot=0.0, from_scale=1.9):
    if t < t0:
        return
    p = (t - t0) / dur
    sc = from_scale - (from_scale - 1) * out_back(p, 1.4)
    paste_center(canvas, im, cx, cy, sc, clamp(p * 2.5), rot)

def pop(canvas, im, cx, cy, t, t0, dur=0.38, rot=0.0):
    if t < t0:
        return
    paste_center(canvas, im, cx, cy, out_back((t - t0) / dur), clamp((t - t0) / dur * 3), rot)

def slide(canvas, im, cx, cy, t, t0, dx=0, dy=60, dur=0.4):
    if t < t0:
        return
    p = out_cubic((t - t0) / dur)
    paste_center(canvas, im, cx + dx * (1 - p), cy + dy * (1 - p), 1.0, p)

def label(text, size, fg, bg, padx=34, pady=18, path=FONT_XB):
    ti = text_img(text, size, fg, path=path)
    w, h = ti.width + padx * 2, ti.height + pady * 2
    im = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), h, fill=bg + (255,))
    im = im.resize((w, h), Image.LANCZOS)
    im.alpha_composite(ti, (padx, pady))
    return im

def wipe_label(canvas, text, size, fg, bg, cx, cy, t, t0, dur=0.35, rot=0.0):
    """색 띠가 왼쪽에서 펼쳐지며 글자가 드러나는 효과"""
    if t < t0:
        return
    im = label(text, size, fg, bg)
    p = out_cubic((t - t0) / dur)
    w = max(1, int(im.width * p))
    part = Image.new("RGBA", im.size, (0, 0, 0, 0))
    part.paste(im.crop((0, 0, w, im.height)), (0, 0))
    paste_center(canvas, part, cx, cy, 1.0, 1.0, rot)

def icon(kind, size=150, color=WHITE, bg=None):
    S = size * 3
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if bg:
        d.ellipse((0, 0, S - 1, S - 1), fill=bg + (255,))
    c = color + (255,)
    u = S / 10
    if kind == "mic":
        d.rounded_rectangle((3.6 * u, 1.6 * u, 6.4 * u, 5.8 * u), 1.4 * u, fill=c)
        d.arc((2.7 * u, 3.4 * u, 7.3 * u, 7.0 * u), 0, 180, fill=c, width=int(0.45 * u))
        d.line((5 * u, 7.0 * u, 5 * u, 8.3 * u), fill=c, width=int(0.45 * u))
        d.line((3.8 * u, 8.3 * u, 6.2 * u, 8.3 * u), fill=c, width=int(0.45 * u))
    elif kind == "note":
        d.ellipse((2.2 * u, 6.0 * u, 4.6 * u, 7.9 * u), fill=c)
        d.ellipse((5.6 * u, 5.2 * u, 8.0 * u, 7.1 * u), fill=c)
        d.rectangle((4.0 * u, 2.4 * u, 4.6 * u, 7.0 * u), fill=c)
        d.rectangle((7.4 * u, 1.6 * u, 8.0 * u, 6.2 * u), fill=c)
        d.polygon([(4.0 * u, 2.4 * u), (8.0 * u, 1.6 * u), (8.0 * u, 2.8 * u), (4.0 * u, 3.6 * u)], fill=c)
    elif kind == "ball":
        d.ellipse((2 * u, 2 * u, 8 * u, 8 * u), outline=c, width=int(0.55 * u))
        d.arc((-1.2 * u, 2 * u, 4.6 * u, 8 * u), -60, 60, fill=c, width=int(0.45 * u))
        d.arc((5.4 * u, 2 * u, 11.2 * u, 8 * u), 120, 240, fill=c, width=int(0.45 * u))
    elif kind == "check":
        d.line([(2.6 * u, 5.2 * u), (4.3 * u, 6.9 * u), (7.6 * u, 3.4 * u)], fill=c, width=int(0.9 * u), joint="curve")
    elif kind == "heart":
        d.ellipse((2.0 * u, 2.6 * u, 5.2 * u, 5.8 * u), fill=c)
        d.ellipse((4.8 * u, 2.6 * u, 8.0 * u, 5.8 * u), fill=c)
        d.polygon([(2.2 * u, 4.9 * u), (7.8 * u, 4.9 * u), (5.0 * u, 8.2 * u)], fill=c)
    return im.resize((size, size), Image.LANCZOS)

def star_img(r, color):
    S = r * 6
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    pts = []
    for i in range(8):
        rr = S / 2 if i % 2 == 0 else S / 7
        a = math.pi / 4 * i - math.pi / 2
        pts.append((c + rr * math.cos(a), c + rr * math.sin(a)))
    d.polygon(pts, fill=color + (255,))
    return im.resize((r * 2, r * 2), Image.LANCZOS)

STARS = {(r, col): star_img(r, col) for r in (18, 26, 36) for col in (YELLOW, WHITE, PINK)}

def sparkles(canvas, t, seed, n=7, area=(0, 0, W, H)):
    rnd = random.Random(seed)
    for i in range(n):
        x = rnd.uniform(area[0], area[2])
        y = rnd.uniform(area[1], area[3])
        r = rnd.choice((18, 26, 36))
        col = rnd.choice((YELLOW, WHITE, PINK))
        ph = rnd.uniform(0, 6.28)
        sc = 0.55 + 0.45 * math.sin(t * 5 + ph)
        paste_center(canvas, STARS[(r, col)], x, y + math.sin(t * 2 + ph) * 10, sc, 0.9, t * 40 + ph * 20)

def chevrons(canvas, t, y, color=YELLOW, n=3, x0=90):
    d = ImageDraw.Draw(canvas)
    for i in range(n):
        k = (t * 2.2 - i * 0.25) % 1.0
        a = int(255 * (0.35 + 0.65 * (1 - k)))
        x = x0 + i * 44
        d.line([(x, y - 22), (x + 22, y), (x, y + 22)], fill=color + (a,), width=10)

def mascot(color, cap, size=230):
    S = size * 2
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((S * .08, S * .14, S * .92, S * .98), fill=color + (255,), outline=NAVY + (255,), width=int(S * .03))
    d.chord((S * .08, S * .14, S * .92, S * .98), 180, 360, fill=cap + (255,), outline=NAVY + (255,), width=int(S * .03))
    for ex in (.38, .62):
        d.ellipse((S * (ex - .04), S * .6, S * (ex + .04), S * .68), fill=NAVY + (255,))
    d.arc((S * .42, S * .66, S * .58, S * .78), 20, 160, fill=NAVY + (255,), width=int(S * .025))
    d.ellipse((S * .24, S * .7, S * .32, S * .75), fill=PINK + (200,))
    d.ellipse((S * .68, S * .7, S * .76, S * .75), fill=PINK + (200,))
    return im.resize((size, size), Image.LANCZOS)

MASCOT_A = mascot(YELLOW, (60, 170, 90))
MASCOT_B = mascot((95, 110, 230), (230, 70, 90))

def card_in(canvas, pid, t, cw, ch, cy, rot=-2.0, focus=(0.5, 0.5), t0=0.0, from_x=W):
    p = out_back((t - t0) / 0.45, 1.2)
    cx = W / 2 + (1 - p) * from_x
    c = card(pid, cw, ch, 1.0 + 0.07 * clamp(t / 5), focus)
    sh = shadow(c.width, c.height)
    paste_center(canvas, sh, cx, cy, 1.0, clamp(p), rot)
    paste_center(canvas, c, cx, cy, 1.0, 1.0, rot)

# ---------------------------------------------------------------- scenes
# 각 장면: (길이, 함수). 함수는 canvas와 장면 내부 시간 t를 받는다.
def s_title(cv, t):
    sparkles(cv, t, 1, 10)
    chevrons(cv, t, 360)
    chevrons(cv, t, 1560, WHITE, x0=820)
    for i, ch in enumerate("평촌제"):
        im = text_img(ch, 300, YELLOW, 14, NAVY)
        slam(cv, im, 250 + i * 290, 820, t, 0.15 + i * 0.28, rot=[6, -4, 5][i])
    wipe_label(cv, "2026. 10. 16 · 평촌중학교", 54, NAVY, WHITE, W / 2, 1080, t, 1.15)
    wipe_label(cv, "우리들의 Festival!", 70, WHITE, PINK, W / 2, 1210, t, 1.5, rot=-3)
    for k, (m, x, ph) in enumerate(((MASCOT_A, 300, 0), (MASCOT_B, 780, 1.4))):
        if t > 1.9 + k * 0.15:
            p = out_back((t - 1.9 - k * 0.15) / 0.4)
            paste_center(cv, m, x, 1480 - abs(math.sin(t * 5 + ph)) * 40, p, 1.0, math.sin(t * 4 + ph) * 8)

def s_crowd(cv, t):
    card_in(cv, "95cc9f4a", t, 980, 551, 1000, rot=-2)
    wipe_label(cv, "전교생", 64, NAVY, YELLOW, W / 2, 330, t, 0.35, rot=-2)
    n = int(1200 * out_cubic((t - 0.55) / 1.2)) if t > 0.55 else 0
    if t > 0.55:
        num = text_img(f"{n:,}명", 190, WHITE, 12, NAVY)
        sc = 1.0 + (0.18 * (1 - clamp((t - 1.75) / 0.25)) if t > 1.75 else 0)
        paste_center(cv, num, W / 2, 510, sc)
        paste_center(cv, text_img("약", 60, YELLOW), 150, 470)
    slide(cv, text_img("다 함께 모이려고", 78, WHITE, 6, NAVY), W / 2, 1390, t, 1.9)
    slam(cv, text_img("학교 밖 무대로!", 104, YELLOW, 9, NAVY), W / 2, 1520, t, 2.3, rot=-2)
    sparkles(cv, t, 2, 5, (60, 1300, 1020, 1650))

def s_teachers(cv, t):
    card_in(cv, "c47a47d1", t, 900, 675, 980, rot=2)
    slide(cv, text_img("새 공간을 찾고", 80, WHITE, 6, NAVY), W / 2, 330, t, 0.3)
    wipe_label(cv, "이동 · 인솔 · 안전관리까지", 64, NAVY, YELLOW, W / 2, 470, t, 0.7)
    slide(cv, text_img("선생님들의 준비 덕분에", 82, WHITE, 6, NAVY), W / 2, 1470, t, 1.6)
    pop(cv, icon("heart", 120, WHITE, PINK), 900, 1300, t, 2.1, rot=12)

def s_mc(cv, t):
    card_in(cv, "d02fef9e", t, 980, 551, 980, rot=-2, focus=(0.55, 0.6))
    pop(cv, icon("mic", 150, WHITE, ORANGE), 180, 380, t, 0.3, rot=-10)
    slide(cv, text_img("학생이 초대하고", 88, WHITE, 6, NAVY), W / 2 + 60, 380, t, 0.45, dx=80, dy=0)
    slam(cv, text_img("학생이 진행한 무대", 104, YELLOW, 9, NAVY), W / 2, 1420, t, 1.3, rot=-2)
    wipe_label(cv, "학생자치회", 56, NAVY, WHITE, W / 2, 1570, t, 1.9)

def s_question(cv, t):
    card_in(cv, "544b11d4", t, 860, 645, 900, rot=2, focus=(0.5, 0.55))
    wipe_label(cv, "학생의 질문에, 교육감의 답", 58, NAVY, YELLOW, W / 2, 360, t, 0.35)
    slide(cv, text_img("누구나 평생 즐기도록", 74, WHITE, 6, NAVY), W / 2, 1335, t, 1.0)
    for i, (k, txt, col, x) in enumerate((("note", "악기 하나", PINK, 290), ("ball", "운동 한 가지", ORANGE, 790))):
        t0 = 1.6 + i * 0.45
        pop(cv, icon(k, 150, WHITE, col), x, 1500, t, t0, rot=math.sin(t * 3 + i) * 8)
        slam(cv, text_img(txt, 70, YELLOW, 6, NAVY), x, 1640, t, t0 + 0.15)

MONTAGE = (("6619ff4e", "춤!", (0.5, 0.6)), ("f7a6362a", "노래!", (0.5, 0.5)), ("a803c3dd", "환호!", (0.5, 0.55)))

def s_stage(cv, t):
    i = min(2, int(t / 1.6))
    lt = t - i * 1.6
    pid, word, focus = MONTAGE[i]
    card_in(cv, pid, lt, 980, 551, 1000, rot=[-2, 2, -1][i], focus=focus, from_x=W if i == 0 else 0)
    if i > 0 and lt < 0.12:
        ImageDraw.Draw(cv).rectangle((0, 0, W, H), fill=(255, 255, 255, int(200 * (1 - lt / 0.12))))
    slide(cv, text_img("몇 분의 공연 뒤에", 76, WHITE, 6, NAVY), W / 2, 320, t, 0.2)
    slam(cv, text_img("수많은 연습", 110, YELLOW, 9, NAVY), W / 2, 470, t, 0.5)
    slam(cv, text_img(word, 170, WHITE, 12, PINK), W / 2, 1450, lt, 0.1, rot=[-6, 5, -4][i])
    sparkles(cv, t, 6 + i, 6, (60, 1300, 1020, 1700))

def s_together(cv, t):
    card_in(cv, "a751c399", t, 980, 551, 980, rot=1.5)
    wipe_label(cv, "공연하는 친구", 64, NAVY, YELLOW, 330, 330, t, 0.3, rot=-3)
    pop(cv, text_img("+", 90, WHITE, 6, NAVY), W / 2 + 40, 420, t, 0.6)
    wipe_label(cv, "응원하는 친구", 64, WHITE, PINK, 740, 510, t, 0.75, rot=3)
    slam(cv, text_img("함께 만든 평촌제", 108, YELLOW, 9, NAVY), W / 2, 1450, t, 1.4, rot=-2)
    pop(cv, icon("heart", 110, WHITE, PINK), 920, 1300, t, 1.9, rot=-10)

def s_safety(cv, t):
    card_in(cv, "dd8ce3cd", t, 900, 675, 960, rot=-2)
    slam(cv, text_img("끝까지 안전하게", 100, WHITE, 8, NAVY), W / 2, 370, t, 0.3)
    for i, txt in enumerate(("순서대로 이동", "귀가 확인까지")):
        t0 = 1.0 + i * 0.5
        pop(cv, icon("check", 100, WHITE, (40, 190, 120)), 260, 1420 + i * 140, t, t0)
        slide(cv, text_img(txt, 72, WHITE, 6, NAVY), 600, 1420 + i * 140, t, t0 + 0.08, dx=-60, dy=0)

COLLAGE = ("95cc9f4a", "c47a47d1", "d02fef9e", "544b11d4", "6619ff4e", "a803c3dd", "f7a6362a", "a751c399")

@lru_cache(None)
def thumb(pid):
    return card(pid, 300, 225, 1.0, border=8)

def s_ending(cv, t):
    # 1) 사진 조각들이 날아와 콜라주 → 2) 단체사진 위 인사 카드
    if t < 2.3:
        rnd = random.Random(9)
        for i, pid in enumerate(COLLAGE):
            gx = 210 + (i % 3) * 330 - (60 if i // 3 == 2 else 0)
            gy = 560 + (i // 3) * 300
            if i >= 6:
                gx = 380 + (i - 6) * 330
            sx, sy = rnd.choice((-500, W + 500)), rnd.uniform(-300, H + 300)
            t0 = i * 0.12
            p = out_back((t - t0) / 0.45, 1.1)
            if t < t0:
                continue
            rot = rnd.uniform(-9, 9)
            paste_center(cv, thumb(pid), sx + (gx - sx) * p, sy + (gy - sy) * p, 1.0, 1.0, rot * p + (1 - p) * 60)
        slam(cv, text_img("오늘의 박수가", 92, WHITE, 8, NAVY), W / 2, 330, t, 0.9)
        slam(cv, text_img("오래 힘이 되기를", 92, YELLOW, 8, NAVY), W / 2, 1560, t, 1.3)
    else:
        lt = t - 2.3
        p = out_cubic(lt / 0.6)
        gp = photo("f33c6000")
        zoom = 1.08 - 0.08 * clamp(lt / 3.5)
        gw = int(W * zoom)
        gh = int(gp.height * gw / gp.width)
        g = gp.resize((gw, gh), Image.BILINEAR).convert("RGBA")
        fade = Image.linear_gradient("L").resize((gw, 420))
        alpha = Image.new("L", (gw, gh), 255)
        alpha.paste(fade.point(lambda v: 255 - v), (0, gh - 420))
        g.putalpha(alpha.point(lambda v: int(v * p)))
        cv.alpha_composite(Image.new("RGBA", (W, H), NAVY + (int(255 * p),)))
        cv.alpha_composite(g, ((W - gw) // 2, 60 - (gh - 1440) // 2))
        slam(cv, text_img("문화 · 예술 · 체육을", 76, WHITE, 6, NAVY), W / 2, 1290, lt, 0.5)
        slam(cv, text_img("마음껏 누리도록", 96, YELLOW, 8, NAVY), W / 2, 1400, lt, 0.8)
        wipe_label(cv, "경기도의원 이채명", 72, NAVY, WHITE, W / 2, 1560, lt, 1.4)
        slide(cv, text_img("경기도교육청 예산결산특별위원장", 44, WHITE, 4, NAVY, FONT_B), W / 2, 1665, lt, 1.7)
        sparkles(cv, lt, 12, 6, (80, 1200, 1000, 1500))

SCENES = [
    (3.4, s_title, None),
    (4.6, s_crowd, "95cc9f4a"),
    (4.4, s_teachers, "c47a47d1"),
    (4.4, s_mc, "d02fef9e"),
    (4.8, s_question, "544b11d4"),
    (4.8, s_stage, "6619ff4e"),
    (4.0, s_together, "a751c399"),
    (3.4, s_safety, "dd8ce3cd"),
    (6.2, s_ending, None),
]
STARTS = np.cumsum([0] + [d for d, _, _ in SCENES])
TOTAL = float(STARTS[-1])
TRANS = 0.32

# 화면 흔들림이 생기는 '쾅' 시점 (전역 시간)
def slam_times():
    s = STARTS
    ts = [s[0] + 0.15 + i * 0.28 for i in range(3)]
    ts += [s[1] + 1.75, s[1] + 2.3, s[3] + 1.3, s[5] + 0.5, s[5] + 0.1, s[5] + 1.7, s[5] + 3.3,
           s[6] + 1.4, s[7] + 0.3, s[8] + 0.9, s[8] + 1.3, s[8] + 2.8, s[8] + 3.1]
    return sorted(ts)

SLAMS = slam_times()

def render_scene(idx, t):
    _, fn, bgk = SCENES[idx]
    cv = background(bgk).copy()
    fn(cv, t)
    return cv

def frame(gt):
    idx = int(np.searchsorted(STARTS, gt, side="right") - 1)
    idx = min(idx, len(SCENES) - 1)
    lt = gt - STARTS[idx]
    cur = render_scene(idx, lt)
    if idx > 0 and lt < TRANS:
        e = in_out(lt / TRANS)
        prev = render_scene(idx - 1, STARTS[idx] - STARTS[idx - 1] + lt)
        out = Image.new("RGBA", (W, H))
        out.paste(prev, (int(-W * e), 0))
        out.paste(cur, (int(W * (1 - e)), 0))
        cur = out
    # 쾅 효과: 짧은 흔들림
    shake = 0
    for st in SLAMS:
        if 0 <= gt - st - 0.12 < 0.22:
            shake = max(shake, 16 * (1 - (gt - st - 0.12) / 0.22))
    if shake:
        rnd = random.Random(int(gt * 1000))
        dx, dy = rnd.uniform(-shake, shake), rnd.uniform(-shake, shake)
        bg = Image.new("RGBA", (W, H), NAVY + (255,))
        bg.paste(cur, (int(dx), int(dy)))
        cur = bg
    # 시작 페이드인, 끝 페이드아웃
    fade = min(clamp(gt / 0.25), clamp((TOTAL - gt) / 0.6))
    if fade < 1:
        cur = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), cur, fade)
    return cur.convert("RGB")

# ---------------------------------------------------------------- audio
def make_audio(path):
    n = int(TOTAL * SR)
    t = np.arange(n) / SR
    mix = np.zeros(n)
    rng = np.random.default_rng(3)
    bpm = 120
    beat = 60 / bpm

    def add(sig, at, gain=1.0):
        i = int(at * SR)
        if i >= n:
            return
        j = min(n, i + len(sig))
        mix[i:j] += sig[: j - i] * gain

    def env(length, a=0.005, d=0.2):
        tt = np.arange(int(length * SR)) / SR
        return np.minimum(1, tt / a) * np.exp(-tt / d)

    # 드럼
    k_t = np.arange(int(0.35 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-k_t * 35)) / SR) * np.exp(-k_t * 8)
    hp = butter(4, 7000, "hp", fs=SR, output="sos")
    bp = butter(2, [1500, 5000], "bp", fs=SR, output="sos")
    hat = sosfilt(hp, rng.standard_normal(int(0.06 * SR))) * env(0.06, 0.001, 0.015)
    snare = sosfilt(bp, rng.standard_normal(int(0.2 * SR))) * env(0.2, 0.001, 0.05)
    snare += np.sin(2 * np.pi * 190 * np.arange(len(snare)) / SR) * env(0.2, 0.001, 0.04) * 0.5

    chords = [(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [60, 64, 69]), (41, [60, 65, 69])]  # C G Am F
    mtof = lambda m: 440 * 2 ** ((m - 69) / 12)
    bar = beat * 4
    music_start = 0.15
    b = 0
    while music_start + b * beat < TOTAL:
        at = music_start + b * beat
        intro = at < 3.4
        add(kick, at, 0.0 if (intro and b % 2) else 0.9)
        if b % 2 == 1 and not intro:
            add(snare, at, 0.45)
        add(hat, at + beat / 2, 0.25)
        if not intro:
            add(hat, at, 0.12)
        root, notes = chords[int((at - music_start) / bar) % 4]
        # 베이스 (8분음표)
        for h in range(2):
            bt = np.arange(int(beat / 2 * SR)) / SR
            f = mtof(root - 12 + (12 if h == 1 else 0))
            bass = (np.sin(2 * np.pi * f * bt) + 0.3 * np.sin(4 * np.pi * f * bt)) * np.exp(-bt * 6)
            add(bass, at + h * beat / 2, 0.28)
        # 아르페지오 플럭
        for h in range(2):
            m = notes[(b * 2 + h) % 3] + 12
            pt = np.arange(int(0.3 * SR)) / SR
            f = mtof(m)
            pl = (np.sin(2 * np.pi * f * pt) + 0.25 * np.sin(6 * np.pi * f * pt)) * np.exp(-pt * 12)
            add(pl, at + h * beat / 2, 0.11)
        b += 1
    # 패드 (화음)
    pad = np.zeros(n)
    for ci in range(int(TOTAL / bar) + 2):
        at = music_start + ci * bar
        root, notes = chords[ci % 4]
        seg = np.arange(int(bar * SR)) / SR
        s = sum(np.sin(2 * np.pi * mtof(m) * seg) + 0.15 * np.sin(4 * np.pi * mtof(m) * seg) for m in notes)
        s *= np.minimum(1, seg / 0.3) * np.minimum(1, (bar - seg) / 0.3)
        i = int(at * SR)
        j = min(n, i + len(s))
        if i < n:
            pad[i:j] += s[: j - i]
    mix += sosfilt(butter(2, 1800, "lp", fs=SR, output="sos"), pad) * 0.045
    mix *= float(os.environ.get("MUSIC", "1"))  # MUSIC=0 이면 효과음만 (앱에서 음악을 따로 넣을 때)

    # 효과음
    def whoosh(length=0.45):
        m = int(length * SR)
        noise = rng.standard_normal(m)
        out = np.zeros(m)
        chunks = 12
        for c in range(chunks):
            lo = 300 + 3500 * (c / chunks)
            sos = butter(2, [lo, lo * 2.2], "bp", fs=SR, output="sos")
            a, z = c * m // chunks, (c + 1) * m // chunks
            out[a:z] = sosfilt(sos, noise[a:z])
        e = np.sin(np.pi * np.arange(m) / m) ** 2
        return out * e

    def thud():
        tt = np.arange(int(0.4 * SR)) / SR
        body = np.sin(2 * np.pi * np.cumsum(40 + 90 * np.exp(-tt * 25)) / SR) * np.exp(-tt * 7)
        crack = sosfilt(butter(2, 2500, "hp", fs=SR, output="sos"), rng.standard_normal(len(tt))) * np.exp(-tt * 40)
        return body + crack * 0.5

    def popfx():
        tt = np.arange(int(0.12 * SR)) / SR
        return np.sin(2 * np.pi * np.cumsum(500 + 900 * tt / 0.12) / SR) * np.exp(-tt * 30)

    def chime():
        tt = np.arange(int(1.6 * SR)) / SR
        return sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt * d) for f, d in ((1046, 2.5), (1568, 3), (2093, 4)))

    for s in STARTS[1:-1]:
        add(whoosh(), s - 0.12, 0.5)
    for st in SLAMS:
        add(thud(), st + 0.1, 0.55)
    S = STARTS
    for at in (S[0] + 1.15, S[0] + 1.5, S[2] + 2.1, S[3] + 0.3, S[4] + 1.6, S[4] + 2.05, S[6] + 0.6,
               S[6] + 1.9, S[7] + 1.0, S[7] + 1.5):
        add(popfx(), at, 0.35)
    for i in range(8):
        add(popfx(), S[8] + i * 0.12 + 0.2, 0.2)
    add(chime(), S[8] + 3.7, 0.25)

    fade = np.minimum(1, np.minimum(t / 0.05, (TOTAL - t) / 1.2))
    mix *= np.clip(fade, 0, 1)
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.89
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    wav = OUT.rsplit(".", 1)[0] + "_audio.wav"
    make_audio(wav)
    if os.environ.get("AUDIO_ONLY"):
        sys.exit()
    nframes = int(TOTAL * FPS)
    if len(sys.argv) > 4:  # 미리보기: 지정 시각 프레임만 PNG로 저장
        for s in sys.argv[4].split(","):
            frame(float(s)).save(f"{OUT}_{s}.png")
        sys.exit()
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT,
    ], stdin=subprocess.PIPE)
    for f in range(nframes):
        proc.stdin.write(frame(f / FPS).tobytes())
        if f % 150 == 0:
            print(f"{f}/{nframes}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("done", TOTAL)
