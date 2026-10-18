"""Vinyl Visualizer render engine.

Why it is fast: everything that never moves (background, vinyl, cover, title,
vignette, grain) is composed ONCE into a single base frame. Per video frame only
two small patches change:
  * the spinning label in the middle of the record (rotation is periodic, so a
    full turn is pre-rendered and reused);
  * the strip with the progress bar, timer and the dot waveform (each dot is a
    pre-rendered sprite, so a frame is ~80 tiny memory copies).
Python sends only these patches (~0.7 MB per 1440p frame) to ffmpeg, which
pastes them over the cached base frame and encodes on the GPU (NVENC).
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
CACHE_DIR = os.path.join(ROOT, "cache")
OUTPUT_DIR = os.path.join(ROOT, "output")

NO_WINDOW = 0x08000000 if os.name == "nt" else 0

RESOLUTIONS = {"720p": 720, "1080p": 1080, "1440p": 1440, "4k": 2160}

DEFAULTS = {
    "audio": "",
    "cover": "assets/demo_cover.jpg",
    "title": "Artist Name - Song Name",
    "title_font": "fonts/Super Magic.ttf",
    "title_size": 33,
    "title_color": "#ffffff",
    "title_shadow": 0.6,
    "title_offset": 0,
    "bg_mode": "gradient",      # gradient | image | cover
    "bg_style": "linear",       # linear | radial
    "bg_top": "#0b6079",
    "bg_bottom": "#7a0009",
    "bg_image": "",
    "bg_blur": 0,
    "bg_darken": 0.0,
    "vignette": 0.5,
    "grain": 0.35,
    "accent": "#1ea7cf",
    "track_color": "#a3061a",
    "track_opacity": 1.0,
    "wave_color": "#ffffff",
    "wave_opacity": 1.0,
    "ring_color": "#38332a",
    "disc_color": "#191919",
    "disc_sheen": 0.6,
    "timer_font": "fonts/F25_Bank_Printer.otf",
    "timer_size": 20,
    "timer_total": True,
    "rpm": 5,
    "spin_dir": 1,
    "sticker": True,
    "sticker_path": "assets/sticker_eac.png",
    "shadow": 0.55,
    "wave_bars": 80,
    "wave_height": 64,
    "wave_sens": 0.85,
    "wave_smooth": 0.55,
    "resolution": "1440p",
    "fps": 30,
    "quality": 18,
    "codec": "h264",
    "output": "",
}

# Layout measured from the reference frame, in 2560x1440 pixels.
LAYOUT = dict(
    cover_x=773, cover_y=413, cover_size=608,
    disc_cx=1486, disc_cy=724, disc_r=300,
    ring_r=132, label_r=90, hole_r=6,
    bar_y=1098, ui_x0=714, ui_x1=1842,
    track_th=4, played_th=6, timer_gap=64,
    wave_y=1158,
)


# ----------------------------------------------------------------- helpers

def resolve(path):
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def merged(cfg):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


def hex_rgb(h):
    h = (h or "#000000").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def find_ffmpeg(name="ffmpeg"):
    local = os.path.join(ROOT, "ffmpeg", name + (".exe" if os.name == "nt" else ""))
    if os.path.isfile(local):
        return local
    found = shutil.which(name)
    if not found:
        raise RuntimeError("Не найден %s. Установите ffmpeg или положите %s.exe в папку VinylRenderer\\ffmpeg" % (name, name))
    return found


_img_cache = {}


def load_image(path, mode="RGB"):
    p = resolve(path)
    key = (p, os.path.getmtime(p), mode)
    if key not in _img_cache:
        if len(_img_cache) > 12:
            _img_cache.clear()
        im = ImageOps.exif_transpose(Image.open(p))
        _img_cache[key] = im.convert(mode)
    return _img_cache[key]


def square(im):
    w, h = im.size
    m = min(w, h)
    return im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))


def cover_fit(im, W, H):
    w, h = im.size
    k = max(W / w, H / h)
    nw, nh = max(W, round(w * k)), max(H, round(h * k))
    im = im.resize((nw, nh), Image.LANCZOS)
    return im.crop(((nw - W) // 2, (nh - H) // 2, (nw - W) // 2 + W, (nh - H) // 2 + H))


_font_cache = {}


def get_font(path, size):
    p = resolve(path)
    key = (p, int(size))
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(p, max(4, int(size)))
        except Exception:
            _font_cache[key] = ImageFont.truetype("arial.ttf", max(4, int(size)))
    return _font_cache[key]


def missing_glyphs(path, text):
    """Characters of `text` the font cannot draw (rendered as empty)."""
    try:
        f = get_font(path, 48)
    except Exception:
        return []
    miss = []
    for ch in sorted(set(text)):
        if ch.isspace():
            continue
        if f.getmask(ch).getbbox() is None:
            miss.append(ch)
    return miss


def text_image(text, font, spacing):
    """White-on-black L image with the text and its top-left offset."""
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = probe.multiline_textbbox((0, 0), text, font=font, align="center", spacing=spacing)
    bb = (int(math.floor(bb[0])), int(math.floor(bb[1])), int(math.ceil(bb[2])), int(math.ceil(bb[3])))
    pad = 4
    im = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
    ImageDraw.Draw(im).multiline_text((pad - bb[0], pad - bb[1]), text, font=font, fill=255,
                                      align="center", spacing=spacing)
    return im


def to_f(im):
    return np.asarray(im, dtype=np.float32) / 255.0


def blend_into(dst, x, y, rgb, a):
    """dst[y:, x:] = dst*(1-a) + rgb*a, clipped to dst. rgb (h,w,3) or (3,), a (h,w)."""
    h, w = a.shape
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    aa = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
    src = rgb if np.ndim(rgb) == 1 else rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d += (src - d) * aa


def capsule_alpha(px, py, ax, ay, bx, by, r):
    """Anti-aliased alpha of a capsule (segment a-b with radius r) at points px,py."""
    vx, vy = bx - ax, by - ay
    ll = vx * vx + vy * vy
    if ll < 1e-9:
        t = 0.0
    else:
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / ll, 0.0, 1.0)
    d = np.hypot(px - (ax + t * vx), py - (ay + t * vy)) - r
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def even_floor(v):
    return int(math.floor(v / 2.0)) * 2


def even_ceil(v):
    return int(math.ceil(v / 2.0)) * 2


def fmt_time(sec, hours):
    sec = max(0, int(sec))
    if hours:
        return "%d:%02d:%02d" % (sec // 3600, sec // 60 % 60, sec % 60)
    return "%02d:%02d" % (sec // 60, sec % 60)


# ------------------------------------------------------------------- audio

def probe_audio(path):
    ffprobe = find_ffmpeg("ffprobe")
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "format=duration:stream=codec_name,sample_rate,channels", "-of", "json",
                          resolve(path)], capture_output=True, text=True, creationflags=NO_WINDOW)
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError("В файле нет аудио: %s" % path)
    return {"duration": float(info.get("format", {}).get("duration") or 0),
            "codec": streams[0].get("codec_name", ""),
            "sample_rate": int(streams[0].get("sample_rate") or 0)}


ANALYSIS_SR = 22050
NFFT = 2048
FMIN, FMAX = 45.0, 10500.0
TILT_DB_OCT = 2.5


def band_centers(nbars):
    edges = np.geomspace(FMIN, FMAX, nbars + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def analysis_path(path, fps, nbars):
    p = resolve(path)
    st = os.stat(p)
    key = "%s|%d|%f|%s|%s|v3" % (os.path.abspath(p), st.st_size, st.st_mtime, fps, nbars)
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode("utf8")).hexdigest() + ".npz")


def cached_analysis(path, fps, nbars):
    try:
        cp = analysis_path(path, fps, nbars)
        if os.path.isfile(cp):
            d = np.load(cp)
            return {"db": d["db"], "duration": float(d["duration"])}
    except Exception:
        pass
    return None


def analyze_audio(path, fps, nbars, progress=None, cancel=None):
    """Spectrum (dB per bar) for every video frame. Streams the audio, so a
    3-hour file never has to fit in memory. Results are cached on disk."""
    hit = cached_analysis(path, fps, nbars)
    if hit is not None:
        return hit
    total = probe_audio(path)["duration"] or 1.0
    ffmpeg = find_ffmpeg()
    proc = subprocess.Popen([ffmpeg, "-v", "error", "-i", resolve(path), "-vn", "-ac", "1",
                             "-ar", str(ANALYSIS_SR), "-f", "f32le", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    hop = ANALYSIS_SR / float(fps)
    win = np.hanning(NFFT).astype(np.float32)
    norm = 2.0 / win.sum()
    edges, _ = band_centers(nbars)
    binhz = ANALYSIS_SR / NFFT
    lo_bins = edges[:-1] / binhz
    hi_bins = edges[1:] / binhz
    wide = (hi_bins - lo_bins) >= 2.0
    starts = np.clip(np.ceil(lo_bins).astype(np.int64), 0, NFFT // 2)
    starts = np.maximum.accumulate(starts)
    counts = np.maximum(1, np.diff(np.append(starts, NFFT // 2 + 1))).astype(np.float32)
    centers = np.sqrt(lo_bins * hi_bins)
    c_lo = np.clip(np.floor(centers).astype(np.int64), 0, NFFT // 2 - 1)
    c_fr = (centers - c_lo).astype(np.float32)
    offs = np.arange(NFFT, dtype=np.int64)

    buf = np.zeros(NFFT // 2, np.float32)   # leading pad: window is centred on the frame time
    buf_start = 0
    n_samples = 0
    fi = 0
    chunks = []
    chunk_bytes = ANALYSIS_SR * 4 * 20
    eof = False
    try:
        while not eof:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Отменено")
            data = proc.stdout.read(chunk_bytes)
            if not data:
                eof = True
                buf = np.concatenate([buf, np.zeros(NFFT, np.float32)])
            else:
                arr = np.frombuffer(data[:len(data) // 4 * 4], np.float32)
                n_samples += arr.size
                buf = np.concatenate([buf, arr])
            avail = buf_start + buf.size
            if eof:
                last = int(math.ceil(n_samples / hop))
            else:
                last = int((avail - NFFT) / hop)
                while last >= fi and int(round(last * hop)) + NFFT > avail:
                    last -= 1
                last += 1
            while fi < last:
                f1 = min(last, fi + 2048)
                pos = np.rint(np.arange(fi, f1) * hop).astype(np.int64) - buf_start
                frames = buf[np.minimum(pos[:, None] + offs, buf.size - 1)] * win
                mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) * norm
                vrms = np.sqrt(np.add.reduceat(mag * mag, starts, axis=1) / counts)
                vint = mag[:, c_lo] * (1 - c_fr) + mag[:, c_lo + 1] * c_fr
                val = np.where(wide[None, :], vrms, vint)
                chunks.append((20.0 * np.log10(val + 1e-7)).astype(np.float16))
                fi = f1
            keep = int(round(fi * hop)) - buf_start
            if keep > 0:
                buf = buf[keep:]
                buf_start += keep
            if progress:
                progress(min(0.999, n_samples / ANALYSIS_SR / total))
    finally:
        proc.stdout.close()
        proc.wait()
    if fi == 0:
        raise RuntimeError("Не удалось прочитать аудио")
    db = np.concatenate(chunks)
    duration = n_samples / float(ANALYSIS_SR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        np.savez(analysis_path(path, fps, nbars), db=db, duration=duration)
    except Exception:
        pass
    return {"db": db, "duration": duration}


def _tilt(nbars):
    _, fc = band_centers(nbars)
    return (TILT_DB_OCT * np.log2(fc / 500.0)).astype(np.float32)[None, :]


def db_ceiling(db):
    x = db[::max(1, db.shape[0] // 20000)].astype(np.float32) + _tilt(db.shape[1])
    return float(np.percentile(x, 99.6))


def levels_from_db(db, fps, sens=1.0, smooth=0.55, ceil=None):
    """0..1 bar heights per frame: tilt for highs, global normalisation, a
    curve that keeps quiet bars as dots, fast attack and smooth release."""
    nbars = db.shape[1]
    x = db.astype(np.float32) + _tilt(nbars)
    if ceil is None:
        ceil = db_ceiling(db)
    rng = 40.0
    v = np.clip((x - (ceil - rng)) / rng, 0.0, 1.0) ** 2.2
    v = np.clip(v * float(sens), 0.0, 1.0)
    if nbars >= 3:
        v[:, 1:-1] = 0.2 * v[:, :-2] + 0.6 * v[:, 1:-1] + 0.2 * v[:, 2:]
    rel = float(np.clip(smooth, 0.0, 0.97)) ** (30.0 / fps)
    att = 0.75
    out = np.empty_like(v)
    prev = np.zeros(nbars, np.float32)
    for i in range(v.shape[0]):
        cur = v[i]
        prev = np.where(cur > prev, prev + (cur - prev) * att, prev * rel + cur * (1 - rel))
        out[i] = prev
    return out


def synthetic_levels(nframes, nbars, seed=7):
    rng = np.random.default_rng(seed)
    b = np.linspace(0, 1, nbars)
    shape = 0.55 * np.exp(-b * 4) + 0.25 * np.exp(-((b - 0.3) / 0.08) ** 2)
    v = shape[None, :] * (0.5 + rng.random((nframes, nbars)))
    return np.clip(v ** 1.6, 0, 1).astype(np.float32)


# ------------------------------------------------------------------- scene

class Scene:
    """Static layers + everything needed to produce the dynamic patches."""

    def __init__(self, cfg, H, fps, duration):
        cfg = merged(cfg)
        self.cfg = cfg
        self.H = H
        self.W = W = int(round(H * 16 / 9 / 2)) * 2
        self.fps = fps
        self.duration = max(0.04, duration)
        self.nframes_total = max(1, int(math.ceil(self.duration * fps - 1e-6)))
        s = self.s = H / 1440.0
        L = LAYOUT

        cover_src = square(load_image(cfg["cover"]))
        self.cover_src = cover_src

        # ---------------- background
        bg = self._background(cover_src)
        # post: vignette multiply + static grain
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx = (xx + 0.5 - W / 2) / (W / 2)
        dy = (yy + 0.5 - H / 2) / (H / 2)
        r = np.sqrt(dx * dx * 0.92 + dy * dy)
        t = np.clip((r - 0.35) / 1.1, 0, 1)
        self.V = (1.0 - float(cfg["vignette"]) * t * t * (3 - 2 * t)).astype(np.float32)
        del yy, xx, dx, dy, r, t
        g = float(cfg["grain"]) * 0.05
        self.G = (np.random.default_rng(1234).standard_normal((H, W), dtype=np.float32) * g) if g > 0 else np.zeros((H, W), np.float32)

        # ---------------- ground shadow
        full = bg
        sh = float(cfg["shadow"])
        if sh > 0:
            gx0 = (L["cover_x"] - 90) * s
            gx1 = (L["disc_cx"] + L["disc_r"] + 170) * s
            gy = (L["cover_y"] + L["cover_size"] + 6) * s
            pad = int(60 * s) + 4
            w = int(gx1 - gx0) + 2 * pad
            h = int(24 * s) + 2 * pad
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).ellipse((pad, pad, w - pad, h - pad), fill=255)
            m = m.filter(ImageFilter.GaussianBlur(16 * s))
            blend_into(full, int(gx0) - pad, int(gy - h / 2), np.zeros(3, np.float32), to_f(m) * sh)

        # ---------------- vinyl
        self.cx = cx = even_floor(L["disc_cx"] * s + 1)
        self.cy = cy = even_floor(L["disc_cy"] * s + 1)
        dr = L["disc_r"] * s
        cov_right = int(round(L["cover_x"] * s)) + int(round(L["cover_size"] * s))
        # the visible part of the record (the rest hides under the opaque sleeve) is redrawn every frame
        dx0 = even_floor(max(cx - dr - 2, cov_right - 4))
        dy0 = even_floor(cy - dr - 2)
        self.D = (dx0, dy0, even_ceil(cx + dr + 2) - dx0, even_ceil(cy + dr + 2) - dy0)
        dw, dh = self.D[2], self.D[3]
        self.under_D = full[dy0:dy0 + dh, dx0:dx0 + dw].copy()
        self._draw_disc(full)

        # ---------------- cover layer (drawn above the record)
        top_rgb, top_a, (tx, ty) = self._cover_layer(cover_src)
        top_D_rgb = np.zeros((dh, dw, 3), np.float32)
        top_D_a = np.zeros((dh, dw), np.float32)
        self._crop_layer(top_rgb, top_a, tx, ty, dx0, dy0, dw, dh, top_D_rgb, top_D_a)
        self.top_D = (top_D_rgb, top_D_a)

        # ---------------- label region (inside the record region, its own rotation cycle)
        self.label_r = L["label_r"] * s
        rn = 4 * int(math.ceil((self.label_r + 3) / 2.0))
        self.R = (cx - rn // 2, cy - rn // 2, rn, rn)
        rx, ry = self.R[0], self.R[1]
        assert rx >= dx0 and ry >= dy0
        self.under_R = full[ry:ry + rn, rx:rx + rn].copy()
        top_R_rgb = np.zeros((rn, rn, 3), np.float32)
        top_R_a = np.zeros((rn, rn), np.float32)
        self._crop_layer(top_rgb, top_a, tx, ty, rx, ry, rn, rn, top_R_rgb, top_R_a)
        self.top_R = (top_R_rgb, top_R_a)
        self._label_prep(cover_src)

        # full static frame (rotation 0) and the base frame for ffmpeg
        lab_rgb, lab_a = self._label_rgba(0.0)
        blend_into(full, rx, ry, lab_rgb, lab_a)
        blend_into(full, tx, ty, top_rgb, top_a)
        self.pre = full
        self.base = self._post(full, 0, 0)

        # one turn = lab_n frames; the record texture repeats every half turn
        rpm = float(cfg["rpm"])
        self.lab_n = max(1, int(round(fps * 60.0 / rpm))) if rpm > 0.01 else 1
        self.disc_n = self.lab_n // 2 if self.lab_n % 2 == 0 else self.lab_n
        self.lab_cache = self.disc_cache = None
        self.disc_cache_path = None

        # ---------------- UI strip (dynamic)
        self._ui_prep()

        # ---------------- atlas = what python sends per frame: UI strip on top, record below
        uw, uh = self.U[2], self.U[3]
        self.ady = uh + 8
        self.aw = max(uw, dw)
        self.ah = self.ady + dh
        self.atlas = np.zeros((self.ah, self.aw, 3), np.uint8)
        self.atlas[:uh, :uw] = self.P_under_u8
        self.prev_lev = None
        self.prev_sec = None
        self.levels_idx = None

    # ---- layers
    def _background(self, cover_src):
        cfg, W, H, s = self.cfg, self.W, self.H, self.s
        mode = cfg["bg_mode"]
        img = None
        if mode == "image" and cfg["bg_image"] and os.path.isfile(resolve(cfg["bg_image"])):
            img = cover_fit(load_image(cfg["bg_image"]), W, H)
            blur = float(cfg["bg_blur"]) * s
        elif mode == "cover":
            img = cover_fit(cover_src, W, H)
            blur = (40 + float(cfg["bg_blur"])) * s
        if img is not None:
            if blur > 0.3:
                small = img.resize((max(16, W // 4), max(9, H // 4)), Image.BILINEAR)
                small = small.filter(ImageFilter.GaussianBlur(blur / 4))
                img = small.resize((W, H), Image.BICUBIC)
            bg = to_f(img).copy()
            bg *= (1.0 - float(cfg["bg_darken"]))
            return bg
        c0, c1 = hex_rgb(cfg["bg_top"]), hex_rgb(cfg["bg_bottom"])
        if cfg["bg_style"] == "radial":
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            dx = (xx - W / 2) / (W * 0.62)
            dy = (yy + H * 0.12) / (H * 1.15)
            t = np.clip(np.sqrt(dx * dx + dy * dy), 0, 1)
            t = t * t * (3 - 2 * t)
            return (c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]).astype(np.float32)
        t = (np.arange(H, dtype=np.float32) + 0.5) / H
        col = c0[None, :] * (1 - t[:, None]) + c1[None, :] * t[:, None]
        return np.repeat(col[:, None, :], W, axis=1).astype(np.float32)

    def _disc_fields(self, X, Y):
        """The record at points X,Y (relative to its centre): static colour and
        alpha, plus the pieces of the texture that turns with it."""
        cfg, s, L = self.cfg, self.s, LAYOUT
        r = L["disc_r"] * s
        ring_r = L["ring_r"] * s
        d = np.hypot(X, Y)
        th = np.arctan2(Y, X)   # y points down, so the angle grows clockwise
        a = np.clip(r - d + 0.5, 0, 1).astype(np.float32)
        lum = np.zeros_like(d)
        # grooves: faint concentric lines
        P = 14.5 * s
        inner, outer = ring_r + 7 * s, r - 6 * s
        fr = np.mod(d - inner, P)
        dist = np.minimum(fr, P - fr)
        sig = max(0.55, 0.7 * s)
        lum += 0.03 * np.exp(-0.5 * (dist / sig) ** 2) * ((d > inner) & (d < outer))
        # rim highlight and a dark line around the label ring
        lum += 0.05 * np.exp(-0.5 * ((d - (r - 1.3 * s)) / max(0.6, 0.9 * s)) ** 2)
        lum -= 0.025 * np.exp(-0.5 * ((d - (ring_r + 2.5 * s)) / max(0.6, 1.2 * s)) ** 2)
        rgb = hex_rgb(cfg["disc_color"])[None, None, :] + lum[..., None]
        ra = np.clip(ring_r - d + 0.5, 0, 1)[..., None]
        rgb = (rgb * (1 - ra) + hex_rgb(cfg["ring_color"])[None, None, :] * ra).astype(np.float32)
        # turning texture only on the grooved part, never under the label patch
        zone = np.clip((d - (ring_r + 8 * s)) / (10 * s), 0, 1) * np.clip((r - 3 * s - d) / (6 * s), 0, 1)
        # glossy reflection, a bit stronger in the middle of the grooved area
        mid = 0.65 + 0.35 * np.sin(np.pi * np.clip((d - ring_r) / (r - ring_r), 0, 1))
        sheen = (float(cfg["disc_sheen"]) * 0.2 * zone * mid).astype(np.float32)
        # uneven grooves: sum of A_m(r) cos(m(th - turn) + P_m(r)), stored as cos/sin parts
        # so any rotation is just a weighted sum. Only even m: the texture repeats every half turn.
        rng = np.random.default_rng(77)
        rr = d / r
        terms = []
        for m in (2, 4, 6, 10):
            f1, f2, p1, p2 = rng.uniform(1.5, 5.0), rng.uniform(2.0, 7.0), rng.uniform(0, 6.3), rng.uniform(0, 6.3)
            amp = 0.018 * zone * (0.55 + 0.45 * np.sin(2 * np.pi * f1 * rr + p1))
            ph = 2.5 * np.sin(2 * np.pi * f2 * rr + p2)
            terms.append((m, (amp * np.cos(m * th + ph)).astype(np.float32),
                          (amp * np.sin(m * th + ph)).astype(np.float32)))
        return rgb, a, th.astype(np.float32), sheen, terms

    @staticmethod
    def _disc_turn(th, sheen, terms, turn):
        """Brightness of the turning texture after `turn` radians clockwise."""
        c2 = np.cos(th - turn + 0.75)
        c2 *= c2
        lum = sheen * (c2 * c2 * c2)       # two soft highlights opposite each other
        for m, C, S in terms:
            lum += C * math.cos(m * turn) + S * math.sin(m * turn)
        return lum

    def _draw_disc(self, full):
        s, L = self.s, LAYOUT
        r = L["disc_r"] * s
        n = 2 * int(math.ceil(r)) + 4
        x0, y0 = self.cx - n // 2, self.cy - n // 2
        c = (np.arange(n, dtype=np.float32) + 0.5 - n / 2)
        X, Y = np.meshgrid(c, c)
        rgb, a, th, sheen, terms = self._disc_fields(X, Y)
        blend_into(full, x0, y0, rgb + self._disc_turn(th, sheen, terms, 0.0)[..., None], a)
        # the same fields over the redrawn region
        dx, dy, dw, dh = self.D
        X, Y = np.meshgrid(np.arange(dx, dx + dw, dtype=np.float32) + 0.5 - self.cx,
                           np.arange(dy, dy + dh, dtype=np.float32) + 0.5 - self.cy)
        self.disc_D = self._disc_fields(X, Y)

    def _turn(self, k, n):
        return 2 * math.pi * k / n * (1 if int(self.cfg["spin_dir"]) >= 0 else -1)

    def disc_patch(self, k):
        k %= self.disc_n
        if self.disc_cache is not None and self.disc_have[k]:
            return self.disc_cache[k]
        rgb, a, th, sheen, terms = self.disc_D
        lum = self._disc_turn(th, sheen, terms, self._turn(k, self.lab_n))
        c = self.under_D.copy()
        c += (rgb + lum[..., None] - c) * a[..., None]
        trgb, ta = self.top_D
        c += (trgb - c) * ta[..., None]
        out = self._post(c, self.D[0], self.D[1])
        if self.disc_cache is not None:
            self.disc_cache[k] = out
            self.disc_have[k] = True
        return out

    def prefill(self, report=None, cancel=None):
        """Pre-render one turn of the record and the label (in parallel threads)."""
        from concurrent.futures import ThreadPoolExecutor
        rn = self.R[2]
        dw, dh = self.D[2], self.D[3]
        self.lab_cache = np.empty((self.lab_n, rn, rn, 3), np.uint8)
        self.lab_have = np.zeros(self.lab_n, bool)
        shape = (self.disc_n, dh, dw, 3)
        if self.disc_n * dh * dw * 3 > 600e6:     # big (4K / 60 fps / slow spin): keep it in a file
            os.makedirs(os.path.join(CACHE_DIR, "job"), exist_ok=True)
            self.disc_cache_path = os.path.join(CACHE_DIR, "job", "disc_cycle.bin")
            self.disc_cache = np.memmap(self.disc_cache_path, np.uint8, "w+", shape=shape)
        else:
            self.disc_cache = np.empty(shape, np.uint8)
        self.disc_have = np.zeros(self.disc_n, bool)
        tasks = [(self.disc_patch, k) for k in range(self.disc_n)] + [(self.label_patch, k) for k in range(self.lab_n)]
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            futs = [ex.submit(f, k) for f, k in tasks]
            for n, fu in enumerate(futs):
                if cancel is not None and cancel.is_set():
                    ex.shutdown(cancel_futures=True)
                    raise RuntimeError("Отменено")
                fu.result()
                if report and n % 25 == 0:
                    report("build", n / len(futs), {})

    def release(self):
        """Free the pre-rendered turn (and its temp file, if any)."""
        self.disc_cache = self.lab_cache = None
        if self.disc_cache_path:
            import gc
            gc.collect()
            try:
                os.remove(self.disc_cache_path)
            except OSError:
                pass
            self.disc_cache_path = None

    def _cover_layer(self, cover_src):
        cfg, s, L = self.cfg, self.s, LAYOUT
        cs = int(round(L["cover_size"] * s))
        cx0, cy0 = int(round(L["cover_x"] * s)), int(round(L["cover_y"] * s))
        m = int(50 * s) + 4
        w = cs + 2 * m
        rgb = np.zeros((w, w, 3), np.float32)
        a = np.zeros((w, w), np.float32)
        # drop shadow of the sleeve
        sm = Image.new("L", (w, w), 0)
        off = int(round(6 * s))
        ImageDraw.Draw(sm).rectangle((m + off, m + off, m + off + cs - 1, m + off + cs - 1), fill=255)
        sm = to_f(sm.filter(ImageFilter.GaussianBlur(12 * s))) * 0.5
        a[:] = sm
        # sleeve
        cov = cover_src.resize((cs, cs), Image.LANCZOS).convert("RGB")
        cov_f = to_f(cov).copy()
        # sticker / logo
        if cfg["sticker"] and cfg["sticker_path"] and os.path.isfile(resolve(cfg["sticker_path"])):
            st = load_image(cfg["sticker_path"], "RGBA")
            sw = max(6, int(round(cs * 0.053)))
            sh_ = max(4, int(round(sw * st.size[1] / st.size[0])))
            st = st.resize((sw, sh_), Image.LANCZOS)
            stf = to_f(st)
            blend_into(cov_f, int(round(cs * 0.128)), int(round(cs * 0.962)) - sh_, stf[..., :3], stf[..., 3])
        # title
        title = (cfg["title"] or "").replace("\r", "")
        if title.strip():
            size = float(cfg["title_size"]) * s
            font = get_font(cfg["title_font"], size)
            tim = text_image(title, font, int(size * 0.25))
            if tim.size[0] > cs * 0.92:   # auto-fit long titles
                font = get_font(cfg["title_font"], size * cs * 0.92 / tim.size[0])
                tim = text_image(title, font, int(size * 0.25))
            tx = cs // 2 - tim.size[0] // 2
            ty = cs // 2 - tim.size[1] // 2 + int(round(float(cfg["title_offset"]) * s))
            shv = float(cfg["title_shadow"])
            if shv > 0:
                p = int(10 * s) + 2
                big = Image.new("L", (tim.size[0] + 2 * p, tim.size[1] + 2 * p), 0)
                big.paste(tim, (p, p))
                big = to_f(big.filter(ImageFilter.GaussianBlur(4 * s))) * shv
                blend_into(cov_f, tx - p, ty - p + int(round(2 * s)), np.zeros(3, np.float32), big)
            blend_into(cov_f, tx, ty, hex_rgb(cfg["title_color"]), to_f(tim))
        rgb[m:m + cs, m:m + cs] = cov_f
        a[m:m + cs, m:m + cs] = 1.0
        return rgb, a, (cx0 - m, cy0 - m)

    @staticmethod
    def _crop_layer(rgb, a, lx, ly, x, y, w, h, out_rgb, out_a):
        """Copy the part of a layer (at lx,ly) that falls inside rect x,y,w,h."""
        H_, W_ = a.shape
        sx0, sy0 = max(x, lx), max(y, ly)
        sx1, sy1 = min(x + w, lx + W_), min(y + h, ly + H_)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        out_rgb[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = rgb[sy0 - ly:sy1 - ly, sx0 - lx:sx1 - lx]
        out_a[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = a[sy0 - ly:sy1 - ly, sx0 - lx:sx1 - lx]

    def _label_prep(self, cover_src):
        rn = self.R[2]
        self.lab_img = cover_src.resize((rn, rn), Image.LANCZOS).convert("RGB")
        c = np.arange(rn, dtype=np.float32) + 0.5 - rn / 2
        X, Y = np.meshgrid(c, c)
        d = np.hypot(X, Y)
        lr = self.label_r
        self.lab_mask = np.clip(lr - d + 0.5, 0, 1).astype(np.float32)
        edge = 1.0 - 0.35 * np.exp(-0.5 * ((d - lr) / max(0.6, 1.2 * self.s)) ** 2)
        hole = np.clip(LAYOUT["hole_r"] * self.s - d + 0.5, 0, 1)
        self.lab_shade = (edge * (1 - hole)).astype(np.float32)[..., None]

    def _label_rgba(self, angle):
        im = self.lab_img.rotate(angle, resample=Image.BICUBIC) if angle else self.lab_img
        return to_f(im) * self.lab_shade, self.lab_mask

    def label_patch(self, k):
        k %= self.lab_n
        if self.lab_cache is not None and self.lab_have[k]:
            return self.lab_cache[k]
        angle = -360.0 * k / self.lab_n * (1 if int(self.cfg["spin_dir"]) >= 0 else -1)
        rgb, a = self._label_rgba(angle)
        c = self.under_R.copy()
        c += (rgb - c) * a[..., None]
        trgb, ta = self.top_R
        c += (trgb - c) * ta[..., None]
        out = self._post(c, self.R[0], self.R[1])
        if self.lab_cache is not None:
            self.lab_cache[k] = out
            self.lab_have[k] = True
        return out

    def _post_f(self, c, x, y):
        h, w = c.shape[:2]
        return np.clip(c * self.V[y:y + h, x:x + w, None] + self.G[y:y + h, x:x + w, None], 0, 1)

    def _post(self, c, x, y):
        return (self._post_f(c, x, y) * 255.0 + 0.5).astype(np.uint8)

    def _post_col(self, col, x, y, w, h):
        return np.clip(col[None, None, :] * self.V[y:y + h, x:x + w, None] + self.G[y:y + h, x:x + w, None], 0, 1)

    # ---- UI strip
    def _ui_prep(self):
        cfg, s, L = self.cfg, self.s, LAYOUT
        x0, x1 = L["ui_x0"] * s, L["ui_x1"] * s
        bar_y = L["bar_y"] * s
        self.hours = self.duration >= 3600

        # timer metrics
        tsize = float(cfg["timer_size"]) * s
        self.tfont = font = get_font(cfg["timer_font"], tsize)
        digits = "0123456789"
        dw = [font.getlength(d) for d in digits]
        wide = digits[int(np.argmax(dw))]
        tpl = fmt_time(self.duration, self.hours)
        tpl = (tpl + " - " + tpl) if cfg["timer_total"] else tpl
        twmax = font.getlength("".join(wide if ch.isdigit() else ch for ch in tpl))
        self.timer_x0 = x1 - twmax
        db = font.getbbox("0:", anchor="ls")
        self.timer_base = bar_y - (db[1] + db[3]) / 2.0
        asc, desc = font.getmetrics()
        t_top = self.timer_base - asc - 2
        t_bot = self.timer_base + desc + 2

        played_th = max(3.0, L["played_th"] * s)
        track_th = max(2.0, L["track_th"] * s)
        self.bar_x0 = x0
        self.bar_x1 = max(x0 + 40 * s, self.timer_x0 - L["timer_gap"] * s)
        self.played_r = played_th / 2
        self.track_r = track_th / 2
        self.bar_y = bar_y

        # waveform geometry
        nb = int(cfg["wave_bars"])
        pitch = (x1 - x0) / (nb - 1 + 0.68)
        dot = 0.68 * pitch
        hmax = max(dot + 2, float(cfg["wave_height"]) * s)
        wy = L["wave_y"] * s
        self.wave_dot, self.wave_hmax, self.wave_y = dot, hmax, wy
        cxs = x0 + dot / 2 + np.arange(nb) * pitch
        sw = int(math.floor(pitch))
        starts = np.floor(cxs - sw / 2).astype(int)

        # region U
        top = min(t_top, bar_y - played_th) - 2
        bottom = wy + hmax / 2 + 3
        left = min(x0, starts[0]) - 4
        right = max(x1, starts[-1] + sw) + 4
        ux, uy = even_floor(left), even_floor(top)
        uw, uh = even_ceil(right - ux), even_ceil(bottom - uy)
        self.U = (ux, uy, uw, uh)
        under = self.pre[uy:uy + uh, ux:ux + uw]
        Pu = self._post_f(under, ux, uy)
        self.P_under_u8 = (Pu * 255 + 0.5).astype(np.uint8)

        # progress bar rows
        by0 = int(math.floor(bar_y - played_th / 2)) - 2 - uy
        by1 = int(math.ceil(bar_y + played_th / 2)) + 2 - uy
        bx1 = min(uw, int(math.ceil(self.bar_x1)) + 3 - ux)
        self.bar_rows = (by0, by1, bx1)
        ys = np.arange(uy + by0, uy + by1, dtype=np.float32) + 0.5
        xs = np.arange(ux, ux + bx1, dtype=np.float32) + 0.5
        X, Y = np.meshgrid(xs, ys)
        Pb = Pu[by0:by1, :bx1]
        trk = hex_rgb(cfg["track_color"])
        a_tr = capsule_alpha(X, Y, x0 + self.track_r, bar_y, self.bar_x1 - self.track_r, bar_y, self.track_r)
        a_tr *= float(cfg["track_opacity"])
        Pt_col = self._post_col(trk, ux, uy + by0, bx1, by1 - by0)
        self.P_track = Pb + (Pt_col - Pb) * a_tr[..., None]
        self.P_acc = self._post_col(hex_rgb(cfg["accent"]), ux, uy + by0, bx1, by1 - by0)
        a_pl = capsule_alpha(X, Y, x0 + self.played_r, bar_y, self.bar_x1 - self.played_r, bar_y, self.played_r)
        self.P_played_u8 = ((self.P_track + (self.P_acc - self.P_track) * a_pl[..., None]) * 255 + 0.5).astype(np.uint8)
        self.P_track_u8 = (self.P_track * 255 + 0.5).astype(np.uint8)
        self.bar_X, self.bar_Y = X, Y

        # timer box
        tb_x0 = max(0, int(math.floor(self.timer_x0)) - 3 - ux)
        tb_x1 = uw
        tb_y0 = max(0, int(math.floor(t_top)) - uy)
        tb_y1 = min(uh, int(math.ceil(t_bot)) - uy)
        self.tbox = (tb_x0, tb_y0, tb_x1, tb_y1)
        self.P_tbox = Pu[tb_y0:tb_y1, tb_x0:tb_x1]
        self.P_tcol = self._post_col(hex_rgb(cfg["accent"]), ux + tb_x0, uy + tb_y0, tb_x1 - tb_x0, tb_y1 - tb_y0)

        # waveform sprites: every bar slot x every height level
        wy0 = int(math.floor(wy - hmax / 2)) - 2 - uy
        wy1 = int(math.ceil(wy + hmax / 2)) + 2 - uy
        self.wave_rows = (wy0, wy1)
        nlev = min(128, max(2, int(round(hmax - dot)) + 1))   # ~1px steps, capped to bound memory
        self.nlev = nlev
        heights = dot + np.arange(nlev, dtype=np.float32) * (hmax - dot) / (nlev - 1)
        hh = wy1 - wy0
        ys = (np.arange(wy0, wy1, dtype=np.float32) + uy + 0.5)[None, :, None]
        half = ((heights - dot) / 2)[:, None, None]
        wcol = hex_rgb(cfg["wave_color"])
        wop = float(cfg["wave_opacity"])
        spr = np.empty((nb, nlev, hh, sw, 3), np.uint8)
        for b in range(nb):
            st = starts[b] - ux
            xs_ = (np.arange(sw, dtype=np.float32) + starts[b] + 0.5)[None, None, :]
            dy = np.maximum(np.abs(ys - wy) - half, 0)
            dd = np.hypot(xs_ - cxs[b], dy) - dot / 2
            a = np.clip(0.5 - dd, 0, 1) * wop
            base = Pu[wy0:wy1, st:st + sw]
            pc = self._post_col(wcol, starts[b], uy + wy0, sw, hh)
            spr[b] = (base[None] + (pc - base)[None] * a[..., None]) * 255 + 0.5
        self.spr = spr
        self.slot_x = (starts - ux).tolist()
        self.slot_w = sw

    def set_levels(self, levels):
        """levels: float (frames, bars) 0..1 for the whole track."""
        dt = np.uint8 if self.nlev <= 256 else np.uint16
        self.levels_idx = np.rint(np.clip(levels, 0, 1) * (self.nlev - 1)).astype(dt)

    # ---- per frame
    def fill_atlas(self, i):
        A = self.atlas
        # waveform: only slots whose height changed
        li = min(i, self.levels_idx.shape[0] - 1)
        lev = self.levels_idx[li]
        if self.prev_lev is None:
            changed = range(len(lev))
        else:
            changed = np.flatnonzero(lev != self.prev_lev).tolist()
        wy0, wy1 = self.wave_rows
        sw, spr, sx = self.slot_w, self.spr, self.slot_x
        for b in changed:
            x = sx[b]
            A[wy0:wy1, x:x + sw] = spr[b, lev[b]]
        self.prev_lev = lev

        # progress bar
        by0, by1, bx1 = self.bar_rows
        ux = self.U[0]
        p = min(1.0, i / max(1, self.nframes_total - 1))
        length = self.bar_x1 - self.bar_x0
        xe = self.bar_x0 + max(length * p, 2 * self.played_r)
        w0 = int(min(bx1, max(0, math.floor(xe - 2 * self.played_r) - 2 - ux)))
        w1 = int(min(bx1, max(0, math.ceil(xe) + 2 - ux)))
        A[by0:by1, :w0] = self.P_played_u8[:, :w0]
        A[by0:by1, w0:bx1] = self.P_track_u8[:, w0:bx1]
        if w1 > w0:
            X, Y = self.bar_X[:, w0:w1], self.bar_Y[:, w0:w1]
            a = capsule_alpha(X, Y, self.bar_x0 + self.played_r, self.bar_y, xe - self.played_r, self.bar_y, self.played_r)
            pt = self.P_track[:, w0:w1]
            A[by0:by1, w0:w1] = ((pt + (self.P_acc[:, w0:w1] - pt) * a[..., None]) * 255 + 0.5).astype(np.uint8)

        # timer (changes once a second)
        sec = int(i / self.fps)
        if sec != self.prev_sec:
            self.prev_sec = sec
            txt = fmt_time(sec, self.hours)
            if self.cfg["timer_total"]:
                txt += " - " + fmt_time(self.duration, self.hours)
            tb_x0, tb_y0, tb_x1, tb_y1 = self.tbox
            m = Image.new("L", (tb_x1 - tb_x0, tb_y1 - tb_y0), 0)
            ImageDraw.Draw(m).text((self.timer_x0 - (self.U[0] + tb_x0), self.timer_base - (self.U[1] + tb_y0)),
                                   txt, font=self.tfont, fill=255, anchor="ls")
            a = to_f(m)[..., None]
            A[tb_y0:tb_y1, tb_x0:tb_x1] = ((self.P_tbox + (self.P_tcol - self.P_tbox) * a) * 255 + 0.5).astype(np.uint8)

        # spinning record, then the label on top of it
        dx, dy, dw, dh = self.D
        ady = self.ady
        A[ady:ady + dh, :dw] = self.disc_patch(i)
        rx, ry, rn, _ = self.R
        lx, ly = rx - dx, ady + ry - dy
        A[ly:ly + rn, lx:lx + rn] = self.label_patch(i)
        return A

    def compose(self, i):
        """Full RGB frame (used for previews)."""
        A = self.fill_atlas(i)
        f = self.base.copy()
        ux, uy, uw, uh = self.U
        f[uy:uy + uh, ux:ux + uw] = A[:uh, :uw]
        dx, dy, dw, dh = self.D
        f[dy:dy + dh, dx:dx + dw] = A[self.ady:self.ady + dh, :dw]
        return f


# ------------------------------------------------------------------ encode

_encoders = None


def available_encoders():
    global _encoders
    if _encoders is not None:
        return _encoders
    ffmpeg = find_ffmpeg()
    res = {}
    for name in ["h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_amf", "hevc_amf"]:
        r = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.1",
                            "-c:v", name, "-f", "null", "-"], capture_output=True, creationflags=NO_WINDOW)
        res[name] = r.returncode == 0
    # compositing on the GPU (base frame lives in video memory, nvenc reads it directly)
    r = subprocess.run([ffmpeg, "-v", "error", "-init_hw_device", "cuda=cu", "-filter_hw_device", "cu",
                        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.1",
                        "-f", "lavfi", "-i", "color=c=white:s=32x32:d=0.1",
                        "-filter_complex", "[0:v]format=yuv420p,hwupload_cuda[a];[1:v]format=yuv420p,hwupload_cuda[b];"
                                           "[a][b]overlay_cuda=8:8[v]",
                        "-map", "[v]", "-c:v", "h264_nvenc", "-f", "null", "-"],
                       capture_output=True, creationflags=NO_WINDOW)
    res["cuda"] = r.returncode == 0
    _encoders = res
    return res


def pick_encoder(codec):
    enc = available_encoders()
    for hw in ("nvenc", "qsv", "amf"):
        name = "%s_%s" % (codec, hw)
        if enc.get(name):
            return name
    if codec != "h264":
        return pick_encoder("h264")
    return "libx264"


def encoder_args(name, q, fps, zero_delay):
    """Constant QP: static areas cost almost nothing, moving parts keep quality.
    P-frames get a slightly higher QP so the static grain is not re-coded every frame."""
    g = str(int(fps) * 10)   # long GOP: the picture is mostly static, keyframes are the main cost
    q = int(q)
    if name.endswith("_nvenc"):
        a = ["-c:v", name, "-preset", "p2", "-tune", "hq", "-rc", "constqp",
             "-init_qpI", str(q), "-init_qpP", str(q + 3), "-init_qpB", str(q + 5), "-g", g]
        if zero_delay:
            # GPU frames must be encoded in order, right away: no B-frames / lookahead
            a += ["-bf", "0", "-rc-lookahead", "0"]
        if name.startswith("h264"):
            a += ["-profile:v", "high"]
        return a
    if name.endswith("_qsv"):
        return ["-c:v", name, "-preset", "faster", "-global_quality", str(q + 3), "-g", g]
    if name.endswith("_amf"):
        return ["-c:v", name, "-quality", "speed", "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q + 3), "-g", g]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(q + 2), "-g", g]


def _encode(sc, cfg, out_path, f0, f1, enc, gpu, audio_copy, report, cancel, tmpdir, safe=False):
    """Pipe the per-frame patches into ffmpeg. Returns (ok, error_text, seconds)."""
    fps = sc.fps
    nframes = f1 - f0
    base_png = os.path.join(tmpdir, "base.png")
    Image.fromarray(sc.base).save(base_png, compress_level=1)
    ffmpeg = find_ffmpeg()
    ux, uy, uw, uh = sc.U
    dx, dy, dw, dh = sc.D
    conv = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
    if gpu:
        # base frame stays in video memory; only the two patches are uploaded per frame
        fc = ("[0:v]%s,hwupload_cuda,loop=loop=-1:size=1:start=0,settb=1/%d,setpts=N[bg];"
              "[1:v]split=2[a][b];[a]crop=%d:%d:0:0,%s,hwupload_cuda[ui];[b]crop=%d:%d:0:%d,%s,hwupload_cuda[dk];"
              "[bg][ui]overlay_cuda=%d:%d:shortest=1[t];[t][dk]overlay_cuda=%d:%d[v]") % (
            conv, fps, uw, uh, conv, dw, dh, sc.ady, conv, ux, uy, dx, dy)
    else:
        fc = ("[0:v]%s,loop=loop=-1:size=1:start=0,settb=1/%d,setpts=N[bg];"
              "[1:v]split=2[a][b];[a]crop=%d:%d:0:0,%s[ui];[b]crop=%d:%d:0:%d,%s[dk];"
              "[bg][ui]overlay=%d:%d:shortest=1:format=yuv420[t];"
              "[t][dk]overlay=%d:%d:format=yuv420,format=yuv420p[v]") % (
            conv, fps, uw, uh, conv, dw, dh, sc.ady, conv, ux, uy, dx, dy)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if gpu:
        cmd += ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu"]
    cmd += ["-i", base_png,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "%dx%d" % (sc.aw, sc.ah),
            "-framerate", str(fps), "-i", "pipe:0"]
    if f0 > 0:
        cmd += ["-ss", "%.3f" % (f0 / fps)]
    cmd += ["-i", resolve(cfg["audio"]), "-filter_complex", fc, "-map", "[v]", "-map", "2:a:0", "-r", str(fps)]
    cmd += encoder_args(enc, cfg["quality"], fps, zero_delay=gpu)
    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    cmd += ["-c:a", "copy"] if audio_copy else ["-c:a", "aac", "-b:a", "320k"]
    cmd += ["-t", "%.3f" % (nframes / fps), out_path]

    # fresh state: the first frame redraws every patch
    sc.prev_lev = None
    sc.prev_sec = None
    log_path = os.path.join(tmpdir, "ffmpeg.log")
    logf = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logf,
                            bufsize=0, creationflags=NO_WINDOW)
    write = proc.stdin.write
    t_start = time.time()
    last = 0.0
    ok = False
    try:
        for i in range(f0, f1):
            if cancel.is_set():
                break
            write(sc.fill_atlas(i).data)
            now = time.time()
            if now - last > 0.25:
                last = now
                done = i - f0 + 1
                el = now - t_start
                speed = done / el if el > 0 else 0
                report("render", done / nframes, {"fps": speed, "elapsed": el, "safe": safe,
                                                  "eta": (nframes - done) / speed if speed else 0,
                                                  "frame": done, "frames": nframes})
        if not cancel.is_set():
            proc.stdin.close()
            proc.wait()
            ok = proc.returncode == 0
    except (BrokenPipeError, OSError):
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        logf.close()
    err = ""
    if not ok:
        try:
            with open(log_path, "rb") as fh:
                err = fh.read().decode("utf8", "replace")[-1500:]
        except Exception:
            pass
    return ok, err, time.time() - t_start


def verify_output(sc, path, f0, nframes, samples=6, report=None):
    """Decode a few frames of the finished video and check that each one shows
    exactly the frame it should (spinning label + waveform + timer)."""
    if nframes < 16:
        return True
    ffmpeg = find_ffmpeg()
    fps = sc.fps
    ux, uy, uw, uh = sc.U
    dx, dy, dw, dh = sc.D
    W, H = sc.W, sc.H
    bad = 0
    ks = np.unique(np.linspace(5, nframes - 6, samples).astype(int))
    for n, k in enumerate(ks):
        if report:
            report("verify", n / len(ks), {})
        t = max(0.0, (k - 0.3) / fps)
        p = subprocess.run([ffmpeg, "-v", "error", "-ss", "%.4f" % t, "-i", path, "-frames:v", "1", "-an",
                            "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
                            "-f", "rawvideo", "-"], capture_output=True, creationflags=NO_WINDOW)
        if len(p.stdout) != W * H * 3:
            bad += 1
            continue
        fr = np.frombuffer(p.stdout, np.uint8).reshape(H, W, 3).astype(np.int16)
        d_disc = fr[dy:dy + dh, dx:dx + dw]
        d_ui = fr[uy:uy + uh, ux:ux + uw]
        errs = []
        for j in range(k - 3, k + 4):
            sc.prev_lev = None
            sc.prev_sec = None
            A = sc.fill_atlas(f0 + j)
            errs.append(np.abs(d_disc - A[sc.ady:sc.ady + dh, :dw]).mean() + np.abs(d_ui - A[:uh, :uw]).mean())
        if errs[3] > min(errs) + 0.3 or errs[3] > 15:
            bad += 1
    return bad == 0


def render(cfg, out_path, progress=None, cancel=None, start=0.0, length=None, force_cpu=False):
    """Render the video. start/length (seconds) render only a piece (test clip).
    progress(stage, fraction, info) is called along the way."""
    cfg = merged(cfg)
    cancel = cancel or threading.Event()
    report = progress or (lambda *a, **k: None)
    if not cfg["audio"] or not os.path.isfile(resolve(cfg["audio"])):
        raise RuntimeError("Выберите аудиофайл")
    if not cfg["cover"] or not os.path.isfile(resolve(cfg["cover"])):
        raise RuntimeError("Выберите обложку")
    fps = int(cfg["fps"])
    H = RESOLUTIONS.get(cfg["resolution"], 1440)
    nb = int(cfg["wave_bars"])
    t0 = time.time()

    report("analyze", 0.0, {})
    an = analyze_audio(cfg["audio"], fps, nb, progress=lambda f: report("analyze", f, {}), cancel=cancel)
    duration = an["duration"]
    report("build", 0.0, {})
    lv = levels_from_db(an["db"], fps, cfg["wave_sens"], cfg["wave_smooth"])
    sc = Scene(cfg, H, fps, duration)
    sc.set_levels(lv)
    try:
        return _render_scene(sc, cfg, out_path, report, cancel, start, length, force_cpu, t0)
    finally:
        sc.release()


def _render_scene(sc, cfg, out_path, report, cancel, start, length, force_cpu, t0):
    fps = sc.fps
    sc.prefill(report, cancel)

    f0 = int(round(start * fps))
    f0 = min(max(0, f0), sc.nframes_total - 1)
    f1 = sc.nframes_total if length is None else min(sc.nframes_total, f0 + int(round(length * fps)))
    nframes = f1 - f0

    tmpdir = os.path.join(CACHE_DIR, "job")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ainfo = probe_audio(cfg["audio"])
    audio_copy = ainfo["codec"] == "aac" and length is None and f0 == 0
    codec = cfg.get("codec", "h264")
    if codec not in ("h264", "hevc"):
        codec = "h264"
    enc = pick_encoder(codec)
    gpu = (not force_cpu) and enc.endswith("_nvenc") and bool(available_encoders().get("cuda"))

    ok, err, secs = _encode(sc, cfg, out_path, f0, f1, enc, gpu, audio_copy, report, cancel, tmpdir)
    verified = None
    if ok and gpu:
        verified = verify_output(sc, out_path, f0, nframes, report=report)
        if not verified and not cancel.is_set():
            # fall back to compositing on the CPU (slower, but frame-exact with any encoder)
            gpu = False
            ok, err, secs = _encode(sc, cfg, out_path, f0, f1, enc, False, audio_copy, report, cancel, tmpdir, safe=True)
            verified = verify_output(sc, out_path, f0, nframes, report=report) if ok else None
    if not ok:
        if cancel.is_set():
            try:
                os.remove(out_path)
            except OSError:
                pass
            raise RuntimeError("Отменено")
        raise RuntimeError("ffmpeg завершился с ошибкой:\n" + err)
    info = {"output": out_path, "frames": nframes, "seconds": time.time() - t0,
            "encoder": enc + (" + CUDA" if gpu else ""), "verified": verified,
            "render_fps": nframes / max(1e-6, secs), "duration": nframes / fps}
    report("done", 1.0, info)
    return info


def preview_image(cfg, t_frac=0.35, H=720):
    """One still frame for the settings preview, rendered by the same code."""
    cfg = merged(cfg)
    fps = int(cfg["fps"])
    nb = int(cfg["wave_bars"])
    an = None
    if cfg["audio"] and os.path.isfile(resolve(cfg["audio"])):
        an = cached_analysis(cfg["audio"], fps, nb)
    duration = an["duration"] if an else 3 * 60 + 25.0
    sc = Scene(cfg, H, fps, duration)
    i = int(t_frac * (sc.nframes_total - 1))
    if an:
        lo = max(0, i - fps * 3)
        lv = levels_from_db(an["db"][lo:i + 1], fps, cfg["wave_sens"], cfg["wave_smooth"], db_ceiling(an["db"]))
        full = np.zeros((i + 1, nb), np.float32)
        full[lo:] = lv
    else:
        full = synthetic_levels(i + 1, nb)
    sc.set_levels(full)
    return Image.fromarray(sc.compose(i)), (an is not None)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("settings")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--length", type=float, default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    with open(args.settings, encoding="utf8") as fh:
        cfg = json.load(fh)
    if args.preview:
        im, _ = preview_image(cfg, args.start, H=RESOLUTIONS.get(merged(cfg)["resolution"], 1440))
        im.save(args.out)
        sys.exit(0)

    def pr(stage, f, info):
        if stage == "render":
            sys.stdout.write("\r%s %5.1f%%  %.0f fps  ETA %.0fs   " % (stage, f * 100, info["fps"], info["eta"]))
        else:
            sys.stdout.write("\r%s %5.1f%%          " % (stage, f * 100))
        sys.stdout.flush()

    res = render(cfg, args.out, pr, start=args.start, length=args.length, force_cpu=args.cpu)
    print("\n", res)
