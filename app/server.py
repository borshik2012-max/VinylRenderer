"""Local web UI for the Vinyl Visualizer. Run start.bat, the browser opens by itself."""
import base64
import http.server
import io
import json
import mimetypes
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

UI_FILE = os.path.join(E.APP_DIR, "ui.html")
SETTINGS_FILE = os.path.join(E.ROOT, "settings.json")
PRESETS_FILE = os.path.join(E.ROOT, "my_presets.json")
TOKEN = secrets.token_hex(16)

state_lock = threading.Lock()
job = {"id": 0, "stage": "idle", "progress": 0.0, "info": {}, "error": None, "kind": None, "output": None}
job_cancel = threading.Event()
audio_job = {"key": None, "stage": "idle", "progress": 0.0, "duration": None, "error": None}
encoders = {}


# ------------------------------------------------------------------ helpers

def load_json(path, default):
    try:
        with open(path, encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def rel_or_abs(path):
    """Store files inside VinylRenderer as relative paths (portable folder)."""
    if not path:
        return ""
    ap = os.path.abspath(path)
    root = os.path.abspath(E.ROOT)
    if ap.lower().startswith(root.lower() + os.sep):
        return os.path.relpath(ap, root).replace("\\", "/")
    return ap.replace("\\", "/")


_fonts = None


def list_fonts():
    global _fonts
    if _fonts is not None:
        return _fonts
    out, seen = [], set()
    fdir = os.path.join(E.ROOT, "fonts")
    for f in sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []:
        if f.lower().endswith((".ttf", ".otf", ".ttc")):
            p = os.path.join(fdir, f)
            try:
                fam, sty = E.ImageFont.truetype(p, 12).getname()
                name = fam + ("" if sty in ("Regular", "Normal") else " " + sty)
            except Exception:
                name = os.path.splitext(f)[0]
            out.append({"name": "★ " + name, "path": "fonts/" + f})
            seen.add(os.path.abspath(p).lower())
    if os.name == "nt":
        import winreg
        windir = os.environ.get("WINDIR", r"C:\Windows")
        sources = [(winreg.HKEY_LOCAL_MACHINE, os.path.join(windir, "Fonts")),
                   (winreg.HKEY_CURRENT_USER, os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"))]
        for hive, base in sources:
            try:
                key = winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
            except OSError:
                continue
            i = 0
            while True:
                try:
                    name, val, _ = winreg.EnumValue(key, i)
                except OSError:
                    break
                i += 1
                if not isinstance(val, str) or not val.lower().endswith((".ttf", ".otf", ".ttc")):
                    continue
                p = val if os.path.isabs(val) else os.path.join(base, val)
                if not os.path.isfile(p) or os.path.abspath(p).lower() in seen:
                    continue
                seen.add(os.path.abspath(p).lower())
                name = name.replace(" (TrueType)", "").replace(" (OpenType)", "").strip()
                out.append({"name": name, "path": p.replace("\\", "/")})
    head = [f for f in out if f["name"].startswith("★")]
    rest = sorted([f for f in out if not f["name"].startswith("★")], key=lambda f: f["name"].lower())
    _fonts = head + rest
    return _fonts


pick_lock = threading.Lock()
FILETYPES = {
    "audio": [("Аудио", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.wma *.aif *.aiff"), ("Все файлы", "*.*")],
    "image": [("Картинки", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Все файлы", "*.*")],
    "font": [("Шрифты", "*.ttf *.otf *.ttc"), ("Все файлы", "*.*")],
}


def pick_file(kind, current=""):
    import tkinter as tk
    from tkinter import filedialog
    with pick_lock:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        cur = E.resolve(current) if current else ""
        initdir = os.path.dirname(cur) if cur and os.path.isdir(os.path.dirname(cur)) else os.path.expanduser("~")
        try:
            if kind == "output":
                name = os.path.basename(cur) if cur else "video.mp4"
                path = filedialog.asksaveasfilename(parent=root, title="Куда сохранить видео", defaultextension=".mp4",
                                                    filetypes=[("MP4 видео", "*.mp4")], initialdir=initdir,
                                                    initialfile=name)
            else:
                titles = {"audio": "Выберите аудио", "image": "Выберите картинку", "font": "Выберите шрифт"}
                path = filedialog.askopenfilename(parent=root, title=titles.get(kind, "Файл"),
                                                  filetypes=FILETYPES.get(kind, [("Все файлы", "*.*")]),
                                                  initialdir=initdir)
        finally:
            root.destroy()
    return path or ""


def default_output(cfg):
    base = os.path.splitext(os.path.basename(E.resolve(cfg.get("audio", ""))))[0] or "video"
    base = "".join(c for c in base if c not in '<>:"/\\|?*').strip() or "video"
    os.makedirs(E.OUTPUT_DIR, exist_ok=True)
    p = os.path.join(E.OUTPUT_DIR, base + ".mp4")
    n = 2
    while os.path.exists(p):
        p = os.path.join(E.OUTPUT_DIR, "%s (%d).mp4" % (base, n))
        n += 1
    return p


# ------------------------------------------------------------------ jobs

def start_audio_analysis(cfg):
    cfg = E.merged(cfg)
    audio = cfg["audio"]
    if not audio or not os.path.isfile(E.resolve(audio)):
        with state_lock:
            audio_job.update(key=None, stage="idle", progress=0.0, duration=None, error=None)
        return
    key = (audio, int(cfg["fps"]), int(cfg["wave_bars"]))
    with state_lock:
        if audio_job["key"] == key and audio_job["stage"] in ("running", "done"):
            return
        audio_job.update(key=key, stage="running", progress=0.0, duration=None, error=None)

    def run():
        try:
            def pr(f):
                with state_lock:
                    if audio_job["key"] == key:
                        audio_job["progress"] = f
            an = E.analyze_audio(audio, key[1], key[2], progress=pr)
            with state_lock:
                if audio_job["key"] == key:
                    audio_job.update(stage="done", progress=1.0, duration=an["duration"])
        except Exception as ex:
            with state_lock:
                if audio_job["key"] == key:
                    audio_job.update(stage="error", error=str(ex))
    threading.Thread(target=run, daemon=True).start()


def start_render(cfg, test, t_frac):
    with state_lock:
        if job["stage"] in ("analyze", "build", "render", "verify", "starting"):
            raise RuntimeError("Рендер уже идёт")
        job_cancel.clear()
        job.update(id=job["id"] + 1, stage="starting", progress=0.0, info={}, error=None,
                   kind="test" if test else "full", output=None)
        jid = job["id"]
    cfg = E.merged(cfg)

    def pr(stage, f, info):
        with state_lock:
            if job["id"] == jid:
                job.update(stage=stage, progress=f, info=info)

    def run():
        try:
            if test:
                out = os.path.join(E.CACHE_DIR, "test_clip.mp4")
                an = E.analyze_audio(cfg["audio"], int(cfg["fps"]), int(cfg["wave_bars"]),
                                     progress=lambda f: pr("analyze", f, {}), cancel=job_cancel)
                start = max(0.0, min(an["duration"] - 15, t_frac * an["duration"]))
                info = E.render(cfg, out, pr, job_cancel, start=start, length=15)
            else:
                out = E.resolve(cfg["output"]) if cfg.get("output") else default_output(cfg)
                info = E.render(cfg, out, pr, job_cancel)
            with state_lock:
                if job["id"] == jid:
                    job.update(stage="done", progress=1.0, info=info, output=out)
        except Exception as ex:
            traceback.print_exc()
            with state_lock:
                if job["id"] == jid:
                    job.update(stage="cancelled" if job_cancel.is_set() else "error", error=str(ex))
    threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------ http

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf8")
        elif isinstance(body, str):
            body = body.encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authorized(self, query=None):
        tok = self.headers.get("X-Token") or (query or {}).get("token", [""])[0]
        return secrets.compare_digest(tok or "", TOKEN)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf8") or "{}") if n else {}

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/":
            with open(UI_FILE, encoding="utf8") as fh:
                html = fh.read().replace("__TOKEN__", TOKEN)
            return self._send(200, html, "text/html; charset=utf-8")
        if not self._authorized(q):
            return self._send(403, {"error": "forbidden"})
        if url.path == "/api/state":
            with state_lock:
                return self._send(200, {"settings": E.merged(load_json(SETTINGS_FILE, {})),
                                        "defaults": E.DEFAULTS, "job": dict(job), "audio": dict(audio_job, key=None),
                                        "encoders": encoders, "presets": load_json(PRESETS_FILE, [])})
        if url.path == "/api/status":
            with state_lock:
                return self._send(200, {"job": dict(job), "audio": dict(audio_job, key=None)})
        if url.path == "/api/fonts":
            return self._send(200, list_fonts())
        if url.path == "/file":
            return self._serve_file(E.resolve(q.get("path", [""])[0]))
        return self._send(404, {"error": "not found"})

    do_HEAD = do_GET

    def _serve_file(self, path):
        if not path or not os.path.isfile(path):
            return self._send(404, {"error": "no file"})
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            else:
                start = max(0, size - int(b))
            end = min(end, size - 1)
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionError, OSError):
                pass

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if not self._authorized():
            return self._send(403, {"error": "forbidden"})
        try:
            data = self._body()
            if url.path == "/api/settings":
                save_json(SETTINGS_FILE, data.get("settings", {}))
                start_audio_analysis(data.get("settings", {}))
                return self._send(200, {"ok": True})
            if url.path == "/api/presets":
                save_json(PRESETS_FILE, data.get("presets", []))
                return self._send(200, {"ok": True})
            if url.path == "/api/pick":
                p = pick_file(data.get("kind", ""), data.get("current", ""))
                return self._send(200, {"path": rel_or_abs(p) if p else ""})
            if url.path == "/api/preview":
                cfg = data.get("settings", {})
                im, real = E.preview_image(cfg, float(data.get("t", 0.35)), H=int(data.get("h", 1080)))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=90)
                m = E.merged(cfg)
                warn = {"title": E.missing_glyphs(m["title_font"], m["title"]),
                        "timer": E.missing_glyphs(m["timer_font"], "0123456789:-")}
                return self._send(200, {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                                        "real_audio": real, "missing": warn})
            if url.path == "/api/render":
                start_render(data.get("settings", {}), bool(data.get("test")), float(data.get("t", 0.35)))
                return self._send(200, {"ok": True})
            if url.path == "/api/cancel":
                job_cancel.set()
                return self._send(200, {"ok": True})
            if url.path == "/api/reveal":
                p = E.resolve(data.get("path", ""))
                if os.path.isfile(p) and os.name == "nt":
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
                elif os.path.isdir(p) and os.name == "nt":
                    subprocess.Popen(["explorer", os.path.normpath(p)])
                return self._send(200, {"ok": True})
            if url.path == "/api/open":
                p = E.resolve(data.get("path", ""))
                if os.path.isfile(p) and os.name == "nt":
                    os.startfile(p)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as ex:
            traceback.print_exc()
            return self._send(500, {"error": str(ex)})


def free_port(start=8765):
    for p in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("Нет свободного порта")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.makedirs(E.CACHE_DIR, exist_ok=True)
    os.makedirs(E.OUTPUT_DIR, exist_ok=True)

    def detect():
        try:
            encoders.update(E.available_encoders())
        except Exception as ex:
            encoders["error"] = str(ex)
        list_fonts()
    threading.Thread(target=detect, daemon=True).start()
    start_audio_analysis(load_json(SETTINGS_FILE, {}))

    port = None
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    port = port or free_port()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    url = "http://127.0.0.1:%d/" % port
    print("Vinyl Visualizer запущен: %s" % url)
    print("Не закрывайте это окно, пока работаете. Закрыть программу: Ctrl+C или закрыть окно.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        job_cancel.set()


if __name__ == "__main__":
    main()
