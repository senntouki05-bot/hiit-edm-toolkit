#!/usr/bin/env python3
"""
hiit-cut — Sunoで作ったEDMを、HIITのタイムライン（秒数）どおりに組み直す。

仕組み:
  1. 曲を解析: BPM・拍・小節頭（4/4前提）・小節ごとの音量（RMS）
  2. 小節を「高(ドロップ)/中/低(ブレイク)」に分類し、連続する塊(run)にまとめる
  3. プロトコル（例: 30低→20中→10全力 ×5）の各区間へ、小節頭から切った素材を並べる
     - 可能なら曲全体を「区間秒数が整数小節になるBPM」へ軽くタイムストレッチ（継ぎ目が拍に乗る）
     - できなければ小節頭から「ちょうどN秒」を切る（秒数は必ず固定・継ぎ目が拍から外れることあり）
  4. 任意でビープ（3-2-1）とコーチ音声（macOS say）を秒数ぴったりに重ねる
  5. -14 LUFS に正規化して WAV + MP3 を書き出す（出力の長さ = 合計秒数ちょうど、サンプル単位で検証）

使い方:  bash h cut <曲.mp3> --preset tabata [--out DIR] [--beep] [--voice] [--tolerance 10]
         bash h analyze <曲.mp3>
         bash h ui
"""
from __future__ import annotations
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "out"
SR = 44100
FADE_MS = 30  # 継ぎ目のフェード（重ねずに突き合わせ＝長さを1サンプルも変えない）

PRESETS = {
    "tabata":   dict(name="Tabata 20/10 ×8",            warm=30, cool=30, rounds=8,  segs=[(20, "work"), (10, "rest")]),
    "t102030":  dict(name="10-20-30 ×5",                 warm=30, cool=30, rounds=5,  segs=[(30, "low"), (20, "mid"), (10, "work")]),
    "r3030":    dict(name="30/30 ×8",                    warm=30, cool=30, rounds=8,  segs=[(30, "work"), (30, "rest")]),
    "r4020":    dict(name="40/20 ×8",                    warm=30, cool=30, rounds=8,  segs=[(40, "work"), (20, "rest")]),
    "r4515":    dict(name="45/15 ×8",                    warm=30, cool=30, rounds=8,  segs=[(45, "work"), (15, "rest")]),
    "sevenmin": dict(name="7-Minute 30/10 ×12",          warm=0,  cool=0,  rounds=12, segs=[(30, "work"), (10, "rest")]),
    "emom":     dict(name="EMOM 40+20 ×10",              warm=30, cool=30, rounds=10, segs=[(40, "work"), (20, "rest")]),
    "beginner": dict(name="初心者 20/40 ×8",              warm=60, cool=60, rounds=8,  segs=[(20, "work"), (40, "rest")]),
    "sit":      dict(name="SIT 30/240 ×4",               warm=120, cool=120, rounds=4, segs=[(30, "work"), (240, "rest")]),
    "nor44":    dict(name="Norwegian 4×4 (240/180)",     warm=300, cool=180, rounds=4, segs=[(240, "mid"), (180, "rest")]),
}
# 区間種別 → 素材の好み（先頭から順に試す）
PREFER = {
    "intro": ["low", "mid", "high"],
    "outro": ["low", "mid", "high"],
    "work":  ["high", "mid", "low"],
    "mid":   ["mid", "high", "low"],
    "low":   ["low", "mid", "high"],
    "rest":  ["low", "mid", "high"],
}


# ---------------------------------------------------------------- 解析
@dataclass
class Bar:
    idx: int
    start: float
    end: float
    rms_db: float
    cls: str  # high / mid / low


@dataclass
class Analysis:
    path: str
    duration: float
    bpm: float
    beat_phase: int
    bars: list
    runs: list  # [{cls, bars:[idx...], start, end}]

    def to_json(self):
        d = asdict(self)
        return d


def decode_to_wav(src: Path, dst: Path, sr: int = SR):
    """何でも ffmpeg で WAV(float32, stereo) に落とす"""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "2", "-ar", str(sr),
                    "-c:a", "pcm_f32le", str(dst)], check=True)


def load_stereo(path: Path):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, sr
    return y  # (n, 2)


def analyze(src: Path, workdir: Path) -> tuple[Analysis, Path]:
    import librosa
    wav = workdir / "src.wav"
    decode_to_wav(src, wav)
    y = load_stereo(wav)
    mono = y.mean(axis=1)
    dur = len(mono) / SR
    # 拍: 大域テンポ推定 → 周期×位相のグリッド探索でオンセット合計を最大化（一定テンポ前提のEDM向け）
    hop = 512
    onset = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=hop)
    tempo0 = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=SR, hop_length=hop))[0])
    fps = SR / hop
    t_frames = np.arange(len(onset)) / fps

    def score(bpm_, ph_):
        per = 60.0 / bpm_
        grid = np.arange(ph_, t_frames[-1], per)
        idx = np.clip(np.round(grid * fps).astype(int), 0, len(onset) - 1)
        return float(onset[idx].mean())

    best = (-1, tempo0, 0.0)
    for mult in (0.5, 1.0, 2.0):  # 半分/倍テンポの取り違え → EDMの実用域 100〜200 BPM に入る候補だけ
        base = tempo0 * mult
        if not 100 <= base < 200:
            continue
        for bpm_ in np.arange(base * 0.97, base * 1.03, 0.05):
            per = 60.0 / bpm_
            for ph_ in np.arange(0, per, per / 32):
                sc = score(bpm_, ph_)
                if sc > best[0]:
                    best = (sc, float(bpm_), float(ph_))
    # 微調整（±0.05 の中を 0.005 刻み）
    _, bpm, ph = best
    for bpm_ in np.arange(bpm - 0.05, bpm + 0.05, 0.005):
        per = 60.0 / bpm_
        for ph_ in np.arange(max(0, ph - per / 32), ph + per / 32, per / 256):
            sc = score(bpm_, ph_)
            if sc > best[0]:
                best = (sc, float(bpm_), float(ph_))
    _, bpm, ph = best
    ibi = 60.0 / bpm
    beats = np.arange(ph, dur, ibi)
    beat_frames = np.clip(np.round(beats * fps).astype(int), 0, len(onset) - 1)
    if len(beats) < 8:
        raise SystemExit("拍が検出できません（曲が短すぎる／無音）")
    # 小節頭の位相: 4拍ごとのオンセット強度が最大になる位相
    ob = onset[np.clip(beat_frames, 0, len(onset) - 1)]
    phase = int(np.argmax([ob[p::4].mean() for p in range(4)]))
    # 小節（4拍）の一覧
    bars = []
    i = phase
    while i + 4 <= len(beats):
        s, e = float(beats[i]), float(beats[i] + 4 * ibi)
        a, b = int(s * SR), int(min(e, dur) * SR)
        if b - a < SR * 0.5:
            i += 4
            continue
        seg = mono[a:b]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        bars.append(Bar(len(bars), s, e, 20 * math.log10(rms), ""))
        i += 4
    if len(bars) < 4:
        raise SystemExit("小節が4つ未満です")
    # 分類（40/65パーセンタイル）
    db = np.array([b.rms_db for b in bars])
    lo_t, hi_t = np.percentile(db, 40), np.percentile(db, 65)
    for b in bars:
        b.cls = "high" if b.rms_db >= hi_t else ("low" if b.rms_db <= lo_t else "mid")
    # 1小節だけ孤立した分類はならす（前後が同じなら合わせる）
    for k in range(1, len(bars) - 1):
        if bars[k - 1].cls == bars[k + 1].cls != bars[k].cls:
            bars[k].cls = bars[k - 1].cls
    # run
    runs = []
    for b in bars:
        if runs and runs[-1]["cls"] == b.cls:
            runs[-1]["bars"].append(b.idx); runs[-1]["end"] = b.end
        else:
            runs.append({"cls": b.cls, "bars": [b.idx], "start": b.start, "end": b.end})
    return Analysis(str(src), dur, bpm, phase, bars, runs), wav


# ---------------------------------------------------------------- 計画
def expand_protocol(p: dict) -> list[tuple[str, str, int]]:
    """(label, type, seconds) の列に展開"""
    out = []
    if p["warm"] > 0:
        out.append(("Intro", "intro", int(p["warm"])))
    for r in range(1, int(p["rounds"]) + 1):
        for sec, t in p["segs"]:
            out.append((f"R{r} {t}", t, int(sec)))
    if p["cool"] > 0:
        out.append(("Outro", "outro", int(p["cool"])))
    return out


def choose_grid_bpm(bpm: float, segments, tolerance_pct: float):
    """全区間が整数小節になるBPM候補のうち、最も近いもの（許容%以内）。無ければ None"""
    g = 0
    for _, _, sec in segments:
        g = math.gcd(g, sec)
    if g <= 0:
        return None
    cands = []
    k = 1
    while 240 * k / g < 400:
        cands.append(240 * k / g)
        k += 1
    best = min(cands, key=lambda c: abs(c / bpm - 1))
    if abs(best / bpm - 1) * 100 <= tolerance_pct:
        return best
    return None


# ---------------------------------------------------------------- 組み立て
class Picker:
    """種別ごとに、素材runから小節を順に取り出す（同じ種別が続いても同じ所ばかり使わない）"""
    def __init__(self, an: Analysis):
        self.an = an
        self.cursor = {}

    def runs_of(self, cls, want_edge=None):
        rs = [r for r in self.an.runs if r["cls"] == cls]
        if not rs:
            return []
        # 長い順。intro/outroは曲頭/曲末に近いものを優先
        if want_edge == "start":
            rs.sort(key=lambda r: r["start"])
        elif want_edge == "end":
            rs.sort(key=lambda r: -r["end"])
        else:
            rs.sort(key=lambda r: -len(r["bars"]))
        return rs

    def bars_for(self, seg_type: str, n_bars_needed: int, edge=None):
        """小節idxの列を返す（run内を巡回して必要数を満たす）"""
        for cls in PREFER[seg_type]:
            rs = self.runs_of(cls, edge)
            if not rs:
                continue
            run = rs[0]
            key = (cls, id(run))
            off = self.cursor.get(key, 0)
            idxs = [run["bars"][(off + i) % len(run["bars"])] for i in range(n_bars_needed)]
            self.cursor[key] = (off + n_bars_needed) % len(run["bars"])
            return idxs, cls
        raise RuntimeError("素材が見つかりません")


def fade_edges(x: np.ndarray, n: int):
    n = min(n, len(x) // 2)
    if n <= 0:
        return x
    ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def assemble(an: Analysis, y: np.ndarray, segments, picker: Picker, log):
    """各区間をちょうど sec 秒で切り出して連結。返り値: (audio, cue_sheet)"""
    sec_per_bar = 240.0 / an.bpm
    fade_n = int(SR * FADE_MS / 1000)
    parts, cues, t = [], [], 0
    for label, typ, sec in segments:
        n_samples = sec * SR
        n_bars = max(1, math.ceil(sec / sec_per_bar) + 1)
        edge = "start" if typ == "intro" else ("end" if typ == "outro" else None)
        idxs, used_cls = picker.bars_for(typ, n_bars, edge)
        chunks = []
        got = 0
        for i in idxs:
            b = an.bars[i]
            a, e = int(b.start * SR), int(b.end * SR)
            chunks.append(y[a:e])
            got += e - a
            if got >= n_samples:
                break
        buf = np.concatenate(chunks)[:n_samples].copy()
        if len(buf) < n_samples:  # 念のため（曲末で足りない）
            buf = np.pad(buf, ((0, n_samples - len(buf)), (0, 0)))
        fade_edges(buf, fade_n)
        parts.append(buf)
        cues.append(dict(label=label, type=typ, start=t, end=t + sec, source_cls=used_cls,
                         source_bars=idxs[:len(chunks)], source_time=round(an.bars[idxs[0]].start, 2)))
        log(f"  {t:5d}s  {label:<12} {sec:3d}s  ← {used_cls:<4} 小節{idxs[0]}〜（{an.bars[idxs[0]].start:.1f}s〜）")
        t += sec
    return np.concatenate(parts), cues


def add_cues(audio: np.ndarray, cues, beep: bool, voice: bool, log):
    """ビープ（3-2-1 + GO）とコーチ音声を秒数ぴったりに重ねる"""
    def tone(freq, ms, gain=0.35):
        n = int(SR * ms / 1000)
        tt = np.arange(n) / SR
        w = np.sin(2 * np.pi * freq * tt).astype(np.float32) * gain
        env = np.minimum(1, np.minimum(tt / 0.005, (n / SR - tt) / 0.02))
        return (w * env)[:, None].repeat(2, axis=1)

    def mix(at_sec, snd):
        a = int(at_sec * SR)
        if a < 0:
            return
        e = min(len(audio), a + len(snd))
        audio[a:e] += snd[:e - a]

    voices = {}
    if voice and shutil.which("say"):
        tmp = Path(tempfile.mkdtemp())
        for word in ("Go", "Rest", "Easy", "Steady", "Done"):
            f = tmp / f"{word}.wav"
            try:
                subprocess.run(["say", "-v", "Samantha", "-r", "200", "-o", str(f), "--data-format=LEF32@44100", word],
                               check=True, capture_output=True)
                import soundfile as sf
                v, _ = sf.read(str(f), dtype="float32", always_2d=True)
                voices[word] = (v.mean(axis=1, keepdims=True).repeat(2, axis=1) * 0.9)
            except Exception as ex:  # sayが無い/失敗 → 無言で進めず記録する
                log(f"  ⚠️ 音声 '{word}' の生成に失敗: {ex}")
    elif voice:
        log("  ⚠️ macOSの say が見つからないためコーチ音声はスキップ")

    for c in cues:
        typ = c["type"]
        if typ == "work":
            if beep:
                for k in (3, 2, 1):
                    mix(c["start"] - k, tone(880, 80))
                mix(c["start"], tone(1320, 220, 0.4))
            if "Go" in voices:
                mix(c["start"], voices["Go"])
        elif typ in ("rest", "low") and c["label"] != "Intro":
            if beep:
                mix(c["start"], tone(440, 160, 0.3))
            w = "Rest" if typ == "rest" else "Easy"
            if w in voices:
                mix(c["start"], voices[w])
        elif typ == "mid" and "Steady" in voices:
            mix(c["start"], voices["Steady"])
        elif typ == "outro" and "Done" in voices:
            mix(c["start"], voices["Done"])
    return audio


def stretch_wav(src: Path, dst: Path, ratio: float):
    """atempo（WSOLA）で速度 ratio 倍。ピッチ不変。"""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-filter:a", f"atempo={ratio:.6f}",
                    "-ar", str(SR), "-ac", "2", "-c:a", "pcm_f32le", str(dst)], check=True)


def finalize(wav_in: Path, mp3_out: Path, wav_out: Path, lufs: float = -14.0):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_in),
                    "-af", f"loudnorm=I={lufs}:TP=-1.0:LRA=11", "-ar", str(SR),
                    "-c:a", "pcm_s16le", str(wav_out)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_out), "-c:a", "libmp3lame", "-b:a", "320k",
                    str(mp3_out)], check=True)


# ---------------------------------------------------------------- メイン処理
def run_cut(src: Path, protocol: dict, out_dir: Path, beep=False, voice=False, tolerance=10.0, log=print) -> dict:
    import soundfile as sf
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hiitcut_"))
    log(f"▶ 解析: {src.name}")
    an, wav = analyze(src, work)
    segments = expand_protocol(protocol)
    total = sum(s for _, _, s in segments)
    log(f"  BPM {an.bpm:.2f} / 尺 {an.duration:.1f}s / 小節 {len(an.bars)} / "
        f"高{sum(b.cls=='high' for b in an.bars)} 中{sum(b.cls=='mid' for b in an.bars)} 低{sum(b.cls=='low' for b in an.bars)}")
    grid = choose_grid_bpm(an.bpm, segments, tolerance)
    mode = "exact-cut"
    ratio = 1.0
    if grid:
        ratio = grid / an.bpm
        if abs(ratio - 1) > 1e-4:
            log(f"  ストレッチ: {an.bpm:.2f} → {grid:.0f} BPM（×{ratio:.4f}・全区間が整数小節に乗る）")
            sw = work / "stretched.wav"
            stretch_wav(wav, sw, ratio)
            wav = sw
            for b in an.bars:
                b.start /= ratio; b.end /= ratio
            for r in an.runs:
                r["start"] /= ratio; r["end"] /= ratio
            an.bpm = grid
            an.duration /= ratio
        mode = "grid"
    else:
        log(f"  ストレッチ不可（許容{tolerance}%内に整数小節BPMなし）→ 小節頭から秒数ちょうどで切る")
    y = load_stereo(wav)
    log("▶ 組み立て")
    audio, cues = assemble(an, y, segments, Picker(an), log)
    assert len(audio) == total * SR, (len(audio), total * SR)
    audio = add_cues(audio, cues, beep, voice, log)
    peak = float(np.abs(audio).max())
    if peak > 0.99:
        audio *= 0.99 / peak
    raw = work / "assembled.wav"
    sf.write(str(raw), audio, SR, subtype="FLOAT")
    stem = src.stem + "_" + protocol.get("key", "custom")
    mp3, wavo = out_dir / f"{stem}.mp3", out_dir / f"{stem}.wav"
    finalize(raw, mp3, wavo)
    # 長さ検証（出力ファイルを実測）
    info = sf.info(str(wavo))
    assert info.frames == total * SR, (info.frames, total * SR)
    report = dict(source=str(src), protocol=protocol, total_seconds=total, mode=mode, stretch_ratio=ratio,
                  bpm_detected=round(an.bpm / ratio, 2), bpm_used=round(an.bpm, 2), cues=cues,
                  mp3=str(mp3), wav=str(wavo), elapsed=round(time.time() - t0, 1))
    (out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    log(f"✅ 出力 {mp3}（{total}s ちょうど・{report['elapsed']}s）")
    shutil.rmtree(work, ignore_errors=True)
    return report


def parse_protocol(args) -> dict:
    if args.preset:
        p = dict(PRESETS[args.preset]); p["key"] = args.preset
    else:
        p = dict(name="custom", key="custom", warm=30, cool=30, rounds=8, segs=[(30, "work"), (10, "rest")])
    if args.warm is not None: p["warm"] = args.warm
    if args.cool is not None: p["cool"] = args.cool
    if args.rounds is not None: p["rounds"] = args.rounds
    if args.segs:
        segs = []
        for tok in args.segs.split(","):
            sec, t = tok.split(":")
            assert t in PREFER, t
            segs.append((int(sec), t))
        p["segs"] = segs; p["key"] = "custom"
    return p


# ---------------------------------------------------------------- UI
def serve(port: int):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from email.parser import BytesParser
    from email.policy import default as email_default
    import html as H

    PAGE = """<!DOCTYPE html><html lang=ja><meta charset=utf-8><title>hiit-cut</title>
<style>body{font:14px/1.6 -apple-system,"Hiragino Sans",sans-serif;background:#0d0f14;color:#e7eaf0;max-width:760px;margin:30px auto;padding:0 16px}
h1{font-size:20px}label{display:block;margin:10px 0 2px;color:#8b93a7;font-size:12px}input,select{width:100%%;padding:7px;background:#0b0d12;color:#e7eaf0;border:1px solid #262c3a;border-radius:6px}
.row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}button{margin-top:14px;background:#ff4d6d;color:#fff;border:0;padding:10px 18px;border-radius:6px;font-weight:600;font-size:14px}
pre{background:#0b0d12;border:1px solid #262c3a;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:12px}a{color:#4de0ff}.chk{display:flex;gap:6px;align-items:center;margin-top:8px}.chk input{width:auto}small{color:#8b93a7}</style>
<h1>hiit-cut — Sunoの曲をHIITの秒数どおりに組み直す</h1>
<form method=post action=/run enctype=multipart/form-data>
<label>曲（mp3/wav/m4a・複数可＝1曲ずつ別ファイルに出力）</label><input type=file name=files multiple required accept=audio/*>
<label>プロトコル</label><select name=preset>%(presets)s<option value="">カスタム（下の区間を使う）</option></select>
<div class=row><div><label>ウォームアップ(s) 空=プリセット値</label><input name=warm type=number min=0></div>
<div><label>クールダウン(s)</label><input name=cool type=number min=0></div><div><label>ラウンド数</label><input name=rounds type=number min=1></div></div>
<label>カスタム区間（秒:種別 をカンマ区切り・種別= work/mid/low/rest）例 30:low,20:mid,10:work</label><input name=segs placeholder="空=プリセットの区間">
<div class=row><div><label>ストレッチ許容(%%)（整数小節BPMへ寄せる上限・0で常に秒切り）</label><input name=tolerance type=number value=10 min=0 max=40></div></div>
<div class=chk><input type=checkbox name=beep id=b checked><label for=b style="margin:0">3-2-1ビープ＋GO音</label></div>
<div class=chk><input type=checkbox name=voice id=v><label for=v style="margin:0">コーチ音声（macOS say: Go / Rest / Easy）</label></div>
<button>組み立てる</button></form>
<p><small>出力先: %(out)s ／ 秒数はサンプル単位で固定（出力後に実測検証）。ドロップの「聴こえ方」は曲の解析結果次第なので、最初の1本は必ず耳で確認。</small></p>
%(result)s</html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _page(self, result=""):
            presets = "".join(f'<option value="{k}">{H.escape(v["name"])}</option>' for k, v in PRESETS.items())
            body = PAGE % dict(presets=presets, out=H.escape(str(OUT_ROOT)), result=result)
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            if self.path.startswith("/out/"):
                f = (OUT_ROOT / self.path[5:].split("?")[0]).resolve()
                if OUT_ROOT.resolve() in f.parents and f.is_file():
                    self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(f.stat().st_size)); self.end_headers()
                    self.wfile.write(f.read_bytes()); return
                self.send_response(404); self.end_headers(); return
            self._page()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            msg = BytesParser(policy=email_default).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
            fields, files = {}, []
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                fn = part.get_filename()
                if fn:
                    files.append((fn, part.get_payload(decode=True)))
                else:
                    fields[name] = part.get_content().strip()
            ns = argparse.Namespace(preset=fields.get("preset") or None,
                                    warm=int(fields["warm"]) if fields.get("warm") else None,
                                    cool=int(fields["cool"]) if fields.get("cool") else None,
                                    rounds=int(fields["rounds"]) if fields.get("rounds") else None,
                                    segs=fields.get("segs") or None)
            proto = parse_protocol(ns)
            logs = []
            results = []
            tmp = Path(tempfile.mkdtemp())
            for fn, data in files:
                if not data:
                    continue
                src = tmp / Path(fn).name
                src.write_bytes(data)
                try:
                    rep = run_cut(src, proto, OUT_ROOT / proto["key"], beep="beep" in fields, voice="voice" in fields,
                                  tolerance=float(fields.get("tolerance") or 10), log=logs.append)
                    rel = Path(rep["mp3"]).relative_to(OUT_ROOT)
                    results.append(f'<li><a href="/out/{rel}">{H.escape(rel.name)}</a>（{rep["total_seconds"]}s・{rep["mode"]}・BPM {rep["bpm_detected"]}→{rep["bpm_used"]}）</li>')
                except Exception as ex:
                    logs.append(f"❌ {fn}: {ex}")
            res = "<h2>結果</h2><ul>" + "".join(results) + "</ul><pre>" + H.escape("\n".join(logs)) + "</pre>"
            self._page(res)

    OUT_ROOT.mkdir(exist_ok=True)
    print(f"hiit-cut UI: http://127.0.0.1:{port}/  （Ctrl+C で終了）")
    subprocess.Popen(["open", f"http://127.0.0.1:{port}/"]) if shutil.which("open") else None
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut"); c.add_argument("src"); c.add_argument("--preset", choices=PRESETS.keys())
    c.add_argument("--warm", type=int); c.add_argument("--cool", type=int); c.add_argument("--rounds", type=int)
    c.add_argument("--segs", help="例 30:low,20:mid,10:work"); c.add_argument("--out")
    c.add_argument("--beep", action="store_true"); c.add_argument("--voice", action="store_true")
    c.add_argument("--tolerance", type=float, default=10.0)
    a = sub.add_parser("analyze"); a.add_argument("src")
    u = sub.add_parser("ui"); u.add_argument("--port", type=int, default=8871)
    args = ap.parse_args()
    if args.cmd == "ui":
        serve(args.port)
    elif args.cmd == "analyze":
        work = Path(tempfile.mkdtemp())
        an, _ = analyze(Path(args.src), work)
        print(f"BPM {an.bpm:.2f}  尺 {an.duration:.1f}s  小節 {len(an.bars)}  位相 {an.beat_phase}")
        for r in an.runs:
            print(f"  {r['cls']:<4} {r['start']:7.1f}s〜{r['end']:7.1f}s  {len(r['bars']):3d}小節")
        shutil.rmtree(work, ignore_errors=True)
    else:
        proto = parse_protocol(args)
        out = Path(args.out) if args.out else OUT_ROOT / proto["key"]
        run_cut(Path(args.src), proto, out, beep=args.beep, voice=args.voice, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
