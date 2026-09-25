"""合成EDM（128BPM・8小節ごとに ドロップ/ブレイク 交互）を作り、
   (1) 出力長がサンプル単位で合計秒数と一致 (2) 全力区間の音量 > 休憩区間 (3) 境界の直前/直後で音量が切り替わる
   を grid（ストレッチ）と exact-cut の両モードで検査する。"""
import sys, math, tempfile, json
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hiit_cut as hc

SR = hc.SR

def synth(path: Path, bpm=128.0, bars=48):
    spb = 60 / bpm
    n = int(bars * 4 * spb * SR)
    t = np.arange(n) / SR
    y = np.zeros(n, dtype=np.float32)
    # パッド（常時・小さい）
    y += 0.05 * np.sin(2 * np.pi * 110 * t)
    for beat in range(bars * 4):
        bar = beat // 4
        loud = (bar // 8) % 2 == 1  # 8小節ごとに交互
        a = int(beat * spb * SR)
        k = np.arange(int(0.25 * SR)) / SR
        kick = np.sin(2 * np.pi * (55 + 120 * np.exp(-k * 30)) * k) * np.exp(-k * 12)
        y[a:a + len(kick)] += (0.9 if loud else 0.25) * kick
        if loud:  # ソウ＋ハット
            seg = slice(a, int((beat + 1) * spb * SR))
            tt = t[seg]
            y[seg] += 0.3 * (2 * ((tt * 220) % 1) - 1) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2 * tt)))
            h = np.random.default_rng(beat).standard_normal(int(0.03 * SR)).astype(np.float32) * 0.15
            y[a + int(spb * SR / 2): a + int(spb * SR / 2) + len(h)] += h
    y = np.clip(y, -1, 1)
    sf.write(str(path), np.stack([y, y], axis=1), SR)

def rms_db(x):
    return 20 * math.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)

def check(report, wav):
    y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == SR
    assert len(y) == report["total_seconds"] * SR, "長さ不一致"
    m = y.mean(axis=1)
    work = [c for c in report["cues"] if c["type"] == "work"]
    rest = [c for c in report["cues"] if c["type"] == "rest"]
    lw = np.mean([rms_db(m[c["start"] * SR:c["end"] * SR]) for c in work])
    lr = np.mean([rms_db(m[c["start"] * SR:c["end"] * SR]) for c in rest])
    assert lw - lr > 6, f"全力/休憩の音量差が不足: {lw:.1f} vs {lr:.1f}"
    # 境界: 全力開始の直後0.5s vs 直前0.5s
    ok = 0
    for c in work:
        s = c["start"] * SR
        after, before = rms_db(m[s: s + SR // 2]), rms_db(m[s - SR // 2: s])
        if after - before > 4: ok += 1
    assert ok >= len(work) - 1, f"境界で切り替わらない区間が多い: {ok}/{len(work)}"
    return lw, lr, ok, len(work)

def main():
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "synth128.wav"
    synth(src)
    p = dict(hc.PRESETS["tabata"]); p["key"] = "tabata"
    passed = 0
    for tol, want_mode in ((15.0, "grid"), (0.0, "exact-cut")):
        rep = hc.run_cut(src, p, tmp / f"out_{want_mode}", beep=True, voice=False, tolerance=tol, log=lambda *_: None)
        assert rep["mode"] == want_mode, rep["mode"]
        assert abs(rep["bpm_detected"] - 128) < 1.5, rep["bpm_detected"]
        if want_mode == "grid":
            assert rep["bpm_used"] in (120, 144), rep["bpm_used"]
        lw, lr, ok, n = check(rep, Path(rep["wav"]))
        print(f"✅ {want_mode:<9} 長さ={rep['total_seconds']}s一致  BPM {rep['bpm_detected']}→{rep['bpm_used']}  全力{lw:.1f}dB/休憩{lr:.1f}dB  境界{ok}/{n}  {rep['elapsed']}s")
        passed += 1
    # 10-20-30（3種別）も長さだけ確認
    p2 = dict(hc.PRESETS["t102030"]); p2["key"] = "t102030"
    rep = hc.run_cut(src, p2, tmp / "out_3", beep=False, voice=False, tolerance=15.0, log=lambda *_: None)
    y, _ = sf.read(rep["wav"], dtype="float32", always_2d=True)
    assert len(y) == rep["total_seconds"] * SR
    print(f"✅ 10-20-30   長さ={rep['total_seconds']}s一致  mode={rep['mode']}")
    print(f"全 {passed + 1} 件パス")

if __name__ == "__main__":
    main()
