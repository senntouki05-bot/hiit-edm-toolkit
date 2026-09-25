#!/bin/bash
# hiit-cut 入口（cd不要）:  bash ~/dev/hiit-edm-prompts/h <ui|cut|analyze|test> [...]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "venvがありません: /opt/homebrew/bin/python3.12 -m venv $REPO/.venv && $REPO/.venv/bin/pip install librosa soundfile numpy"; exit 1; }
case "${1:-}" in
  test) exec "$PY" "$REPO/tests/test_cut.py" ;;
  ui|cut|analyze) exec "$PY" "$REPO/hiit_cut.py" "$@" ;;
  *) echo "使い方: bash $REPO/h <ui|cut|analyze|test>"; echo "  ui                    ブラウザUI（曲をドラッグ→プロトコル選択→組み立て）"; echo "  cut <曲> --preset tabata [--beep] [--voice] [--segs 30:low,20:mid,10:work]"; echo "  analyze <曲>          BPM・小節・高/中/低の分布だけ見る"; echo "  test                  合成曲で自己テスト"; exit 1 ;;
esac
