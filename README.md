# ライントレースの簡単なシミュレーション

## 環境構築
```bash
git clone https://github.com/yuto0o/line-trace-omni-simulation.git
```
### uv 環境ない人（以下のコマンドでバージョン情報が出ない人）
```bash
uv --version
```
https://docs.astral.sh/uv/getting-started/installation/
このリンクなどからインストールしてください。
以下のコマンドでもよいです。
```
pip install uv
```

### uv 環境ある人
以下のコマンドで環境構築が完了します
```bash
uv sync
```

## 実行
```
uv run python main.py
```

