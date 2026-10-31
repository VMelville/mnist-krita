# Kritaで手書き数字を認識する

Kritaの標準レイヤーだけで、28×28入力 → 72ニューロンのReLU → 10出力を計算します。文書は40×40ピクセルです。

## 試す

`digit_recognition.kra` をKritaで開き、最下部の `DRAW HERE` レイヤーを選びます。左上28×28へ白で数字を描きます。元の数字を消すときは不透明な黒で塗ります。1600%程度へズームすると描きやすくなります。

下段の0〜9の棒は各数字のスコアです。表示範囲は−12〜+12、範囲外はクリップします。確率ではありません。入力はR=G=Bの白黒、不透明を保ってください。文書の色空間・深度は変更しません。

## 学習してから再生成

Windows、Krita 5.2.2、uvで動作確認しています。プロジェクトルートで実行します。

```powershell
uv run --locked train.py
uv run build.py
```

`train.py` がMNISTを取得してチェックサムを検査し、55,000件で学習、5,000件で検証します。検証精度でcheckpointを選んだ後に、テスト10,000件を評価します。テスト結果によるcheckpointの選び直しは行いません。

学習した重みはローカルの `model.npz` に保存し、`build.py` がそれを読み込んでKritaのレイヤーへ変換します。重みJSONの配布は不要です。`model.npz` はgit管理・配布対象から除外します。

Python 3.12と学習用ライブラリは、`train.py` の依存指定と `train.py.lock` に従ってuvが用意します。学習条件は `config.json` で設定します。既定は45 epoch、batch 256、AdamW、学習率0.003→0.0001、dropout 0.05、最大±1画素の位置ずらしです。

MNISTは `.cache/mnist/`、その回の分割は `.cache/split.npz`、epochごとの検証結果・採用epoch・テスト精度・実行バージョンは `.cache/training.json` に保存します。いずれもローカル生成物です。

既定のKritaの場所は `C:\Program Files\Krita (x64)\bin` です。異なる場合は指定します。

```powershell
uv run build.py --krita-bin 'D:\Apps\Krita\bin'
```

同じパスを環境変数 `KRITA_BIN` に設定することもできます。出力先の変更は `--output path/to/file.kra` です。既定ではルートの `digit_recognition.kra` を上書きします。kritarunnerは同時に複数起動せず、完了を待ってから次を実行してください。

学習にはNumPyとPyTorchを使います。KRA生成側はKritaの内蔵Pythonを使い、NumPyを追加インストールする必要はありません。生成後には初期入力の出力・棒・入力表示を学習側の期待値と照合し、保存KRAを開き直して再検査します。結果は標準出力へ表示し、一時ファイルは実行後に削除します。

## ファイル

- `digit_recognition.kra`：完成品
- `build.py`：uvから実行する起動スクリプト
- `generate.py`：Krita内蔵Pythonで実行するレイヤー生成処理
- `train.py`：MNISTの取得・学習・評価・重み出力
- `train.py.lock`：学習用の依存関係を固定
- `config.json`：学習条件（重み・seedは含まない）
- `README.md`、`.gitignore`
- `LICENSE`：MIT License（Copyright (c) 2026 VMelville）
