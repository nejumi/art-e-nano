# 環境構築ガイド (参加者向け)

当日までに以下の手順を完了し、**事前確認スクリプトが PASS することを確認してください**。明日公開する版は、主に環境構築と事前確認を目的にしています。学習設定や評価の細部はワークショップ当日まで調整される可能性があります。

所要時間の目安: 15〜30 分 (ほとんどはアカウント作成とデータダウンロードの待ち時間です)

## 1. 必要なアカウントと API キー

### Weights & Biases (必須)

学習基盤 (Serverless RL) と実験管理に使います。

1. [wandb.ai](https://wandb.ai) でアカウントを作成 (会社のチームに所属している場合はそのアカウントで OK)
2. [wandb.ai/authorize](https://wandb.ai/authorize) で API キーを取得
3. Serverless RL は W&B Training の利用枠が必要です。`.env` の `WANDB_ENTITY` には、個人ユーザー名ではなく、利用可能な Team entity 名を指定してください。Team entity であれば基本的には任意のチームで構いません。チームがない場合は [W&B の手順](https://docs.wandb.ai/ja/platform/app/settings-page/teams#%E5%85%B1%E5%90%8C%E4%BD%9C%E6%A5%AD%E7%94%A8%E3%81%AE%E3%83%81%E3%83%BC%E3%83%A0%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B) に従って作成してください。

> `WANDB_ENTITY` に個人アカウント名を指定すると、通常の W&B ログ記録はできても、Serverless RL のモデル登録で `Error code: 524 origin_response_timeout` や権限エラーになる場合があります。W&B Training が有効な Team entity を使ってください。

### OpenAI (任意)

回答の正誤判定 (LLM-as-a-judge) や RULER を使う場合に利用します。OpenAI API キーを用意できない場合も、`JUDGE_MODE=heuristic` でヒューリスティック判定に切り替えて参加できます。当日の標準手順は RULER を使わない構成で進めます。

OpenAI API キーを使う場合:

1. [platform.openai.com](https://platform.openai.com) でアカウントを作成
2. [API keys](https://platform.openai.com/api-keys) でキーを発行
3. 課金設定と利用上限を確認

## 2. ツールのインストール

### uv (Python パッケージマネージャ)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Python 3.11 以上が必要ですが、無ければ uv が自動でダウンロードします。

## 3. リポジトリの取得と依存関係のインストール

```bash
git clone https://github.com/nejumi/art-e-nano
cd art-e-nano
uv sync
```

## 4. 環境変数の設定

```bash
cp .env.sample .env
```

`.env` をエディタで開き、自分のキーを設定します。

OpenAI API キーを使う場合:

```bash
WANDB_API_KEY=<1. で取得した W&B の API キー>
WANDB_ENTITY=<利用可能な Team entity 名>
WANDB_PROJECT=art-e-nano
OPENAI_API_KEY=<1. で取得した OpenAI の API キー>
JUDGE_MODE=auto
```

OpenAI API キーを使わない場合:

```bash
WANDB_API_KEY=<1. で取得した W&B の API キー>
WANDB_ENTITY=<利用可能な Team entity 名>
WANDB_PROJECT=art-e-nano
JUDGE_MODE=heuristic
# OPENAI_API_KEY は未設定またはコメントアウトのままにしてください
```

> `.env` は `.gitignore` 済みです。API キーは絶対にコミットしないでください。

## 5. 事前確認スクリプトの実行

```bash
uv run python scripts/preflight_check.py
```

当日使う主要な API を実際に呼び出して検証します (約 2〜5 分。初回はメール DB の構築で数分かかります)。OpenAI API キーがない場合は、`.env` で `JUDGE_MODE=heuristic` を設定してから実行してください。

```
============================================================
ART-E nano ハンズオン事前確認
============================================================

[1. Python バージョン]
  ✓ PASS  Python 3.11 以上  (検出: 3.12.x)

[2. 環境変数 (.env)]
  ✓ PASS  WANDB_API_KEY が設定されている
  - SKIP  OPENAI_API_KEY: 未設定 (JUDGE_MODE=heuristic のため heuristic judge にフォールバック)
  ...

[7. W&B Serverless RL (モデル登録 → 推論 → 削除)]
  ✓ PASS  Serverless RL へのモデル登録
  ✓ PASS  W&B Inference での推論

============================================================
すべてのチェックに合格しました。当日はこのまま学習を実行できます:
  uv run python -m art_e.train
```

`すべてのチェックに合格しました` と表示されれば準備完了です。

## トラブルシューティング

| 症状 | 対処 |
| --- | --- |
| `WANDB_API_KEY が有効` で FAIL | キーの値を再確認。`wandb.ai/authorize` で再発行して `.env` を更新 |
| `OPENAI_API_KEY が設定されている` で FAIL | `JUDGE_MODE=llm` の場合は OpenAI API キーが必須です。キーがない場合は `.env` で `JUDGE_MODE=heuristic` に変更 |
| `ジャッジ判定` が heuristic fallback | OpenAI API キー未設定、quota / 利用上限、または `JUDGE_MODE=heuristic` により heuristic judge を使用中。学習自体は続行可能 |
| `Serverless RL へのモデル登録` で FAIL (403) | 指定した `WANDB_ENTITY` が Team entity であることと、W&B Training の利用権限があることを確認。チームがない場合は W&B のチーム作成手順に従って作成 |
| `Serverless RL へのモデル登録` で FAIL (`524 origin_response_timeout`) | `.env` の `WANDB_ENTITY` に個人ユーザー名を指定していないか確認。W&B Training が有効な Team entity 名に変更して再実行 |
| データセットのダウンロードが遅い / 失敗 | ネットワークを確認して再実行。再実行時はキャッシュから再開されます |
| 社内プロキシ環境で SSL エラー | `HTTPS_PROXY` / `SSL_CERT_FILE` を設定するか、別ネットワークで実行 |

解決しない場合は、エラーメッセージ全文を添えて主催者まで連絡してください。

## (参考) 当日やること

```bash
# 学習の実行 (所要時間は設定により変わります)
uv run python -m art_e.train

# 進捗は W&B のダッシュボードで確認
# https://wandb.ai/<WANDB_ENTITY>/<WANDB_PROJECT>
```
