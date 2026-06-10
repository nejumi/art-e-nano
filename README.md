# ART•E nano — W&B Serverless RL ハンズオン

メール検索エージェントを強化学習 (RL) で鍛える [OpenPipe ART-E](https://github.com/OpenPipe/ART/tree/art-e/examples/art-e) を、**1〜2 時間のハンズオンで完走できる規模に軽量化**したプロジェクトです。学習には [W&B Training (Serverless RL)](https://docs.wandb.ai/serverless-rl/) を使うため、GPU の用意は不要です。

> 現在の公開版は、参加者の環境構築と事前確認を優先したプレリリースです。学習設定・評価指標・当日の説明内容は、ワークショップ本番まで継続して調整します。

## なにをするのか

[Enron メールデータセット](https://huggingface.co/datasets/corbt/enron_emails_sample_questions) を題材に、メールボックスを検索して質問に答えるエージェントを GRPO で学習します。

| ツール | 説明 |
| --- | --- |
| `search_emails` | キーワード・差出人・日付などでメールを全文検索 |
| `read_email` | message_id を指定してメール本文を読む |
| `return_final_answer` | 出典 (message_id) 付きで最終回答を返す |

## ワークショップ設計（検証済み）

| 項目 | 内容 |
| --- | --- |
| **主指標** | `val/answer_correct` = OpenAI ジャッジによる回答正答率 |
| **検証** | `eval_temperature=0.7` + 固定 96 問 (seed=42) |
| **学習報酬** | ART-E 準拠のルーブリック報酬 (-2〜+2)。RULER 使用時も置き換えず補助加点として混ぜる |
| **Judge** | OpenAI (`gpt-5.5`, `reasoning_effort=low`) を使用。`JUDGE_MODE=auto` では、障害時のみ heuristic にフォールバック |
| **学習長** | step 0 のベースライン評価 + 複数回の更新。W&B で挙動を観察する |

このハンズオンでは、単一の最高スコアを追うよりも、W&B 上で次の変化を観察することを重視します。

- `val/answer_correct` がベースラインから改善しているか
- `val/sources_correct` が改善し、正しいメールを見つける行動が増えているか
- `val/returned_i_dont_know` が下がる、または急増していないか
- 学習を続けると必ず良くなるわけではないことを、checkpoint ごとの挙動から確認する

## セットアップ

[SETUP.md](SETUP.md) を参照。

```bash
uv run python scripts/preflight_check.py
```

## 学習の実行

```bash
MODEL_NAME=art-e-nano-$(date +%Y%m%d-%H%M) uv run python -m art_e.train
```

環境変数 (任意):

| 変数 | デフォルト | 説明 |
| --- | --- | --- |
| `MODEL_NAME` | `art-e-nano-workshop` | **毎回ユニークな名前推奨** |
| `FRESH_START` | `true` | 同名モデルを削除してベースから再開 |
| `JUDGE_MODE` | `auto` | 通常は `auto` のままで OK。`llm` / `heuristic` も可 |
| `JUDGE_REASONING_EFFORT` | `low` | OpenAI ジャッジの推論量 |
| `USE_RULER` | `false` | RULER を補助加点として使う |
| `RULER_WEIGHT` | `0.2` | `rubric_reward + RULER_WEIGHT * ruler_score` |
| `IDK_PENALTY` | `0.2` | `"I don't know"` への退避を抑えるペナルティ |
| `SIMPLE_REWARD` | `false` | 簡易報酬に切り替える実験用フラグ |

終了時に **EVALUATION SUMMARY** が表示されます。スコアは環境やモデル提供状況で変動するため、README には固定の期待値を書かず、W&B 上で自分の Run の変化を確認します。

### W&B で見るべきメトリクス

| メトリクス | 意味 |
| --- | --- |
| **`val/answer_correct`** | 主指標。OpenAI ジャッジによる正答率 |
| `val/sources_correct` | 正しいメールを引用できた率 |
| `val/task_success` | 正答または正しいメールを引用できた率 |
| `val/returned_i_dont_know` | 急増 = 方策崩壊の兆候 |

## プロンプトのみのモデルとの比較 (オプション)

```bash
uv run python -m art_e.evaluate.benchmark_prompted_models
```

## ライセンス

[LICENSE](LICENSE) に従います。
