"""ハンズオン事前確認スクリプト。

実行方法:
    uv run python scripts/preflight_check.py

このスクリプトがすべて PASS すれば、当日のハンズオン (art_e/train.py) は
そのまま動きます。当日使う API を実際に呼んで検証します:

  1. Python バージョン
  2. .env の必須環境変数
  3. OpenAI API (ジャッジモデル。使えない場合は heuristic judge にフォールバック可能)
  4. W&B 認証 (WANDB_API_KEY)
  5. Hugging Face データセットのダウンロード (学習データ)
  6. メール DB (SQLite) の構築と検索ツールの動作確認
  7. W&B Serverless RL (モデル登録 → 推論 1 回 → 削除)

オプション:
    --skip-serverless   手順 7 をスキップ (W&B Training の枠を使いたくない場合)
"""

import argparse
import asyncio
import os
import sys
import traceback

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m- SKIP\033[0m"

results: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str = ""):
    print(f"  {PASS if ok else FAIL}  {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, ok))


def report_optional(name: str, ok: bool, detail: str = ""):
    marker = PASS if ok else SKIP
    print(f"  {marker}  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str):
    print(f"\n[{title}]")


def check_python() -> bool:
    section("1. Python バージョン")
    ok = sys.version_info >= (3, 11)
    report(
        "Python 3.11 以上",
        ok,
        f"検出: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    return ok


def check_env_vars() -> bool:
    section("2. 環境変数 (.env)")
    from dotenv import load_dotenv

    load_dotenv()

    all_ok = True
    for var, required in [
        ("WANDB_API_KEY", True),
        ("OPENAI_API_KEY", True),
        ("WANDB_ENTITY", False),
        ("WANDB_PROJECT", False),
    ]:
        value = os.getenv(var, "")
        ok = bool(value)
        if required:
            report(f"{var} が設定されている", ok)
            all_ok = all_ok and ok
        else:
            label = "設定済み" if ok else "未設定 (デフォルト値を使用)"
            print(f"  {PASS if ok else SKIP}  {var}: {label}")
    return all_ok


async def check_openai_judge() -> bool:
    section("3. OpenAI API (ジャッジモデル)")
    if not os.getenv("OPENAI_API_KEY"):
        report(
            "OPENAI_API_KEY",
            False,
            "イベント案内に従って OpenAI API キーを設定してください",
        )
        return False

    try:
        from art_e.rollout import (
            JUDGE_MODEL,
            JUDGE_REASONING_EFFORT,
            determine_if_answer_is_correct,
            get_active_judge_mode,
        )
        from art_e.data.types_enron import SyntheticQuery

        dummy = SyntheticQuery(
            id=0,
            question="What time is the meeting?",
            answer="3pm",
            message_ids=["<dummy>"],
            how_realistic=1.0,
            inbox_address="test@example.com",
            query_date="2001-01-01",
        )
        correct = await determine_if_answer_is_correct("The meeting is at 3pm.", dummy)
        incorrect = await determine_if_answer_is_correct("The meeting is at 9am.", dummy)
        ok = correct is True and incorrect is False
        mode = get_active_judge_mode()
        label = (
            f"ジャッジ判定 ({JUDGE_MODEL}, reasoning={JUDGE_REASONING_EFFORT})"
            if mode == "llm"
            else "ジャッジ判定 (heuristic fallback)"
        )
        report_optional(label, ok)
        return True
    except Exception as e:
        report_optional(
            "ジャッジモデルの呼び出し",
            False,
            f"{type(e).__name__}: {e}。当日は heuristic judge で続行できます",
        )
        return True


async def check_wandb_auth() -> bool:
    section("4. W&B 認証")
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.wandb.ai/graphql",
                auth=("api", os.environ["WANDB_API_KEY"]),
                json={"query": "query { viewer { username entity } }"},
                timeout=30,
            )
        viewer = resp.json().get("data", {}).get("viewer") or {}
        username = viewer.get("username")
        ok = resp.status_code == 200 and bool(username)
        report("WANDB_API_KEY が有効", ok, f"user: {username}" if ok else f"HTTP {resp.status_code}")
        return ok
    except Exception as e:
        report("WANDB_API_KEY が有効", False, f"{type(e).__name__}: {e}")
        return False


def check_hf_dataset() -> bool:
    section("5. Hugging Face データセット (学習/検証データ)")
    try:
        from art_e.data.query_iterators import load_synthetic_queries

        train = load_synthetic_queries(split="train", limit=2)
        test = load_synthetic_queries(split="test", limit=2)
        ok = len(train) == 2 and len(test) == 2
        report("corbt/enron_emails_sample_questions のダウンロード", ok)
        return ok
    except Exception as e:
        report("データセットのダウンロード", False, f"{type(e).__name__}: {e}")
        return False


def check_email_db() -> bool:
    section("6. メール DB (SQLite) の構築 + 検索ツール")
    try:
        from art_e.data.local_email_db import DEFAULT_DB_PATH, generate_database

        if not os.path.exists(DEFAULT_DB_PATH):
            print("  メール DB が見つからないため構築します (初回のみ、数分かかります)...")
        generate_database()
        report("メール DB の構築", os.path.exists(DEFAULT_DB_PATH))

        from art_e.data.query_iterators import load_synthetic_queries
        from art_e.email_search_tools import read_email, search_emails

        query = load_synthetic_queries(split="test", limit=1)[0]
        email = read_email(query.message_ids[0])
        ok_read = email is not None and bool(email.body)
        report("read_email ツール", ok_read)

        hits = search_emails(inbox=query.inbox_address, keywords=[(email.subject or "the").split()[0]])
        report("search_emails ツール", isinstance(hits, list))
        return ok_read
    except Exception as e:
        traceback.print_exc()
        report("メール DB / 検索ツール", False, f"{type(e).__name__}: {e}")
        return False


async def check_serverless_rl() -> bool:
    section("7. W&B Serverless RL (モデル登録 → 推論 → 削除)")
    try:
        import art
        from art.serverless.backend import ServerlessBackend

        from art_e.project_types import ProjectPolicyConfig

        backend = ServerlessBackend()
        model = art.TrainableModel(
            name="preflight-check",
            project=os.getenv("WANDB_PROJECT", "art-e-nano"),
            entity=os.getenv("WANDB_ENTITY"),
            base_model="OpenPipe/Qwen3-14B-Instruct",
            config=ProjectPolicyConfig(),
        )
        await model.register(backend)
        report("Serverless RL へのモデル登録", True, f"model id: {model.id}")

        response = await model.openai_client().chat.completions.create(
            model=model.get_inference_name(),
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_completion_tokens=10,
            timeout=120,
        )
        content = (response.choices[0].message.content or "").strip()
        report("W&B Inference での推論", bool(content), f"応答: {content[:40]!r}")

        # 確認用モデルを削除してクリーンアップ
        if model.id is not None:
            await backend._client.models.delete(model_id=model.id)
            print("  (preflight-check モデルを削除しました)")
        await backend.close()
        return True
    except Exception as e:
        traceback.print_exc()
        report("Serverless RL", False, f"{type(e).__name__}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-serverless", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("ART-E nano ハンズオン事前確認")
    print("=" * 60)

    check_python()
    env_ok = check_env_vars()

    if not env_ok:
        print("\n環境変数が不足しています。.env.sample を .env にコピーして値を設定してください。")
        sys.exit(1)

    await check_openai_judge()
    await check_wandb_auth()
    check_hf_dataset()
    check_email_db()

    if args.skip_serverless:
        section("7. W&B Serverless RL")
        print(f"  {SKIP}  --skip-serverless が指定されたためスキップ")
    else:
        await check_serverless_rl()

    print("\n" + "=" * 60)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"NG: {len(failed)} 件のチェックに失敗しました:")
        for name in failed:
            print(f"  - {name}")
        print("SETUP.md のトラブルシューティングを確認してください。")
        sys.exit(1)
    else:
        print("すべてのチェックに合格しました。当日はこのまま学習を実行できます:")
        print("  uv run python -m art_e.train")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
