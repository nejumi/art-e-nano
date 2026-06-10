"""プロンプトのみ (RL なし) の商用モデルをベンチマークする比較用スクリプト。

実行方法:
    uv run python -m art_e.evaluate.benchmark_prompted_models

学習済みエージェントとの比較対象として、最新の OpenAI モデルを同じ
検証セット・同じツールで評価します。
"""

import asyncio
import os

import weave
from dotenv import load_dotenv

import art

from art_e.data.local_email_db import generate_database
from art_e.evaluate.benchmark import benchmark_model
from art_e.project_types import ProjectPolicyConfig

load_dotenv()

# 比較対象のモデル (OpenAI API)。必要に応じて追加・削除してください。
PROMPTED_MODELS = [
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-4.1",
]


async def main(limit: int = 48):
    weave_project = os.getenv("WANDB_PROJECT", "art-e-nano")
    if entity := os.getenv("WANDB_ENTITY"):
        weave_project = f"{entity}/{weave_project}"
    weave.init(weave_project)

    generate_database()

    for model_name in PROMPTED_MODELS:
        model = art.Model(
            name=model_name,
            project=os.getenv("WANDB_PROJECT", "art-e-nano"),
            # 商用モデルはネイティブのツールコールで評価する
            config=ProjectPolicyConfig(max_turns=10, use_tools=True),
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
        )
        results = await benchmark_model(model, limit=limit, log=False)
        print(f"\n=== {model_name} ===")
        print(results.select("answer_correct", "reward", "num_turns"))


if __name__ == "__main__":
    asyncio.run(main())
