import polars as pl

import art

from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.rollout import rollout

# 全 run で同じ検証セットを使う (seed 固定・シャッフルなし)
DEFAULT_VAL_SEED = 42


def load_validation_scenarios(
    limit: int = 48,
    seed: int = DEFAULT_VAL_SEED,
) -> list[SyntheticQuery]:
    return load_synthetic_queries(
        split="test",
        limit=limit,
        shuffle=False,
        seed=seed,
    )


async def benchmark_model(
    model: art.Model,
    limit: int = 48,
    scenarios: list[SyntheticQuery] | None = None,
    swallow_exceptions: bool = True,
    log: bool = True,
) -> pl.DataFrame:
    """検証セット (test split) でモデルを低温度 (eval_temperature) で評価する。

    temperature=1 では同一チェックポイントでも正答率が大きく振れ、
    temperature=0 では Qwen3 が即「わかりません」に収束するため、
    デフォルト 0.3 で安定性と探索のバランスを取る。

    log=True かつモデルがバックエンドに登録済みであれば、結果を W&B にも記録する
    (split="val" として記録され、W&B のチャートで学習の進捗として確認できる)。
    """
    val_scenarios = scenarios or load_validation_scenarios(limit=limit)
    val_trajectories = await art.gather_trajectories(
        (
            rollout(model, scenario, for_training=False)
            for scenario in val_scenarios
        ),
        pbar_desc=f"validation {model.name}",
        max_exceptions=limit if swallow_exceptions else 0,
    )

    valid_trajectories = [t for t in val_trajectories if isinstance(t, art.Trajectory)]

    if log and model._backend is not None:
        await model.log(valid_trajectories, split="val")

    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))

    return avg_metrics
