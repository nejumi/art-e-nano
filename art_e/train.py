"""ART-E nano: W&B Serverless RL でメール検索エージェントを学習する。

実行方法:
    uv run python -m art_e.train

設計 (オリジナル ART-E 準拠):
- 報酬はルーブリック (-2〜+2)、正誤判定は OpenAI LLM ジャッジ
- 検証は eval_temperature=0.7 + 固定 96 問を 3 step ごと
- 学習率 1.2e-5 / 48 rollouts/step × 30 step
"""

import asyncio
import os
import time
from typing import List

import wandb
import weave
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

import art
from art.rewards import ruler_score_group
from art.utils import iterate_dataset
from art.utils.strip_logprobs import strip_logprobs

from art_e.data.local_email_db import generate_database
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.evaluate.benchmark import benchmark_model, load_validation_scenarios
from art_e.project_types import ProjectPolicyConfig, TrainingConfig
from art_e.rollout import get_active_judge_mode, rollout

load_dotenv()

BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"
RULER_JUDGE = os.getenv("RULER_JUDGE_MODEL", "openai/gpt-4.1-mini")

model = art.TrainableModel(
    name=os.getenv("MODEL_NAME", "art-e-nano-workshop"),
    project=os.getenv("WANDB_PROJECT", "art-e-nano"),
    entity=os.getenv("WANDB_ENTITY"),
    base_model=BASE_MODEL,
    config=ProjectPolicyConfig(
        max_turns=6,
        stupid_simple_reward_fn=os.getenv("SIMPLE_REWARD", "false").lower() in ("1", "true", "yes"),
        use_ruler=os.getenv("USE_RULER", "false").lower() in ("1", "true", "yes"),
        ruler_weight=float(os.getenv("RULER_WEIGHT", "0.2")),
        idk_penalty=float(os.getenv("IDK_PENALTY", "0.2")),
        training_config=TrainingConfig(),
    ),
)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=10, max=120), reraise=True)
async def register_model_with_retry(model: art.TrainableModel, backend) -> None:
    await model.register(backend)


def init_wandb_and_weave(model: art.TrainableModel) -> None:
    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=model.project,
        name=model.name,
        id=model.name,
        resume="allow",
    )
    model._wandb_run = run

    run.define_metric("training_step")
    run.define_metric("time/wall_clock_sec")
    for section in (
        "reward", "loss", "throughput", "costs", "time", "data",
        "train", "val", "test", "discarded",
    ):
        run.define_metric(f"{section}/*", step_metric="training_step")

    weave.init(
        f"{model.entity or os.getenv('WANDB_ENTITY')}/{model.project}",
        settings={"print_call_link": False},
        global_postprocess_output=strip_logprobs,
    )


def _metric_value(avg_metrics, name: str) -> float:
    row = avg_metrics.row(0, named=True)
    val = row.get(name)
    return float(val) if val is not None else 0.0


def _apply_env_overrides(tcfg: TrainingConfig, policy: ProjectPolicyConfig) -> None:
    """実験時にコードを触らずハイパラを振るための最小限の override。"""
    int_overrides = {
        "TRAJECTORIES_PER_GROUP": "trajectories_per_group",
        "GROUPS_PER_STEP": "groups_per_step",
        "EVAL_STEPS": "eval_steps",
        "VAL_SET_SIZE": "val_set_size",
        "TRAINING_DATASET_SIZE": "training_dataset_size",
        "MAX_STEPS": "max_steps",
        "EARLY_STOP_PATIENCE": "early_stop_patience",
        "SEED": "seed",
    }
    for env_name, field_name in int_overrides.items():
        if env_name in os.environ:
            setattr(tcfg, field_name, int(os.environ[env_name]))

    if "LEARNING_RATE" in os.environ:
        tcfg.learning_rate = float(os.environ["LEARNING_RATE"])
    if "EVAL_TEMPERATURE" in os.environ:
        policy.eval_temperature = float(os.environ["EVAL_TEMPERATURE"])
    if "RULER_WEIGHT" in os.environ:
        policy.ruler_weight = float(os.environ["RULER_WEIGHT"])
    if "IDK_PENALTY" in os.environ:
        policy.idk_penalty = float(os.environ["IDK_PENALTY"])


async def run_training(model: art.TrainableModel):
    generate_database()

    assert isinstance(model.config, ProjectPolicyConfig)
    assert model.config.training_config is not None
    tcfg = model.config.training_config
    _apply_env_overrides(tcfg, model.config)

    from art.serverless.backend import ServerlessBackend

    backend = ServerlessBackend()

    fresh_start = os.getenv("FRESH_START", "true").lower() in ("1", "true", "yes")
    if fresh_start:
        try:
            await backend.delete(model)
            print(f"Deleted existing Serverless model '{model.name}' (FRESH_START=true)")
        except Exception:
            pass

    await register_model_with_retry(model, backend)
    init_wandb_and_weave(model)

    val_scenarios = load_validation_scenarios(limit=tcfg.val_set_size, seed=tcfg.seed)
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train",
        limit=tcfg.training_dataset_size,
        shuffle=True,
        seed=tcfg.seed,
    )

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)} (fixed seed={tcfg.seed})")
    print(f"Judge mode: {get_active_judge_mode()} (JUDGE_MODE={os.getenv('JUDGE_MODE', 'auto')})")
    print(f"RULER: {model.config.use_ruler} (judge={RULER_JUDGE})")
    if model.config.use_ruler:
        print(f"RULER mix: reward = rubric_reward + {model.config.ruler_weight} * ruler_score")
    print(f"Reward: {'simple' if model.config.stupid_simple_reward_fn else 'rubric'}")
    print(f"IDK penalty: {model.config.idk_penalty}")
    print(
        "Config: "
        f"traj/group={tcfg.trajectories_per_group}, groups/step={tcfg.groups_per_step}, "
        f"lr={tcfg.learning_rate}, max_steps={tcfg.max_steps}, val={tcfg.val_set_size}, "
        f"eval_temp={model.config.eval_temperature}"
    )

    async def score_group(group: art.TrajectoryGroup) -> art.TrajectoryGroup | None:
        if not model.config.use_ruler:
            return group
        scored_group = await ruler_score_group(
            group,
            judge_model=RULER_JUDGE,
            swallow_exceptions=True,
        )
        if scored_group is None:
            return None
        for traj in scored_group.trajectories:
            independent_reward = float(
                traj.metrics.get("independent_reward", traj.reward)
            )
            ruler_score = float(traj.metrics.get("ruler_score", 0.0))
            mixed_reward = independent_reward + model.config.ruler_weight * ruler_score
            traj.metrics["mixed_reward"] = mixed_reward
            traj.reward = mixed_reward
        return scored_group

    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=tcfg.groups_per_step,
        num_epochs=tcfg.num_epochs,
        initial_step=await model.get_step(),
    )

    eval_history: list[tuple[int, float, float, float, float]] = []
    best_val = -1.0
    best_step = -1
    no_improve = 0
    training_start = time.monotonic()

    for batch in train_iterator:
        if batch.step >= tcfg.max_steps:
            print(f"\nReached max_steps={tcfg.max_steps}, stopping.")
            break

        step_start = time.monotonic()

        if batch.step % tcfg.eval_steps == 0:
            print(f"\n--- Evaluating at step {batch.step} (eval_temperature={model.config.eval_temperature}) ---")
            avg_metrics = await benchmark_model(
                model,
                scenarios=val_scenarios,
                log=True,
            )
            acc = _metric_value(avg_metrics, "answer_correct")
            task = _metric_value(avg_metrics, "task_success")
            src = _metric_value(avg_metrics, "sources_correct")
            idk = _metric_value(avg_metrics, "returned_i_dont_know")
            turns = _metric_value(avg_metrics, "num_turns")
            eval_history.append((batch.step, acc, task, src, idk, turns))
            print(avg_metrics.select(
                "answer_correct", "task_success", "sources_correct",
                "returned_i_dont_know", "num_turns", "reward",
            ))

            # ワークショップ指標: answer_correct (OpenAI ジャッジによる正答率)
            score = acc
            if score > best_val + 0.005:
                best_val = score
                best_step = batch.step
                no_improve = 0
                print(f"  ★ new best: {best_val:.1%} at step {best_step}")
            else:
                no_improve += 1
                if no_improve >= tcfg.early_stop_patience and batch.step > 0:
                    print(
                        f"\nEarly stopping: no improvement for {tcfg.early_stop_patience} evals "
                        f"(best {best_val:.1%} at step {best_step})"
                    )
                    break

        eval_done = time.monotonic()

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, for_training=True)
                    for _ in range(tcfg.trajectories_per_group)
                )
                for scenario in batch.items
            ),
            pbar_desc=f"step {batch.step} rollouts",
            after_each=score_group if model.config.use_ruler else None,
            max_exceptions=tcfg.groups_per_step * tcfg.trajectories_per_group // 4,
        )
        groups = [g for g in groups if g is not None]
        if not groups:
            print(f"[step {batch.step}] All groups failed scoring, skipping train.")
            continue

        rollouts_done = time.monotonic()

        train_kwargs: dict = {"learning_rate": tcfg.learning_rate}
        if "PPO_EPSILON" in os.environ:
            train_kwargs["ppo"] = True
            train_kwargs["epsilon"] = float(os.environ["PPO_EPSILON"])
        result = await backend.train(model, groups, **train_kwargs)
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")

        now = time.monotonic()
        print(
            f"[step {batch.step}] eval: {eval_done - step_start:.0f}s, "
            f"rollouts: {rollouts_done - eval_done:.0f}s, "
            f"train: {now - rollouts_done:.0f}s, "
            f"elapsed: {(now - training_start) / 60:.1f}min"
        )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (fixed val set, eval_temperature={})".format(model.config.eval_temperature))
    print("=" * 60)
    if eval_history:
        baseline = eval_history[0][1]
        print(f"{'step':>6}  {'acc':>7}  {'task':>7}  {'src':>7}  {'IDK%':>7}  {'turns':>7}")
        for step, acc, task, src, idk, turns in eval_history:
            marker = " ← best" if step == best_step else ""
            print(f"{step:>6}  {acc:>6.1%}  {task:>6.1%}  {src:>6.1%}  {idk:>6.1%}  {turns:>7.1f}{marker}")
        print("-" * 60)
        print(f"Baseline answer_correct (step 0): {baseline:.1%}")
        print(f"Best answer_correct (step {best_step}): {best_val:.1%}  (Δ {best_val - baseline:+.1%})")
        if best_step == eval_history[-1][0]:
            print("Final step matched best checkpoint.")
        elif best_val - baseline >= 0.05:
            print(
                f"Note: peak was at step {best_step}. "
                "For deployment, use that checkpoint (not necessarily the last step)."
            )
    print(f"\nTotal wall-clock: {(time.monotonic() - training_start) / 60:.1f} min")


if __name__ == "__main__":
    asyncio.run(run_training(model))
