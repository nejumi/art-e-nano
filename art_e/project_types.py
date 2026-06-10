from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """オリジナル ART-E のレシピを 1-2 時間に縮めたデフォルト値。

    方針 (オリジナル準拠):
    - 報酬はルーブリック (-2〜+2、IDK に部分点、誤答にペナルティ)
    - 正誤判定は OpenAI LLM ジャッジ (学習・検証とも)
    - 学習率はオリジナルと同じ 1.2e-5
    - 12 groups × 4 traj = 48 rollouts/step を 30 step (≈1,440 rollouts)
    - 検証は固定 seed の 96 問を 3 step ごとに実施
    """

    trajectories_per_group: int = 4
    groups_per_step: int = 12
    learning_rate: float = 1.2e-5
    eval_steps: int = 3
    val_set_size: int = 96
    training_dataset_size: int = 360
    num_epochs: int = 1
    max_steps: int = 30
    early_stop_patience: int = 4
    seed: int = 42


class ProjectPolicyConfig(BaseModel):
    max_turns: int = 6
    max_tokens: int = 2048
    stupid_simple_reward_fn: bool = False
    use_ruler: bool = False
    # RULER は主報酬を置き換えず、成功/不成功ベースの報酬に小さく足す。
    ruler_weight: float = 0.2
    idk_penalty: float = 0.2
    # 検証時のサンプリング温度。0.0 だと Qwen3 は即「わかりません」に
    # 寄りやすく、1.0 だと評価が荒れるため 0.7 を採用。
    eval_temperature: float = 0.7
    # ツールの呼び出し方式。
    # False: JSON テキスト (学習信号が安定、probs_corr ≈ 1)
    # True:  ネイティブ tool_calls (プロンプトのみ比較用)
    use_tools: bool = False

    training_config: TrainingConfig | None = None
