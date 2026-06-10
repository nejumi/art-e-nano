import json
import os
import re
import textwrap
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, List

import weave
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

import art
from art import Trajectory
from art.utils import limit_concurrency

from art_e.data.types_enron import SyntheticQuery
from art_e.email_search_tools import read_email, search_emails
from art_e.project_types import ProjectPolicyConfig

# 回答の正誤判定 (LLM-as-a-judge) に使うモデル。
# オリジナルの art-e は gemini-2.0-flash / gpt-4o を使っていたが、最新モデルに更新。
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.5")
# auto: LLM を試し、quota 等で失敗したら heuristic に切替
# llm / heuristic で固定も可
JUDGE_MODE = os.getenv("JUDGE_MODE", "auto").lower()

_judge_client: AsyncOpenAI | None = None
_llm_judge_available: bool | None = None
_judge_mode_logged = False


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s:./-]", "", text)
    return text


def heuristic_answer_match(ai_answer: str, correct_answer: str) -> bool:
    """API 不要の正答判定。検証曲線の再現性を優先する。"""
    ai = normalize_answer(ai_answer)
    correct = normalize_answer(correct_answer)
    if not ai or not correct:
        return False
    if ai == correct or correct in ai or ai in correct:
        return True
    return SequenceMatcher(None, ai, correct).ratio() >= 0.72


def get_active_judge_mode() -> str:
    if JUDGE_MODE == "heuristic":
        return "heuristic"
    if JUDGE_MODE == "llm":
        return "llm"
    if _llm_judge_available is False:
        return "heuristic"
    return "llm"


def get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncOpenAI()  # OPENAI_API_KEY を使用
    return _judge_client


# ツール定義。inbox 引数はロールアウト側でユーザーのアドレスを自動的に渡すため、
# モデルに見せるスキーマからは削除する。
search_tool = convert_to_openai_tool(search_emails)
del search_tool["function"]["parameters"]["properties"]["inbox"]
search_tool["function"]["parameters"]["required"].remove("inbox")


def return_final_answer(answer: str, sources: List[str] | None) -> str:
    """
    This function is used to return the final answer to the user's query.
    It should be called with the answer and the sources. If you cannot find the answer, you should return "I don't know" with an empty list of sources.

    Args:
        answer: (str) the answer to the user's query. If you cannot find the answer, you should return "I don't know" with an empty list of sources.
        sources: (list[str]) a list of message ids that are relevant to the query. Usually there will be only one.

    Returns:
        (str) the final answer to the user's query
    """
    ...


tools: list[ChatCompletionToolParam] = [
    search_tool,
    convert_to_openai_tool(read_email),
    convert_to_openai_tool(return_final_answer),
]  # type: ignore


@dataclass
class FinalRubric:
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_metrics(self) -> dict[str, float | int]:
        metrics = {k: int(v) for k, v in asdict(self).items()}
        metrics["task_success"] = int(
            self.answer_correct or (self.attempted_answer and self.sources_correct)
        )
        return metrics


def calculate_reward(
    policy_config: ProjectPolicyConfig, rubric: FinalRubric, traj: Trajectory
) -> float:
    # nano 版のシンプル報酬: 正答率の最大化だけに集中する。
    #   正解:                +1.0
    #   回答したが誤り:        0.0
    #   IDK / ターン切れ:    -0.2  (「わかりません」への逃避を防ぐ)
    #   フォーマットエラー:   -1.0
    # オリジナルのルーブリック報酬 (下) は「わかりません」に部分点を与えるため
    # ハルシネーション抑制には優れるが、短いハンズオンでは IDK への退避が先に
    # 学習されて正答率が一時的に下がるディップが発生する。
    if policy_config.stupid_simple_reward_fn:
        if rubric.answer_correct:
            return 1.0
        if (
            rubric.cant_parse_tool_call
            or rubric.bad_tool_call_name
            or rubric.bad_tool_call_args
        ):
            return -1.0
        if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
            return -0.2
        return 0.0

    # 注意: 部分報酬の合計は常に 0.5 未満になるようにする。
    partial_rewards = 0
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0

    # フォーマットエラー: 報酬は -2 〜 -1
    if rubric.cant_parse_tool_call:
        return -2 + partial_rewards

    if rubric.bad_tool_call_name:
        return -1.9 + partial_rewards

    if rubric.bad_tool_call_args:
        return -1.8 + partial_rewards

    # フォーマットは正しいが回答が間違い: 報酬は -1 〜 0
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1 + partial_rewards

    # 回答を返さなかった: 報酬は 0 〜 1
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        return 0 + partial_rewards

    # 回答が正しい: 報酬は 1 〜 2
    if rubric.answer_correct:
        reward = 1
        reward += 0.3 if rubric.sources_correct else 0

        # 余計な出典を含めなかったことへの追加報酬。
        reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0

        # 少ないターン数で回答できたことへの追加報酬。
        reward += 0.1 * (1 - rubric.num_turns / policy_config.max_turns)
        return reward

    traj.logs.append(f"Rubric: {rubric}")
    traj.logs.append("Rubric not handled properly")
    raise ValueError("Rubric is not handled properly")


def tool_response(response: Any, tool_call_id: str | None) -> ChatCompletionMessageParam:
    """ツールの実行結果を会話に追加するメッセージを生成する。

    ネイティブツールコール時は role=tool、JSON テキスト方式では role=user で返す。
    """
    if tool_call_id is not None:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(response),
        }
    return {
        "role": "user",
        "content": json.dumps(response),
    }


@weave.op
async def determine_if_answer_is_correct(answer: str, query: SyntheticQuery) -> bool:
    global _llm_judge_available, _judge_mode_logged

    mode = get_active_judge_mode()
    if mode == "heuristic":
        if not _judge_mode_logged:
            print("[judge] Using heuristic answer matching (JUDGE_MODE=heuristic or LLM unavailable)")
            _judge_mode_logged = True
        return heuristic_answer_match(answer, query.answer)

    system_prompt = (
        "You will be given an question and two different answers to the question, "
        "the correct answer and the answer given by an AI. Your job is to determine "
        "if the answer given by the AI is correct. Return True if the answer is "
        "semantically similar to the correct answer, and False otherwise. "
        "Return only the word True or False, no other text."
    )

    try:
        response = await get_judge_client().chat.completions.create(
            model=JUDGE_MODEL,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Question: {query.question}\nCorrect answer: {query.answer}\nAI answer: {answer}",
                },
            ],
        )
        _llm_judge_available = True
        content = response.choices[0].message.content or ""
        return content.strip().lower().startswith("t")
    except RateLimitError as e:
        if JUDGE_MODE == "llm":
            raise
        _llm_judge_available = False
        if not _judge_mode_logged:
            print(f"[judge] LLM unavailable ({e}), falling back to heuristic matching")
            _judge_mode_logged = True
        return heuristic_answer_match(answer, query.answer)


@limit_concurrency(32, derive_key=lambda model, scenario, **kwargs: model.name)
@weave.op
async def rollout(
    model: art.Model,
    scenario: SyntheticQuery,
    *,
    for_training: bool = True,
) -> Trajectory:
    rubric = FinalRubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"email_inbox": scenario.inbox_address, "scenario_id": scenario.id},
    )
    assert isinstance(model.config, ProjectPolicyConfig)

    system_prompt = textwrap.dedent(f"""\
        You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {model.config.max_turns} turns to find the answer, so if your first search doesn't find the answer, you can try with different keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}
    """)

    if model.config.use_tools:
        traj.tools = tools
    else:
        # ツールを JSON テキストで呼び出す方式 (オリジナル ART-E の既定)。
        # assistant メッセージが純テキストになるため、学習時のトークン列が
        # サンプリング時と完全に一致し、学習が安定する。
        system_prompt += textwrap.dedent(f"""\

            Here are the tools you can use:
            {json.dumps(tools, ensure_ascii=False)}

            Respond with a valid JSON object with the following fields:
            - tool_name: (str) the name of the tool to use
            - tool_args: (JSON) the arguments to pass to the tool

            For example, to read a specific email, you should respond with:
            {{
                "tool_name": "read_email",
                "tool_args": {{
                    "message_id": "<12635597.1075855702772.JavaMail.evans@thyme>"
                }}
            }}
        """)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]

    client = model.openai_client()

    while True:
        rubric.num_turns += 1

        if rubric.num_turns > model.config.max_turns:
            rubric.ran_out_of_turns = True
            break

        extra_params: dict[str, Any] = {}
        # 学習ロールアウト: temperature=1 (GRPO の多様性)
        # 検証: eval_temperature (低め固定。0=即IDK、1=評価が再現不能)
        if for_training and model.trainable:
            extra_params["temperature"] = 1.0
        else:
            extra_params["temperature"] = model.config.eval_temperature
        if model.config.use_tools:
            extra_params["tools"] = tools
            if not model.trainable:
                # プロンプトのみの比較用モデルにはツール呼び出しを強制する。
                # (注意: tool_choice に明示的な None を渡すと W&B Inference (vLLM) の
                #  ツールコールパーサが無効になるため、未指定との分岐にしている)
                extra_params["tool_choice"] = "required"

        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=traj.messages(),
            max_completion_tokens=model.config.max_tokens,
            timeout=600,
            **extra_params,
        )

        if response.usage is not None:
            rubric.prompt_tokens += response.usage.prompt_tokens
            rubric.completion_tokens += response.usage.completion_tokens

        choice = response.choices[0]
        assert isinstance(choice, Choice)

        # ロールアウトは 1 ターンに 1 ツール呼び出しのみ扱うため、並列呼び出しは先頭だけ残す。
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 1:
            choice.message.tool_calls = choice.message.tool_calls[:1]

        traj.messages_and_choices.append(choice)

        tool_call_id = None
        if model.config.use_tools:
            tool_call = (
                choice.message.tool_calls[0] if choice.message.tool_calls else None
            )
            if tool_call is None:
                rubric.bad_tool_call_args = True
                break
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
                assert isinstance(tool_args, dict)
            except Exception:
                rubric.bad_tool_call_args = True
                break
        else:
            # JSON テキストからツール呼び出しをパースする
            raw_content = choice.message.content
            if raw_content is None:
                rubric.cant_parse_tool_call = True
                break
            start_index = raw_content.find("{")
            end_index = raw_content.rfind("}")
            if not (start_index != -1 and end_index != -1 and start_index < end_index):
                rubric.cant_parse_tool_call = True
                break
            try:
                parsed = json.loads(raw_content[start_index : end_index + 1])
            except Exception as e:
                traj.logs.append(f"Error parsing tool call: {e}")
                rubric.cant_parse_tool_call = True
                break
            if "tool_args" not in parsed or not isinstance(parsed.get("tool_args"), dict):
                rubric.bad_tool_call_args = True
                traj.logs.append(f"Tool call missing tool_args: {parsed}")
                break
            tool_name = parsed.get("tool_name")
            tool_args = parsed["tool_args"]

        match tool_name:
            case "search_emails":
                try:
                    search_results = search_emails(
                        **tool_args,
                        inbox=scenario.inbox_address,
                    )
                    traj.messages_and_choices.append(
                        tool_response(
                            [asdict(r) for r in search_results],
                            tool_call_id,
                        )
                    )
                    for r in search_results:
                        if r.message_id == scenario.message_ids[0]:
                            rubric.ever_found_right_email = True
                except Exception as e:
                    rubric.bad_tool_call_args = True
                    traj.logs.append(f"Error searching emails: {e}")
                    break
            case "read_email":
                message_id_to_read = tool_args.get("message_id")
                if not isinstance(message_id_to_read, str):
                    rubric.bad_tool_call_args = True
                    break
                if message_id_to_read == scenario.message_ids[0]:
                    rubric.ever_read_right_email = True
                email_content = read_email(message_id_to_read)
                if email_content is None:
                    traj.messages_and_choices.append(
                        tool_response({"error": "Email not found"}, tool_call_id)
                    )
                    rubric.ever_tried_to_read_invalid_email = True
                else:
                    traj.messages_and_choices.append(
                        tool_response(email_content.model_dump(), tool_call_id)
                    )
            case "return_final_answer":
                final_answer = tool_args.get("answer")
                final_sources = tool_args.get("sources")

                if (
                    final_answer is None
                    or final_sources is None
                    or not isinstance(final_sources, list)
                ):
                    rubric.bad_tool_call_args = True
                    break

                rubric.num_sources = len(final_sources)

                if final_answer == "I don't know":
                    rubric.returned_i_dont_know = True
                else:
                    rubric.attempted_answer = True
                    # RULER 学習時は judge を検証だけに使い、学習ロールアウトの API コストを抑える
                    skip_judge = (
                        for_training
                        and model.config.use_ruler
                        and model.trainable
                    )
                    if skip_judge:
                        rubric.sources_correct = (
                            scenario.message_ids[0] in final_sources
                        )
                    else:
                        rubric.answer_correct = await determine_if_answer_is_correct(
                            final_answer, scenario
                        )
                        rubric.sources_correct = (
                            scenario.message_ids[0] in final_sources
                        )
                break
            case _:
                rubric.bad_tool_call_name = True
                break

    traj.reward = calculate_reward(model.config, rubric, traj)
    traj.metrics = rubric.to_metrics()
    return traj.finish()


if __name__ == "__main__":
    import asyncio

    import yaml
    from dotenv import load_dotenv

    from art_e.data.query_iterators import load_synthetic_queries

    load_dotenv()

    traj = asyncio.run(
        rollout(
            art.Model(
                name="gpt-5.5",
                project="art-e-nano",
                config=ProjectPolicyConfig(use_tools=True),
                inference_api_key=os.getenv("OPENAI_API_KEY"),
                inference_base_url="https://api.openai.com/v1",
            ),
            load_synthetic_queries(split="test", limit=1)[0],
        )
    )
    print(yaml.dump(traj.for_logging()))
