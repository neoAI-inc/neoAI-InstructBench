import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from utils.closed_validators import validate_closed_instruction
from utils.llm_utils import build_prompt_from_row, get_client

JUDGE_PROMPT = """あなたは厳格かつ一貫した採点者です。以下の情報に基づき、「評価対象の指示」に対して、「モデルの出力」が従っているかのみを評価してください。その他の指示は評価対象に含めません。

[元の依頼文]
{raw_input_text}

[モデルに与えた最終プロンプト]
{prompt_text}

[評価対象の指示]
{constraint}

[モデルの出力]
{response_text}

評価方針:
- 評価対象は上記の「評価対象の指示」に限定します。指示に直接関係しない観点は無視してください。
- 指示の意図を正しく汲み取り、「モデルの出力」がその意図に明確かつ十分に応えているかを判断してください。
- 部分的・曖昧・条件付きの遵守は 0 とし、明確に満たす場合のみ 1 とします。

出力要件:
1) 最初に「理由: 」で 1〜3 文の根拠を簡潔に述べる。
2) 最終行を必ず「点数: 1」(従っている) または「点数: 0」(従っていない) のみで記載する。
3) それ以外の出力や装飾は行わない。
"""


def parse_score(judge_text: str) -> int:
    m = re.search(r"点数\s*[:：]\s*([01])\s*$", (judge_text or "").strip(), re.MULTILINE)
    return int(m.group(1)) if m else 0


def judge_responses(
    judge_model: str,
    max_tokens: int,
    inference_path: Path,
    output_path: Path,
) -> None:
    client, caller = get_client(judge_model)

    with inference_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            inf_obj = json.loads(line)
            obj_id = inf_obj.get("id")
            prompt_text = build_prompt_from_row(inf_obj)
            response_text = inf_obj.get("response", "")

            instructions = inf_obj.get("instructions", [])
            if not isinstance(instructions, list):
                instructions = []

            scores_sum = 0
            for inst in instructions:
                if not isinstance(inst, dict):
                    continue

                inst_text = str(inst.get("instruction", "")).strip()
                if not inst_text:
                    inst["judge_model"] = judge_model
                    inst["judge_response"] = "Skipped: missing instruction"
                    inst["score"] = 0
                    continue

                eval_type = inst.get("eval_type", "open")
                eval_config = inst.get("eval_config")

                if eval_type == "closed" and eval_config:
                    try:
                        score = validate_closed_instruction(eval_config, response_text)
                        inst["score"] = score
                        scores_sum += score
                    except Exception as exc:
                        print(f"Error in rule-based validator for id={obj_id}: {exc}")
                        inst["score"] = 0
                    continue

                raw_input_text = (str(inf_obj.get("raw_input", "")) or "").strip() or "（未提供）"
                eval_prompt = JUDGE_PROMPT.format(
                    raw_input_text=raw_input_text,
                    prompt_text=prompt_text,
                    constraint=inst_text,
                    response_text=str(response_text),
                )

                judge_out = None
                for attempt in range(3):
                    try:
                        judge_out = caller(client, judge_model, eval_prompt, max_tokens)
                        break
                    except Exception as exc:
                        if attempt == 2:
                            judge_out = f"Error: {exc}"
                            print(f"Error judging id={obj_id}: {exc}")
                        time.sleep(0.5)

                score = parse_score(judge_out or "")
                scores_sum += score

                inst["judge_model"] = judge_model
                inst["judge_response"] = judge_out
                inst["score"] = score

            overall_score = round(scores_sum / len(instructions), 3) if instructions else 0.0

            out_obj: Dict[str, Any] = dict(inf_obj)
            out_obj["response"] = response_text
            out_obj["overall_score"] = overall_score
            out_obj["num_instructions"] = len(instructions)

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            time.sleep(0.05)

            status = "success" if overall_score >= 0.0 else "error"
            print(
                f"Judged {line_num}: id={inf_obj["id"]}, score={overall_score} ({scores_sum}/{len(instructions)}), status={status}"
            )

    print(f"Judge completed. Output: {output_path}")


def analyze_results(input_path: Path, output_path: Path) -> Dict[str, Any]:
    total_prompts = 0
    prompt_passes = 0
    total_instructions = 0
    instruction_passes = 0

    closed_total = 0
    closed_passes = 0
    open_total = 0
    open_passes = 0

    per_inst_count: Dict[int, Dict[str, int]] = {}
    per_category: Dict[str, Dict[str, int]] = {}
    per_eval_type: Dict[str, Dict[str, int]] = {}

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj: Dict[str, Any] = json.loads(line)
            total_prompts += 1

            overall_score = float(obj.get("overall_score", 0))
            if overall_score == 1.0:
                prompt_passes += 1

            inst_list: List[Dict[str, Any]] = obj.get("instructions") or []
            if not isinstance(inst_list, list):
                inst_list = []

            n_insts = len(inst_list)
            bucket = per_inst_count.setdefault(n_insts, {"total": 0, "passes": 0})
            bucket["total"] += 1
            if overall_score == 1.0:
                bucket["passes"] += 1

            for inst in inst_list:
                if not isinstance(inst, dict):
                    continue
                score = int(inst.get("score", 0))
                total_instructions += 1
                instruction_passes += score

                eval_type = inst.get("eval_type", "open")
                if eval_type == "closed":
                    closed_total += 1
                    closed_passes += score
                else:
                    open_total += 1
                    open_passes += score

                et_bucket = per_eval_type.setdefault(eval_type, {"total": 0, "passes": 0})
                et_bucket["total"] += 1
                et_bucket["passes"] += score

                cat = inst.get("category", "UNKNOWN")
                c_bucket = per_category.setdefault(cat, {"total": 0, "passes": 0})
                c_bucket["total"] += 1
                c_bucket["passes"] += score

    prompt_accuracy = prompt_passes / total_prompts if total_prompts > 0 else 0.0
    instruction_accuracy = instruction_passes / total_instructions if total_instructions > 0 else 0.0

    prompt_accuracy_by_inst_count: Dict[str, Dict[str, Any]] = {}
    for k, stats in sorted(per_inst_count.items()):
        acc = stats["passes"] / stats["total"] if stats["total"] > 0 else 0.0
        prompt_accuracy_by_inst_count[str(k)] = {
            "total": stats["total"],
            "passes": stats["passes"],
            "accuracy": acc,
        }

    instruction_accuracy_by_category: Dict[str, Dict[str, Any]] = {}
    for cat, stats in sorted(per_category.items()):
        acc = stats["passes"] / stats["total"] if stats["total"] > 0 else 0.0
        instruction_accuracy_by_category[cat] = {
            "total": stats["total"],
            "passes": stats["passes"],
            "accuracy": acc,
        }

    closed_accuracy = closed_passes / closed_total if closed_total > 0 else 0.0
    open_accuracy = open_passes / open_total if open_total > 0 else 0.0

    instruction_accuracy_by_eval_type: Dict[str, Dict[str, Any]] = {}
    for eval_type, stats in sorted(per_eval_type.items()):
        acc = stats["passes"] / stats["total"] if stats["total"] > 0 else 0.0
        instruction_accuracy_by_eval_type[eval_type] = {
            "total": stats["total"],
            "passes": stats["passes"],
            "accuracy": acc,
        }

    summary = {
        "total_prompts": total_prompts,
        "prompt_passes": prompt_passes,
        "prompt_accuracy": prompt_accuracy,
        "total_instructions": total_instructions,
        "instruction_passes": instruction_passes,
        "instruction_accuracy": instruction_accuracy,
        "closed_total": closed_total,
        "closed_passes": closed_passes,
        "closed_accuracy": closed_accuracy,
        "open_total": open_total,
        "open_passes": open_passes,
        "open_accuracy": open_accuracy,
        "prompt_accuracy_by_inst_count": prompt_accuracy_by_inst_count,
        "instruction_accuracy_by_category": instruction_accuracy_by_category,
        "instruction_accuracy_by_eval_type": instruction_accuracy_by_eval_type,
    }

    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Results ===")
    print(f"Prompt accuracy: {summary['prompt_accuracy']:.4f} ({summary['prompt_passes']}/{summary['total_prompts']})")
    print(
        f"Instruction accuracy: {summary['instruction_accuracy']:.4f} ({summary['instruction_passes']}/{summary['total_instructions']})"
    )

    print("\n--- By eval_type ---")
    print(f"Closed-end: {summary['closed_accuracy']:.4f} ({summary['closed_passes']}/{summary['closed_total']})")
    print(f"Open-end: {summary['open_accuracy']:.4f} ({summary['open_passes']}/{summary['open_total']})")

    if summary.get("prompt_accuracy_by_inst_count"):
        print("\nBy instruction count:")
        for k in sorted(summary["prompt_accuracy_by_inst_count"].keys(), key=int):
            stats = summary["prompt_accuracy_by_inst_count"][k]
            print(f"  n={k}: {stats['accuracy']:.4f} ({stats['passes']}/{stats['total']})")

    if summary.get("instruction_accuracy_by_category"):
        print("\nBy category:")
        for cat in sorted(summary["instruction_accuracy_by_category"].keys(), key=str.lower):
            stats = summary["instruction_accuracy_by_category"][cat]
            print(f"  {cat}: {stats['accuracy']:.4f} ({stats['passes']}/{stats['total']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate instruction adherence")
    parser.add_argument("--eval-model", required=True, help="Judge model name (e.g., gpt-5, openai/gpt-oss-120b)")
    parser.add_argument("--inference", required=True, help="Path to inference results JSONL")
    parser.add_argument("--max_tokens", type=int, default=10240, help="Max tokens for judge")
    args = parser.parse_args()

    inference_path = Path(args.inference)
    results_dir = inference_path.parent
    results_dir.mkdir(exist_ok=True)

    inference_stem = inference_path.stem
    judged_path = results_dir / f"{inference_stem}_judged.jsonl"
    analysis_path = results_dir / f"{inference_stem}_analysis.json"

    print(f"Judge: {args.eval_model}")
    judge_responses(
        judge_model=args.eval_model,
        max_tokens=args.max_tokens,
        inference_path=inference_path,
        output_path=judged_path,
    )

    summary = analyze_results(judged_path, analysis_path)
    print_summary(summary)
    print(f"\nSaved: {analysis_path}")


if __name__ == "__main__":
    main()
