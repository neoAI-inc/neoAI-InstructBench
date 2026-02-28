import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from utils.llm_utils import build_prompt_from_row, get_client


def process_file(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_tokens: int,
) -> None:
    client, caller = get_client(model_name)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate[str](fin, start=1):
            line = line.strip()
            if not line:
                continue

            obj: Dict[str, Any] = json.loads(line)
            prompt_text = build_prompt_from_row(obj)

            response_text = None
            for attempt in range(3):
                try:
                    response_text = caller(client, model_name, prompt_text, max_tokens)
                    if response_text and response_text.strip():
                        obj["response"] = response_text
                        break
                    else:
                        print(f"Empty response on attempt {attempt + 1}/3 for line {line_num}, retrying...")
                        response_text = None
                except Exception as exc:
                    if attempt == 2:
                        obj["error"] = str(exc)
                        print(f"Error line {line_num}: {exc}")
                    else:
                        print(f"Error on attempt {attempt + 1}/3 for line {line_num}: {exc}, retrying...")
                    time.sleep(0.5)

            obj["model_name"] = model_name
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            time.sleep(0.1)

            status = "success" if response_text else "error"
            item_id = obj["id"]
            print(f"Processed {line_num}: id={item_id}, status={status}")

    print(f"Completed. Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Get LLM responses for instruction benchmark")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-5, openai/gpt-oss-120b)")
    parser.add_argument("--input", default="data/neoai-instruct-bench.jsonl", help="Input JSONL path")
    parser.add_argument("--output", help="Output JSONL path (default: results/{model}.jsonl)")
    parser.add_argument("--max_tokens", type=int, default=10240, help="Max output tokens")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        model_stem = args.model.split("@")[0].split("/")[-1]
        output_path = results_dir / f"{model_stem}.jsonl"

    print(f"Model: {args.model}")
    process_file(
        input_path=input_path,
        output_path=output_path,
        model_name=args.model,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
