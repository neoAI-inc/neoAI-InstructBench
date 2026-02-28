# neoAI-InstructBench

A Japanese benchmark for evaluating LLMs' ability to follow complex, multi-constraint instructions based on real-world scenarios.

## Usage

### 1. Setup

```bash
uv sync
```

Create a `.env` file with your API key:

```
OPENAI_API_KEY=your-api-key
# or
OPENROUTER_API_KEY=your-api-key
```

### 2. Get LLM Responses

```bash
uv run python scripts/get_response.py --model gpt-5.2
```

Output: `results/gpt-5.jsonl`

### 3. Evaluate Responses

```bash
uv run python scripts/evaluate.py --eval-model gpt-5.1 --inference results/gpt-5.2.jsonl
```

Output: `results/gpt-5.2_judged.jsonl` and `results/gpt-5.2_analysis.json`

## Citation

If you use this benchmark in your research, we would appreciate a citation:

```bibtex
@inproceedings{neoaiinstructbench,
  title={neoAI-InstructBench: 実践的シナリオに基づく日本語複合指示追従ベンチマーク},
  author={川本 稔己 and 板井 孝樹 and 大槻 真輝},
  booktitle={言語処理学会 第32回年次大会},
  year={2026}
}
```

## License

This project is licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
