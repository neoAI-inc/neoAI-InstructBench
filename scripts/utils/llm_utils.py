import os
from typing import Any, Callable, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openrouter import OpenRouter

load_dotenv()


def call_openai(client, model_name: str, prompt: str, max_tokens: int, reasoning: str = "high") -> str:
    params = {
        "model": model_name,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": reasoning},
    }
    response = client.responses.create(**params)
    return response.output_text


def call_openrouter(client, model_name: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.send(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        reasoning={
            "effort": "high",
            "exclude": True,
        },
    )
    return response.choices[0].message.content


def get_client(model_name: str) -> Tuple[Any, Callable]:
    # OpenRouter models (format: provider/model or provider/model:tag, e.g., openai/gpt-oss-120b or openai/gpt-oss-20b:free)
    if "/" in model_name:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. " "Please set it in your .env file."
            )
        openrouter_client = OpenRouter(api_key=api_key)
        return openrouter_client, call_openrouter

    if model_name.startswith("gpt-"):
        openai_client = OpenAI()
        return openai_client, call_openai

    raise ValueError(f"Unsupported model name: {model_name}")


def build_prompt_from_row(obj: Dict[str, Any]) -> str:
    if obj.get("input"):
        return str(obj["input"])

    raw_input = str(obj.get("raw_input", "")).strip()
    instructions = obj.get("instructions", [])
    inst_texts = []
    if isinstance(instructions, list):
        for item in instructions:
            if isinstance(item, dict):
                text = item.get("instruction")
                if text:
                    inst_texts.append(str(text).strip())
    return " ".join([raw_input] + inst_texts).strip()
