import inspect
import json
import re
from typing import Any, Dict, List

import spacy
import yaml

nlp = spacy.load("ja_ginza")


def char_count_max(response: str, params: int) -> int:
    """Check if response has at most N characters."""
    return 1 if len(response) <= params else 0


def char_count_min(response: str, params: int) -> int:
    """Check if response has at least N characters."""
    return 1 if len(response) >= params else 0


def char_count_range(response: str, params: Dict[str, int]) -> int:
    """Check if response has between min and max characters (inclusive)."""
    char_count = len(response)
    min_chars = params["min"]
    max_chars = params["max"]
    return 1 if min_chars <= char_count <= max_chars else 0


def first_line_char_count_max(response: str, params: int) -> int:
    """Check if the first line of response has at most N characters."""
    lines = response.split("\n")
    return 1 if lines and len(lines[0]) <= params else 0


def table_items_char_count_max(response: str, params: int) -> int:
    """Check if all table items are within character limit."""
    lines = response.split("\n")
    for line in lines:
        if "\t" in line:
            items = line.split("\t")
            for item in items:
                if len(item.strip()) > params:
                    return 0
    return 1


def forbidden_words(response: str, params: List[str]) -> int:
    """Check if response does not contain any of the forbidden words."""
    return 0 if any(word in response for word in params) else 1


def forbidden_words_en(response: str, params: List[str]) -> int:
    """Check if response does not contain any of the forbidden English words (case-insensitive)."""
    response_lower = response.lower()
    return 0 if any(word.lower() in response_lower for word in params) else 1


def required_word_count(response: str, params: Dict[str, Any]) -> int:
    """Check if response contains a specific word exactly N times."""
    word = params["word"]
    expected = params["count"]
    return 1 if response.count(word) == expected else 0


def required_words_total(response: str, params: Dict[str, Any]) -> int:
    """Check if the total count of specified words in response equals the expected total."""
    words = params["words"]
    total = params["total"]
    actual_total = sum(response.count(w) for w in words)
    return 1 if actual_total == total else 0


def required_words_each(response: str, params: Dict[str, Any]) -> int:
    """Check if each of the specified words appears exactly N times in response."""
    words = params["words"]
    count_each = params["count_each"]
    return 1 if all(response.count(w) == count_each for w in words) else 0


def required_words_all(response: str, params: List[str]) -> int:
    """Check if all specified words are present in response."""
    return 1 if all(word in response for word in params) else 0


def required_words_any(response: str, params: List[str]) -> int:
    """Check if any of the specified words are present in response."""
    return 1 if any(word in response for word in params) else 0


def must_contain_forbidden(response: str, params: Dict[str, List[str]]) -> int:
    """Check if response contains any word from the must_contain list AND contains no word from the forbidden list."""
    has_required = required_words_any(response, params["must_contain"])
    no_forbidden = forbidden_words(response, params["forbidden"])

    return 1 if (has_required == 1 and no_forbidden == 1) else 0


def no_katakana(response: str) -> int:
    """Check if response contains no katakana characters."""
    return 0 if re.search(r"[\u30A0-\u30FF]", response) else 1


def no_alphabet(response: str) -> int:
    """Check if response contains no alphabet characters."""
    return 0 if re.search(r"[A-Za-z]", response) else 1


def no_halfwidth_numbers(response: str) -> int:
    """Check if response contains no halfwidth numbers."""
    return 0 if re.search(r"[0-9]", response) else 1


def only_hiragana_kanji_punctuation(response: str) -> int:
    """Check if response contains only hiragana, kanji, and punctuation (、。)."""
    return 1 if re.match(r"^[\u3040-\u309F\u4E00-\u9FFF、。]+$", response) else 0


def only_hiragana_punctuation(response: str) -> int:
    """Check if response contains only hiragana and punctuation (、。)."""
    return 1 if re.match(r"^[\u3040-\u309F、。]+$", response) else 0


def only_hiragana_kanji_punctuation_newline(response: str) -> int:
    """Check if response contains only hiragana, kanji, punctuation (、。), and newlines."""
    return 1 if re.match(r"^[\u3040-\u309F\u4E00-\u9FFF、。\n]+$", response) else 0


def kanji_more_than_hiragana(response: str) -> int:
    """Check if kanji count is more than hiragana count."""
    kanji_count = len(re.findall(r"[\u4E00-\u9FFF]", response))
    hiragana_count = len(re.findall(r"[\u3040-\u309F]", response))
    return 1 if kanji_count > hiragana_count else 0


def number_usage_exact(response: str, params: int) -> int:
    """Check if response contains exactly N number sequences (halfwidth or fullwidth)."""
    number_count = len(re.findall(r"[0-9０-９]+", response))
    return 1 if number_count == params else 0


def alphabet_word_count(response: str, params: int) -> int:
    """Check if response contains exactly N words with alphabet characters."""
    words_with_alphabet = re.findall(r"\S*[A-Za-z]+\S*", response)
    return 1 if len(words_with_alphabet) == params else 0


def katakana_word_max(response: str, params: int) -> int:
    """Check if response contains at most N katakana words."""
    katakana_words = re.findall(r"[\u30A0-\u30FF]+", response)
    return 1 if len(katakana_words) <= params else 0


def no_html_tags(response: str, params: List[str]) -> int:
    """Check if response does not contain any of the specified HTML tags."""
    for tag in params:
        if re.search(rf"<\s*{tag}[\s>]", response, re.IGNORECASE):
            return 0
    return 1


def no_inline_blocks(response: str) -> int:
    """Check if response contains no backticks (no inline code or code blocks)."""
    return 0 if "`" in response else 1


def no_math_expressions(response: str) -> int:
    """Check if response contains no mathematical expressions or symbols."""
    # Check for clearly mathematical symbols (excluding hyphen to avoid false positives in compound words)
    if re.search(r"[\$\\\{\}\_\^\∫∑∏√±≤≥≠∞×÷]", response):
        return 0
    # Check for actual mathematical operators with numbers (e.g., "2+3", "x=5", "10*2", "8/4")
    if re.search(r"\d\s*[\+\*\/=]\s*\d", response):
        return 0
    return 1


def no_markdown_bold_italic(response: str) -> int:
    """Check if response contains no markdown bold (**text**) or italic (*text*)."""
    if re.search(r"\*\*[^*]+\*\*", response) or re.search(r"(?<!\*)\*[^*]+\*(?!\*)", response):
        return 0
    return 1


def wrapped_in_code_block(response: str) -> int:
    """Check if response is wrapped in a code block (starts and ends with ```)."""
    return 1 if response.startswith("```") and response.endswith("```") else 0


def first_line_inline_code_only(response: str) -> int:
    """Check if first line contains only inline code (backticks) and no explanation."""
    lines = response.split("\n")
    if not lines:
        return 0
    first_line = lines[0].strip()
    if not first_line.startswith("`") or not first_line.endswith("`"):
        return 0
    if len(first_line) < 3:
        return 0
    return 1


def format_json_array(response: str, params: Dict[str, Any]) -> int:
    """Check if response is a valid JSON array with required keys in each object."""
    try:
        data = json.loads(response.strip())
        if not isinstance(data, list):
            return 0

        required_keys = params["keys"]
        if required_keys and data and isinstance(data[0], dict):
            return 1 if all(k in data[0] for k in required_keys) else 0
        return 1
    except (json.JSONDecodeError, KeyError, IndexError):
        return 0


def format_yaml(response: str, params: Dict[str, Any]) -> int:
    """Check if response is valid YAML with required keys."""
    try:

        data = yaml.safe_load(response)

        required_keys = params["keys"]
        if not required_keys:
            return 1

        if isinstance(data, dict):
            return 1 if all(k in data for k in required_keys) else 0
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return 1 if all(k in data[0] for k in required_keys) else 0
        return 1
    except Exception:
        return 0


def format_csv(response: str, params: Dict[str, Any]) -> int:
    """Check if response is valid CSV format with required columns and consistent comma counts."""
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if not lines:
        return 0

    expected_columns = params["columns"]
    first_line = lines[0]

    if not all(col in first_line for col in expected_columns):
        return 0

    comma_counts = [line.count(",") for line in lines]
    return 1 if len(set(comma_counts)) == 1 else 0


def format_tsv(response: str, params: Dict[str, Any]) -> int:
    """Check if response is valid TSV format with required columns and consistent tab counts."""
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if not lines:
        return 0

    expected_columns = params["columns"]
    first_line = lines[0]

    if not all(col in first_line for col in expected_columns):
        return 0

    tab_counts = [line.count("\t") for line in lines]
    return 1 if len(set(tab_counts)) == 1 else 0


def line_count_exact(response: str, params: int) -> int:
    """Check if response has exactly N lines."""
    lines = response.split("\n")
    return 1 if len(lines) == params else 0


def line_count_max(response: str, params: int) -> int:
    """Check if response has at most N lines."""
    lines = response.split("\n")
    return 1 if len(lines) <= params else 0


def no_newlines(response: str) -> int:
    """Check if response contains no newlines."""
    return 0 if "\n" in response else 1


def double_newlines_only(response: str) -> int:
    """Check if all newlines in response are exactly double newlines (2 consecutive newlines)."""
    if "\n\n\n" in response:
        return 0
    temp = re.sub(r"\n\n", "<<DOUBLE>>", response)
    return 0 if "\n" in temp else 1


def triple_newlines_only(response: str) -> int:
    """Check if all newlines are exactly triple newlines (3 consecutive newlines)."""
    if "\n\n\n\n" in response:
        return 0
    temp = re.sub(r"\n\n\n", "<<TRIPLE>>", response)
    return 0 if "\n" in temp else 1


def paragraph_count(response: str, params: int) -> int:
    """Check if response has exactly N paragraphs (lines)."""
    return 1 if len(response.split("\n")) == params else 0


def paragraph_separator(response: str, params: Dict[str, Any]) -> int:
    """Check if response is divided into N paragraphs by the specified separator."""
    separator = params["separator"]
    expected_count = params["count"]
    separator_count = response.count(separator)

    if expected_count:
        expected_separators = expected_count - 1
        return 1 if separator_count == expected_separators else 0
    return 1 if separator_count > 0 else 0


def arrow_connected_single_line(response: str, params: str) -> int:
    """Check if response is a single line containing the specified arrow character."""
    lines = response.split("\n")
    return 1 if len(lines) == 1 and params in response else 0


def no_empty_lines(response: str) -> int:
    """Check if response contains no empty lines."""
    lines = response.split("\n")
    return 0 if any(not line.strip() for line in lines) else 1


def bullet_count(response: str, params: int) -> int:
    """Check if response has exactly N bullet list items (lines starting with '- ')."""
    bullet_pattern = r"^-\s"
    lines = response.split("\n")
    bullet_count = sum(1 for line in lines if re.match(bullet_pattern, line))
    return 1 if bullet_count == params else 0


def bullet_list_format(response: str) -> int:
    """Check if response is formatted as a bullet list (at least 50% of lines are bullet items)."""
    lines = response.split("\n")
    bullet_pattern = re.compile(r"^-\s")
    bullet_lines = sum(1 for line in lines if bullet_pattern.match(line))
    return 1 if bullet_lines >= len(lines) * 0.5 else 0


def numbered_list_count(response: str, params: int) -> int:
    """Check if response has exactly N numbered list items."""
    numbered_items = re.findall(r"^\s*\d+\.\s", response, re.MULTILINE)
    return 1 if len(numbered_items) == params else 0


def numbered_list_format(response: str) -> int:
    """Check if response is formatted as a numbered list."""
    numbered_items = re.findall(r"^\s*\d+\.\s", response, re.MULTILINE)
    return 1 if len(numbered_items) > 0 else 0


def sentence_ending(response: str, params: str) -> int:
    """Check if all sentences in response end with the specified ending."""
    doc = nlp(response)
    sentences = [str(sent).strip() for sent in doc.sents if str(sent).strip()]
    return 1 if all(s.endswith(params) for s in sentences) else 0


def sentence_ending_pattern(response: str, params: List[str]) -> int:
    """Check if all sentences in response end with one of the specified ending patterns."""
    doc = nlp(response)
    sentences = [str(sent).strip() for sent in doc.sents if str(sent).strip()]
    return 1 if all(any(s.endswith(p) for p in params) for s in sentences) else 0


def sentence_count_max(response: str, params: int) -> int:
    """Check if response has at most N sentences."""
    doc = nlp(response)
    sentence_count = len(list(doc.sents))
    return 1 if sentence_count <= params else 0


def taigendome_count(response: str, params: int) -> int:
    """Check if response has exactly N sentences ending with a noun (体言止め)."""
    doc = nlp(response)
    sents = [sent for sent in doc.sents if str(sent).strip()]

    taigendome_count = 0
    for sent in sents:
        content_tokens = [t for t in sent if t.pos_ not in ["PUNCT", "SPACE"]]
        if content_tokens and all(t.pos_ == "NOUN" for t in content_tokens):
            continue
        for token in reversed(list(sent)):
            if token.pos_ not in ["PUNCT", "SPACE"]:
                if token.pos_ == "NOUN":
                    taigendome_count += 1
                break

    return 1 if taigendome_count == params else 0


def punctuation_comma_period(response: str) -> int:
    """Check if response uses comma and period (,) instead of Japanese punctuation (、。)."""
    has_jp_punct = "、" in response or "。" in response
    return 0 if has_jp_punct else 1


def word_endings(response: str) -> int:
    """Check if all elements (separated by → or newlines) end with a content word (noun, verb, adjective, etc.)."""
    elements = [e.strip() for e in re.split(r"[→\n]", response) if e.strip()]
    for element in elements:
        doc = nlp(element)
        tokens = [token for token in doc if token.pos_ not in ["PUNCT", "SPACE"]]
        if not tokens:
            continue
        last_token = tokens[-1]
        if last_token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
            continue
        else:
            return 0
    return 1


# Markdown/HTML structure validators
def h1_section_count(response: str, params: int) -> int:
    """Check if response has exactly N h1 sections (Markdown format: # heading)."""
    # Remove code blocks (```...```) and inline code (`...`) to avoid counting # in code
    text_without_code_blocks = re.sub(r"```[\s\S]*?```", "", response)
    text_without_inline_code = re.sub(r"`[^`]+`", "", text_without_code_blocks)
    h1_pattern = r"^#\s+[^\n]+"
    h1_count = len(re.findall(h1_pattern, text_without_inline_code, re.MULTILINE))
    return 1 if h1_count == params else 0


def h2_section_count(response: str, params: int) -> int:
    """Check if response has exactly N h2 sections (Markdown format: ## heading)."""
    h2_pattern = r"^##\s+[^\n]+"
    h2_count = len(re.findall(h2_pattern, response, re.MULTILINE))
    return 1 if h2_count == params else 0


def markdown_h1_h3_only(response: str) -> int:
    """Check if response uses only h1 and h3, not h2. Both h1 and h3 must be present."""
    # Check if h2 exists (not allowed)
    if re.search(r"^##\s+[^\n]+", response, re.MULTILINE):
        return 0
    # Check if both h1 and h3 exist
    has_h1 = re.search(r"^#\s+[^\n]+", response, re.MULTILINE)
    has_h3 = re.search(r"^###\s+[^\n]+", response, re.MULTILINE)
    return 1 if (has_h1 and has_h3) else 0


def specific_h1_headings(response: str, params: List[str]) -> int:
    """Check if response contains all specified h1 headings (HTML format)."""
    for heading in params:
        pattern = rf"<\s*h1\s*>\s*{re.escape(heading)}\s*<\s*/\s*h1\s*>"
        if not re.search(pattern, response, re.DOTALL):
            return 0
    return 1


def lines_per_section(response: str, params: int) -> int:
    """Check if each section (separated by h1 tags) has exactly N lines."""
    h1_pattern = r"<h1>.*?</h1>"
    parts = re.split(h1_pattern, response, flags=re.DOTALL)

    sections = []
    for part in parts[1:]:  # Skip content before first h1
        section_lines = [line for line in part.split("\n") if line.strip()]
        if section_lines:
            sections.append(section_lines)

    if not sections:
        return 0

    return 1 if all(len(sec) == params for sec in sections) else 0


def first_line_yes_no_only(response: str) -> int:
    """Check if first line is only 'Yes' or 'No'."""
    lines = response.split("\n")
    if not lines:
        return 0
    first_line = lines[0].strip()
    return 1 if first_line in ["Yes", "No"] else 0


def wrapped_in_delimiter(response: str, params: str) -> int:
    """Check if response starts and ends with the specified delimiter."""
    return 1 if response.startswith(params) and response.endswith(params) else 0


def function_then_double_newline(response: str) -> int:
    """Check if response has a function/formula on the first line followed by double newline and explanation."""
    if "\n\n" not in response:
        return 0
    parts = response.split("\n\n", 1)
    if len(parts) < 2 or not parts[1].strip():
        return 0
    if parts[1].startswith("\n"):
        return 0
    return 1


def line_prefix_pattern(response: str, params: List[str]) -> int:
    """Check if all lines in response start with one of the specified prefixes."""
    lines = response.split("\n")
    for line in lines:
        if not any(line.startswith(prefix) for prefix in params):
            return 0
    return 1


def line_count_exact_with_emoji_start(response: str, params: int) -> int:
    """Check if response has exactly N lines and each line starts with an emoji."""
    lines = response.split("\n")
    if len(lines) != params:
        return 0
    emoji_pattern = r"^[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\u2600-\u27BF]"
    return 1 if all(re.match(emoji_pattern, line) for line in lines) else 0


def vertical_reading_matches(response: str, params: str) -> int:
    """Check if first characters of each line/paragraph read vertically match the target."""
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    if len(lines) < len(params):
        return 0
    vertical_text = "".join(line[0] if line else "" for line in lines[: len(params)])
    return 1 if vertical_text == params else 0


def validate_closed_instruction(eval_config: Dict[str, Any], response: str) -> int:
    """Main validator dispatcher using dynamic function lookup."""
    func_name = eval_config["function"]
    if not isinstance(func_name, str):
        raise ValueError(f"Invalid function name type: {type(func_name)}, expected str")

    params = eval_config.get("params")

    validator_func = globals().get(func_name)
    if not validator_func:
        raise ValueError(f"Validator function '{func_name}' not found. Did you implement it in closed_validators.py?")
    if not callable(validator_func):
        raise ValueError(f"'{func_name}' exists but is not a callable function")

    try:
        sig = inspect.signature(validator_func)
        if len(sig.parameters) == 1:
            result = validator_func(response)
        else:
            result = validator_func(response, params)
        return int(result) if isinstance(result, (int, bool)) else 0
    except Exception as e:
        raise RuntimeError(f"Error executing validator '{func_name}': {str(e)}") from e
