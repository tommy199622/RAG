"""
Conversation / Model Context

負責：
1. 從使用者問題判斷型號
2. 更新目前型號
3. 判斷是否為目前型號查詢
4. 將代詞問題改寫成完整問題
"""

import re
from typing import Optional


# --------------------------------------------------
# Model normalization
# --------------------------------------------------

def normalize_model(model: str) -> str:

    model = model.strip().upper()

    model = model.replace(" ", "")
    model = model.replace("_", "-")

    # BP-A6 -> BPA6
    model = re.sub(
        r"^BP-([A-Z0-9].*)$",
        r"BP\1",
        model
    )

    return model


# --------------------------------------------------
# Extract model from user question
# --------------------------------------------------

MODEL_PATTERNS = [

    # BPA6 / BPA3
    r"\bBP[A-Z]\d+\b",

    # BP3GU1-7B
    r"\bBP\d+GU\d+-\d+[A-Z]\b",

    # B2
    r"\bBP[A-Z]\d+\b",
]


def extract_model(
    question: str
) -> Optional[str]:

    text = question.upper()

    for pattern in MODEL_PATTERNS:

        match = re.search(
            pattern,
            text
        )

        if match:

            return normalize_model(
                match.group(0)
            )

    # 支援：
    # BP-A6
    match = re.search(
        r"\bBP-([A-Z]\d+)\b",
        text
    )

    if match:

        return normalize_model(
            "BP" + match.group(1)
        )

    return None


# --------------------------------------------------
# Is model switch / model setting?
# --------------------------------------------------

def is_model_setting_question(
    question: str
) -> bool:

    keywords = [
        "切換",
        "改成",
        "改為",
        "設定",
        "指定",
        "使用",
        "討論",
        "以",
        "接下來",
        "目前使用",
    ]

    has_model = (
        extract_model(question)
        is not None
    )

    has_keyword = any(
        keyword in question
        for keyword in keywords
    )

    return has_model and has_keyword


# --------------------------------------------------
# Is asking current model?
# --------------------------------------------------

def is_current_model_question(
    question: str
) -> bool:

    keywords = [
        "目前型號",
        "現在型號",
        "當前型號",
        "目前是哪個型號",
        "現在是哪個型號",
        "我們討論哪個型號",
        "現在討論哪個型號",
    ]

    return any(
        keyword in question
        for keyword in keywords
    )


# --------------------------------------------------
# Is casual conversation?
# --------------------------------------------------

def is_casual_question(
    question: str
) -> bool:

    text = question.strip()

    casual_patterns = [
        "你好",
        "嗨",
        "hello",
        "hi",
        "謝謝",
        "感謝",
        "可以嗎",
        "好的",
        "了解",
        "OK",
        "ok",
    ]

    return text.lower() in [
        x.lower()
        for x in casual_patterns
    ]


# --------------------------------------------------
# Rewrite question with model context
# --------------------------------------------------

def rewrite_question(
    question: str,
    current_model: Optional[str]
) -> str:

    if not current_model:
        return question

    # 如果問題本身已經有型號
    explicit_model = extract_model(
        question
    )

    if explicit_model:

        return question

    # 加入目前型號
    return (
        f"產品型號為 {current_model}。"
        f"使用者問題：{question}"
    )