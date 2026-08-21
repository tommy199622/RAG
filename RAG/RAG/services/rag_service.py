"""
RAG Service

負責：

Question
    |
Retriever
    |
Context
    |
Prompt
    |
LLM
    |
Response


"""

import time
from typing import Optional

from services.index_service import (
    get_vectorstore
)

from services.llm import (
    get_llm
)

from services.prompt import (
    build_prompt
)

from models.chat import (
    ChatResponse
)

from models.source import (
    Source
)

from config import (
    SEARCH_K,
    FETCH_K
)

from utils.logger import (
    logger
)
from services.session_service import (
    get_session,
    set_model,
    add_message,
    get_history
)

from services.conversation import (
    extract_model,
    is_model_setting_question,
    is_current_model_question,
    is_casual_question,
    rewrite_question
)


# =================================
# Context Builder
# =================================

def build_context(docs):

    """
    將搜尋結果整理成 LLM Context

    不直接 join，
    增加來源資訊
    """

    context_list = []


    for doc in docs:


        filename = (
            doc.metadata.get(
                "filename",
                "unknown"
            )
        )


        page = (
            doc.metadata.get(
                "page",
                0
            )
            + 1
        )


        content = doc.page_content



        context_list.append(

            f"""
[文件]
{filename}

[頁碼]
Page {page}

[內容]
{content}

"""

        )


    return "\n".join(
        context_list
    )




# =================================
# Source Builder
# =================================

def build_sources(docs):


    sources = []


    seen = set()


    for doc in docs:


        filename = (
            doc.metadata.get(
                "filename",
                "unknown"
            )
        )


        page = (

            doc.metadata.get(
                "page",
                0
            )

            + 1

        )


        key = (
            filename,
            page
        )


        # 避免重複來源

        if key in seen:

            continue


        seen.add(key)


        sources.append(

            Source(

                filename=filename,

                page=page,

                filepath=
                doc.metadata.get(
                    "filepath"
                ),

                category=
                doc.metadata.get(
                    "category"
                )

            )

        )


    return sources




# =================================
# Main RAG Function
# =================================

def ask(
    question: str,
    session_id: str
):

    start_time = time.time()

    logger.info(
        f"Session: {session_id}"
    )

    logger.info(
        f"Question: {question}"
    )

    # ==================================================
    # Session
    # ==================================================

    session = get_session(
        session_id
    )

    current_model = (
        session.current_model
    )


    # ==================================================
    # 1. Detect explicit model
    # ==================================================

    detected_model = extract_model(
        question
    )


    # ==================================================
    # 2. Model setting
    # ==================================================

    if (
        detected_model
        and is_model_setting_question(question)
    ):

        set_model(
            session_id,
            detected_model
        )

        current_model = detected_model

        add_message(
            session_id,
            "user",
            question
        )

        answer = (
            f"好的，目前討論型號已設定為 "
            f"{current_model}。"
        )

        add_message(
            session_id,
            "assistant",
            answer
        )

        return ChatResponse(
            answer=answer,
            sources=[],
            current_model=current_model
        )


    # ==================================================
    # 3. Current model question
    # ==================================================

    if is_current_model_question(
        question
    ):

        add_message(
            session_id,
            "user",
            question
        )

        if current_model:

            answer = (
                f"目前討論型號為 "
                f"{current_model}。"
            )

        else:

            answer = (
                "目前尚未指定產品型號。"
            )

        add_message(
            session_id,
            "assistant",
            answer
        )

        return ChatResponse(
            answer=answer,
            sources=[],
            current_model=current_model
        )


    # ==================================================
    # 4. Casual conversation
    # ==================================================

    if is_casual_question(
        question
    ):

        add_message(
            session_id,
            "user",
            question
        )

        llm = get_llm()

        response = llm.invoke(
            [
                (
                    "system",
                    "你是一個企業產品問答助手。"
                ),
                (
                    "human",
                    question
                )
            ]
        )

        answer = response.content

        add_message(
            session_id,
            "assistant",
            answer
        )

        return ChatResponse(
            answer=answer,
            sources=[],
            current_model=current_model
        )


    # ==================================================
    # 5. No model
    # ==================================================

    if not current_model:

        return ChatResponse(
            answer=(
                "請先指定產品型號，"
                "例如：「接下來討論 BPA6」。"
            ),
            sources=[],
            current_model=None
        )


    # ==================================================
    # 6. Rewrite question
    # ==================================================

    rewritten_question = rewrite_question(
        question,
        current_model
    )

    logger.info(
        f"Current Model: {current_model}"
    )

    logger.info(
        f"Rewritten Question: "
        f"{rewritten_question}"
    )


    # ==================================================
    # 7. Vector Search
    # ==================================================

    vectorstore = (
        get_vectorstore()
    )

    search_start = time.time()


    # 重要：
    # 使用 model metadata filter
    #
    # 這需要重新建立 FAISS index。
    #

    docs = (
        vectorstore
        .max_marginal_relevance_search(
            rewritten_question,
            k=SEARCH_K,
            fetch_k=FETCH_K,
            filter={
                "model": current_model
            }
        )
    )


    search_time = (
        time.time()
        - search_start
    )

    logger.info(
        f"Retriever Time: "
        f"{search_time:.3f}s"
    )


    # ==================================================
    # 8. No documents
    # ==================================================

    if not docs:

        add_message(
            session_id,
            "user",
            question
        )

        answer = (
            f"在目前型號 {current_model} "
            f"的文件中沒有找到相關資訊。"
        )

        add_message(
            session_id,
            "assistant",
            answer
        )

        return ChatResponse(
            answer=answer,
            sources=[],
            current_model=current_model
        )


    # ==================================================
    # 9. Context
    # ==================================================

    context = build_context(
        docs
    )


    # ==================================================
    # 10. History
    # ==================================================

    history_messages = get_history(
        session_id,
        max_messages=10
    )

    history_text = "\n".join(
        [
            f"{message.role}: "
            f"{message.content}"
            for message
            in history_messages
        ]
    )


    # ==================================================
    # 11. Prompt
    # ==================================================

    messages = build_prompt(
        context=context,
        question=question,
        current_model=current_model,
        history=history_text
    )


    # ==================================================
    # 12. LLM
    # ==================================================

    llm = get_llm()

    llm_start = time.time()

    response = llm.invoke(
        messages
    )

    llm_time = (
        time.time()
        - llm_start
    )

    logger.info(
        f"LLM Time: "
        f"{llm_time:.3f}s"
    )


    # ==================================================
    # 13. Save history
    # ==================================================

    add_message(
        session_id,
        "user",
        question
    )

    add_message(
        session_id,
        "assistant",
        response.content
    )


    # ==================================================
    # 14. Sources
    # ==================================================

    sources = build_sources(
        docs
    )


    # ==================================================
    # 15. Total time
    # ==================================================

    total_time = (
        time.time()
        - start_time
    )

    logger.info(
        f"Total Time: "
        f"{total_time:.3f}s"
    )


    return ChatResponse(
        answer=response.content,
        sources=sources,
        current_model=current_model
    )