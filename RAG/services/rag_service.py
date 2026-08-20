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

    # session_id: Optional[str] = None

):


    start_time = time.time()


    logger.info(

        f"Question: {question}"

    )


    try:


        # -------------------------
        # Vector Search
        # -------------------------

        vectorstore = (
            get_vectorstore()
        )


        search_start = time.time()


        docs = (
            vectorstore
            .max_marginal_relevance_search(

                question,

                k=SEARCH_K,

                fetch_k=FETCH_K

            )
        )


        search_time = (
            time.time()
            -
            search_start
        )


        logger.info(

            f"Retriever Time: {search_time:.3f}s"

        )



        if not docs:


            return ChatResponse(

                answer=
                "文件中沒有找到相關資訊。",

                sources=[]

            )



        # -------------------------
        # Context
        # -------------------------

        context = build_context(
            docs
        )



        # -------------------------
        # Prompt
        # -------------------------

        messages = build_prompt(

            context,

            question

        )



        # -------------------------
        # LLM
        # -------------------------

        llm = get_llm()


        llm_start = time.time()


        response = llm.invoke(

            messages

        )


        llm_time = (

            time.time()

            -

            llm_start

        )


        logger.info(

            f"LLM Time: {llm_time:.3f}s"

        )



        # -------------------------
        # Source
        # -------------------------

        sources = build_sources(

            docs

        )



        total_time = (

            time.time()

            -

            start_time

        )


        logger.info(

            f"Total Time: {total_time:.3f}s"

        )



        return ChatResponse(

            answer=response.content,

            sources=sources

        )



    except Exception as e:


        logger.exception(

            f"RAG Error: {e}"

        )


        raise e