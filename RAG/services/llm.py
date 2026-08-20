"""
Bedrock LLM Service

API 啟動後只建立一次。
"""

from langchain_aws import ChatBedrock

from config import AWS_REGION

from config import MODEL_ID


_llm = None


def get_llm():

    global _llm

    if _llm is None:

        print("Loading Bedrock LLM...")

        _llm = ChatBedrock(

            region_name=AWS_REGION,

            model_id=MODEL_ID

        )

        print("Bedrock Ready")

    return _llm