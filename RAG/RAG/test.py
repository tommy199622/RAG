from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock

# Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# 載入 FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Claude (Bedrock)
llm = ChatBedrock(
    region_name="us-west-2",
    model_id="global.amazon.nova-2-lite-v1:0"
)

while True:

    question = input("\n請輸入問題（exit 離開）： ")

    if question.lower() == "exit":
        break

    # 取回相關文件
    # docs = vectorstore.similarity_search(
    #     question,
    #     k=3
    # )
    docs = vectorstore.max_marginal_relevance_search(
        question,
        k=4,
        fetch_k=10
    )
    
    context = "\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
你是一位企業文件知識庫問答助手。

你的任務是根據提供的文件內容回答使用者問題。

========================
回答規則
========================

1. 僅能使用「文件內容」中的資訊回答。
2. 不可使用你的背景知識補充答案。
3. 不可推測、猜測或自行推導文件沒有提供的資訊。
4. 若文件沒有足夠資訊回答問題，請回答：
   「文件中沒有找到相關資訊。」
5. 若文件存在不同版本或矛盾資訊，請指出差異。
6. 保留文件中的：
   - 產品名稱
   - 型號
   - 錯誤代碼
   - 技術名詞
   不要自行翻譯或修改。
7. 回答需簡潔、明確，避免重複敘述。

========================
文件內容
========================

{context}

========================
使用者問題
========================

{question}

========================
回答格式
========================

答案：
<回答內容>

來源：
- 文件名稱：
- 頁碼：

"""

    response = llm.invoke(prompt)
    print("\n=== 來源 ===")
    for doc in docs:
        print(f"檔案:{doc.metadata['filename']}")
        print(f"Page: {doc.metadata['page']+1}")
        print("-"*60)
    print("\n=== 回答 ===")
    print(response.content)