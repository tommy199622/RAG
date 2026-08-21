from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

PDF_path=r"D:/AIProject/RAG/data"
documents=[]
pdf_dir=Path(PDF_path)
for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Loading:{pdf_file.name}")
    loader=PyMuPDFLoader(str(pdf_file))
    pdf_docs=loader.load()
    for doc in pdf_docs:
        doc.metadata["filename"]=pdf_file.name
        doc.metadata["filepath"] = str(pdf_file)
        doc.metadata["category"] = pdf_file.parent.name
    documents.extend(pdf_docs)


# loader = PyMuPDFLoader(r"D:\AIProject\RAG\IB-BP-B2-BASIC-TW-5121.pdf")
# documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n","\n", "。","！","？","；","，"," "]
)
docs = text_splitter.split_documents(documents)

# texts = [
#     "LangChain 是一個 LLM 框架",
#     "FAISS 是向量資料庫",
#     "BGE-M3 支援中文與英文",
#     "AWS Bedrock 是 Amazon 提供的生成式 AI 平台",
#     "Claude 是 Anthropic 開發的大型語言模型"]
print(f"documents = {len(documents)}")
print(f"docs = {len(docs)}")
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

vectorstore.save_local("faiss_index")

# vectorstore = FAISS.load_local(
#     "faiss_index",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# while True:
#     query = input("\n請輸入問題（輸入 exit 離開）： ")

#     if query.lower() == "exit":
#         break

#     docs = vectorstore.similarity_search(query, k=1)

#     print("\n搜尋結果：")
#     for i, doc in enumerate(docs, start=1):
#         print(f"\n[{i}]")
#         print(doc.page_content)