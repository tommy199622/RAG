"""
Index Service

負責：

1. 讀取 PDF
2. 文件切割
3. Embedding
4. 建立 FAISS
5. 儲存 Index
6. 載入 Index


未來支援：

S3 PDF
S3 FAISS
OpenSearch Vector
"""

import re
from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)


from config import (
    DATA_PATH,
    INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    STORAGE_MODE
)


from services.embedding import (
    get_embeddings
)


from utils.logger import (
    logger
)
from utils.s3 import download_file,list_pdf_files

# ==============================
# Global VectorStore
# ==============================

_vectorstore = None



# ==============================
# PDF Loader
# ==============================
def infer_model_from_filename(
    filename: str
):

    name = filename.upper()

    # -----------------------------------------
    # BPA6 / BPA3
    # IB-BP-A6-BT_4218.pdf
    # BPA3-Basic.pdf
    # -----------------------------------------

    match = re.search(
        r"BP[-_]?([A-Z]\d+)",
        name
    )

    if match:

        return (
            "BP"
            + match.group(1)
        )


    # -----------------------------------------
    # BP3GU1-7B
    # -----------------------------------------

    match = re.search(
        r"(BP\d+GU\d+-\d+[A-Z])",
        name
    )

    if match:

        return match.group(1)


    return None
def load_documents():

    """
    載入所有 PDF

    支援:

    data/
        A.pdf

        folder/
            B.pdf

    """

    documents = []


    if STORAGE_MODE=="LOCAL":

        pdf_files = list(
            DATA_PATH.rglob("*.pdf")
        )

    else:

        pdf_files = list_pdf_files()


    if not pdf_files:

        raise FileNotFoundError(
            "No PDF found"
        )


    logger.info(
        f"Found PDF: {len(pdf_files)}"
    )



    for pdf_file in pdf_files:


        logger.info(
            f"Loading: {pdf_file}"
        )


        if STORAGE_MODE=="LOCAL":
            pdf_path = str(pdf_file)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf",delete=False)
            download_file(pdf_file,tmp.name)
            pdf_path = tmp.name

        loader = PyMuPDFLoader(pdf_path)

        pdf_docs = loader.load()



        for doc in pdf_docs:


            doc.metadata["filename"] = (
                pdf_file.name
            )


            doc.metadata["filepath"] = (
                str(pdf_file)
            )


            doc.metadata["category"] = (
                pdf_file.parent.name
            )


            # PyMuPDF page 從 0 開始
            doc.metadata["page"] = (
                doc.metadata.get(
                    "page",
                    0
                )
            )

            doc.metadata["model"] = (
                infer_model_from_filename(
                    pdf_file.name
                )
            )


        documents.extend(
            pdf_docs
        )


    logger.info(

        f"Loaded pages: {len(documents)}"

    )


    return documents




# ==============================
# Text Split
# ==============================

def split_documents(documents):


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=SEPARATORS

    )


    docs = splitter.split_documents(

        documents

    )


    logger.info(

        f"Chunks created: {len(docs)}"

    )


    return docs




# ==============================
# Build FAISS
# ==============================

def build_index():


    global _vectorstore


    try:

        for doc in docs[:20]:

            logger.info(
                f"PAGE={doc.metadata.get('page')}"
                f"MODEL={doc.metadata.get('model')} "
                f"FILE={doc.metadata.get('filename')} "
            )


        logger.info(
            "===== Build Index Start ====="
        )



        documents = load_documents()



        docs = split_documents(

            documents

        )



        embeddings = get_embeddings()



        vectorstore = FAISS.from_documents(

            docs,

            embeddings

        )



        INDEX_PATH.mkdir(

            exist_ok=True

        )



        vectorstore.save_local(

            str(INDEX_PATH)

        )



        _vectorstore = vectorstore



        logger.info(

            "FAISS Index Saved"

        )



        logger.info(

            "===== Build Index Complete ====="

        )


        return vectorstore



    except Exception as e:


        logger.exception(

            f"Build Index Failed: {e}"

        )


        raise e





# ==============================
# Load FAISS
# ==============================

def load_vectorstore():


    global _vectorstore



    if _vectorstore is not None:


        return _vectorstore




    embeddings = get_embeddings()



    index_file = (

        INDEX_PATH /

        "index.faiss"

    )



    if not index_file.exists():


        raise FileNotFoundError(

            "FAISS index not found"

        )



    logger.info(

        "Loading FAISS Index"

    )



    _vectorstore = FAISS.load_local(

        str(INDEX_PATH),

        embeddings,

        allow_dangerous_deserialization=True

    )



    logger.info(

        "FAISS Loaded"

    )



    return _vectorstore




# ==============================
# Getter
# ==============================

def get_vectorstore():

    global _vectorstore


    if _vectorstore is None:


        return load_vectorstore()



    return _vectorstore