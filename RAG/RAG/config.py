from pathlib import Path

from dotenv import load_dotenv

import os


# Load .env

load_dotenv()



BASE_DIR = Path(__file__).parent



# ==========================
# Path
# ==========================

DATA_PATH = (
    BASE_DIR / "data"
)


INDEX_PATH = (
    BASE_DIR / "faiss_index"
)


LOG_PATH = (
    BASE_DIR / "logs"
)



# ==========================
# Embedding
# ==========================

EMBEDDING_MODEL = os.getenv(

    "EMBEDDING_MODEL",

    "BAAI/bge-m3"

)



# ==========================
# Bedrock
# ==========================

AWS_REGION = os.getenv(

    "AWS_REGION",

    "us-west-2"

)


MODEL_ID = os.getenv(

    "MODEL_ID",

    "global.amazon.nova-2-lite-v1:0"

)



# ==========================
# Storage
# ==========================

STORAGE_MODE = os.getenv(

    "STORAGE_MODE",

    "LOCAL"

)



S3_BUCKET = os.getenv(

    "S3_BUCKET",

    ""

)



S3_PREFIX = os.getenv(

    "S3_PREFIX",

    ""

)



# ==========================
# Chunk
# ==========================

CHUNK_SIZE = int(

    os.getenv(

        "CHUNK_SIZE",

        500

    )

)


CHUNK_OVERLAP = int(

    os.getenv(

        "CHUNK_OVERLAP",

        100

    )

)



SEPARATORS = [

    "\n\n",

    "\n",

    "。",

    "！",

    "？",

    "；",

    "，",

    " "

]



# ==========================
# Retriever
# ==========================

SEARCH_K = int(

    os.getenv(

        "SEARCH_K",

        4

    )

)


FETCH_K = int(

    os.getenv(

        "FETCH_K",

        10

    )

)