import logging
from pathlib import Path

from config import LOG_PATH


# 建立 Log 目錄
LOG_PATH.mkdir(
    exist_ok=True
)


LOG_FILE = LOG_PATH / "rag.log"


logging.basicConfig(

    level=logging.INFO,

    format=
    "%(asctime)s | %(levelname)s | %(message)s",

    handlers=[

        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),

        logging.StreamHandler()

    ]

)


logger = logging.getLogger("RAG")