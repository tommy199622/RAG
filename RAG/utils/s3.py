"""
S3 Utility

功能：

1. 列出 PDF
2. 下載 PDF
3. 上傳 FAISS Index

"""

import boto3

from pathlib import Path

from config import (
    AWS_REGION,
    S3_BUCKET,
    S3_PREFIX
)

from utils.logger import logger



# ==========================
# S3 Client
# ==========================

s3_client = boto3.client(

    "s3",

    region_name=AWS_REGION

)



# ==========================
# List PDF
# ==========================

def list_pdf_files():


    """
    取得 S3 所有 PDF

    回傳:

    [
        "documents/a.pdf",
        "documents/b.pdf"
    ]

    """


    response = s3_client.list_objects_v2(

        Bucket=S3_BUCKET,

        Prefix=S3_PREFIX

    )


    files = []


    for obj in response.get(
        "Contents",
        []
    ):


        key = obj["Key"]


        if key.lower().endswith(
            ".pdf"
        ):

            files.append(key)



    logger.info(

        f"S3 PDF Count: {len(files)}"

    )


    return files





# ==========================
# Download PDF
# ==========================

def download_file(

        s3_key:str,

        local_path:str

):


    """
    S3 PDF 下載

    """


    logger.info(

        f"Download {s3_key}"

    )


    s3_client.download_file(

        S3_BUCKET,

        s3_key,

        local_path

    )





# ==========================
# Upload File
# ==========================

def upload_file(

        local_path:str,

        s3_key:str

):


    """
    上傳檔案到 S3

    """


    logger.info(

        f"Upload {s3_key}"

    )


    s3_client.upload_file(

        local_path,

        S3_BUCKET,

        s3_key

    )





# ==========================
# Upload FAISS
# ==========================

def upload_faiss_index(

        index_path:str

):


    """
    上傳 FAISS Index


    index.faiss

    index.pkl

    """


    path = Path(index_path)



    for file in path.iterdir():


        if file.name.endswith(

            (
                ".faiss",
                ".pkl"
            )

        ):


            upload_file(

                str(file),

                f"faiss/{file.name}"

            )