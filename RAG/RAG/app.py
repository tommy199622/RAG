from contextlib import asynccontextmanager
from fastapi import HTTPException,FastAPI
from models.chat import ChatRequest,ChatResponse
from services.index_service import load_vectorstore,build_index
from services.rag_service import ask
from utils.logger import logger



@asynccontextmanager
async def lifespan(app:FastAPI):


    logger.info(

        "Starting RAG API"

    )


    try:


        load_vectorstore()


        logger.info(

            "FAISS Loaded"

        )


    except Exception as e:


        logger.warning(

            f"FAISS not loaded: {e}"

        )


    yield


    logger.info(

        "Shutdown RAG API"

    )





app = FastAPI(

    title="Enterprise RAG API",

    version="1.0.0",

    lifespan=lifespan

)




@app.get("/")

def root():

    return {

        "service":

        "Enterprise RAG",

        "status":

        "running"

    }




@app.get("/health")

def health():

    return {

        "status":

        "OK"

    }




@app.post(

    "/chat",

    response_model=ChatResponse

)

def chat(

    request:ChatRequest

):


    try:


        result = ask(

            question=request.question,
            session_id=request.session_id
            

        )


        return result



    except Exception as e:


        logger.exception(e)


        raise HTTPException(

            status_code=500,

            detail=str(e)

        )





@app.post("/index/build")

def build():

    try:


        build_index()


        load_vectorstore()


        return {

            "message":

            "Index build success"

        }



    except Exception as e:


        logger.exception(e)


        raise HTTPException(

            status_code=500,

            detail=str(e)

        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )