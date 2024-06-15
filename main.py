from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from uuid import uuid4
import uvicorn
import os

from schema import IngestRequest, QueryModel, QueryLLM, QueryGPT, VectordDB
from utils import masking, unmasking, ingestion_pipeline, generate_chat_gpt, generate_llm
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, JSONLoader
import chromadb
from chromadb import DEFAULT_TENANT
from chromadb.config import Settings
import logging
import json


app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO
    )
logger = logging.getLogger(__name__)

# Chroma Admin Client for creating databases
# adminClient = chromadb.AdminClient(Settings(
#     is_persistent=True,
#     persist_directory="./chroma-db",
#     allow_reset = True
# ))

# # Chroma database get or create
# def get_or_create_chroma_db(vectordb_name):
#     database = f"{vectordb_name}"
#     try:
#         adminClient.get_database(database)
#     except Exception as e:
#         adminClient.create_database(database, DEFAULT_TENANT)
#     return DEFAULT_TENANT, database


def get_chroma_client():
    #_, chroma_database = get_or_create_chroma_db(vectordb_name=vectordb_name)
    client = chromadb.PersistentClient(
        path="./chroma-db",
        settings=Settings(allow_reset=True),
        )
        #database=chroma_database)
    #logger.info(f"using database: {chroma_database}")
    try:
        return client
    except Exception as e:
        logger.error("error: ", str(e))
        raise HTTPException(status_code=400, detail=f"Chroma Client not working, Error: {str(e)}")
        

    # finally:
    #     client.close() 


# Endpoints

@app.post("/ingest-pdf-unmasked")
async def ingest_pdf_unmasked(
    file: UploadFile = File(...),
    request: IngestRequest = Depends(),
    client = Depends(get_chroma_client),
):

    if not file.filename.endswith(".pdf"):
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File format not supported")
    
    try:
        # saving file in temporary location
        temp_file_path = f"temp_files_dir/{file.filename}"
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.file.read())
        logger.info(f"File saved temporarily at {temp_file_path}")

        # loading
        loader = PyMuPDFLoader(temp_file_path)
        data = loader.load()
        logger.info(f"Loaded data from {file.filename}")

        ingestion_pipeline(data=data, request=request, file=file, client=client)
      
        return {"Message": "File Ingested Successfully"}

    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} removed")


@app.post("/ingest-text-unmasked")
async def ingest_text_unmasked(
    file: UploadFile = File(...),
    request: IngestRequest = Depends(),
    client = Depends(get_chroma_client),
):

    if not file.filename.endswith(".txt"):
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File format not supported")
    
    try:
        # saving file in temporary location
        temp_file_path = f"temp_files_dir/{file.filename}"
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.file.read())
        logger.info(f"File saved temporarily at {temp_file_path}")

        # loading
        loader = TextLoader(temp_file_path)
        data = loader.load()
        logger.info(f"Loaded data from {file.filename}")

        ingestion_pipeline(data=data, request=request, file=file, client=client)
     
        return {"Message": "File Ingested Successfully"}

    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} removed")


### json schema is important
@app.post("/ingest-json-unmasked")
async def ingest_json_unmasked(
    content_key: str,
    file: UploadFile = File(...),
    request: IngestRequest = Depends(),
    client = Depends(get_chroma_client),
    
):

    if not file.filename.endswith(".json"):
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File format not supported")
    
    try:
        # saving file in temporary location
        temp_file_path = f"temp_files_dir/{file.filename}"
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.file.read())
        logger.info(f"File saved temporarily at {temp_file_path}")

        # loading
        loader = JSONLoader(
            file_path=temp_file_path, 
            jq_schema='.[]',
            content_key=content_key,
            text_content=True
            )
        data = loader.load()
        logger.info(f"Loaded data from {file.filename}")

        ingestion_pipeline(data=data, request=request, file=file, client=client)

        return {"Message": "File Ingested Successfully"}

    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} removed")



@app.post("/ingest-pdf-masked")
async def ingest_pdf_masked(
    file: UploadFile = File(...),
    mask_source: UploadFile = File(...),
    request: IngestRequest = Depends(),
    client = Depends(get_chroma_client),
):

    if not file.filename.endswith(".pdf"):
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File format not supported")
    
    try:
        # saving file in temporary location
        temp_file_path = f"temp_files_dir/{file.filename}"
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.file.read())
        logger.info(f"File saved temporarily at {temp_file_path}")

        # loading
        loader = PyMuPDFLoader(temp_file_path)
        data = loader.load()
        logger.info(f"Loaded data from {file.filename}")

        # loading mask source (JSON)
        mask_source = await mask_source.read()
        mask_source = json.loads(mask_source)

        # masking
        for page in data:
            page.page_content = masking(page.page_content, mask_source)

        # ingestion pipeline
        ingestion_pipeline(data=data, request=request, file=file, client=client)
      
        return {"Message": "File Ingested Successfully"}

    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} removed")


@app.post("/ingest-text-masked")
async def ingest_text_masked(
    file: UploadFile = File(...),
    mask_source: UploadFile = File(...),
    request: IngestRequest = Depends(),
    client = Depends(get_chroma_client),
):

    if not file.filename.endswith(".txt"):
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File format not supported")
    
    try:
        # saving file in temporary location
        temp_file_path = f"temp_files_dir/{file.filename}"
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.file.read())
        logger.info(f"File saved temporarily at {temp_file_path}")

        # loading
        loader = TextLoader(temp_file_path)
        data = loader.load()
        logger.info(f"Loaded data from {file.filename}")

        # loading mask source (JSON)
        mask_source = await mask_source.read()
        mask_source = json.loads(mask_source)

        # masking
        for page in data:
            page.page_content = masking(page.page_content, mask_source)

        ingestion_pipeline(data=data, request=request, file=file, client=client)
     
        return {"Message": "File Ingested Successfully"}

    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} removed")


### json schema is important
@app.post("/ingest-json-masked")
async def ingest_json_masked(
    content_key: str,
    file: UploadFile = File(...),
    mask_source: UploadFile = File(...),
    request: IngestRequest = Depends(),
    client = Depends(get_chroma_client),
):

    if not file.filename.endswith(".json"):
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File format not supported")
    
    try:
        # saving file in temporary location
        temp_file_path = f"temp_files_dir/{file.filename}"
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.file.read())
        logger.info(f"File saved temporarily at {temp_file_path}")

        # loading
        loader = JSONLoader(
            file_path=temp_file_path, 
            jq_schema='.[]',
            content_key=content_key,
            text_content=True
            )
        data = loader.load()
        logger.info(f"Loaded data from {file.filename}")

        # loading mask source (JSON)
        mask_source = await mask_source.read()
        mask_source = json.loads(mask_source)

        # masking
        for page in data:
            page.page_content = masking(page.page_content, mask_source)

        ingestion_pipeline(data=data, request=request, file=file, client=client)

        return {"Message": "File Ingested Successfully"}

    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} removed")





@app.post("/query-chat-gpt")
async def query_chat_gpt(
    query: QueryGPT = Depends(),
    client = Depends(get_chroma_client)
):
    
    try:
        collection_name = f"{query.collection}"
        collection = client.get_collection(collection_name)
        
        retrieved_docs = collection.query(
            query_texts = query.prompt,
            n_results = query.n_results
        )
       
        formatted_context = "\n\n".join(doc for doc in retrieved_docs['documents'][0])
        formatted_prompt = f"Question: {query.prompt}\n\nProvided Context: {query.context}\n\nRetrieved Context: {formatted_context}"

        res = generate_chat_gpt(prompt=formatted_prompt, temperature=query.temperature)
        return {
            "prompt": query.prompt,
            "response": res,
            "retrieved_results": [doc for doc in retrieved_docs['documents'][0]]
            }
   
    except Exception as e:
        logger.error(f"Error in query_chat_gpt, Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-llm")
async def query_llm(
    structure: UploadFile = File(...),
    query: QueryLLM = Depends(),
    client = Depends(get_chroma_client)
):
    if not structure.filename.endswith(".json"):
        logger.warning(f"Unsupported file format: {structure.filename}")
        raise HTTPException(status_code=400, detail="File format not supported")
    try:
        # loading mask source (JSON)
        structure = await structure.read()
        structure = json.loads(structure)
        prompt = structure["prompt"]

        collection_name = f"{query.collection}"
        collection = client.get_collection(collection_name)
        
        retrieved_docs = collection.query(
            query_texts = prompt,
            n_results = query.n_results
        )
       
        formatted_context = "\n\n".join(doc for doc in retrieved_docs['documents'][0])
        formatted_prompt = f"Question: {prompt}\n\nProvided Context: {query.context}\n\nRetrieved Context: {formatted_context}"

        structure['prompt'] = formatted_prompt
        print(structure['prompt'])
        res = generate_llm(structure=structure)
        
        return res.json() 
    
            # {
            # "prompt": prompt,
            # "response": response,
            # "retrieved_results": [doc for doc in retrieved_docs['documents'][0]]
            # }
   
    except Exception as e:
        logger.error(f"Error in query_chat_gpt, Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unmask")
async def unmask(
    input: str,
    mask_source: UploadFile = File(...)
):
    try:
        # loading mask source (JSON)
        mask_source = await mask_source.read()
        mask_source = json.loads(mask_source)

        text = unmasking(text=input, mask_source=mask_source)

        return {"Unmasked_text": text}
    
    except Exception as e:
        logger.error(f"Error while unmasking, Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/list-all-vector-db")
# async def list_vectordb(
#     client = Depends(get_chroma_client)
# ):
#     available_dbs = ''
#     #available_dbs = client.get_settings()
#     #available_dbs = [db.value for db in VectordDB]
#     return available_dbs



@app.get("/list-all-collections")
async def list_collections(
    vectordb: VectordDB,    
    client = Depends(get_chroma_client)
):
    try:
        return client.list_collections()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection")
async def get_collection(
    collection: str,
    client = Depends(get_chroma_client)
):
    try:
        collection_name = f"{collection}"
        collection = client.get_collection(collection_name)
        
        return collection.get()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/count-collection", description="To count the no. of records in the collection")
async def count_collection(
    collection: str,
    client = Depends(get_chroma_client)
):
    try:
        collection = client.get_collection(name=f"{collection}") 
        return {"collection_count": collection.count()}
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/delete-collection")
async def delete_collection(
    collection: str,
    client = Depends(get_chroma_client)
):
    try:
        client.delete_collection(name=f"{collection}") 
        return {"message": f"Collection {collection} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# use allow-reset=True in settings
@app.post("/delete-vector-db")
async def delete_vectordb(
    client = Depends(get_chroma_client)
):
    try:
        client.reset()
        return {"message":"Database deleted successfully"}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))    



# #######################################################################

# @app.post("/testing-masking-documents")
# async def testing(
#     file: UploadFile = File(...),
#     mask_source: UploadFile = File(...),
#     #request: IngestRequest = Depends(),
#     #client = Depends(get_chroma_client),
# ):
#     print('testing ...')
#     if not file.filename.endswith(".pdf"):
#         logger.warning(f"Unsupported file format: {file.filename}")
#         raise HTTPException(status_code=400, detail="File format not supported")
    
#     try:
#         # saving file in temporary location
#         temp_file_path = f"temp_files_dir/{file.filename}"
#         with open(temp_file_path, 'wb') as temp_file:
#             temp_file.write(file.file.read())
#         logger.info(f"File saved temporarily at {temp_file_path}")

#         # loading
#         loader = PyMuPDFLoader(temp_file_path)
#         data = loader.load()
#         logger.info(f"Loaded data from {file.filename}")

#         # loading mask source (JSON)
#         mask_source = await mask_source.read()
#         mask_source = json.loads(mask_source)

#         for page in data:
#             page.page_content = masking(page.page_content, mask_source)

        

#         return {"Message": "File Ingested Successfully"}

#     except Exception as e:
#         logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
    
#     finally:
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)
#             logger.info(f"Temporary file {temp_file_path} removed")


# @app.post("/query")
# def query(
#     query: QueryModel = Depends(),
#     client = Depends(get_chroma_client)
# ):
    
#     try:
#         collection_name = f"{query.user_id}"
#         collection = client.get_collection(collection_name)
#         if collection:
#             results = collection.query(
#                 query_texts = query.query,
#                 n_results = query.n_results
#             )
            
#             return {"query": query.query, "results": results['documents'][0]}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# #######################################################################


# Run the application
if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, log_config=f"./log.ini")
