import requests
from datetime import datetime, timezone
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb.utils.embedding_functions as embedding_functions


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from openai import OpenAI
from uuid import uuid4
from config import OPENAI_API_KEY, LLM_URL

#from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)


def chunk_text(docs, chunk_method: str, chunk_size: int, chunk_overlap: int):
    
    if chunk_method == "Recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len,
            #is_separator_regex=False,
        )
    elif chunk_method == "Simple":
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            #is_separator_regex=False,
        )

    elif chunk_method == "Token":
        text_splitter = TokenTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
    
    splits = text_splitter.split_documents(docs)

    return splits


def create_metadata(splits, filename):
    # creating metadatas for chunks
    docs = [split.page_content for split in splits]
    metadatas = [split.metadata for split in splits]
    
    # updating source in metadata
    for metadata in metadatas:
        metadata['source'] = filename
        # metadata['ingestionDate'] = datetime.now(timezone.utc).isoformat()
        # metadata['ingestionDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata['ingestionDate'] = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


    ids = [str(uuid4()) for _ in docs]   

    return (docs, metadatas, ids)


def get_embedding_function(embed_method: str):
    if embed_method=="HuggingFaceEmbedding":
        ef = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key="YOUR_API_KEY",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif embed_method=="OpenAIEmbedding":
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
    elif embed_method=="CustomEmbedding":
        ef = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        #ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    elif embed_method=="all-MiniLM-L6-v2":
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


    return ef


def ingestion_pipeline(data, request, file, client):
    # creating chunks
    splits = chunk_text(
        docs = data, 
        chunk_method = request.chunk_method,
        chunk_size = request.chunk_size,
        chunk_overlap = request.chunk_overlap
    )
    logger.info(f"Created chunks for {file.filename}")

    # creating metadatas for chunks  
    docs, metadatas, ids = create_metadata(splits = splits, filename = file.filename)
    logger.info(f"Created metadata for {file.filename}")

    # selecting embedding function 
    ef = get_embedding_function(embed_method = request.embed_method)
    logger.info(f"Selected embedding function: {request.embed_method}")

    # unique collection for each user
    collection_name = f"{request.collection}"
    collection = client.get_or_create_collection(name = collection_name, embedding_function = ef, metadata={"hnsw:space": "cosine"})
    logger.info(f"Using collection: {collection_name}")

    # Ingestion
    collection.add(
        documents = docs,
        metadatas = metadatas,
        ids = ids
    )
    logger.info(f"Successfully ingested file: {file.filename}")


def masking(text, mask_source):
    for item in mask_source:
        text = text.replace(item['clear_text'], item['uuid'])
    return text


def unmasking(text, mask_source):
    for item in mask_source:
        text = text.replace(item['uuid'], item['clear_text'])
    return text


def generate_chat_gpt(
        prompt: str,
        system_prompt='',
        prev_messages=[],
        temperature = 0.4
    ):

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(prev_messages)
    messages.append({"role": "user", "content": prompt.strip()})
    # For CLI info
    logger.info("--Sending request to OpenAI endpoint--")

    response = client.chat.completions.create(
        model=  "gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens = 2000,
        temperature=temperature
    )

    logger.info(response.usage)
    return response.choices[0].message.content


def generate_llm(
        structure  
    ):
    payload = json.dumps(structure)

    headers = {
        'Content-Type': 'application/json'
    }
    
    logger.info("--Sending request to LLM endpoint--")
    response = requests.post(LLM_URL, headers=headers ,data=payload)
    return response

#############################################################
## Langchain Implementation of Embedding Model
def get_embedding_model(embed_method: str):
    
    if embed_method=="HuggingFaceEmbedding":
        embedding_model=HuggingFaceEmbeddings()
    
    return embedding_model
