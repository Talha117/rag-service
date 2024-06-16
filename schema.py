from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum



###########################################################

class EmbedMethod(str, Enum):
    method0 = "all-MiniLM-L6-v2" # working
#    method1 = "CustomEmbedding" # langchain huggingface # not working: "Expected EmbeddingFunction.__call__ to have the following signature: odict_keys(['self', 'input']), got odict_keys(['args', 'kwargs'])"
    method2 = "OpenAIEmbedding"
#    method3 = "HuggingFaceEmbedding" 
    


class ChunkMethod(str, Enum):
    method1 = "Recursive"
    method2 = "Simple"
    method3 = "Token"


class VectordDB(str, Enum):
    db1 = "Chroma"
    db2 = "Pinecone"


class IngestRequest(BaseModel):
    #vectordb_name: str
    collection: str
    chunk_method: ChunkMethod
    chunk_size: int
    chunk_overlap: int
    embed_method: EmbedMethod


class QueryModel(BaseModel):
    user_id: str
    query: str
    n_results: int = 3



class QueryGPT(BaseModel):
    #vectordb_name: str
    collection: str
    prompt: str
    context:str
    n_results: int
    #structure: Optional[Dict[str, Any]] = None
    temperature: float
    embed_method: EmbedMethod

class QueryLLM(BaseModel):
    #vectordb_name: str
    collection: str
    prompt: str
    context: str
    n_results: int
    #structure: Optional[Dict[str, Any]] = None
    #temperature: float
    embed_method: EmbedMethod

############################################################