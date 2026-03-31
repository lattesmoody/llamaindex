# pip install llama-index-vector-stores-qdrant==0.3.0
# pip install llama-index==0.11.11
# pip install llama-index-embeddings-openai
# pip install llama-index-embeddings-huggingface
# https://aka.ms/vs/17/release/vc_redist.x64.exe

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core.schema import Document

from qdrant_client import QdrantClient
client = QdrantClient("http://localhost", port=6333)

# 로컬 임베딩 모델 설정 (한국어 지원)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.embed_model = embed_model

documents = [
    "고양이는 작은 육식동물로, 주로 애완동물로 기릅니다. 민첩하고 장난기 있는 행동으로 유명합니다.",
    "강아지는 충성심이 강하고 친절한 동물로, 흔히 인간의 최고의 친구로 불립니다. 주로 애완동물로 기르고, 동반자로서 유명합니다.",
    "고양이와 강아지는 전 세계적으로 인기 있는 애완동물로, 각각 독특한 특징을 가지고 있습니다."
]
ids = ["doc1", "doc2", "doc3"]

documents = [Document(text=doc, id_=doc_id) for doc, doc_id in zip(documents, ids)]

collection_name = "llama_index_qdrant"


# 이미 존재하면 삭제 후 재생성
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  # bge-m3 모델 차원
)

# Qdrant 벡터 스토어 설정
vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

# StorageContext를 통해 Qdrant에 실제 저장
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 인덱스 생성 및 저장 (Qdrant에 실제 저장)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 인덱스 생성 및 저장 (인메모리 인덱스 생성, Qdrant에 저장 안 됨)
# index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)


# 쿼리 엔진 생성 및 실행
query_engine = index.as_query_engine()
query_text = "고양이"
response = query_engine.query(query_text)

print("[질의 결과]")
print(response)

print("\n[응답에 사용된 문서]")
for i, node in enumerate(response.source_nodes, 1):
    print(f"{i}. {node.text}\n")

# 실제 쿼드런트에 저장된 포인트 수 조회
count = client.count(collection_name=collection_name, exact=True)
print(f"\n[실제 Qdrant 포인트 수] {count.count}")
