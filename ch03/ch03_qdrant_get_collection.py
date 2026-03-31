from qdrant_client import QdrantClient
client = QdrantClient("http://localhost", port=6333)

collection_name = "llama_index_qdrant"

# 컬렉션 정보 조회 (points_count는 최적화된 세그먼트 기준이라 0일 수 있음)
info = client.get_collection(collection_name=collection_name)
print("[컬렉션 정보]")
print(info)

# 실제 저장된 포인트 수 조회
count = client.count(collection_name=collection_name, exact=True)
print(f"\n[실제 포인트 수] {count.count}")
