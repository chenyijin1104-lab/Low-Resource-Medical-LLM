import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ================= 配置区 =================
# 指向你刚刚用 chunker 切片生成的 JSONL 文件
CHUNKS_FILE = "data/kg_data/tcm_massive_chunks.jsonl" 
# ChromaDB 向量数据库持久化存储的目录
DB_PATH = "data/vector_db" 
# 选用 BAAI 开源的顶级轻量化中文 Embedding 模型
MODEL_NAME = "BAAI/bge-small-zh-v1.5" 
# ==========================================

def build_vector_db():
    print(f"🚀 [1/4] 正在加载 Embedding 翻译官模型: {MODEL_NAME}")
    print("（首次运行会自动从 HuggingFace 自动拉取模型权重，请耐心等待）")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\n📂 [2/4] 正在初始化 ChromaDB 向量数据库...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 创建或加载集合 (Collection)，指定使用“余弦相似度 (Cosine Similarity)”
    collection = client.get_or_create_collection(
        name="tcm_ancient_books",
        metadata={"hnsw:space": "cosine"} 
    )

    print("\n📖 [3/4] 正在读取古籍切片数据...")
    documents = []
    metadatas = []
    ids = []

    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ 找不到切片文件 {CHUNKS_FILE}，请确认你已经成功运行了 batch_tcm_chunker.py！")
        return

    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            documents.append(data['text'])
            # 将书名和长度作为元数据 (Metadata) 存入，以后可以支持“只查某本书”的功能
            metadatas.append({"book": data['book'], "length": data['length']})
            ids.append(f"chunk_{i}")

    total_chunks = len(documents)
    print(f"✅ 共读取到 {total_chunks} 个知识块。准备开始向高维空间跃迁！")

    print("\n⚙️ [4/4] 引擎全开：开始计算向量并分批入库（这会极大消耗 CPU/GPU 算力）...")
    # 采用分批写入策略（Batching），每次 5000 条，防止内存撑爆
    batch_size = 5000
    for i in range(0, total_chunks, batch_size):
        batch_docs = documents[i : i+batch_size]
        batch_metas = metadatas[i : i+batch_size]
        batch_ids = ids[i : i+batch_size]

        print(f"   ⏳ 正在处理第 {i} 到 {i+len(batch_docs)} 条数据...")
        
        # 1. 调用大模型将文言文翻译成 768 维特征向量
        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()

        # 2. 将原文、元数据和向量一并砸进 ChromaDB 数据库
        collection.add(
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
    print("\n🎉 伟大的一刻：中医古籍千万级向量库构建彻底完成！")
    print(f"📊 数据库当前总管辖数据量: {collection.count()} 条")

if __name__ == "__main__":
    # 确保数据库存放目录存在
    os.makedirs(DB_PATH, exist_ok=True)
    build_vector_db()