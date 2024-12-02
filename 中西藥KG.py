import os
import pandas as pd
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, StorageContext
from langchain_ollama import ChatOllama, OllamaEmbeddings
from llama_index.core.text_splitter import SentenceSplitter
from neo4j import GraphDatabase
import numpy as np
from llama_index.core.node_parser import SimpleNodeParser

# ===========================
# 1. 初始化 LLM 和嵌入模型
# ===========================

# 初始化 LLM
llm = ChatOllama(model="llama3.1")

# 初始化嵌入模型
embedding_provider = OllamaEmbeddings(model="llama3.1")

# ===========================
# 2. 連接 Neo4j 資料庫
# ===========================

# 連接 Neo4j 資料庫
neo4j_url = 'bolt://localhost:7687'
neo4j_username = 'neo4j'
neo4j_password = 'stockinfo'

driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))

def add_chunk_to_graph(tx, properties):
    tx.run("""
        MERGE (h:Herb {name: $herb_name})
        MERGE (d:Drug {name: $drug_name})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (h)-[:INTERACTS_WITH]->(d)
        MERGE (c)-[:DESCRIBES]->(h)
        MERGE (c)-[:DESCRIBES]->(d)
    """, properties)

# ===========================
# 3. 加載和預處理資料
# ===========================

# 加載 JSON 資料集
data = pd.read_json('medical_dataset.json')

# 資料預處理
data = data[['ID', '中藥中文名', '中藥英文名', '西藥學名', '可能交互作用結果', '建議處理方式']].dropna()

# 將資料轉換為 Document 格式
documents = []
for i, row in data.iterrows():
    content = f"中藥：{row['中藥中文名']}（{row['中藥英文名']}），西藥：{row['西藥學名']}。可能交互作用結果：{row['可能交互作用結果']}。建議處理方式：{row['建議處理方式']}。"
    metadata = {
        'ID': row['ID'],
        'herb_name': row['中藥中文名'],
        'drug_name': row['西藥學名']
    }
    doc = Document(text=content, metadata=metadata)
    documents.append(doc)

# ===========================
# 4. 文本分割
# ===========================

# 初始化文本分割器
parser = SimpleNodeParser()

# 分割文檔
nodes = parser.get_nodes_from_documents(documents)

# ===========================
# 5. 生成嵌入向量並存儲到 Neo4j
# ===========================

with driver.session() as session:
    for node in nodes:
        herb_name = node.metadata.get('herb_name', 'Unknown')
        drug_name = node.metadata.get('drug_name', 'Unknown')
        chunk_id = f"{node.metadata.get('ID', '0')}.{node.metadata.get('page', 0)}"
        print(f"處理區塊 ID：{chunk_id}")
        print("內容：")
        print(node.text)
        print("-" * 50)

        # 生成嵌入向量
        chunk_embedding = embedding_provider.embed_query(node.text)
        # 確保嵌入向量是列表格式，並轉換為可存儲的格式（例如字串）
        chunk_embedding_list = chunk_embedding.tolist() if isinstance(chunk_embedding, np.ndarray) else chunk_embedding

        properties = {
            "herb_name": herb_name,
            "drug_name": drug_name,
            "chunk_id": chunk_id,
            "text": node.text,
            "embedding": chunk_embedding_list  # 需要根據 Neo4j 的存儲方式調整
        }

        # 將區塊添加到圖中
        session.write_transaction(add_chunk_to_graph, properties)
'''
# ===========================
# 6. 建立向量索引
# ===========================

# 在 Neo4j 中創建向量索引
#=====================================================================
# ANN 限定neo4j--version == 5.x
# 本地目前使用neo4j-3.8
with driver.session() as session:
    session.run("""
        CREATE ANN INDEX IF NOT EXISTS chunkVector
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {
        similarity_function: 'cosine',
        vector_dimensions: 1536
        };
    """)

# ===========================
# 7. 創建並保存 Llama-Index 向量索引
# 不知道為啥會卡openai-api問題
# ===========================

# 創建向量存儲索引
index = VectorStoreIndex(nodes)

# 保存索引到磁碟（可選）
storage_context = StorageContext.from_defaults(persist_dir="medical_chunks_index")
index.storage_context.persist(persist_dir="medical_chunks_index")

# ===========================
# 8. 查詢示例
# ===========================

# 加載索引（如果之前已保存）
# storage_context = StorageContext.from_defaults(persist_dir="medical_chunks_index")
# index = VectorStoreIndex.load_from_storage(storage_context)

# 示例查詢
query = "中藥與西藥的潛在交互作用有哪些？"

response = index.query(query)

print("查詢結果：")
print(response)
'''
# 關閉 Neo4j 驅動
driver.close()
