# create_db
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.node_parser import SimpleNodeParser

from neo4j import GraphDatabase
import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv

# %% initial
load_dotenv()
# initial LLM
llm = Ollama(model=os.getenv("CHAT_MODEL_TYPE"))

# initial embedding model
embedding_provider = OllamaEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL_TYPE"),
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# %% connect to Neo4j
# acceess info
neo4j_url = os.getenv("NEO4J_URL")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))

def add_interaction_to_graph(tx, properties):
    tx.run("""
        MERGE (h:Herb {name: $herb_name})
        MERGE (d:Drug {name: $drug_name})
        MERGE (i:interaction {id: $interaction_id})
        SET i.text = $text
        SET i.embedding = $embedding
        MERGE (h)-[:INTERACTS_WITH]->(d)
        MERGE (c)-[:DESCRIBES]->(h)
        MERGE (c)-[:DESCRIBES]->(d)
    """, properties)

# %% load and preprocess
# load .json dataset
data_path = os.path.join("dataset", "medical_dataset.json")
data = pd.read_json(data_path)

# preprocessing
data = data[['ID', '中藥中文名', '中藥英文名', '西藥學名', '可能交互作用結果', '建議處理方式']].dropna()

# convert to documents
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

# %% document splitting
# initial splitter
parser = SimpleNodeParser()

nodes = parser.get_nodes_from_documents(documents)

# %% embedding to Neo4j
with driver.session() as session:
    for node in nodes:
        herb_name = node.metadata.get('herb_name', 'Unknown')
        drug_name = node.metadata.get('drug_name', 'Unknown')
        interaction_id = f"{node.metadata.get('ID', '0')}"
        print(f"交互作用 ID：{interaction_id}")
        print("內容：")
        print(node.text)
        print("-" * 50)

        # embedding generate
        interaction_embedding = embedding_provider.get_text_embedding_batch([node.text], show_progress=True)[0]
        
        properties = {
            "herb_name": herb_name,
            "drug_name": drug_name,
            "interaction_id": interaction_id,
            "text": node.text,
            "embedding": interaction_embedding  # can be change to what Neo4j can store
        }

        # add chunk in graph
        session.write_transaction(add_interaction_to_graph, properties)

# %% create indexes in Neo4j
with driver.session() as session:
    # Create VECTOR INDEX
    session.run("""
    CREATE VECTOR INDEX MedicineInteractionVectorIndex IF NOT EXISTS
    FOR (i:interaction)
    ON i.embedding
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 4096,
        `vector.similarity_function`: 'cosine'
      }
    }
    """)

# close the Neo4j driver
driver.close()
print("knowledge graph has been created")
# %%