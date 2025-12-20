import os
import json
import logging
from tqdm import tqdm # 进度条库，建议 pip install tqdm

# 引入 RestGPT 的组件 (假设你在项目根目录运行)
from restgpt.utils import reduce_openapi_spec  # 确保这里是你上一轮修改过的版本
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

# 配置你的 API Key (如果没有在环境变量设置)
# os.environ["OPENAI_API_KEY"] = "sk-..."

def load_and_merge_json_files(data_folder):
    """
    遍历文件夹，读取所有 JSON，并利用 reduce_openapi_spec 提取 endpoint
    """
    all_endpoints = []
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    print(f"Found {len(files)} files in {data_folder}")

    for filename in tqdm(files, desc="Processing JSONs"):
        file_path = os.path.join(data_folder, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                spec_data = json.load(f)
            
            # 关键步骤：调用修改后的 reduce 函数
            # dereference=False 因为你的数据看起来已经是清洗展开过的
            reduced_spec = reduce_openapi_spec(spec_data, dereference=False)
            
            # reduced_spec.endpoints 是一个列表: [(name, description, detailed_docs), ...]
            all_endpoints.extend(reduced_spec.endpoints)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return all_endpoints

def build_index(endpoints, persist_dir="./data/chroma"):
    """
    将提取出的 endpoints 转换成 LangChain Document 并存入 ChromaDB
    """
    documents = []
    
    print(f"Building index for {len(endpoints)} endpoints...")
    
    for name, description, docs in endpoints:
        # 1. 构造存入向量库的文本 (RestGPT 通常用 name + description)
        # name 类似: "POST https://api.github.com/..."
        # description 类似: "Create an issue..."
        page_content = f"{name}\n{description}"
        
        # 2. 构造元数据 (Metadata)，实际调用 API 时需要完整的 JSON
        metadata = {
            "name": name,
            # 将复杂的 dict 转为字符串存入 metadata，方便后续取出
            "spec": json.dumps(docs) 
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    # 3. 创建向量库 (这里演示用 Chroma，你可以换成 FAISS)
    embeddings = OpenAIEmbeddings() # 也可以换成 HuggingFaceEmbeddings 以节省成本
    
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    vector_store.persist()
    print(f"Index saved to {persist_dir}")

if __name__ == "__main__":
    # 修改这里的路径指向你的 json 文件夹
    MY_DATA_FOLDER = "./my_data"
    PERSIST_DB_PATH = "./specs/my_custom_service/index" # 索引保存路径
    
    # 1. 加载并合并所有文件的 endpoint
    merged_endpoints = load_and_merge_json_files(MY_DATA_FOLDER)
    
    # 2. 建立索引
    if merged_endpoints:
        build_index(merged_endpoints, persist_dir=PERSIST_DB_PATH)
    else:
        print("No endpoints found. Check your JSON format.")