import os
import json
from utils.chatops_utils import reduce_openapi_spec  # 调用你之前修改过的 reduce 函数

def merge_all_jsons(input_folder, output_file):
    merged_spec = {}
    
    files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    print(f"检测到 {len(files)} 个文件，开始合并...")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            
            # 2. 将清洗后的 endpoints 重新组装回字典格式
            # reduced_data.endpoints 是一个列表: [(name, desc, dict), ...]
            reduced_data = reduce_openapi_spec(data, dereference=False)
            
            for name, description, docs in reduced_data.endpoints:
                parts = name.split(" ", 1)
                if len(parts) == 2:
                    method, url = parts
                    
                    if url not in merged_spec:
                        merged_spec[url] = {}
                    merged_spec[url][method.lower()] = docs
                    
        except Exception as e:
            print(f"跳过文件 {filename}: {e}")

    # 3. 保存合并后的文件
    # 修改上面的第 3 步保存逻辑
    
    # 伪造一个标准 OAS 结构
    final_output = {
        "openapi": "3.0.0",
        "info": {
            "title": "chatops",
            "version": "1.0.0",
            "description": "chatops APIs"
        },
        "paths": {}  # 关键在这里
    }

    # 把我们合并的 merged_spec (格式是 url -> methods) 塞进 paths
    # 注意：标准 OAS 的 paths key 通常是相对路径 (例如 /issues)，但你的是绝对路径
    # RestGPT 对此通常不敏感，只要是字符串即可
    final_output["paths"] = merged_spec

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 配置路径
    system = "gitlab"
    INPUT_FOLDER = f"./APIs/{system}"  # 你的原始文件目录
    OUTPUT_FILE = f"./specs/{system}.json" # 输出给 RestGPT 用的文件
    
    merge_all_jsons(INPUT_FOLDER, OUTPUT_FILE)