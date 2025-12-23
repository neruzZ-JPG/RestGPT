import os
import json
import logging
import datetime
import time
import yaml

import spotipy
from langchain.requests import Requests
from langchain.llms import OpenAIChat

from utils import reduce_openapi_spec, ColorPrint
from utils import ReducedOpenAPISpec
from model import RestGPT
import logging.config
import os
from pathlib import Path

logger = logging.getLogger()


def main():
    config = yaml.load(open('config-2.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_BASE"] = config["openai_api_base"]
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ["TMDB_ACCESS_TOKEN"] = config['tmdb_access_token']
    os.environ['SPOTIPY_CLIENT_ID'] = config['spotipy_client_id']
    os.environ['SPOTIPY_CLIENT_SECRET'] = config['spotipy_client_secret']
    os.environ['SPOTIPY_REDIRECT_URI'] = config['spotipy_redirect_uri']
    
    scenario = config["scenario"]
    index = config["index"]
    query = config["query"]

    def setup_logging():
        """设置统一的日志配置"""
        # 创建日志目录（如果不存在）
        log_dir = Path(f"logs/{scenario}")
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f"{index}.log"
        logging.basicConfig(
            level = logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(ColorPrint())],
        )
    
    setup_logging()
    
    scenario = scenario.split("_")[0]
    if scenario == 'tmdb':
        with open("specs/tmdb_oas.json") as f:
            raw_tmdb_api_spec = json.load(f)

        api_spec = reduce_openapi_spec(raw_tmdb_api_spec, only_required=False)

        access_token = os.environ["TMDB_ACCESS_TOKEN"]
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
    elif scenario == 'spotify':
        with open("specs/spotify_oas.json") as f:
            raw_api_spec = json.load(f)

        api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

        scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
        access_token = spotipy.util.prompt_for_user_token(scope=','.join(scopes))
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
    elif scenario in ['github', 'gitlab', 'docker', 'kubernetes', 'jenkins']:
        with open(f"specs/{scenario}.json") as f:
            raw_api_spec = json.load(f)
        endpoints_list = []
        # 新增一个变量来存储 base_url
        detected_base_url = "http://localhost" 

        # 兼容两种格式：
        paths_source = raw_api_spec.get("paths", raw_api_spec) 

        for url, methods in paths_source.items():
            if not isinstance(methods, dict): continue
            
            if url.startswith("http"):
                from urllib.parse import urlparse
                parsed = urlparse(url)
                detected_base_url = f"{parsed.scheme}://{parsed.netloc}"

            for method_name, details in methods.items():
                if method_name.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    # 构造符合 RestGPT 要求的元组：(Name, Description, DocsDict)
                    endpoint_name = f"{method_name.upper()} {url}"
                    description = details.get("summary", "") or details.get("description", "")
                    
                    endpoints_list.append(
                        (endpoint_name, description, details)
                    )
        # 3. 重新赋值给 api_spec
        api_spec = ReducedOpenAPISpec(
            # === 修复点：填入 servers 列表 ===
            servers=[{"url": detected_base_url}], 
            # ==============================
            description=f"{scenario} Data", 
            endpoints=endpoints_list
        )
        if scenario == 'github':
            headers = {
                "Authorization" : "Bearer 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"
            }
        elif scenario == 'gitlab':
            headers = {
                "PRIVATE-TOKEN" : "glpat-g73bERuXv9jomlbbVrk4cW86MQp1OmVxbXU3Cw.01.120ws7pth"
            }
        else:
            headers = {}
        scenario = "chatops"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    requests_wrapper = Requests(headers=headers)

    planner_llm = OpenAIChat(
        model_name="gpt-5.1", 
        temperature=1.0, 
        max_tokens=700,
    )
    tool_llm = OpenAIChat(
        model_name='gpt-5-nano',
        temerature=1.0,
        max_token=700,
    )
    rest_gpt = RestGPT(planner_llm=planner_llm, tool_llm=tool_llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)

    # if scenario == 'tmdb':
    #     query_example = "Give me the number of movies directed by Sofia Coppola"
    # elif scenario == 'spotify':
    #     query_example = "Add Summertime Sadness by Lana Del Rey in my first playlist"
    # else:
    #     query_example = "No query example for your scenario, input your query"
    # print(f"Example instruction: {query_example}")
    # query = input("Please input an instruction (Press ENTER to use the example instruction): ")
    # if query == '':
    #     query = query_example
    
    logger.info(f"Query: {query}")

    start_time = time.time()
    rest_gpt.run(query)
    logger.info(f"Execution Time: {time.time() - start_time}")

if __name__ == '__main__':
    main()
