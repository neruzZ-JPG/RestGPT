"""Quick and dirty representation for OpenAPI specs."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

def dereference_refs(spec_obj: dict, full_spec: dict) -> Union[dict, list]:
    """Try to substitute $refs.

    The goal is to get the complete docs for each endpoint in context for now.

    In the few OpenAPI specs I studied, $refs referenced models
    (or in OpenAPI terms, components) and could be nested. This code most
    likely misses lots of cases.
    """

    def _retrieve_ref_path(path: str, full_spec: dict) -> dict:
        components = path.split("/")
        if components[0] != "#":
            raise RuntimeError(
                "All $refs I've seen so far are uri fragments (start with hash)."
            )
        out = full_spec
        for component in components[1:]:
            out = out[component]
        return out

    def _dereference_refs(
        obj: Union[dict, list], stop: bool = False
    ) -> Union[dict, list]:
        if stop:
            return obj
        obj_out: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "$ref":
                    # stop=True => don't dereference recursively.
                    return _dereference_refs(
                        _retrieve_ref_path(v, full_spec), stop=False
                    )
                elif isinstance(v, list):
                    obj_out[k] = [_dereference_refs(el) for el in v]
                elif isinstance(v, dict):
                    obj_out[k] = _dereference_refs(v)
                else:
                    obj_out[k] = v
            return obj_out
        elif isinstance(obj, list):
            return [_dereference_refs(el) for el in obj]
        else:
            return obj

    return _dereference_refs(spec_obj)


def merge_allof_properties(obj):
    def merge(to_merge):
        merged = {'properties': {}, 'required': [], 'type': 'object'}
        for partial_schema in to_merge:
            if 'allOf' in partial_schema:
                tmp = merge(partial_schema['allOf'])
                merged['properties'].update(tmp['properties'])
                if 'required' in tmp:
                    merged['required'].extend(tmp['required'])
                continue
            if 'properties' in partial_schema:
                merged['properties'].update(partial_schema['properties'])
            if 'required' in partial_schema:
                merged['required'].extend(partial_schema['required'])
        return merged

    def _merge_allof(obj):
        obj_out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'allOf':
                    return _merge_allof(merge(v))
                elif isinstance(v, list):
                    obj_out[k] = [_merge_allof(el) for el in v]
                elif isinstance(v, dict):
                    obj_out[k] = _merge_allof(v)
                else:
                    obj_out[k] = v
            return obj_out
        elif isinstance(obj, list):
            return [_merge_allof(el) for el in obj]
        else:
            return obj

    return _merge_allof(obj)


@dataclass
class ReducedOpenAPISpec:
    servers: List[dict]
    description: str
    endpoints: List[Tuple[str, Union[str, None], dict]]

def reduce_openapi_spec(spec: dict, dereference: bool = False, only_required: bool = True, merge_allof: bool = False) -> ReducedOpenAPISpec:
    """
    Modified to handle your custom cleaned JSON format:
    { "FULL_URL": { "METHOD": { ...details... } } }
    """
    
    # 1. 适配你的数据结构：直接遍历 spec 的 items，而不是 spec["paths"]
    # 注意：你的 key 是完整的 URL (https://...), RestGPT 通常把 server 和 path 分开
    # 这里我们把完整的 URL 当作 endpoint name 处理
    endpoints = []
    
    # 遍历你的根字典
    for url, operations in spec.items():
        # 你的 JSON 中，url 对应的值是一个包含 "post", "get" 等方法的字典
        if not isinstance(operations, dict):
            continue
            
        for operation_name, docs in operations.items():
            # 过滤非 HTTP 方法
            if operation_name.lower() not in ["get", "post", "patch", "delete", "put"]:
                continue
                
            # 构造 endpoint 标识: "POST https://api.github.com/..."
            endpoint_name = f"{operation_name.upper()} {url}"
            
            # 提取描述
            description = docs.get("summary") or docs.get("description") or ""
            
            endpoints.append(
                (endpoint_name, description, docs)
            )

    # 2. Dereference (可选)
    # 鉴于你的 JSON 看起来已经把 schema 展开了（没有看到 $ref），
    # 且你的根对象里没有 "components" 字段，这里的 dereference 会报错，
    # 所以建议默认设置为 False，或者直接跳过。
    if dereference and "components" in spec:
        endpoints = [
            (name, description, dereference_refs(docs, spec))
            for name, description, docs in endpoints
        ]

    # 3. Merge allOf (保持原逻辑，可选)
    if merge_allof:
        endpoints = [
            (name, description, merge_allof_properties(docs))
            for name, description, docs in endpoints
        ]

    # 4. 精简文档 (Keep original logic)
    def reduce_endpoint_docs(docs: dict) -> dict:
        out = {}
        if docs.get("description"):
            out["description"] = docs.get("description")
        if docs.get("summary"):
            out["summary"] = docs.get("summary")
            
        # 处理 Parameters
        if docs.get("parameters"):
            if only_required:
                out["parameters"] = [
                    p for p in docs.get("parameters", [])
                    if p.get("required")
                ]
            else:
                out["parameters"] = docs.get("parameters", [])
        
        # 处理 Request Body
        if docs.get("requestBody"):
            out["requestBody"] = docs.get("requestBody")
            
        # 处理 Responses (优先取 200, 201, 或 default)
        responses = docs.get("responses", {})
        for code in ["200", 201, "201", 200, "default"]:
            if code in responses:
                out["responses"] = responses[code]
                break
        
        return out

    endpoints = [
        (name, description, reduce_endpoint_docs(docs))
        for name, description, docs in endpoints
    ]

    # 5. 返回对象
    # 因为你的 JSON 没有 info 和 servers 字段，我们给默认空值
    return ReducedOpenAPISpec(
        servers=[], # 你的 URL 包含了 server 信息，所以这里留空即可
        description="Custom Processed API Dataset",
        endpoints=endpoints,
    )