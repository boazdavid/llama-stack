def expand_refs(schema: dict, root: dict = None) -> dict:
    if root is None:
        root = schema

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref = schema["$ref"]
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported ref: {ref}")
            key = ref.split("/")[-1]
            return expand_refs(root["$defs"][key], root)
        return {k: expand_refs(v, root) for k, v in schema.items() if k != "$defs"}
    elif isinstance(schema, list):
        return [expand_refs(i, root) for i in schema]
    return schema

