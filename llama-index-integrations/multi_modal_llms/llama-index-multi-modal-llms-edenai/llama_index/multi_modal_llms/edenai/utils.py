from typing import Optional, Dict, Tuple
import json


def parse_edenai_stream_chunk(chunk_str: str) -> Tuple[Optional[str], Optional[Dict]]:
    if not chunk_str.strip():
        return None, None

    try:
        chunk_json = json.loads(chunk_str)
        if "choices" in chunk_json and chunk_json["choices"]:
            choice = chunk_json["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content is not None:
                return content, chunk_json
        return None, chunk_json
    except Exception as e:
        print(f"Error parsing chunk: {e}")
        return None, {"error": str(e)}
