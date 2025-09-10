
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime

def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

@dataclass
class Message:
    role: str
    content: str
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self):
        d = {"role": self.role, "content": self.content, "ts": now_ts()}
        if self.meta:
            d["meta"] = self.meta
        return d

def pretty(obj) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def tokenize(text: str) -> List[str]:
    # Simple tokenizer: lowercase, keep alphanumerics
    return [t for t in re_sub(r'[^a-z0-9\s]', ' ', text.lower()).split() if t]

import re
def re_sub(pattern, repl, text):
    return re.sub(pattern, repl, text)
