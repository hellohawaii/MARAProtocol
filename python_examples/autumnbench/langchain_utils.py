import copy
import hashlib
import json
import logging
import math
import os
import pickle
import sqlite3
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from langchain_core.caches import BaseCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation, LLMResult
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Logging and Cost Calculation ---

MODEL_PRICING = {
    "google/gemini-2.5-pro": {
        "tiers": [
            {"up_to_tokens": 200000, "input": 1.25, "output": 10.00},
            {"up_to_tokens": float("inf"), "input": 2.50, "output": 15.00},
        ]
    },
    "google/gemini-3-pro-preview": {
        "tiers": [
            {"up_to_tokens": 200000, "input": 2.00, "output": 12.00},
            {"up_to_tokens": float("inf"), "input": 4.00, "output": 18.00},
        ]
    },
    "anthropic/claude-opus-4.6": {
        "tiers": [
            {"up_to_tokens": float("inf"), "input": 5.00, "output": 25.00},
        ]
    },
    "openai/gpt-5.4": {
        "tiers": [
            {"up_to_tokens": float("inf"), "input": 2.50, "output": 15.00},
        ]
    },
    "default": {
        "tiers": [
            {"up_to_tokens": float("inf"), "input": 1.00, "output": 3.00}
        ]
    },
}

def get_pricing_for_request(model_name: str, prompt_tokens: int) -> dict:
    """Selects the correct pricing tier based on the number of prompt tokens."""
    model_info = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    tiers = model_info.get("tiers", [])
    for tier in tiers:
        if prompt_tokens < tier["up_to_tokens"]:
            return {"input": tier["input"], "output": tier["output"]}
    if tiers:
        last_tier = tiers[-1]
        return {"input": last_tier["input"], "output": last_tier["output"]}
    return {"input": 1.00, "output": 3.00}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = get_pricing_for_request(model_name, prompt_tokens)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

def setup_llm_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "llm_usage.log")

    logger = logging.getLogger("llm_usage")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

llm_usage_logger = setup_llm_logger()

def log_usage(model: str, prompt_tokens: int, completion_tokens: int):
    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    log_message = (
        f"Model: {model}, "
        f"PromptTokens: {prompt_tokens}, "
        f"CompletionTokens: {completion_tokens}, "
        f"TotalTokens: {total_tokens}, "
        f"EstimatedCostUSD: {cost:.8f}"
    )
    llm_usage_logger.info(log_message)

class TokenUsageCallback(BaseCallbackHandler):
    """Callback to log token usage and cost after each LLM call."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Logs token usage by extracting metadata from the response.
        """
        llm_output = response.llm_output
        if not llm_output:
            return

        token_usage = llm_output.get("token_usage")
        model_name = llm_output.get("model_name")

        if token_usage and model_name:
            log_usage(
                model=model_name,
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
            )

# --- Caching ---

RETURN_VAL_TYPE = Sequence[Generation]

# Use a local db file in the same directory for simplicity in this context, 
# or mirror the structure if needed. 
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), 'llm_cache.db')

class SQLCache(BaseCache):
    """Cache that uses an SQLite database for storage."""

    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS langchain_cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL
                )
                """
            )
            conn.commit()

    def _get_key(self, prompt: str, llm_string: str) -> str:
        processed_prompt = prompt
        try:
            messages = json.loads(prompt)
            if isinstance(messages, list):
                messages_copy = copy.deepcopy(messages)
                for message in messages_copy:
                    if not (isinstance(message, dict) and "kwargs" in message and isinstance(message["kwargs"], dict)):
                        continue

                    kwargs = message["kwargs"]
                    msg_type = kwargs.get("type")

                    kwargs.pop("id", None)

                    if msg_type == "ai":
                        if "response_metadata" in kwargs and isinstance(kwargs["response_metadata"], dict):
                            kwargs["response_metadata"].pop("id", None)
                        kwargs.pop("usage_metadata", None)
                        if "tool_calls" in kwargs and isinstance(kwargs["tool_calls"], list):
                            for tool_call in kwargs["tool_calls"]:
                                if isinstance(tool_call, dict):
                                    tool_call.pop("id", None)
                    elif msg_type == "tool":
                        kwargs.pop("tool_call_id", None)

                processed_prompt = json.dumps(messages_copy, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            pass

        key_source = f"{processed_prompt}---{llm_string}"
        return hashlib.sha256(key_source.encode("utf-8")).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        key = self._get_key(prompt, llm_string)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM langchain_cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                # print(f"--- Cache Hit for key: {key[:8]}... ---")
                try:
                    return pickle.loads(result[0])
                except pickle.UnpicklingError:
                    return None
        # print(f"--- Cache Miss for key: {key[:8]}... ---")
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        key = self._get_key(prompt, llm_string)
        # print(f"--- Updating cache for key: {key[:8]}... ---")
        value = pickle.dumps(return_val)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO langchain_cache (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def clear(self, **kwargs: Any) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM langchain_cache")
            conn.commit()


# --- LLM Factory ---

def get_llm(model: str = "google/gemini-2.5-pro", **kwargs) -> ChatOpenAI:
    """Returns a configured ChatOpenAI instance using OpenRouter."""
    
    # Initialize cache
    set_llm_cache(SQLCache())
    
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=5,
        timeout=3600,
        callbacks=[TokenUsageCallback()],
        **kwargs
    )
