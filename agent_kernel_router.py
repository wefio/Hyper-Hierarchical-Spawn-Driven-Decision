"""ModelRouter — 异构模型路由 + 厂商适配层。

模型注册（YAML）→ 路由（ID/标签/sticky）→ 厂商适配器（请求转换/响应解析）→ API 调用。

每个 provider 有独立适配器，处理思考模式、缓存指标、响应格式差异。
models.yaml 只声明，适配逻辑全在适配器代码里。
"""

from __future__ import annotations
import hashlib
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import yaml

from config import Config


# ============================================================================
# ModelSpec
# ============================================================================

@dataclass
class ModelSpec:
    id: str                              # "deepseek:v4-pro"
    provider: str                        # anthropic | deepseek | openai | local | ...
    name: str                            # model name for API
    api_key: str = ""
    base_url: str = ""
    context_window: int = 200000
    max_output: int = 4096
    rate_limit: float = 1.0
    account_pool: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    tier: str = "sonnet"         # opus | sonnet | haiku | fallback
    fallback: str = ""
    max_concurrent: int = 3
    extra_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(
            id=d.get("id", ""),
            provider=d.get("provider", ""),
            name=d.get("name", ""),
            api_key=os.path.expandvars(d.get("api_key", "")),
            base_url=d.get("base_url", ""),
            context_window=int(d.get("context_window", 200000)),
            max_output=int(d.get("max_output", 4096)),
            rate_limit=float(d.get("rate_limit", 1.0)),
            account_pool=[os.path.expandvars(k) for k in d.get("account_pool", [])],
            tags=d.get("tags", []),
            tier=d.get("tier", "sonnet"),
            fallback=d.get("fallback", ""),
            max_concurrent=int(d.get("max_concurrent", 3)),
            extra_params=d.get("extra_params", {}),
        )


# ============================================================================
# ProviderAdapter — 厂商适配层
# ============================================================================

class ProviderAdapter:
    """厂商适配基类：Anthropic 兼容协议默认行为。"""

    provider: str = "anthropic"

    def build_request(self, kwargs: dict, spec: ModelSpec) -> dict:
        """在 API 调用前转换 kwargs。"""
        if spec.extra_params:
            kwargs["extra_body"] = spec.extra_params
        return kwargs

    def parse_usage(self, response, raw_meta: dict) -> dict:
        """从响应中提取用量指标。返回更新后的 meta dict。"""
        if hasattr(response, 'usage'):
            u = response.usage
            raw_meta["input_tokens"] = getattr(u, 'input_tokens', 0) or 0
            raw_meta["output_tokens"] = getattr(u, 'output_tokens', 0) or 0
            raw_meta["cache_read_tokens"] = getattr(u, 'cache_read_input_tokens', 0) or 0
            raw_meta["cache_write_tokens"] = getattr(u, 'cache_creation_input_tokens', 0) or 0
        return raw_meta

    def cache_control(self) -> dict:
        """返回 cache_control 标记。Anthropic 用 ephemeral，其他厂商默认空。"""
        return {"type": "ephemeral"}

    def post_process(self, response) -> dict:
        extra = {}
        if hasattr(response, 'reasoning_content') and response.reasoning_content:
            extra['reasoning_content'] = response.reasoning_content
        return extra


class DeepSeekAdapter(ProviderAdapter):
    """DeepSeek Anthropic 兼容端点适配。"""

    provider = "deepseek"

    def cache_control(self) -> dict:
        return {}  # DeepSeek 服务端自动缓存，不需要 cache_control 标记

    def build_request(self, kwargs: dict, spec: ModelSpec) -> dict:
        # DeepSeek thinking mode
        thinking = spec.extra_params.get("thinking", None)
        if thinking:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["thinking"] = thinking
        # 其余 extra_params 也 merge
        for k, v in spec.extra_params.items():
            if k != "thinking":
                kwargs.setdefault("extra_body", {})[k] = v
        return kwargs

    def parse_usage(self, response, raw_meta: dict) -> dict:
        if hasattr(response, 'usage'):
            u = response.usage
            raw_meta["input_tokens"] = getattr(u, 'input_tokens', 0) or 0
            raw_meta["output_tokens"] = getattr(u, 'output_tokens', 0) or 0
            # DeepSeek: Anthropic SDK 返回 cache_read_input_tokens，
            #           原生 API 返回 prompt_cache_hit_tokens
            raw_meta["cache_read_tokens"] = (
                getattr(u, 'cache_read_input_tokens', 0)
                or getattr(u, 'prompt_cache_hit_tokens', 0) or 0)
            raw_meta["cache_write_tokens"] = (
                getattr(u, 'cache_creation_input_tokens', 0)
                or getattr(u, 'prompt_cache_miss_tokens', 0) or 0)
        return raw_meta


class LocalAdapter(ProviderAdapter):
    """本地模型适配（无认证，Anthropic 兼容端点）。"""
    provider = "local"

    def cache_control(self) -> dict:
        return {}  # 本地模型不支持服务端缓存

    def build_request(self, kwargs: dict, spec: ModelSpec) -> dict:
        for k, v in spec.extra_params.items():
            kwargs.setdefault("extra_body", {})[k] = v
        return kwargs


class KimiAdapter(ProviderAdapter):
    """Kimi K2 思考模型适配（Anthropic 兼容端点）。

    - 思考模式默认开启，max_tokens 最低 16000
    - reasoning_content 跨 tool call 轮次必须保留
    - temperature 固定 1.0（kimi-k2.6+）
    """

    provider = "kimi"

    def cache_control(self) -> dict:
        return {}

    def build_request(self, kwargs: dict, spec: ModelSpec) -> dict:
        # Kimi K2 推荐 max_tokens >= 16000
        if kwargs.get("max_tokens", 0) < 16000:
            kwargs["max_tokens"] = 16000
        # 思考模式默认开启，不传 thinking 参数即启用
        thinking = spec.extra_params.get("thinking", None)
        if thinking:
            kwargs.setdefault("extra_body", {})["thinking"] = thinking
        for k, v in spec.extra_params.items():
            if k != "thinking":
                kwargs.setdefault("extra_body", {})[k] = v
        return kwargs


class HumanAdapter(ProviderAdapter):
    """人工模型适配 — stdin 阻塞式交互。"""

    provider = "human"

    def build_request(self, kwargs: dict, spec: ModelSpec) -> dict:
        return kwargs


# ============================================================================
# Adapter registry
# ============================================================================

_ADAPTERS: dict[str, ProviderAdapter] = {
    "anthropic": ProviderAdapter(),
    "deepseek": DeepSeekAdapter(),
    "local": LocalAdapter(),
    "kimi": KimiAdapter(),
    "human": HumanAdapter(),
}


def get_adapter(provider: str) -> ProviderAdapter:
    if provider not in _ADAPTERS:
        _ADAPTERS[provider] = ProviderAdapter()  # 默认 Anthropic 兼容
    return _ADAPTERS[provider]


# ============================================================================
# ModelRegistry
# ============================================================================

class ModelRegistry:
    """模型注册表：从 YAML 加载，按 id / tags 索引。"""

    def __init__(self, registry_path: str = None):
        self._models: dict[str, ModelSpec] = {}
        self._by_tag: dict[str, list[str]] = {}
        self._clients: dict[str, anthropic.Anthropic] = {}
        self._pool_cursors: dict[str, int] = {}
        self._lock = threading.Lock()

        default_path = registry_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models.yaml")
        if os.path.exists(default_path):
            self.load(default_path)
        else:
            self._add_default()

    def _add_default(self):
        spec = ModelSpec(
            id="default",
            provider="anthropic",
            name=Config.MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            base_url=Config.ANTHROPIC_BASE_URL,
            context_window=Config.CONTEXT_BUDGET,
            max_output=4096,
            tags=["default"],
        )
        self._models["default"] = spec

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f.read())
        for entry in data.get("models", []):
            spec = ModelSpec.from_dict(entry)
            self._models[spec.id] = spec
            for tag in spec.tags:
                self._by_tag.setdefault(tag, []).append(spec.id)

    def get(self, model_id: str) -> Optional[ModelSpec]:
        return self._models.get(model_id)

    def by_tags(self, tags: list[str], min_context: int = 0) -> list[ModelSpec]:
        scored = []
        for spec in self._models.values():
            if spec.context_window < min_context:
                continue
            score = len(set(tags) & set(spec.tags))
            if score > 0:
                scored.append((score, spec.context_window, spec))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [s for _, _, s in scored]

    def by_tier(self, tier: str, min_context: int = 0) -> list[ModelSpec]:
        """返回指定 tier 的所有模型（按 context_window 降序）。"""
        result = [s for s in self._models.values()
                  if s.tier == tier and s.context_window >= min_context]
        result.sort(key=lambda s: s.context_window, reverse=True)
        return result

    @property
    def tiers(self) -> dict[str, list[str]]:
        """返回 {tier: [model_id, ...]} 映射。"""
        result: dict[str, list[str]] = {}
        for s in self._models.values():
            result.setdefault(s.tier, []).append(s.id)
        return result

    def get_client(self, spec: ModelSpec) -> anthropic.Anthropic:
        if spec.provider == "human":
            return None
        key = self._next_key(spec)
        cache_key = f"{spec.id}:{key[:12]}"
        with self._lock:
            if cache_key not in self._clients:
                saved = os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
                try:
                    self._clients[cache_key] = anthropic.Anthropic(
                        api_key=key,
                        base_url=spec.base_url or Config.ANTHROPIC_BASE_URL,
                    )
                finally:
                    if saved is not None:
                        os.environ["ANTHROPIC_AUTH_TOKEN"] = saved
            return self._clients[cache_key]

    def _next_key(self, spec: ModelSpec) -> str:
        if spec.account_pool:
            with self._lock:
                cur = self._pool_cursors.get(spec.id, 0)
                key = spec.account_pool[cur % len(spec.account_pool)]
                self._pool_cursors[spec.id] = cur + 1
                return key
        return spec.api_key or Config.ANTHROPIC_API_KEY


# ============================================================================
# ModelSemaphore
# ============================================================================

class ModelSemaphore:
    """限制同模型的并发 API 调用数，保证 KV 缓存命中。"""

    def __init__(self):
        self._sem: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()

    def acquire(self, model_id: str, max_concurrent: int = 3):
        with self._lock:
            if model_id not in self._sem:
                self._sem[model_id] = threading.BoundedSemaphore(max_concurrent)
        self._sem[model_id].acquire()

    def release(self, model_id: str):
        if model_id in self._sem:
            self._sem[model_id].release()


# ============================================================================
# ModelRouter
# ============================================================================

class ModelRouter:
    """模型路由 + 厂商适配。"""

    def __init__(self, registry: ModelRegistry = None):
        self.registry = registry or ModelRegistry()
        self._sem = ModelSemaphore()
        self._sticky_map: dict[str, str] = {}

    _FALLBACK_CHAIN = ["opus", "sonnet", "haiku", "fallback"]

    def resolve(self, spec: dict = None,
                sticky_key: str = "",
                required_context: int = 0) -> tuple[anthropic.Anthropic, ModelSpec]:
        """返回 (client, ModelSpec)。

        路由: 精确ID → tier + sticky → 同tier降级 → 跨tier降级 → fallback
        """
        spec = spec or {}
        model = None
        min_ctx = max(required_context, spec.get("min_context", 0))

        if "id" in spec:
            model = self.registry.get(spec["id"])
            if model and model.context_window < min_ctx:
                model = None  # ctx 不够，继续降级

        if not model:
            tier = spec.get("tier", "sonnet")
            # 构建降级链：从指定 tier 开始
            chain = self._FALLBACK_CHAIN[self._FALLBACK_CHAIN.index(tier):] if tier in self._FALLBACK_CHAIN else self._FALLBACK_CHAIN
            for t in chain:
                candidates = self.registry.by_tier(t, min_ctx)
                if candidates:
                    model = self._pick_sticky(candidates, sticky_key, [t])
                    break
            # 最后兜底：任意模型
            if not model:
                all_models = self.registry.by_tier("", min_ctx) or list(self.registry._models.values())
                if all_models:
                    model = all_models[0]

        if not model:
            model = self.registry.get("default")
        if not model:
            raise RuntimeError("No model available")

        if model.provider != "human":
            self._sem.acquire(model.id, model.max_concurrent)
        client = self.registry.get_client(model)
        return client, model

    def release(self, model_id: str):
        self._sem.release(model_id)

    def _pick_sticky(self, candidates: list[ModelSpec],
                     sticky_key: str, tags: list[str]) -> ModelSpec:
        if sticky_key and sticky_key in self._sticky_map:
            model_id = self._sticky_map[sticky_key]
            for c in candidates:
                if c.id == model_id:
                    return c
        chosen = random.choice(candidates)
        if sticky_key:
            self._sticky_map[sticky_key] = chosen.id
        return chosen

    def resolve_for_agent(self, spec: dict = None,
                          sticky_key: str = "",
                          required_context: int = 0) -> dict:
        """返回 Agent 创建所需参数。"""
        client, model = self.resolve(spec, sticky_key, required_context)
        can_spawn = spec.get("can_spawn", True) if spec else True
        return {
            "client": client,
            "model_spec": model,
            "context_budget": model.context_window,
            "can_spawn": can_spawn,
            "model_id": model.id,
            "extra_params": model.extra_params,
        }


# ============================================================================
# Global instance
# ============================================================================

_router: Optional[ModelRouter] = None

def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
