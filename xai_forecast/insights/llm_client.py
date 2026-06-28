"""Thin JSON-only DeepSeek client. Fails loudly if key is absent."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None        # type: ignore[assignment,misc]
    _AsyncOpenAI = None   # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False

MAX_TOKENS_FLASH  = 3000
MAX_TOKENS_PRO    = 4096


class DeepSeekClient:
    """
    Wraps two DeepSeek models:
      - flash: deepseek-v4-flash  (planner, hypothesis, synthesis — high volume)
      - pro:   deepseek-v4-pro    (critic — quality gate)

    Provides both sync (call_*) and async (acall_*) interfaces.
    Requires DEEPSEEK_API_KEY. Raises RuntimeError at construction if unavailable.
    """

    FLASH_MODEL = 'deepseek-v4-flash'
    PRO_MODEL   = 'deepseek-v4-pro'
    BASE_URL    = 'https://api.deepseek.com'

    def __init__(self) -> None:
        key = os.environ.get('DEEPSEEK_API_KEY', '')
        if not key:
            raise RuntimeError(
                'DEEPSEEK_API_KEY is not set. '
                'generate_insights.py requires the LLM — set the key in .env and retry.'
            )
        if not _OPENAI_AVAILABLE or _OpenAI is None:
            raise RuntimeError('openai package is not installed. Run: uv add openai')

        base_url    = os.environ.get('DEEPSEEK_BASE_URL', self.BASE_URL)
        flash_model = os.environ.get('DEEPSEEK_MODEL', self.FLASH_MODEL)
        pro_model   = os.environ.get('DEEPSEEK_CRITIC_MODEL', self.PRO_MODEL)

        self._client       = _OpenAI(api_key=key, base_url=base_url)
        self._async_client = _AsyncOpenAI(api_key=key, base_url=base_url)
        self.flash_model   = flash_model
        self.pro_model     = pro_model

    # ── Sync interface (used by tests and fallback paths) ─────────────────────

    def call_flash(self, system: str, user: dict | str, temperature: float = 0.2) -> dict:
        return self._call(self.flash_model, system, user, MAX_TOKENS_FLASH, temperature)

    def call_pro(self, system: str, user: dict | str, temperature: float = 0.1) -> dict:
        return self._call(self.pro_model, system, user, MAX_TOKENS_PRO, temperature)

    def _call(
        self, model: str, system: str, user: dict | str,
        max_tokens: int, temperature: float,
    ) -> dict:
        user_content = json.dumps(user, ensure_ascii=False) if isinstance(user, dict) else user
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
            timeout=60,
        )
        choice = resp.choices[0]
        if choice.finish_reason == 'length':
            logger.warning('Response truncated (finish_reason=length, model=%s)', model)
        raw = choice.message.content or '{}'
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.error('JSON parse failed (model=%s finish_reason=%s): %s — raw: %s…',
                         model, choice.finish_reason, exc, raw[:300])
            return {}

    # ── Async interface (used by graph.py for concurrent fan-out) ─────────────

    async def acall_flash(self, system: str, user: dict | str, temperature: float = 0.2) -> dict:
        return await self._acall(self.flash_model, system, user, MAX_TOKENS_FLASH, temperature)

    async def acall_pro(self, system: str, user: dict | str, temperature: float = 0.1) -> dict:
        return await self._acall(self.pro_model, system, user, MAX_TOKENS_PRO, temperature)

    async def _acall(
        self, model: str, system: str, user: dict | str,
        max_tokens: int, temperature: float,
    ) -> dict:
        user_content = json.dumps(user, ensure_ascii=False) if isinstance(user, dict) else user
        resp = await self._async_client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
            timeout=60,
        )
        choice = resp.choices[0]
        if choice.finish_reason == 'length':
            logger.warning('Response truncated (finish_reason=length, model=%s)', model)
        raw = choice.message.content or '{}'
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.error('JSON parse failed (model=%s finish_reason=%s): %s — raw: %s…',
                         model, choice.finish_reason, exc, raw[:300])
            return {}
