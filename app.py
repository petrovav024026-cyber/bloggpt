"""
FastAPI сервис для генерации постов с использованием:
- Currents API (актуальные новости по теме как контекст)
- OpenAI API (генерация структурированного контента)
- Пресеты целей (тон, аудитория, UTM, CTA-ротация)
- Эндпоинты здоровья/готовности/версии/пресетов
- Обработка ошибок и CORS
- Запуск через uvicorn

Автор: вы :)
Дата: 2025-08-09
"""

from __future__ import annotations

import os
import re
import math
import json
import random
import logging
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass

import requests
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

# --------------------------- ЛОГИРОВАНИЕ -------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("content-suite")

# ------------------------------ OpenAI ---------------------------------------
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Не установлен пакет openai. Установите: pip install openai") from e

# ----------------------------- tiktoken --------------------------------------
try:
    import tiktoken
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

# =============================================================================
# Конфигурация и пресеты
# =============================================================================

# Ключи из окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY не задан! /ready покажет not_ready.")
if not CURRENTS_API_KEY:
    logger.warning("CURRENTS_API_KEY не задан! Новости Currents не будут использованы.")

# Модели
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# Примерные тарифы (обновите под актуальные)
PRICING_PER_1K = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

# Пресеты
TONE_PRESETS: Dict[str, str] = {
    "friendly_expert": "дружелюбный, экспертный, без канцелярита",
    "witty_brisk": "остроумный, быстрый, без воды, с лёгкими шутками",
    "serious_trust": "деловой, уверенный, даёт ясные выводы",
    "ironic_smart": "ироничный, умный, уважительный, без токсичности",
}

AUDIENCE_PRESETS: Dict[str, str] = {
    "tg_productivity": "подписчики Telegram-канала о продуктивности и саморазвитии",
    "smm_founders": "предприниматели и SMM-специалисты малого бизнеса",
    "devs_ai": "разработчики и энтузиасты ИИ-инструментов",
    "realty_leads": "потенциальные покупатели апартаментов и инвесторы в недвижимость",
}

THEME_PRESETS: Dict[str, Dict[str, Any]] = {
    "leads": {
        "keywords": ["акция", "консультация", "заявка", "скидка"],
        "base_url": "https://example.com/offer",
        "utm": {"source": "telegram", "medium": "social", "campaign": "leads_tg", "content": "{slug}"},
    },
    "traffic": {
        "keywords": ["гайд", "чек-лист", "подборка", "блог"],
        "base_url": "https://example.com/blog",
        "utm": {"source": "telegram", "medium": "social", "campaign": "blog_push", "content": "{slug}"},
    },
    "ironic": {
        "keywords": ["ирония", "холодные факты", "юмор"],
        "base_url": "https://example.com/post",
        "utm": {"source": "telegram", "medium": "social", "campaign": "brand_tone", "content": "{slug}"},
    },
}

# Цели с UTM и ротацией CTA
GOAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "leads": {
        "tone": "serious_trust",
        "audience": "realty_leads",
        "theme": "leads",
        "utm": {
            "base_url": "https://example.com/offer",
            "source": "telegram",
            "medium": "social",
            "campaign": "leadgen_q3",
            "content": "{slug}",
        },
        "cta_variants": [
            "Оставьте заявку — подскажем лучший вариант 🚀",
            "Напишите нам — ответим в течение часа ✉️",
            "Бесплатная консультация — бронируйте слот 📅",
        ],
    },
    "traffic": {
        "tone": "witty_brisk",
        "audience": "tg_productivity",
        "theme": "traffic",
        "utm": {
            "base_url": "https://example.com/blog",
            "source": "telegram",
            "medium": "social",
            "campaign": "blog_push",
            "content": "{slug}",
        },
        "cta_variants": [
            "Читайте полный гайд по ссылке 📚",
            "У нас подборка — загляните 👀",
            "Подробности внутри, жмите ↗",
        ],
    },
    "brand": {
        "tone": "friendly_expert",
        "audience": "smm_founders",
        "theme": "ironic",
        "utm": {
            "base_url": "https://example.com/brand",
            "source": "telegram",
            "medium": "social",
            "campaign": "brand_awareness",
            "content": "{slug}",
        },
        "cta_variants": [
            "Подпишитесь, чтобы не пропустить новое ✨",
            "Сохраняйте пост — пригодится 📌",
            "Расскажите коллегам — пусть тоже знают 🤝",
        ],
    },
    "offer": {
        "tone": "serious_trust",
        "audience": "realty_leads",
        "theme": "leads",
        "utm": {
            "base_url": "https://example.com/offer",
            "source": "telegram",
            "medium": "social",
            "campaign": "special_offer",
            "content": "{slug}",
        },
        "cta_variants": [
            "Забронируйте по акции — мест немного 🎯",
            "Успейте сегодня — предложение ограничено ⏳",
            "Уточните детали — сделаем персональный расчёт 📈",
        ],
    },
    "news": {
        "tone": "friendly_expert",
        "audience": "devs_ai",
        "theme": "traffic",
        "utm": {
            "base_url": "https://example.com/news",
            "source": "telegram",
            "medium": "social",
            "campaign": "news_update",
            "content": "{slug}",
        },
        "cta_variants": [
            "Подробности по ссылке ↗",
            "Читать обновление в блоге 📰",
            "Собрали факты и примеры — заходите 👀",
        ],
    },
    "expert_tip": {
        "tone": "friendly_expert",
        "audience": "tg_productivity",
        "theme": "traffic",
        "utm": {
            "base_url": "https://example.com/tips",
            "source": "telegram",
            "medium": "social",
            "campaign": "expert_tips",
            "content": "{slug}",
        },
        "cta_variants": [
            "Сохраните чек-лист — пригодится 📌",
            "Отправьте коллеге — пусть тоже ускорится ⚡",
            "Ещё инструменты — по ссылке ↗",
        ],
    },
}

# =============================================================================
# Вспомогательные утилиты
# =============================================================================

def estimate_tokens(text: str, model: str = DEFAULT_TEXT_MODEL) -> int:
    """Точный подсчёт через tiktoken (если есть), иначе эвристика ~4 символа/токен."""
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    return max(1, math.ceil(len(text or "") / 4))


def estimate_cost(input_tokens: int, output_tokens: int, model: str = DEFAULT_TEXT_MODEL) -> float:
    price = PRICING_PER_1K.get(model) or {"input": 0.0, "output": 0.0}
    return (input_tokens / 1000) * price["input"] + (output_tokens / 1000) * price["output"]


def slugify(text: str, max_len: int = 80) -> str:
    """Упрощённый слаг: латиница/цифры/дефисы, нижний регистр, обрезка."""
    import unicodedata
    text = unicodedata.normalize("NFKD", text or "").lower()
    text = re.sub(r"[ё]", "e", text)
    text = re.sub(r"[^a-z0-9\-\_\s]+", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:max_len] or "post"


def build_utm(url: str, source="telegram", medium="social", campaign="content", content: Optional[str] = None) -> str:
    """Собирает UTM-ссылку поверх любого базового URL."""
    from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
    parts = list(urlsplit(url))
    q = dict(parse_qsl(parts[3]))
    q.update({"utm_source": source, "utm_medium": medium, "utm_campaign": campaign})
    if content:
        q["utm_content"] = content
    parts[3] = urlencode(q, doseq=True)
    return urlunsplit(parts)


def resolve_preset(value: Optional[str], presets: Dict[str, str], default: Optional[str]) -> str:
    """Если передан ключ пресета — подставляем текст, иначе возвращаем исходное/дефолт."""
    if not value and default:
        return default
    if value in presets:
        return presets[value]
    return value or (default or "")

# =============================================================================
# Pydantic модели запросов/ответов
# =============================================================================

class GenerateRequest(BaseModel):
    topic: str = Field(..., description="Тема поста")
    goal: Optional[Literal["leads", "traffic", "brand", "offer", "news", "expert_tip"]] = Field(
        default=None, description="Цель поста: подставляет тон/ЦА/тему/UTM/CTA-ротацию"
    )
    tone: Optional[str] = Field(default=None, description="Тон (ключ пресета или текст)")
    audience: Optional[str] = Field(default=None, description="Целевая аудитория (ключ пресета или текст)")
    theme: Optional[Literal["leads", "traffic", "ironic"]] = Field(
        default=None, description="Тематика (добавляет ключевые слова и запасные UTM)"
    )
    length: Literal["short", "medium", "long"] = Field(default="medium", description="Длина поста")
    language: str = Field(default="ru", description="Язык генерации (ru/en/...)")
    keywords: Optional[List[str]] = Field(default=None, description="Доп. ключевые слова")
    news_limit: int = Field(default=3, ge=0, le=10, description="Сколько новостей Currents подтянуть в контекст")
    news_language: Optional[str] = Field(default=None, description="Язык новостей Currents (ru/en/...)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Температура генерации")
    make_image: bool = Field(default=False, description="Сгенерировать ли обложку через OpenAI Images")


class NewsItem(BaseModel):
    title: str
    description: Optional[str] = None
    url: Optional[HttpUrl] = None
    author: Optional[str] = None
    published: Optional[str] = None


class GenerateResponse(BaseModel):
    title: str
    meta_description: str
    post: str
    hashtags: List[str]
    cta: str
    image_prompt: str
    image_url: Optional[HttpUrl] = None
    slug: str
    utm_link: Optional[HttpUrl] = None
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    news_used: List[NewsItem] = Field(default_factory=list)

# =============================================================================
# Currents API — получение новостей
# =============================================================================

def fetch_news_from_currents(
    query: str,
    api_key: Optional[str],
    language: Optional[str] = None,
    limit: int = 3,
    timeout_sec: int = 20,
) -> List[NewsItem]:
    """
    Запрашивает новости из Currents API.
    Если ключа нет или произошла ошибка — возвращает пустой список,
    чтобы сервис продолжал работать.
    """
    if not api_key or limit <= 0:
        return []

    url = "https://api.currentsapi.services/v1/search"
    params = {
        "keywords": query,
        "apiKey": api_key,
        "limit": max(1, min(limit, 50)),
    }
    if language:
        params["language"] = language

    try:
        resp = requests.get(url, params=params, timeout=timeout_sec)
        if resp.status_code != 200:
            logger.warning("Currents API non-200: %s %s", resp.status_code, resp.text[:300])
            return []
        data = resp.json()
    except Exception as e:
        logger.warning("Currents API error: %s", e)
        return []

    items: List[NewsItem] = []
    news = (data or {}).get("news") or (data or {}).get("articles") or []
    for n in news[:limit]:
        items.append(
            NewsItem(
                title=str(n.get("title", "")).strip(),
                description=(n.get("description") or n.get("summary") or "")[:600],
                url=n.get("url"),
                author=n.get("author"),
                published=n.get("published") or n.get("publishedAt"),
            )
        )
    return items

# =============================================================================
# Генерация через OpenAI (с контекстом новостей)
# =============================================================================

@dataclass
class PostSpec:
    title: str
    meta_description: str
    post: str
    hashtags: List[str]
    cta: str
    image_prompt: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_estimate: float = 0.0


def make_news_context(news: List[NewsItem]) -> str:
    """Собирает краткий контекст из новостей для подсказки модели (не слишком длинный)."""
    if not news:
        return "Нет актуальных новостей, контент генерируем без новостного контекста."
    lines = ["Актуальные новости по теме (кратко):"]
    for i, n in enumerate(news, 1):
        line = f"{i}) {n.title}"
        if n.description:
            line += f" — {n.description}"
        if n.url:
            line += f" ({n.url})"
        lines.append(line)
    return "\n".join(lines[:12])


def call_openai_structured(
    client: OpenAI,
    model: str,
    topic: str,
    tone: str,
    audience: str,
    length: str,
    language: str,
    keywords: List[str],
    cta_hint: str,
    news_context: str,
    temperature: float = 0.7,
) -> PostSpec:
    """
    Вызывает OpenAI Chat Completions и просит вернуть JSON с нужными полями.
    Включаем в prompt контекст новостей (news_context).
    """
    system_prompt = f"Ты — опытный SMM-редактор. Пиши кратко, чётко и без воды, на языке: {language}."

    user_prompt = f"""
Сгенерируй структурированный пост для Telegram.

Тема: {topic}
Тон: {tone}
Аудитория: {audience}
Длина: {length}
Ключевые слова: {', '.join(keywords) if keywords else '—'}
CTA (ориентир, можно перефразировать близко к смыслу): {cta_hint or '—'}

{news_context}

Верни JSON c полями:
- title
- meta_description
- post
- hashtags (массив 5–10, без # внутри элементов)
- cta
- image_prompt

Требования:
- Заголовок без кавычек
- Хэштеги — в нижнем регистре, без пробелов внутри
- Без комментариев и преамбулы: только корректный JSON
""".strip()

    input_tokens = estimate_tokens(system_prompt + "\n" + user_prompt, model=model)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=900,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    content = (resp.choices[0].message.content or "").strip()

    # Пытаемся распарсить JSON ответа
    try:
        payload = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(content[start : end + 1])
            except Exception:
                raise HTTPException(status_code=500, detail="Не удалось распарсить JSON-ответ модели.")
        else:
            raise HTTPException(status_code=500, detail="Модель вернула неожиданный формат (не JSON).")

    title = str(payload.get("title", "")).strip()
    meta_description = str(payload.get("meta_description", "")).strip()
    post = str(payload.get("post", "")).strip()
    hashtags = payload.get("hashtags", [])
    if not isinstance(hashtags, list):
        hashtags = []
    hashtags = [str(x).lstrip("#").strip() for x in hashtags][:10]
    cta_out = str(payload.get("cta", "")).strip()
    image_prompt = str(payload.get("image_prompt", "")).strip()

    output_tokens = estimate_tokens(content, model=model)
    cost_est = estimate_cost(input_tokens, output_tokens, model=model)

    return PostSpec(
        title=title,
        meta_description=meta_description,
        post=post,
        hashtags=hashtags,
        cta=cta_out,
        image_prompt=image_prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_estimate=round(cost_est, 6),
    )


def generate_image_url(client: OpenAI, prompt: str, size: str = "1024x1024", model: str = DEFAULT_IMAGE_MODEL) -> Optional[str]:
    """Запрашивает у OpenAI Images ссылку на изображение (без скачивания/хранения)."""
    if not prompt:
        return None
    try:
        img = client.images.generate(model=model, prompt=prompt, size=size, n=1)
        # В новых версиях SDK по умолчанию отдается b64_json; если есть url, вернем его
        if getattr(img.data[0], "url", None):
            return img.data[0].url
        if getattr(img.data[0], "b64_json", None):
            # Если пришла base64-картинка, оставим ссылку пустой, чтобы фронт решил, что делать
            return None
        return None
    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        return None

# =============================================================================
# FastAPI приложение
# =============================================================================

app = FastAPI(
    title="Content Suite API (FastAPI)",
    version="1.0.0",
    description="Генерация постов с контекстом новостей (Currents) и OpenAI.",
)
from fastapi.responses import FileResponse, HTMLResponse
import os

@app.get("/", include_in_schema=False)
def ui():
    """Отдаёт index.html из корня репозитория."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<p>index.html not found. Endpoints: /health, /ready, /generate, /docs</p>")

# CORS (удобно для локальной разработки/фронта)
origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]
allow_credentials_flag = False if origins == ["*"] else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials_flag,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_openai_client() -> OpenAI:
    """Ленивая инициализация клиента OpenAI (чтобы /ready мог подсказать отсутствие ключа)."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY не задан.")
    return OpenAI(api_key=OPENAI_API_KEY)

# --------------------- Метаданные/диагностика -------------------------------

@app.get("/health", tags=["meta"])
def health() -> Dict[str, str]:
    """Простой пинг: сервис жив."""
    return {"status": "ok"}

@app.get("/ready", tags=["meta"])
def ready() -> Dict[str, Any]:
    """Проверяем наличие ключей и базовую готовность к работе."""
    return {
        "openai": bool(OPENAI_API_KEY),
        "currents": bool(CURRENTS_API_KEY),
        "text_model": DEFAULT_TEXT_MODEL,
        "image_model": DEFAULT_IMAGE_MODEL,
        "status": "ready" if OPENAI_API_KEY else "not_ready",
    }

@app.get("/version", tags=["meta"])
def version() -> Dict[str, str]:
    return {"version": app.version}

@app.get("/presets/goals", tags=["presets"])
def get_goal_presets() -> Dict[str, Any]:
    """Возвращает цели и их параметры (для фронтов/интерфейсов)."""
    out = {}
    for k, v in GOAL_PRESETS.items():
        out[k] = {
            "tone": v["tone"],
            "audience": v["audience"],
            "theme": v["theme"],
            "utm": v["utm"],
            "cta_variants": v["cta_variants"],
        }
    return out

# --------------------------- Генерация --------------------------------------

@app.post("/generate", response_model=GenerateResponse, tags=["generate"])
def generate(req: GenerateRequest = Body(...)) -> GenerateResponse:
    """
    Основной пайплайн:
    1) Тянем новости Currents по теме (если есть ключ)
    2) Применяем цель-пресет: тон/ЦА/тема/UTM/CTA-ротация
    3) Собираем подсказку с новостным контекстом и вызываем OpenAI
    4) Формируем slug, UTM-ссылку; опционально генерируем image_url
    """
    client = get_openai_client()

    # Новости Currents
    news_items = fetch_news_from_currents(
        query=req.topic,
        api_key=CURRENTS_API_KEY,
        language=req.news_language,
        limit=req.news_limit,
    )
    news_context = make_news_context(news_items)

    # Пресеты по цели
    tone_in = req.tone
    audience_in = req.audience
    theme_in = req.theme
    cta_hint = ""
    utm_link: Optional[str] = None

    if req.goal and req.goal in GOAL_PRESETS:
        gp = GOAL_PRESETS[req.goal]
        tone_in = tone_in or gp["tone"]
        audience_in = audience_in or gp["audience"]
        theme_in = theme_in or gp["theme"]
        cta_hint = random.choice(gp.get("cta_variants", [])) if gp.get("cta_variants") else ""

    # Ключевые слова из темы
    keywords = list(req.keywords or [])
    if theme_in and theme_in in THEME_PRESETS:
        for kw in THEME_PRESETS[theme_in].get("keywords", []):
            if kw not in keywords:
                keywords.append(kw)

    # Генерация OpenAI
    spec = call_openai_structured(
        client=client,
        model=DEFAULT_TEXT_MODEL,
        topic=req.topic,
        tone=resolve_preset(tone_in, TONE_PRESETS, tone_in),
        audience=resolve_preset(audience_in, AUDIENCE_PRESETS, audience_in),
        length=req.length,
        language=req.language,
        keywords=keywords,
        cta_hint=cta_hint,
        news_context=news_context,
        temperature=req.temperature,
    )

    # Слаг
    s = slugify(spec.title or req.topic)

    # UTM: сначала из цели, если нет — из темы (запасной)
    if req.goal and req.goal in GOAL_PRESETS and GOAL_PRESETS[req.goal].get("utm"):
        u = GOAL_PRESETS[req.goal]["utm"]
        utm_link = build_utm(
            u["base_url"],
            source=u.get("source", "telegram"),
            medium=u.get("medium", "social"),
            campaign=u.get("campaign", "content"),
            content=(u.get("content", "{slug}") or "{slug}").format(slug=s),
        )
    elif theme_in and theme_in in THEME_PRESETS:
        utm_cfg = THEME_PRESETS[theme_in].get("utm", {})
        base = THEME_PRESETS[theme_in]["base_url"]
        utm_link = build_utm(
            base,
            source=utm_cfg.get("source", "telegram"),
            medium=utm_cfg.get("medium", "social"),
            campaign=utm_cfg.get("campaign", "content"),
            content=(utm_cfg.get("content", "{slug}") or "{slug}").format(slug=s),
        )

    # Зафиксируем выбранный CTA-вариант, если был
    if cta_hint:
        spec.cta = cta_hint

    # Картинка (опционально)
    image_url = None
    if req.make_image:
        image_url = generate_image_url(client, prompt=spec.image_prompt)

    return GenerateResponse(
        title=spec.title,
        meta_description=spec.meta_description,
        post=spec.post,
        hashtags=spec.hashtags,
        cta=spec.cta,
        image_prompt=spec.image_prompt,
        image_url=image_url,
        slug=s,
        utm_link=utm_link,
        input_tokens=spec.input_tokens,
        output_tokens=spec.output_tokens,
        cost_estimate=spec.cost_estimate,
        news_used=news_items,
    )

# ----------------------- Глобальные обработчики ошибок -----------------------

@app.exception_handler(HTTPException)
def http_exc_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
def generic_exc_handler(_, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# ----------------------------- Запуск через uvicorn --------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run("app:app", host=host, port=port, reload=reload_flag)
