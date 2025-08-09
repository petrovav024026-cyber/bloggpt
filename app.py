# app.py
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

# -----------------------------
# Конфиг
# -----------------------------
APP_NAME = "bloggpt"
VERSION = "1.0.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NEWS_LOCALE = os.getenv("NEWS_LOCALE", "en")  # например: ru, en
NEWS_TIMEFRAME_HOURS = int(os.getenv("NEWS_TIMEFRAME_HOURS", "24"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

# -----------------------------
# Логирование
# -----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(APP_NAME)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title=APP_NAME, version=VERSION)

# -----------------------------
# Схемы
# -----------------------------
class GenerateRequest(BaseModel):
    topic: str = Field(..., description="Тема/запрос для генерации контента")
    max_news: int = Field(3, ge=0, le=10, description="Сколько новостей подтянуть как контекст")
    system_prompt: Optional[str] = Field(
        default="Ты — лаконичный, остроумный автор, смотри в будущее и добавляй уместный юмор.",
        description="Системная роль для модели"
    )
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(500, ge=1, le=2048)
    language: Optional[str] = Field(default="ru", description="Язык ответа")


class NewsItem(BaseModel):
    title: str
    description: Optional[str] = None
    url: Optional[HttpUrl] = None
    published: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None


class GenerateResponse(BaseModel):
    topic: str
    used_model: str
    news_used: List[NewsItem] = []
    content: str
    tokens_info: Optional[Dict[str, Any]] = None


# -----------------------------
# Утилиты
# -----------------------------
def _currents_search(query: str, max_news: int) -> List[NewsItem]:
    """
    Получить свежие новости из Currents API.
    Возвращаем пустой список, если ключ не задан или произошла ошибка.
    """
    if not CURRENTS_API_KEY or max_news <= 0:
        return []

    endpoint = "https://api.currentsapi.services/v1/search"
    # Отрезаем по времени (last N hours)
    start = (datetime.utcnow() - timedelta(hours=NEWS_TIMEFRAME_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "apiKey": CURRENTS_API_KEY,
        "keywords": query,
        "language": NEWS_LOCALE,
        "start_date": start,
        "page_size": max_news
    }

    try:
        r = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
        news = []
        for item in data.get("news", [])[:max_news]:
            news.append(
                NewsItem(
                    title=item.get("title", ""),
                    description=item.get("description"),
                    url=item.get("url"),
                    published=item.get("published"),
                    author=item.get("author"),
                    category=(item.get("category")[0] if isinstance(item.get("category"), list) and item["category"] else None),
                )
            )
        return news
    except Exception as e:
        log.warning("Currents API error: %s", e)
        return []


def _openai_chat(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
    """
    Вызов OpenAI Chat Completions (новый SDK >=1.0).
    Возвращаем словарь с текстом и возможно usage, либо поднимаем HTTPException.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY не задан в переменных окружения.")

    # ленивый импорт, чтобы модуль не требовал ключ на старте
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        return {
            "text": text,
            "usage": getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else None
        }
    except Exception as e:
        log.exception("OpenAI error")
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")


def _build_context_block(news: List[NewsItem]) -> str:
    if not news:
        return "Новостной контекст не найден или пропущен."
    lines = ["Свежие новости по теме (кратко):"]
    for i, n in enumerate(news, 1):
        piece = f"{i}. {n.title}"
        if n.description:
            piece += f" — {n.description}"
        if n.url:
            piece += f" ({n.url})"
        lines.append(piece)
    return "\n".join(lines)


# -----------------------------
# Эндпоинты
# -----------------------------
@app.get("/", tags=["meta"])
def root():
    return {"app": APP_NAME, "version": VERSION, "status": "ok"}


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/ping", tags=["meta"])
def ping():
    return JSONResponse({"pong": True})


@app.get("/status", tags=["meta"])
def status():
    return {
        "app": APP_NAME,
        "version": VERSION,
        "openai_model": OPENAI_MODEL,
        "news_locale": NEWS_LOCALE,
        "news_timeframe_hours": NEWS_TIMEFRAME_HOURS
    }


@app.post("/v1/generate", response_model=GenerateResponse, tags=["generate"])
def generate(req: GenerateRequest):
    """
    Генерация текста по теме с подложкой из свежих новостей Currents API (если ключ задан).
    """
    # 1) Подтягиваем новости
    news = _currents_search(req.topic, req.max_news)

    # 2) Формируем сообщения для модели
    context_block = _build_context_block(news)
    user_prompt = (
        f"Тема: {req.topic}\n\n"
        f"{context_block}\n\n"
        f"Задача: создай лаконичный, живой текст (1–3 абзаца) на языке '{req.language}', "
        f"с аккуратной ссылкой на тренды без воды. Уместный юмор — по вкусу. "
        f"Смотри в будущее, избегай дат, которые быстро устареют."
    )

    messages = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 3) Вызов OpenAI
    result = _openai_chat(messages, temperature=req.temperature, max_tokens=req.max_tokens)

    return GenerateResponse(
        topic=req.topic,
        used_model=OPENAI_MODEL,
        news_used=news,
        content=result["text"],
        tokens_info=result.get("usage"),
    )


# -----------------------------
# Локальный запуск
# -----------------------------
if __name__ == "__main__":
    # локально: python app.py
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
