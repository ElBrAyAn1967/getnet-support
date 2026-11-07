# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Knowledge agent tools (T-5..T-8). Getnet product answers come only from the indexed Getnet corpus."""

from datetime import date, timedelta
from typing import Any

import httpx
from ddgs import DDGS
from langchain_core.tools import tool

from app.http import client
from app.rag.index import search

# Web results from other acquirers never answer a merchant of Getnet (G-4).
COMPETITORS = ("stone", "cielo", "pagseguro", "pagbank", "mercadopago", "mercado pago", "sumup", "infinitepay",
               "userede", "rede.com", "ton.com")
_PTAX = (
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaPeriodo("
    "moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
)


@tool
def search_getnet(query: str) -> dict[str, Any]:
    """Search Getnet's official Brazilian site and machine manuals. Use for ANY question about Getnet products,
    machines, fees, Pix, anticipation, crediário, payment links. Write the query in Brazilian Portuguese."""
    hits = search(query)
    return {"results": [
        {"title": h.title, "url": h.url, "fetched_at": h.fetched_at.isoformat(), "text": h.body} for h in hits
    ]}


@tool
def get_weather(city: str, day_offset: int = 0) -> dict[str, Any]:
    """Weather forecast for a city. day_offset: 0 today, 1 tomorrow (max 6)."""
    geo = client().get(
        "https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "pt"}
    ).json()
    if not geo.get("results"):
        return {"error": f"city not found: {city}"}
    place = geo["results"][0]
    daily = client().get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 7,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    }).json()["daily"]
    i = max(0, min(day_offset, len(daily["time"]) - 1))
    return {
        "city": f"{place['name']}, {place.get('admin1', '')}", "date": daily["time"][i],
        "max_c": daily["temperature_2m_max"][i], "min_c": daily["temperature_2m_min"][i],
        "rain_probability_pct": daily["precipitation_probability_max"][i],
        "source": "Open-Meteo", "url": "https://open-meteo.com/",
    }


@tool
def get_exchange_rate(currency: str = "EUR") -> dict[str, Any]:
    """Official exchange rate (PTAX, Banco Central do Brasil) of a currency in Brazilian reais (BRL)."""
    end = date.today()
    rows = client().get(_PTAX, params={
        "@moeda": f"'{currency.upper()}'", "@dataInicial": f"'{(end - timedelta(days=7)):%m-%d-%Y}'",
        "@dataFinalCotacao": f"'{end:%m-%d-%Y}'", "$format": "json",
        "$select": "cotacaoCompra,cotacaoVenda,dataHoraCotacao,tipoBoletim",
    }).json().get("value", [])
    if not rows:
        return {"error": f"no PTAX quote for {currency}"}
    last = rows[-1]
    return {
        "currency": currency.upper(), "brl_buy": last["cotacaoCompra"], "brl_sell": last["cotacaoVenda"],
        "quoted_at": last["dataHoraCotacao"][:16], "bulletin": last["tipoBoletim"],
        "source": "Banco Central do Brasil (PTAX)", "url": "https://www.bcb.gov.br/estabilidadefinanceira/historicocotacoes",
    }


@tool
def web_search(query: str) -> dict[str, Any]:
    """General web search for topics outside Getnet (news, general knowledge). Never for Getnet products."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as exc:  # noqa: BLE001 — any search-provider failure degrades to a notice
        return {"error": f"web search unavailable ({type(exc).__name__})"}
    kept = [r for r in results if not any(c in (r.get("href", "") + r.get("title", "")).lower() for c in COMPETITORS)]
    return {"results": [{"title": r["title"], "url": r["href"], "text": r["body"]} for r in kept[:3]]}


KNOWLEDGE_TOOLS = [search_getnet, get_weather, get_exchange_rate, web_search]
EXTERNAL_TOOLS = {"get_weather", "get_exchange_rate", "web_search"}
TOOL_ERRORS = (httpx.HTTPError, KeyError, ValueError)
