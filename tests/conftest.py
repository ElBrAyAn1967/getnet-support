# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Unit and graph tests run with no network, no database and no API key (A-19): sockets are blocked, the database
and outside APIs are replaced at their single entry points, and the model is the deterministic fake."""

import re
import socket
from collections.abc import Iterator
from datetime import date
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

import app.tools.knowledge as kn
from app import repo
from app.config import Settings, get_settings
from app.graph.build import build, run_turn
from app.graph.state import SERDE, TurnState
from app.rag.index import Hit

CORPUS = [
    ("https://site.getnet.com.br/maquininha/get-classica/", "Get Clássica",
     "Get Clássica: maquininha com chip e Wi-Fi, recebimento em 2 dias úteis, imprime o comprovante."),
    ("https://site.getnet.com.br/maquininha/get-smart/", "Get Smart",
     "Get Smart: maquininha smart com tela touch, aceita Pix por QR Code e imprime comprovante."),
    ("https://site.getnet.com.br/pix-na-maquininha-getnet/", "Pix na maquininha",
     "Com Pix na maquininha você recebe o pagamento na hora, direto na sua Get Conta, "
     "sem conta bancária de outro banco."),
    ("https://site.getnet.com.br/antecipacao/", "Antecipação de vendas",
     "A antecipação de vendas permite receber antes o valor das vendas parceladas no crédito, "
     "com taxa por antecipação."),
    ("https://site.getnet.com.br/crediario/", "Crediário — Ofereça parcelamento",
     "Ofereça parcelamento: em até 48x para o seu cliente. Cada emissor de cartão estabelece o limite de parcelas."),
    ("https://site.getnet.com.br/pagamentos-pelo-whatsapp/", "Pagamentos pelo WhatsApp",
     "Com o Link de Pagamento você vende pelo WhatsApp: gere o link e envie ao cliente."),
    ("https://site.getnet.com.br/duvidas/operacional/", "Transação não autorizada",
     "Quando a transação é recusada ou não autorizada, peça outro cartão ao cliente ou outra forma de pagamento."),
]
_WORD = re.compile(r"\w{4,}")


def fake_search(query: str, k: int | None = None) -> list[Hit]:
    words = set(_WORD.findall(query.lower()))
    return [Hit(url=u, title=t, fetched_at=date.today(), body=b, similarity=0.6, score=0.03)
            for u, t, b in CORPUS if words & set(_WORD.findall(f"{t} {b}".lower()))][: k or 5]


def _outside_apis(request: httpx.Request) -> httpx.Response:
    if "geocoding" in request.url.host:
        return httpx.Response(200, json={"results": [
            {"name": "Porto Alegre", "admin1": "Rio Grande do Sul", "latitude": -30.03, "longitude": -51.23}]})
    if "open-meteo" in request.url.host:
        return httpx.Response(200, json={"daily": {
            "time": ["2026-11-06", "2026-11-07"], "temperature_2m_max": [27.1, 29.4],
            "temperature_2m_min": [18.0, 19.2], "precipitation_probability_max": [10, 65]}})
    if "bcb.gov.br" in request.url.host:
        return httpx.Response(200, json={"value": [{
            "cotacaoCompra": 6.1300, "cotacaoVenda": 6.1317, "dataHoraCotacao": "2026-11-06 13:06:13",
            "tipoBoletim": "Fechamento"}]})
    return httpx.Response(404)


class _FakeDDGS:
    def text(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return [{"title": "Mercado Pago", "href": "https://www.mercadopago.com.br", "body": "competitor"},
                {"title": "Notícia", "href": "https://example.org/n", "body": f"Resultado sobre {query}."}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_: Any, **__: Any) -> None:
        raise RuntimeError("unit tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture(autouse=True)
def settings() -> Iterator[Settings]:
    s = get_settings()
    s.auth_secret, s.llm_provider, s.hitl_sync_wait_s = SecretStr("t" * 40), "fake", 0.0
    # psycopg talks through libpq (C), which the socket guard cannot see: an unresolvable host makes any
    # accidental database access fail at once instead of reaching whatever Postgres runs on this machine.
    s.database_url = "postgresql://unit-tests@no-database.invalid/none"
    yield s


@pytest.fixture
def cases(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the database behind search and cases; returns the cases opened during the test."""
    opened: list[dict[str, Any]] = []

    def create_case(user_id: str, conversation_id: str, reason: str, summary: str,
                    action: dict[str, Any] | None = None) -> dict[str, Any]:
        case = {"id": f"case_{len(opened) + 1}", "status": "pending_approval" if action else "open"}
        opened.append({**case, "user_id": user_id, "reason": reason, "summary": summary, "action": action})
        return case

    monkeypatch.setattr(kn, "search", fake_search)
    monkeypatch.setattr(kn, "client", lambda: httpx.Client(transport=httpx.MockTransport(_outside_apis)))
    monkeypatch.setattr(kn, "DDGS", _FakeDDGS)
    monkeypatch.setattr(repo, "create_case", create_case)
    monkeypatch.setattr(repo, "get_case", lambda cid: next((c for c in opened if c["id"] == cid), None))

    def merge_action(case_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        case = next(c for c in opened if c["id"] == case_id)
        case["action"] = {**(case["action"] or {}), **fields}
        return dict(case["action"])

    monkeypatch.setattr(repo, "merge_action", merge_action)
    return opened


@pytest.fixture
def saver() -> InMemorySaver:
    return InMemorySaver(serde=SERDE)


@pytest.fixture
def graph(cases: list[dict[str, Any]], saver: InMemorySaver) -> CompiledStateGraph[TurnState]:
    return build(saver)


def ask(graph: CompiledStateGraph[TurnState], text: str, user: str = "cliente1988", thread: str = "c_test") -> Any:
    return run_turn(graph, text, user, thread)
