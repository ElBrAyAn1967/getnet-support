# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Pure units: redaction, grounding, business days, identity, startup guard, ingestion, prompts."""

import json
import re
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr

from app.config import Mode, Settings
from app.graph.guardrails import figures, grounding, guardrail_out
from app.graph.state import Evidence, RouteDecision
from app.prompts import load, versions
from app.rag import ingest
from app.security.auth import current_principal, hash_password, issue_token, verify_password
from app.security.redact import redact
from app.tools import backend

# --- redaction (G-1) -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "kind"), [
    ("meu CPF é 123.456.789-09", "cpf"), ("CNPJ 12.345.678/0001-95", "cnpj"),
    ("cartão 4111 1111 1111 1111", "card"), ("mail ana@padaria.com.br", "email"), ("ligue (51) 99876-5432", "phone"),
])
def test_redact_hides_each_kind(text: str, kind: str) -> None:
    out = redact(text)
    assert kind in out.kinds and f"[{kind.upper()}]" in out.text


@pytest.mark.parametrize("text", ["R$ 1.290,00 em 3x", "vendi 4111 1111 1111 1112 reais", "dia 2026-11-05"])
def test_redact_leaves_amounts_dates_and_non_luhn_numbers(text: str) -> None:
    assert redact(text).kinds == frozenset()


# --- grounding (G-7) -------------------------------------------------------------------------------------------


def test_figures_normalize_brazilian_and_english_formats() -> None:
    assert figures("R$ 1.290,00 e 845,20 e 1290.00 e 48x e [3]") == {Decimal("1290.00"), Decimal("845.20"),
                                                                       Decimal(48)}


def _state(answer: str, evidence: str) -> Any:
    route = RouteDecision(scope="getnet", steps=["knowledge"], standalone_question="q", language="en",
                          confidence=0.9, injection_suspected=False)
    return {"answer": answer, "question": "q", "notices": [], "route": route, "messages": [],
            "evidence": [Evidence(agent="knowledge", tool="search_getnet", content=evidence, source_n=1)]}


def test_grounding_removes_only_the_sentence_with_an_unsupported_figure() -> None:
    out = grounding(_state("You can sell in up to 48x [1]. The fee is 2.99%.", "em até 48x"))
    assert "48x" in out["answer"] and "2.99" not in out["answer"] and "figure_unverified" in out["notices"]


def test_grounding_keeps_a_fully_supported_answer_untouched() -> None:
    assert grounding(_state("Up to 48x, decided by 2 parties [1].", "em até 48x")) == {}


def test_competitor_sentences_are_removed_from_the_reply() -> None:
    out = guardrail_out({"answer": "Getnet offers Pix. Stone is cheaper.", "notices": []})  # type: ignore[typeddict-item]
    assert out["answer"] == "Getnet offers Pix."


# --- simulated backend: dates relative to today (A-11) ---------------------------------------------------------


@pytest.mark.parametrize("weekday_offset", range(7))
def test_yesterdays_sales_exist_whatever_day_it_is(weekday_offset: int) -> None:
    today = date(2026, 11, 2) + timedelta(days=weekday_offset)
    rows = backend.settlements(backend.MERCHANTS["cliente1988"], -1, -1, today)
    assert rows and {r["sale_date"] for r in rows} == {(today - timedelta(days=1)).isoformat()}


def test_deposits_skip_the_weekend() -> None:
    friday = date(2026, 11, 6)
    assert backend.add_business_days(friday, 1) == date(2026, 11, 9)
    assert backend.add_business_days(date(2026, 11, 7), 0) == date(2026, 11, 9)


# --- identity (S-1..S-3) ---------------------------------------------------------------------------------------


def test_passwords_are_salted_scrypt() -> None:
    a, b = hash_password("s3cret"), hash_password("s3cret")
    assert a != b and verify_password("s3cret", a) and not verify_password("wrong", a)


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_a_valid_token_identifies_the_user(settings: Settings) -> None:
    principal = current_principal(_bearer(issue_token("cliente1988", "customer", settings)), settings)
    assert (principal.user_id, principal.role) == ("cliente1988", "customer")


@pytest.mark.parametrize("forge", ["none_alg", "other_secret", "expired", "missing"])
def test_forged_expired_or_missing_tokens_are_rejected(settings: Settings, forge: str) -> None:
    now = int(time.time())
    tokens = {
        "none_alg": jwt.encode({"sub": "cliente1988", "exp": now + 60}, None, algorithm="none"),
        "other_secret": jwt.encode({"sub": "cliente1988", "exp": now + 60}, "x" * 40, algorithm="HS256"),
        "expired": jwt.encode({"sub": "cliente1988", "exp": now - 1}, "t" * 40, algorithm="HS256"),
    }
    with pytest.raises(HTTPException) as err:
        current_principal(_bearer(tokens[forge]) if forge in tokens else None, settings)
    assert err.value.status_code == 401


# --- startup guard (A-07, S-2) ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("overrides", "message"), [
    ({"app_mode": Mode.PRODUCTION}, "AUTH_SECRET"),
    ({"app_mode": Mode.PRODUCTION, "auth_secret": SecretStr("short")}, "AUTH_SECRET"),
    ({"app_mode": Mode.PRODUCTION, "auth_secret": SecretStr("k" * 40), "llm_provider": "fake"}, "fake model"),
    ({"llm_provider": "anthropic"}, "ANTHROPIC_API_KEY"),
    ({"llm_provider": "openai", "llm_model": "gpt-x"}, "OPENAI_API_KEY"),
    ({"llm_provider": "azure_openai", "llm_model": "my-deployment"}, "AZURE_OPENAI_ENDPOINT"),
    ({"llm_provider": "google_genai", "llm_model": "gemini-x"}, "GOOGLE_API_KEY"),
])
def test_unsafe_configurations_refuse_to_start(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        Settings(_env_file=None, **overrides).validate_for_startup()  # type: ignore[call-arg]


PROVIDERS = {  # provider → (credentials it reads, model name, class built)
    "anthropic": ({"ANTHROPIC_API_KEY": "k"}, "claude-opus-5-5", "ChatAnthropic"),
    "openai": ({"OPENAI_API_KEY": "k"}, "gpt-x", "ChatOpenAI"),
    "azure_openai": ({"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
                      "OPENAI_API_VERSION": "2024-10-21"}, "my-deployment", "AzureChatOpenAI"),
    "google_genai": ({"GOOGLE_API_KEY": "k"}, "gemini-x", "ChatGoogleGenerativeAI"),
    "bedrock": ({"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "k"},
                "anthropic.claude-x", "ChatBedrockConverse"),
    "ollama": ({}, "llama-x", "ChatOllama"),
}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_supported_provider_builds_through_the_one_factory(
    provider: str, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluator can run the same graph on the model their company uses: switching is configuration only."""
    from app import llm  # noqa: PLC0415

    env, name, cls = PROVIDERS[provider]
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, "llm_model", name)
    llm.model.cache_clear()
    try:
        settings.validate_for_startup()
        built = llm.model()
        assert type(built).__name__ == cls
        assert built.bind_tools([]) is not None and built.with_structured_output(RouteDecision) is not None
    finally:
        llm.model.cache_clear()


def test_demo_mode_starts_without_secrets() -> None:
    Settings(_env_file=None, app_mode=Mode.DEMO).validate_for_startup()  # type: ignore[call-arg]


# --- ingestion (R-1) -------------------------------------------------------------------------------------------

PAGE = b"""<html><head><title>Crediario | Getnet</title></head><body>
<header><a>Crediario Parcelamento em ate 48x</a></header><nav class="main-menu"><li>Maquininhas e mais</li></nav>
<h2>Ofereca parcelamento</h2><p>em ate 48x para o seu cliente, conforme o emissor do cartao escolhido.</p>
<p>em ate 48x para o seu cliente, conforme o emissor do cartao escolhido.</p>
<h2>Fale com a gente</h2><p>Ok</p><footer>Getnet 2026</footer></body></html>"""


def test_html_extraction_drops_chrome_and_keeps_headings_inline() -> None:
    page = ingest._extract_html("https://site.getnet.com.br/crediario/", PAGE)
    chunks = ingest.chunk(page)
    assert page.title == "Crediario"
    assert [c.body for c in chunks] == ["Ofereca parcelamento: em ate 48x para o seu cliente, conforme o emissor do "
                                        "cartao escolhido."]


FAQ_PAGE = b"""<html><head><title>Pix | Getnet</title></head><body><header>menu</header>
<div class="c-more-answers"><button>Preciso de conta?</button>
<div class="o-modal"><div class="o-modal__content--faq">
<p>O Pix cai na hora na sua Get Conta, sem custo, para vendas na maquininha Getnet.</p>
</div></div></div></body></html>"""


def test_faq_answers_inside_modal_elements_are_kept() -> None:
    """The site keeps FAQ answers in `o-modal` elements; a class-name noise filter once deleted all of them."""
    chunks = ingest.chunk(ingest._extract_html("https://site.getnet.com.br/duvidas/pix/", FAQ_PAGE))
    assert chunks and "Get Conta" in chunks[0].body


def test_an_empty_extraction_never_replaces_a_good_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "DATA", tmp_path)
    url = "https://site.getnet.com.br/duvidas/pix/"
    good = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=FAQ_PAGE, headers={"content-type": "text/html"})))
    blank = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=b"<html><body></body></html>", headers={"content-type": "text/html"})))
    first, second = ingest.fetch(good, url), ingest.fetch(blank, url)
    assert first and second and second.blocks == first.blocks


def test_every_committed_snapshot_is_a_usable_fallback() -> None:
    """The snapshots are what the index falls back to when the site is down: none may be empty or templated."""
    snapshots = sorted((ingest.DATA / "snapshot").glob("*.json"))
    assert len(snapshots) >= len(ingest.read_sources())
    for path in snapshots:
        page = json.loads(path.read_text())
        assert page["blocks"], f"empty snapshot {page['url']}"
        assert "{{" not in path.read_text(), f"unrendered template in {page['url']}"


def test_site_wide_blocks_are_dropped_as_boilerplate() -> None:
    pages = [ingest.Page(f"u{i}", "t", "2026-11-06", [("h", "Baixe o app"), ("h", f"conteudo {i}")]) for i in range(5)]
    assert all(b == [("h", f"conteudo {i}")] for i, b in enumerate(p.blocks for p in ingest.drop_boilerplate(pages)))


def test_a_source_that_fails_falls_back_to_its_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "DATA", tmp_path)
    url = "https://site.getnet.com.br/crediario/"
    ok = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=PAGE,
                                                                              headers={"content-type": "text/html"})))
    down = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    first, second = ingest.fetch(ok, url), ingest.fetch(down, url)
    assert first and second and second.blocks == first.blocks


# --- notifications and alerts ---------------------------------------------------------------------------------


def test_a_new_case_is_posted_to_teams_with_only_redacted_fields(settings: Settings,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from app import notify  # noqa: PLC0415

    sent: list[bytes] = []
    teams = httpx.Client(transport=httpx.MockTransport(lambda r: (sent.append(r.content), httpx.Response(200))[1]))
    monkeypatch.setattr(notify, "client", lambda: teams)
    settings.teams_webhook_url = SecretStr("https://example.webhook.office.com/x")
    notify.case_opened("case_1", "user_request", "Merchant asked for a human.")
    settings.teams_webhook_url = None
    assert sent and b"case_1" in sent[0] and b"Merchant asked" in sent[0]


def test_a_failing_teams_webhook_never_breaks_the_turn(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import notify  # noqa: PLC0415

    monkeypatch.setattr(notify, "client", lambda: httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(500))))
    settings.teams_webhook_url = SecretStr("https://example.webhook.office.com/x")
    notify.case_opened("case_1", "tool_failure", "down")  # must not raise
    settings.teams_webhook_url = None


def test_alert_rules_only_use_metrics_the_api_exposes() -> None:
    from app import observability  # noqa: PLC0415

    rules = (Path(__file__).parents[1] / "deploy" / "prometheus" / "alerts.yml").read_text()
    exposed = {m._name for m in vars(observability).values() if hasattr(m, "_name")}
    used = set(re.findall(r"\b(chat_\w+|router_\w+|grounding_\w+|escalations\w*|llm_\w+)", rules))
    base = {u.removesuffix("_total").removesuffix("_bucket") for u in used}
    assert base <= exposed, base - exposed


# --- prompts ---------------------------------------------------------------------------------------------------


def test_every_prompt_is_versioned_and_renders() -> None:
    assert set(versions()) == {"router", "knowledge", "support", "escalation", "personality"}
    load("router").text.format(today="2026-11-06", min_confidence=0.55)
    load("personality").text.format(language="pt")
    for name in ("knowledge", "support", "escalation"):
        load(name).text.format(today="2026-11-06")
