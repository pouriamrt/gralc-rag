"""UMLS Terminology Services REST-API client.

Provides authenticated access to concept search, relation lookup, and
other UMLS endpoints with automatic ticket management and rate limiting.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_MIN_INTERVAL: float = 1.0 / 20  # 20 requests/second cap


class UMLSClient:
    """Thin wrapper around the UMLS REST API.

    Parameters
    ----------
    api_key:
        UMLS API key obtained from https://uts.nlm.nih.gov/uts/profile.
    """

    AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("A non-empty UMLS API key is required.")
        self._api_key = api_key
        self._tgt: str | None = None
        self._last_call_ts: float = 0.0

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def get_tgt(self) -> str:
        """Obtain (or return cached) Ticket Granting Ticket from the CAS server."""
        if self._tgt is not None:
            return self._tgt

        self._rate_limit()
        resp = requests.post(
            self.AUTH_URL,
            data={"apikey": self._api_key},
            timeout=30,
        )
        resp.raise_for_status()

        # The TGT URL is embedded in the HTML response inside a <form action="…">
        html = resp.text
        try:
            tgt_url = html.split('action="')[1].split('"')[0]
        except (IndexError, AttributeError) as exc:
            raise RuntimeError(
                "Failed to parse TGT URL from UMLS auth response."
            ) from exc

        self._tgt = tgt_url
        logger.debug("Obtained UMLS TGT: %s", tgt_url)
        return tgt_url

    def get_service_ticket(self) -> str:
        """Get a single-use Service Ticket from the current TGT."""
        tgt_url = self.get_tgt()

        self._rate_limit()
        resp = requests.post(
            tgt_url,
            data={"service": "http://umlsks.nlm.nih.gov"},
            timeout=30,
        )
        resp.raise_for_status()
        service_ticket = resp.text.strip()
        logger.debug("Service ticket: %s", service_ticket)
        return service_ticket

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def search_concept(self, term: str) -> list[dict[str, Any]]:
        """Search UMLS for *term* and return matching concepts.

        Each result dict contains keys ``ui`` (CUI), ``name``, and
        ``rootSource``.
        """
        ticket = self.get_service_ticket()
        url = f"{self.BASE_URL}/search/current"
        params = {
            "string": term,
            "ticket": ticket,
            "pageSize": 25,
        }

        self._rate_limit()
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results: list[dict[str, Any]] = []
        for item in data.get("result", {}).get("results", []):
            # Skip the "NO RESULTS" sentinel that the API returns
            if item.get("ui") == "NONE":
                continue
            results.append(
                {
                    "ui": item.get("ui", ""),
                    "name": item.get("name", ""),
                    "rootSource": item.get("rootSource", ""),
                }
            )
        return results

    def get_concept_relations(self, cui: str) -> list[dict[str, Any]]:
        """Return relations for the concept identified by *cui*.

        Each result dict contains keys ``relatedIdName``,
        ``relationLabel``, and ``relatedId`` (full URI).
        """
        ticket = self.get_service_ticket()
        url = f"{self.BASE_URL}/content/current/CUI/{cui}/relations"
        params = {
            "ticket": ticket,
            "pageSize": 25,
        }

        self._rate_limit()
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        relations: list[dict[str, Any]] = []
        for item in data.get("result", []):
            relations.append(
                {
                    "relatedIdName": item.get("relatedIdName", ""),
                    "relationLabel": item.get("relationLabel", ""),
                    "relatedId": item.get("relatedId", ""),
                }
            )
        return relations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until at least ``_MIN_INTERVAL`` seconds since last call."""
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call_ts = time.monotonic()

    def invalidate_tgt(self) -> None:
        """Clear the cached TGT (e.g. after an auth error)."""
        self._tgt = None
