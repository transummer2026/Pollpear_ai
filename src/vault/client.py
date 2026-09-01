"""Minimal HashiCorp Vault client (AppRole login + KV v2 reads)."""

import os
import threading

import httpx

# Per-request timeout (seconds) for every Vault HTTP call.
_TIMEOUT = 5.0


class VaultError(RuntimeError):
    """Raised when Vault is unreachable or a required secret is missing."""

    pass


class VaultClient:
    """Reads KV v2 secrets from Vault, logging in lazily on first use.

    The AppRole client token is fetched on the first read and then cached
    for the lifetime of the instance; access is serialized with a lock so
    concurrent readers trigger at most one login.
    """

    def __init__(self, addr: str, role_id: str, secret_id: str) -> None:
        self._addr = addr.rstrip("/")
        self._role_id = role_id
        self._secret_id = secret_id
        self._token: str | None = None
        self._lock = threading.Lock()

    def _login(self) -> str:
        """Exchange the AppRole role_id/secret_id for a short-lived client token."""
        resp = httpx.post(
            f"{self._addr}/v1/auth/approle/login",
            json={"role_id": self._role_id, "secret_id": self._secret_id},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["auth"]["client_token"]

    def read_kv(self, path: str) -> dict[str, str]:
        """Return the key/value pairs stored at KV v2 mount ``secret`` under ``path``."""
        with self._lock:
            # Log in once on first read; later calls reuse the cached token.
            if self._token is None:
                self._token = self._login()
        resp = httpx.get(
            f"{self._addr}/v1/secret/data/{path}",
            headers={"X-Vault-Token": self._token},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        # KV v2 wraps the secret twice: response.data.data holds the actual pairs.
        return resp.json()["data"]["data"]


def from_env() -> VaultClient:
    """Build a client from the VAULT_ADDR/ROLE_ID/SECRET_ID env vars.

    Raises ``VaultError`` listing every bootstrap credential that is unset.
    """
    addr = os.environ.get("VAULT_ADDR")
    role_id = os.environ.get("ROLE_ID")
    secret_id = os.environ.get("SECRET_ID")
    missing = [
        name
        for name, value in (
            ("VAULT_ADDR", addr),
            ("ROLE_ID", role_id),
            ("SECRET_ID", secret_id),
        )
        if not value
    ]
    if missing:
        raise VaultError(f"Missing Vault bootstrap creds: {', '.join(missing)}")
    return VaultClient(addr, role_id, secret_id)
