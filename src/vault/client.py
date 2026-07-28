import os
import threading

import httpx

_TIMEOUT = 5.0


class VaultError(RuntimeError):
    pass


class VaultClient:
    def __init__(self, addr: str, role_id: str, secret_id: str) -> None:
        self._addr = addr.rstrip("/")
        self._role_id = role_id
        self._secret_id = secret_id
        self._token: str | None = None
        self._lock = threading.Lock()

    def _login(self) -> str:
        resp = httpx.post(
            f"{self._addr}/v1/auth/approle/login",
            json={"role_id": self._role_id, "secret_id": self._secret_id},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["auth"]["client_token"]

    def read_kv(self, path: str) -> dict[str, str]:
        with self._lock:
            if self._token is None:
                self._token = self._login()
        resp = httpx.get(
            f"{self._addr}/v1/secret/data/{path}",
            headers={"X-Vault-Token": self._token},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["data"]["data"]


def from_env() -> VaultClient:
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
