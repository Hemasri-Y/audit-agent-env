"""
AuditAgentEnv — Docker / Remote HTTP Client
=============================================
Lightweight helper to run a container and talk to its HTTP API.
Provides async reset, step, close expected by inference.py.
"""

import asyncio
import subprocess
from typing import Optional

import httpx


class AuditAgentEnvEnv:

    def __init__(self, base_url: str, container_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.container_id = container_id
        self._client = httpx.AsyncClient(timeout=60)
        self._closed = False

    @classmethod
    async def from_docker_image(cls, image_name: str) -> "AuditAgentEnvEnv":
        proc = subprocess.run(
            ["docker", "run", "-d", "-P", image_name],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr.strip()}")

        container_id = proc.stdout.strip()

        proc2 = subprocess.run(
            ["docker", "port", container_id, "7860/tcp"],
            capture_output=True, text=True,
        )
        if proc2.returncode != 0:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
            raise RuntimeError(f"docker port query failed: {proc2.stderr.strip()}")

        mapping = proc2.stdout.strip()
        if ":" not in mapping:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
            raise RuntimeError(f"unexpected docker port output: {mapping}")

        host_port = mapping.split(":")[-1]
        base_url = f"http://127.0.0.1:{host_port}"

        # Wait up to 60 seconds for the container to become healthy
        check_client = httpx.Client(timeout=2)
        for attempt in range(30):
            try:
                r = check_client.get(f"{base_url}/")
                if r.status_code == 200:
                    check_client.close()
                    print(f"[DEBUG] Container healthy after ~{attempt * 2}s", flush=True)
                    return cls(base_url=base_url, container_id=container_id)
            except Exception:
                pass
            await asyncio.sleep(2)

        check_client.close()
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
        raise RuntimeError("Container did not become healthy within 60 seconds")

    async def reset(self, payload: dict) -> dict:
        r = await self._client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    async def step(self, action: dict) -> dict:
        r = await self._client.post(f"{self.base_url}/step", json=action)
        r.raise_for_status()
        return r.json()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self._client.aclose()
        finally:
            if self.container_id:
                subprocess.run(["docker", "stop", self.container_id], capture_output=True)
                subprocess.run(["docker", "rm", self.container_id], capture_output=True)
