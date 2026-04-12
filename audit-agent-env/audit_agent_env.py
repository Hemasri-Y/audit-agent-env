import asyncio
import subprocess
from typing import Optional

import httpx


class AuditAgentEnvEnv:
    """Lightweight helper to run a container and talk to its HTTP API.

    Uses the Docker CLI to start the image with published ports and then
    discovers the mapped host port for container port 7860. Provides async
    `reset`, `step`, and `close` methods expected by `inference.py`.
    """

    def __init__(self, base_url: str, container_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.container_id = container_id
        self._client = httpx.AsyncClient(timeout=30)

    @classmethod
    async def from_docker_image(cls, image_name: str) -> "AuditAgentEnvEnv":
        # Run container detached and publish ports (-P)
        proc = subprocess.run(["docker", "run", "-d", "-P", image_name], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr.strip()}")

        container_id = proc.stdout.strip()

        # Query mapped host port for container port 7860/tcp
        proc2 = subprocess.run(["docker", "port", container_id, "7860/tcp"], capture_output=True, text=True)
        if proc2.returncode != 0:
            # cleanup
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
            raise RuntimeError(f"docker port query failed: {proc2.stderr.strip()}")

        mapping = proc2.stdout.strip()  # e.g. '0.0.0.0:32768' or ':::32768'
        if ":" not in mapping:
            # unknown mapping format
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
            raise RuntimeError(f"unexpected docker port output: {mapping}")

        host_port = mapping.split(":")[-1]
        base_url = f"http://127.0.0.1:{host_port}"

        # wait for health
        client = httpx.Client(timeout=1)
        for _ in range(20):
            try:
                r = client.get(f"{base_url}/")
                if r.status_code == 200:
                    client.close()
                    return cls(base_url=base_url, container_id=container_id)
            except Exception:
                pass
            await asyncio.sleep(0.5)

        client.close()
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
        raise RuntimeError("Container did not become healthy in time")

    async def reset(self, payload):
        r = await self._client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    async def step(self, action):
        r = await self._client.post(f"{self.base_url}/step", json=action)
        r.raise_for_status()
        return r.json()

    async def close(self):
        try:
            await self._client.aclose()
        finally:
            if self.container_id:
                subprocess.run(["docker", "stop", self.container_id], capture_output=True)
                subprocess.run(["docker", "rm", self.container_id], capture_output=True)
