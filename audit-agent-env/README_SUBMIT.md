# Hackathon Submission Checklist

Follow these steps to prepare and submit this project.

1. Clean workspace
   - Ensure `.venv/` is not included in the submission (it's in `.gitignore`).

2. Required environment variables
   - `HF_TOKEN` (or `API_KEY`): your Hugging Face API token. Must be set before running `inference.py`.
   - Optional: `IMAGE_NAME` if you want the script to run against a Docker image (remote mode).

3. Install and run locally
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   $env:HF_TOKEN="hf_your_token_here"
   python inference.py
   ```

4. Run with Docker (server)
   ```bash
   cd audit-agent-env
   docker build -t audit-agent-env .
   docker run -p 7860:7860 audit-agent-env
   # then in another shell (local mode uses HF_TOKEN only):
   HF_TOKEN=hf_your_token python inference.py
   ```

5. Remote/docker mode (inference against container)
   - Set `IMAGE_NAME` env var to the published image (e.g. `docker.io/username/image:tag`) and run `inference.py`.

6. Notes & hardening applied
   - `inference.py` masks the API key in logs and catches top-level exceptions to print tracebacks.
   - `audit_agent_env.py` provides a minimal `AuditAgentEnvEnv.from_docker_image()` helper for remote mode.
   - `httpx` added to `requirements.txt` for remote HTTP client.

7. Common issues
   - If the LLM returns billing errors (HTTP 402), the script will log those and continue; you need valid credits or a PRO subscription.
   - Make sure ports (7860) are available if using Docker.

8. Submit
   - Create the repo archive or push to the hackathon submission target excluding `.venv/`.

Good luck! If you want, I can also create a small `run_submission.sh` that automates the venv setup and a dry run.
