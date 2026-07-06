FROM python:3.12-slim

WORKDIR /app

# Install the package the same way pip users get it — from pyproject.toml.
# create_app() imports passport.api and sparks_bus ships as package data, so
# copying agentb/ alone produces an image that dies on ModuleNotFoundError
# at `python -m agentb.server` (shipped broken until v4.9.5 — no CI ran it).
COPY pyproject.toml README.md ./
COPY agentb/ agentb/
COPY passport/ passport/
COPY sparks_bus/ sparks_bus/
RUN pip install --no-cache-dir .

# Run as a non-root user; ~/.agentb is the default data dir.
RUN useradd --create-home mnemo
USER mnemo

COPY agentb.yaml.example agentb.yaml

EXPOSE 50001

# NOTE (v4.9.5 fail-closed auth): the baked example config binds 0.0.0.0 with
# no auth_token, so the server will REFUSE to serve until you provide auth.
# Mount a real config over /app/agentb.yaml (set server.auth_token, or
# server.allow_unauthenticated: true if a gatekeeper sits in front):
#   docker run -v $PWD/agentb.yaml:/app/agentb.yaml -p 50001:50001 <img>
CMD ["python", "-m", "agentb.server"]
