# Installation

The README covers the recommended Docker-based setup. This page keeps the full detail:
every agent CLI pin, the Headless runtime, what `agsci doctor` checks, the Docker image
build options, the optional native Autoillustrate install, and the credentials the
containers expect.

## Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

`uv sync` creates `.venv`, installs the project dependencies, and installs the
`agsci` console script (`agsci run`, `agsci doctor`, `agsci inspect`). Check the
host before the first run:

```bash
agsci doctor
```

It reports Python, Git, Docker (CLI and daemon), Node/npm, the agent image,
visible GPUs, installed agent CLIs, the pipeline config and task presets, and
temporary storage. Pass the same arguments you would give `agsci run` to also
check that specific run: the selected agent's CLI and credentials, the
configured image, auth root and env file, the GPUs the preset requires (and that
the image sees them through Docker), and whether the task scaffold is ready or
will be drafted:

```bash
agsci doctor examples/grpo preset=grpo
```

Missing optional pieces are warnings; the command exits non-zero only when a
required check fails.

## Agent CLIs and the Headless runtime

Agentic Scientist invokes every supported coding agent through
[Headless CLI](https://github.com/RobertTLange/headless-cli) with full tool
permissions (`--allow yolo`). Native runs automatically resolve the pinned
Headless revision through `npx`; container runs use the same revision built
into the agent image. Git and Node.js/npm are required for native runs. You can
verify the pinned runtime directly:

```bash
npx -y git+https://github.com/RobertTLange/headless-cli.git#93cd9b06b85f848af1308c41e018991b33907c5e --version
```

The Docker image already contains every backend CLI below. For native
(non-containerized) runs, install the backend CLIs you plan to use:

```bash
# Claude Code
npm install -g @anthropic-ai/claude-code@2.1.285

# Codex CLI
npm i -g @openai/codex@0.153.4

# Cursor CLI
curl https://cursor.com/install -fsS | bash

# OpenCode
npm i -g opencode-ai@1.18.29

# Gemini CLI
npm i -g @google/gemini-cli

# pi
npm install -g @earendil-works/pi-coding-agent@0.85.1
```

Verify:

```bash
claude --version
codex --version
cursor-agent --version
opencode --version
gemini --version
pi --version
```

`agsci doctor` runs these and the other host checks in one go.

## Docker image

The build installs the Autoillustrate release pinned by `ARG AUTOILLUSTRATE_VERSION`
in the Dockerfile from npm; no submodule checkout is needed:

```bash
docker build -t agentic-scientist/agent-cli:latest -f src/docker/agent-cli/Dockerfile .
```

The image is required for orchestrated `generate-concept-figure` requests. Its
Autoillustrate phase agents run in a dedicated one-shot container even when the
outer paper workflow runs natively.

The image also includes TeX Live (`pdflatex` + `bibtex`) so `paper.enabled=true`
works when running in `container.enabled=true` mode.

If default Docker networking cannot reach package mirrors:

```bash
docker build --network=host -t agentic-scientist/agent-cli:latest -f src/docker/agent-cli/Dockerfile .
```

If you need custom Debian mirrors:

```bash
docker build --network=host \
  --build-arg DEBIAN_MIRROR=https://deb.debian.org/debian \
  --build-arg DEBIAN_SECURITY_MIRROR=https://security.debian.org/debian-security \
  -t agentic-scientist/agent-cli:latest \
  -f src/docker/agent-cli/Dockerfile .
```

Avoid `https://ftp.us.debian.org/debian` here. Country `ftp.*` mirrors are not a
safe HTTPS default; if you need a country mirror, prefer plain HTTP or stick
with `deb.debian.org`.

Existing images and containers retain the revision they were built from;
rebuild the image after changing anything it installs.

## Credentials

Create a `.env` file at repo root with the API keys your agent needs. The
pipeline mounts it into every agent container by default
(`container.env_file=.env`):

```bash
# Required keys for common containerized agents
GEMINI_API_KEY=...
CURSOR_API_KEY=...
CLAUDE_CODE_OAUTH_TOKEN=...

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2
```

Do not commit `.env` or any real API key.

For `agent.profile=<name>`, keep `~/.codex/<name>.config.toml` (and its model
catalog, if any) under `container.auth_root` (defaults to `~`). The Docker
runner copies those files into the run-scoped container home alongside Codex
auth files. Containers only forward `OPENAI_API_KEY` and `CODEX_API_KEY` to
Codex, so a profile whose `model_provider` sets a different `env_key` cannot
authenticate inside the concept-figure, ablation, or post-hoc containers.

Export `AGENT_CHATGPT_SUBSCRIPTION=1` for a Pi/OpenCode GPT-5.6-Sol subscription
run; it removes paid credentials from the agent containers and uses the existing
Codex ChatGPT login (`codex login`). The final AI review remains a separate
API-key-backed pipeline phase. Billing modes are described in
[agent_billing.md](agent_billing.md).

## Container options

Both `agsci run` and the standalone `agsci investigate` accept the same container block.
The pipeline defaults differ from the investigation defaults: `agsci run` enables the
container, uses host networking, and mounts `.env`; `agsci investigate` runs natively
unless told otherwise.

| Key | `agsci investigate` default | `agsci run` default | Description |
|-----|---------|---------|-------------|
| `container.enabled` | `false` | `true` | Run inside a Docker container |
| `container.image` | `agentic-scientist/agent-cli:latest` | same | Docker image to use |
| `container.network` | `bridge` | `host` | Docker network mode (`bridge`, `host`) |
| `container.auth_root` | `~` | same | Auth credentials root directory |
| `container.env_file` | `""` | `.env` | Path to `.env` file for the container |
| `container.paper_enabled` | n/a | `null` (follows `container.enabled`) | Run only the paper phase in Docker; set `true` for a native run with a paper (its evidence audit and LaTeX build need the image) |

`container.network=host` is often required on locked-down machines where the
default bridge network cannot reach the provider APIs. The post-hoc phase keeps
its own narrower `posthoc.container.network=bridge` default; override it to
`host` only when the selected agent requires it.

## Optional: native Autoillustrate

The paper orchestrator runs Autoillustrate inside the Docker image above, so
nothing on the host is required for pipeline runs. To use the CLI directly on
the host instead, install the same pinned npm release into your user-local npm
prefix:

```bash
./scripts/install_autoillustrate.sh
export PATH="$HOME/.local/bin:$PATH"
autoillustrate doctor --agent "${AGENT:-claude}"
```

The upstream [Autoillustrate README](https://github.com/SakanaAI/Autoillustrate#readme)
documents the single-command `npm install -g autoillustrate` route and its
prerequisites (draw.io desktop CLI, Node, Python, Headless, API keys).

On Linux with Nix already installed but without Draw.io, opt into installing
the pinned Draw.io and Xvfb packages. This writes to the user Nix profile and
creates the Draw.io wrapper under `~/.local/bin` regardless of
`AUTOILLUSTRATE_PREFIX`:

```bash
./scripts/install_autoillustrate.sh --install-drawio-with-nix
```

The bootstrap deliberately does not install Nix or pipe a remote installer to
a shell. Install Nix separately if you choose this path.

On macOS, install the native app instead (`brew install --cask drawio`) and
follow `autoillustrate doctor` if its CLI is not already on `PATH`.

Set `AUTOILLUSTRATE_PREFIX` to use a prefix other than `~/.local` and
`AUTOILLUSTRATE_VERSION` to install a release other than the image pin. The CLI
creates its hashed Python environment on first use.
