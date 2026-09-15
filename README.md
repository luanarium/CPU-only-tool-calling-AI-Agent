# CPU-only-tool-calling-AI-Agent
A setup for running LLMs entirely on your own machine with real tool-calling support: llama-server serving quantized models, wired up to the llm library so it can actually call Python functions.

# How-to

# Local AI Agent — llm + llama-server Setup

This sets up [`llm`](https://llm.datasette.io) to run local GGUF models through **llama-server**, registered as an OpenAI-compatible endpoint — giving full sampler control (`temperature`, `frequency_penalty`, `presence_penalty`, `top_p`, `max_tokens`, etc.) and tool/function calling. `llama_cpp.server` is covered too, but only as a fallback for non-tool use — it does not support tool calling at all, confirmed by testing.

> **Why not the `llm-llama-cpp` plugin's direct bindings?** That plugin's `Options` schema only exposes `n_gpu_layers`, `n_ctx`, `max_tokens`, `verbose`, `no_gpu` — no sampler params at all. That means no `temperature`, no `repeat_penalty`, nothing to control repetition or randomness, and no tool calling. It's a dead end for actual generation quality, so this README skips it entirely in favor of the server-backed setup below.

---

## 1. Install system dependencies

```bash
sudo apt update && sudo apt install build-essential
```

Install llama.cpp — this gives you the `llama-server` binary, `llama-cli`, etc. The install script's URL/behavior can drift, so get the current release directly from GitHub instead:

```bash
curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases | grep '"tag_name": "b' | head -1
```

That prints the current build tag (e.g. `b10795`). List its assets and find the `ubuntu-x64.tar.gz` one:

```bash
curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/<TAG> | grep "browser_download_url"
```

Download and extract it (substitute the real tag/filename from the previous step):

```bash
wget https://github.com/ggml-org/llama.cpp/releases/download/<TAG>/llama-<TAG>-bin-ubuntu-x64.tar.gz
```

```bash
tar -xzf llama-<TAG>-bin-ubuntu-x64.tar.gz
```

Confirm where the binary landed — this has varied across releases:

```bash
find . -name "llama-server" -type f
```

Note: this asset is CPU-only (no CUDA/Vulkan). Fine for this setup; build from source with the appropriate `-DGGML_*` flag if you need GPU offload beyond `-ngl`.

Install `uv` (Python tooling):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

```bash
source ~/.bashrc
```

---

## 2. Set up the project

```bash
mkdir local-ai-agent && cd local-ai-agent
```

```bash
uv init
```

```bash
uv add llm llama-cpp-python
```

> **Version check for tool calling.** `llm`'s tool-calling support (`-T`, `--functions`, `llm.Toolbox`) is a relatively recent addition. Confirm you're on a current release:
> ```bash
> uv run llm --version
> ```
> If it's noticeably old, upgrade:
> ```bash
> uv add --upgrade llm
> ```

---

## 3. Download models

```bash
uv tool install "huggingface_hub[cli]"
```

General instruct model:

```bash
uv run --with huggingface_hub python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct-GGUF', filename='qwen2.5-0.5b-instruct-q4_k_m.gguf', local_dir='.')"
```

Coder model:

```bash
uv run --with huggingface_hub python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF', filename='qwen2.5-coder-0.5b-instruct-q4_k_m.gguf', local_dir='.')"
```

---

## 4. Run the server and register it with `llm`

**Use `llama-server` (the binary from step 1) with `--jinja` — this is required for tool calling, confirmed by testing.** `llama_cpp.server`'s bundled server (`--chat_format chatml` or any other value) silently drops the `tools` field before the model ever sees it — verified directly: a request with a `tools` array produced `prompt_tokens: 20` (no schema injected, plain refusal text) against `llama_cpp.server`, versus `prompt_tokens: 159` and a correct structured `tool_calls` response against `llama-server --jinja` for the identical request. This isn't a model capability gap; it's specific to that codepath. Only use `llama_cpp.server` if you don't need tools at all (see the fallback note at the end of this section).

```bash
./llama-<TAG>/llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080 --ctx-size 4000 -ngl 1 --jinja
```

This exposes `http://localhost:8080/v1/chat/completions`, an OpenAI-compatible endpoint with the full sampling schema (`temperature`, `frequency_penalty`, `presence_penalty`, `top_p`, `max_tokens`, etc.) plus working tool calling.

**Confirm tool-call parsing is actually active.** Check the startup log for a `Chat format:` line:

- `Chat format: Qwen 2.5` — native handler, best reliability. Expected for this model.
- `Chat format: Generic` — the template wasn't recognized; tool calling still works but is less token-efficient and less reliable.

You can also inspect the active template directly:

```bash
curl http://localhost:8080/props
```

**Sanity-check tool-schema injection directly**, bypassing `llm` entirely — useful any time tool behavior looks inconsistent, to confirm the server side rather than guessing:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-0.5b-instruct-q4_k_m", "messages": [{"role": "user", "content": "What time is it right now?"}], "tools": [{"type": "function", "function": {"name": "get_current_time", "description": "Get the current system date and time.", "parameters": {"type": "object", "properties": {}}}}]}'
```

A working response has `"finish_reason": "tool_calls"` and a `tool_calls` array, with `prompt_tokens` well above the bare message length (the tool schema and Qwen2.5's `<tools>`/`<tool_call>` instruction block add real bulk). A `prompt_tokens` count barely above the raw message, plain-text refusal content, and no `tool_calls` field means the schema never reached the model — check which server/flags are actually running.

**Register it with `llm`** by creating `extra-openai-models.yaml` in your `llm` config dir (find it via `uv run llm logs path`, then use the same directory):

```yaml
- model_id: local-instruct
  model_name: qwen2.5-0.5b-instruct-q4_k_m
  api_base: "http://localhost:8080/v1"
  api_key_name: null
  supports_tools: true
```

`supports_tools: true` is required for tool calling — without it, `llm` refuses tool-calling requests client-side (`Error: OpenAI Chat: <model> does not support tools`) before ever contacting the server, regardless of whether the model or server can actually handle it.

Write it with:

```bash
mkdir -p ~/.config/io.datasette.llm
```

```bash
printf '%s\n' \
  '- model_id: local-instruct' \
  '  model_name: qwen2.5-0.5b-instruct-q4_k_m' \
  '  api_base: "http://localhost:8080/v1"' \
  '  api_key_name: null' \
  '  supports_tools: true' \
  > ~/.config/io.datasette.llm/extra-openai-models.yaml
```

Verify it parsed and registered:

```bash
uv run --with llm python -c "import yaml; print(yaml.safe_load(open('$HOME/.config/io.datasette.llm/extra-openai-models.yaml')))"
```

```bash
uv run llm models | grep local-instruct
```

Repeat for the coder model (or any other GGUF) with a different `model_id` and a server running on a different `--port`.

### Fallback: `llama_cpp.server` (no tool calling)

If you only need sampler control (`temperature`, `frequency_penalty`, etc.) and don't need tools at all, `llama-cpp-python`'s bundled server is a lighter-weight alternative to building/downloading the `llama-server` binary — no separate binary to manage:

```bash
uv run --with 'llama-cpp-python[server]' python -m llama_cpp.server \
  --model qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --n_gpu_layers 1 \
  --n_ctx 4000 \
  --port 8080 \
  --chat_format chatml
```

Do not register a model on this backend with `supports_tools: true` — confirmed by testing, it silently drops any `tools` array sent to it regardless of `--chat_format` value, so `llm` would report success while no tool call ever actually happens.

---

## 5. Usage

### Method A — Run from anywhere (no project folder needed)

`uvx` builds its own ephemeral environment on every invocation, so this needs no `uv init` / `uv add` at all — it's fully independent of the project folder in step 2. The server just needs to be running (step 4) and the model registered in `extra-openai-models.yaml` — that config is global, so it doesn't matter which method started the server or did the registration.

```bash
uvx --with llm llm -m local-instruct "Why is the sky blue?" \
  -o temperature 0.7 \
  -o frequency_penalty 0.3 \
  -o presence_penalty 0.3
```

> **Config is shared regardless of method.** `~/.config/io.datasette.llm/` (`extra-openai-models.yaml`, `logs.db`) is one global location `llm` reads from no matter how `llm` itself was launched — `uvx`, `uv run` in a project (Method B), or a plain `uv tool install`. Registration (step 4) only needs to happen once; it's immediately visible under every method.
>
> **`uvx` is cached, not reinstalled each time** — `uv` caches resolved environments/wheels, so repeat invocations are fast even though no project files are created. If you'd rather not resolve the environment on every call, install it once as a persistent tool instead:
> ```bash
> uv tool install llm
> ```
> After that, just run `llm -m local-instruct "..."` directly, with no `uvx`/`uv run` prefix needed.

### Method B — Project folder workflow

Once initial setup (steps 2–4) is done, your daily routine is just:

```bash
cd local-ai-agent
```

Make sure the server from step 4 is running, then:

```bash
uv run llm -m local-instruct "Why is the sky blue?" \
  -o temperature 0.7 \
  -o frequency_penalty 0.3 \
  -o presence_penalty 0.3
```

You never need to re-run `uv init`, `uv add`, or re-download models — they persist in the project folder.

### Python API

```python
import llm
model = llm.get_model("local-instruct")
response = model.prompt(
    "Why is the sky blue?",
    system="Answer in one sentence.",
    temperature=0.7,
    frequency_penalty=0.3,
    presence_penalty=0.3,
)
print(response.text())
```

Omit `max_tokens` for uncapped output — it's bounded only by `--ctx-size` on the server.

---

## 6. Tool calling

`llm`'s tool support works the same regardless of backend — it converts your Python function into a JSON Schema tool definition, sends it alongside the prompt, and executes the function if the model requests it. The backend must be `llama-server` with `--jinja` (step 4) — `llama_cpp.server` does not support this at all, confirmed by testing, regardless of its `--chat_format` setting.

**CLI**, using `llm`'s `--functions` flag with an inline function:

```bash
uv run llm -m local-instruct --functions '
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny and 72F in {location}"
' "What is the weather in Seattle?"
```

**Python API**:

```python
import llm

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny and 72F in {location}"

model = llm.get_model("local-instruct")
chain_response = model.chain(
    "What is the weather in Seattle?",
    tools=[get_weather],
)
print(chain_response.text())
```

### Tool-call reliability at small model sizes

Two distinct failure modes show up at small parameter counts (confirmed by repeated testing on Qwen2.5-0.5B), and they need different fixes:

**1. Whether the tool gets called at all is noisy at default sampling settings.** The same exact prompt, run repeatedly with no `-o temperature` flag, called the tool roughly half the time and asked a clarifying question (rather than using the documented default argument) the other half. This is a sampling-decision problem, not a capability ceiling — running the identical prompt 10 times at `-o temperature 0.0` produced 10/10 correct tool calls. For any tool-reliant prompt, prefer low/zero temperature:

```bash
uv run llm -m local-instruct --functions tools.py -o temperature 0.0 "What time is it right now?"
```

Note the tension this creates: low temperature stabilizes tool-call decisions, but it's the opposite of the `temperature 0.7`/`frequency_penalty`/`presence_penalty` settings used earlier in this README to fix repetition loops in open-ended generation. There isn't one setting that's right for both — pick per use case (tool-heavy vs. conversational), rather than assuming one temperature value works everywhere.

**2. Reasoning about a correct tool result afterward is a separate, harder problem — temperature does not fix it.** Even with a perfectly correct tool call and result, the model can still misdescribe or mis-transform that result in its final answer (e.g. converting `14:30` to 12-hour time incorrectly, or describing `22:56` as being in "24-hour clock (12-hour clock with AM/PM)"). This is a reasoning/accuracy failure, not a sampling-noise failure, so lowering temperature does not reliably fix it. The effective fix is pushing the computation into the tool itself rather than the model's free-text reasoning — e.g. add a `format` parameter to the tool so the model only has to pick the right argument, rather than doing the conversion in its answer:

```python
import datetime

def get_current_time(format: str = "24h") -> str:
    """Get the current system time. Set format to '12h' for 12-hour clock with AM/PM, or '24h' for 24-hour clock."""
    now = datetime.datetime.now()
    if format == "12h":
        return now.strftime("%I:%M %p")
    return now.strftime("%H:%M:%S")
```

As a general principle for evaluating small local models: the model's job should be *deciding which tool to call and with what arguments* — that's the part it's most reliable at, even at small sizes once sampling is tuned. Anything requiring multi-step reasoning on a result (arithmetic, format conversion, filtering) is more robust pushed into the tool as another argument or a second tool call, rather than trusted to the model's own reasoning after the fact.

### Working-directory tools

Since `--functions` accepts arbitrary Python, and that Python runs in your actual shell environment, tools aren't limited to toy examples — they can read files, list directories, or run commands, turning `llm` into a working-directory-aware assistant rather than just a chat window. Nothing is automatic: the model only gets the capabilities you explicitly write functions for.

From inside `local-ai-agent`:

```bash
uv run llm -m local-instruct --functions '
import os

def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    return "\n".join(os.listdir(directory))

def read_file(path: str) -> str:
    """Read the contents of a text file."""
    with open(path) as f:
        return f.read()
' "What Python files are in this directory, and what does the main one do?"
```

The model calls `list_files`, sees the results, decides to call `read_file` on whatever looks relevant, and answers using the actual file contents — grounded in the real filesystem, not guesses.

**For repeated use, a `--functions` file beats inline strings.** Keep a `tools.py` in the project folder with your working-directory-aware functions, versioned alongside everything else, and point `llm` at it directly:

```bash
uv run llm -m local-instruct --functions tools.py "..."
```

> **Safety note.** Exposing tools — especially anything that reads/writes files or runs shell commands — means the model's output is effectively deciding what code executes on your machine. Keep functions narrow and specific (read-only where possible) rather than something broad like `run_shell(cmd: str)`, especially while you're still validating reliability on a 0.5B model.

---

## Troubleshooting notes

- **`ValidationError: Extra inputs are not permitted` for `temperature`/`repeat_penalty`** — you're calling a model registered through the `llm-llama-cpp` plugin's direct bindings instead of the server-backed setup above. That plugin's `Options` schema doesn't expose sampler params at all — switch to the server-backed registration instead.
- **`llama-server: command not found`** — the binary isn't on your `PATH` and probably isn't downloaded yet. Follow step 1's GitHub release download rather than assuming an install script placed it — asset naming and script availability both drift over time. Or use `llama_cpp.server` as a temporary fallback if you don't need tools right now (see step 4's fallback section) — no separate binary required.
- **`TypeError: 'NoneType' object is not iterable` from `register_models`** — `extra-openai-models.yaml` exists but is empty. `yaml.safe_load()` on an empty file returns `None`. Rewrite the file with actual YAML list content (see step 4).
- **Repetition loops on small models** — set `temperature` (e.g. `0.7`) and `frequency_penalty`/`presence_penalty` (e.g. `0.3`) via the server-backed endpoint.
- **Model responds with plain text instead of a tool call, or `tool_calls` comes back empty** — confirmed cause: the server is `llama_cpp.server`, which silently drops the `tools` field regardless of `--chat_format`. Switch to `llama-server --jinja` (step 4) — there's no fix on the `llama_cpp.server` side, it doesn't support tool-schema injection at all. If you're already on `llama-server`, confirm `--jinja` is actually present in the running command and check the `Chat format:` startup log line isn't blank/unrecognized.
- **`Error: OpenAI Chat: <model> does not support tools`** — this is `llm` refusing client-side before any request is sent, not the server rejecting anything. Add `supports_tools: true` to that model's entry in `extra-openai-models.yaml` (step 4) and retry.
- **`ValueError: no signature found for builtin type <class 'datetime.datetime'>`** (or similar for other stdlib classes) — `llm`'s `--functions` auto-discovery inspects every callable in the file's module scope as a potential tool, not just the function(s) you intended. `from datetime import datetime` pulls the `datetime` class itself into scope, and `llm` tries (and fails) to treat it as a tool. Fix: `import datetime` instead, and reference `datetime.datetime.now()` inside your function body — this keeps only your actual function at module scope.
- **Same prompt sometimes calls the tool, sometimes doesn't (or asks a clarifying question instead of using a documented default)** — this is sampling noise on the decision boundary, not a broken tool. Add `-o temperature 0.0` for tool-reliant prompts; see "Tool-call reliability at small model sizes" in §6 for the full finding and its trade-off against repetition-loop settings used elsewhere in this README.
- **Tool call and result are correct, but the model's final answer misdescribes or mis-transforms the result** (wrong unit conversion, self-contradictory description, etc.) — this is a reasoning-accuracy issue, not sampling noise; lowering temperature does not fix it. Push the computation into the tool itself (e.g. an explicit `format` parameter) rather than relying on the model to transform the result correctly in free text. See §6.

---

## Offline / backup notes

For portable, reproducible environments (air-gapped or otherwise):

Build an explicit wheelhouse for this project:

```bash
uv export --format requirements-txt | uv pip download -r /dev/stdin -d ./wheelhouse
```

Install fully offline later, elsewhere:

```bash
uv pip install --no-index --find-links ./wheelhouse -r requirements.txt
```

Confirm a backup is complete (fails loudly if anything's missing):

```bash
uv sync --offline
```

Keep `./wheelhouse` alongside `uv.lock` — the lockfile pins exact versions/hashes, so the pair gives you a fully reproducible, offline-installable environment rather than relying on `uv`'s global cache (`uv cache dir`), which mixes in packages from unrelated projects.

### Moving the backup to removable media

A wheelhouse is just a flat folder of `.whl`/`.tar.gz` files — no database, no symlinks, no baked-in absolute paths — so it copies cleanly to a USB drive or external disk.

Archive it as one file:

```bash
tar czf wheelhouse-backup.tar.gz wheelhouse uv.lock pyproject.toml
```

Copy to mounted removable media:

```bash
cp wheelhouse-backup.tar.gz /media/your-usb/
```

**Restoring on another machine, fully offline:**

```bash
tar xzf wheelhouse-backup.tar.gz
```

```bash
uv sync --offline --find-links ./wheelhouse
```

Notes:

- **Architecture matters.** Wheels are often platform-specific (e.g. `manylinux_x86_64` vs `macosx_arm64`). If the drive needs to serve different machines, download with `--python-platform`/`--python-version` flags on `uv pip download` so the wheelhouse contains wheels for the *target* machine, not just the one that built it. A single-architecture homelab (all x86_64 Linux) doesn't need to worry about this.
- **Always include `uv.lock`.** Without it, the wheelhouse is just "some package files," not a reproducible environment — the lockfile is what tells `uv` exactly which versions/hashes to expect.
- **Check size before copying** with `du -sh ./wheelhouse`; export with `--no-dev` if you only need runtime dependencies and want to trim dev/test tooling from the backup.
