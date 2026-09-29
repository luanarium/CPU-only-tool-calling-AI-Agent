## contents

- [CPU-only tool-calling AI agent](#cpu-only-tool-calling-ai-agent)
- [YouTube transcript summarizer](#8-youtube-transcript-summarizer)
- [Offline / backup notes](#offline--backup-notes)

# CPU-only-tool-calling-AI-Agent

A setup for running LLMs entirely on your own machine with real tool-calling support: llama-server serving quantized models, wired up to the llm library so it can actually call Python functions.

## Ideal use cases for this kind of agent

Querying live system state (time, disk, memory, service health) through small deterministic Python tools rather than asking the model to compute or reason over raw data; turning terse fact strings into natural-language reports, the same way a shipboard computer might narrate sensor readings; and exploring where small open-weight models genuinely hold up under real tool-calling conditions versus where they need the surrounding system to do the heavy lifting instead.

## How-to

This runs a local Qwen2.5-0.5B-Instruct model with `llama-server` (llama.cpp) and registers it with [`llm`](https://llm.datasette.io) as the OpenAI-compatible model `qwen-clean-server` (any name will suffice, this is just what I used). That setup gives you:

- sampler control (`temperature`, `frequency_penalty`, `presence_penalty`, `top_p`, `max_tokens`)
- tool calling from a `tools.py` file
- a YouTube transcript summarizer

> **Why `llama-server`?** The `llm-llama-cpp` plugin exposes no sampler options and has no tool calling. The `llama_cpp.server` bundled with `llama-cpp-python` silently drops tool definitions, which was confirmed by testing. `llama-server --jinja` supports both.

Final layout:

```
project/
├── llama-b10795/                        # llama.cpp binaries
├── qwen2.5-0.5b-instruct-q4_k_m.gguf    # model
├── tools.py                             # tools for llm
├── yt-summarize.sh                      # YouTube summarizer
└── pyproject.toml, uv.lock, .venv/, src/
```

Tested with: llama.cpp **b10795** (CPU build), `llm` **0.33**, Python **3.10**, `uv` **0.12**, Ubuntu x64.

---

## 1. Install uv and create the project

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

```bash
source ~/.bashrc
```

Create the project:

```bash
mkdir project
```

```bash
cd project
```

```bash
uv init --package --python 3.10
```

```bash
uv add "llm==0.33"
```

All remaining commands run from inside `project/`.

---

## 2. Download llama.cpp

```bash
wget https://github.com/ggml-org/llama.cpp/releases/download/b10795/llama-b10795-bin-ubuntu-x64.tar.gz
```

```bash
tar -xzf llama-b10795-bin-ubuntu-x64.tar.gz
```

This creates `llama-b10795/`, with `llama-server` directly inside it. The build is CPU-only.

---

## 3. Download the model

```bash
uv run --with huggingface_hub python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct-GGUF', filename='qwen2.5-0.5b-instruct-q4_k_m.gguf', local_dir='.')"
```

Optionally verify the download:

```bash
sha256sum qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Expected: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

---

## 4. Start the server

Run the server in its own terminal and leave it running:

```bash
./llama-b10795/llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8081 --ctx-size 4000 -ngl 1 --jinja
```

- `--jinja` is **required** for tool calling.
- `--ctx-size 4000` is rounded up to 4096 by the server. Prompt and output together must fit in it.

Check that tool calling works on the server side:

```bash
curl -s http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen2.5-0.5b-instruct-q4_k_m", "messages": [{"role": "user", "content": "What time is it right now?"}], "tools": [{"type": "function", "function": {"name": "get_current_time", "description": "Get the current system date and time.", "parameters": {"type": "object", "properties": {}}}}]}'
```

A working response contains `"finish_reason": "tool_calls"` and about `"prompt_tokens": 159`. A prompt of only about 20 tokens with a plain-text reply means the tool schema never reached the model; check that `--jinja` is set.

---

## 5. Register the server with `llm`

```bash
mkdir -p ~/.config/io.datasette.llm
```

```bash
cat > ~/.config/io.datasette.llm/extra-openai-models.yaml << 'EOF'
- model_id: qwen-clean-server
  model_name: qwen2.5-0.5b-instruct-q4_k_m
  api_base: "http://localhost:8081/v1"
  api_key_name: null
  supports_tools: true
EOF
```

`supports_tools: true` is required. Without it, `llm` refuses tool calls itself, before contacting the server.

Verify:

```bash
uv run llm models | grep qwen-clean-server
```

Expected: `OpenAI Chat: qwen-clean-server`

This config is global: it applies to every `llm` on the machine, not just this project.

---

## 6. Usage

```bash
uv run llm -m qwen-clean-server "Why is the sky blue?" -o temperature 0.7 -o frequency_penalty 0.3 -o presence_penalty 0.3
```

These settings prevent the repetition loops a 0.5B model falls into during open-ended text. For tool calls, use `temperature 0.0` instead (see §7).

From Python:

```python
import llm

model = llm.get_model("qwen-clean-server")
response = model.prompt(
    "Why is the sky blue?",
    system="Answer in one sentence.",
    temperature=0.7,
    frequency_penalty=0.3,
    presence_penalty=0.3,
)
print(response.text())
```

---

## 7. Tool calling

`llm --functions tools.py` exposes every function in the file to the model as a tool. When the model calls one, `llm` runs it and passes the result back.

Create `tools.py`:

```python
# tools.py
import datetime


def get_current_time(format: str = "24h") -> str:
    """Get the current system time. Set format to '12h' for 12-hour clock with AM/PM, or '24h' for 24-hour clock."""
    now = datetime.datetime.now()
    if format == "12h":
        return now.strftime("%I:%M %p")
    return now.strftime("%H:%M:%S")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Supports +, -, *, /, **, and parentheses only."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](eval_node(node.operand))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expression, mode="eval")
    return str(eval_node(tree.body))


def days_until(date_str: str) -> str:
    """Calculate days remaining until a given date. Format: YYYY-MM-DD."""
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.date.today()
    delta = (target - today).days
    return f"{delta} days" if delta >= 0 else f"{-delta} days ago"


def days_until_named_holiday(name: str) -> str:
    """Calculate days remaining until a named holiday this year. Supports: christmas, new year's, halloween."""
    today = datetime.date.today()
    holidays = {
        "christmas": (12, 25),
        "new year's": (1, 1),
        "halloween": (10, 31),
    }
    key = name.strip().lower()
    if key not in holidays:
        return f"Unknown holiday: {name}"
    month, day = holidays[key]
    target = datetime.date(today.year, month, day)
    if target < today:
        target = datetime.date(today.year + 1, month, day)
    delta = (target - today).days
    return f"{delta} days"
```

Try it:

```bash
uv run llm -m qwen-clean-server --functions tools.py -o temperature 0.0 "What time is it right now?"
```

```bash
uv run llm -m qwen-clean-server --functions tools.py -o temperature 0.0 "What is (17 * 23) + 4?"
```

Add `--td` (tools debug) to see each tool call and its result.

### Lessons from testing on a 0.5B model

- **Use `-o temperature 0.0` for tool prompts.** At default settings, the same prompt called the tool only about half the time. At `0.0` it went 10/10.
- **Put the computation in the tool, not in the model.** The model reliably picks a tool and its arguments, but it often garbles results. For example, it converted `14:30` to 12-hour time incorrectly. So `get_current_time` takes a `format` argument, and `calculate` does the arithmetic.
- **Keep module scope clean.** `--functions` treats *every* callable at the top level of the file as a tool. Use `import datetime`, not `from datetime import datetime`, and put other imports inside the function that needs them, as `calculate` does.
- **`calculate` never uses `eval()`.** It walks the AST and allows only arithmetic, so the model can't run arbitrary code through it.

> **Safety:** tools run as real Python on your machine, and the model decides when to call them. Keep them narrow and read-only where possible.

---

## 8. YouTube transcript summarizer

This fetches a video's auto-generated English subtitles, strips the timing markup and duplicate lines, and asks the model for a summary. No video is downloaded.

Install `yt-dlp`:

```bash
uv tool install yt-dlp
```

Create `yt-summarize.sh`:

```bash
#!/bin/bash
# Usage: ./yt-summarize.sh "https://youtube.com/watch?v=..."

yt-dlp --write-auto-sub --skip-download --sub-lang en --sub-format vtt -o "/tmp/yt-transcript.%(ext)s" "$1" \
  && sed -e '/-->/d' -e '/^WEBVTT/d' -e '/^$/d' -e 's/<[^>]*>//g' /tmp/yt-transcript.en.vtt \
  | awk '!seen[$0]++' \
  | uv run llm -m qwen-clean-server "Summarize the following"
```

```bash
chmod +x yt-summarize.sh
```

Run it (with the server up):

```bash
./yt-summarize.sh "https://www.youtube.com/watch?v=bk-p1ZURaSE"
```

What each stage does:

1. `yt-dlp` saves only the subtitles, to `/tmp/yt-transcript.en.vtt`.
2. `sed` removes the header, the timestamp lines, blank lines and inline tags.
3. `awk` drops repeated lines. Auto-captions repeat each line as they scroll.
4. `llm` summarizes the cleaned text from stdin.

Limits: long videos can exceed the 4096-token context; raise `--ctx-size` if you have the RAM. Two runs at once overwrite each other's `/tmp` file.
My tests found this works for videos of up to ~33 minutes in length.

---

## Troubleshooting

- **`Error: OpenAI Chat: <model> does not support tools`:** add `supports_tools: true` to the YAML (§5).
- **`TypeError: 'NoneType' object is not iterable` from `register_models`:** the YAML file is empty. Rewrite it with the command in §5.
- **Plain-text reply instead of a tool call:** check that `--jinja` is on the server command, then run the `curl` check in §4.
- **`ValueError: no signature found for builtin type ...`:** an import in `tools.py` put a class at the top level of the file. Use `import datetime` (§7).
- **The tool is only sometimes called:** add `-o temperature 0.0`.
- **The tool result is right but the answer is wrong:** lowering the temperature won't help. Move that step into the tool.
- **Repetition loops:** use `-o temperature 0.7 -o frequency_penalty 0.3 -o presence_penalty 0.3`.
- **Connection errors from `llm`:** the server isn't running, or it's not on port 8081.

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
