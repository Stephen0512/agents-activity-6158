#!/usr/bin/env python3
"""Skeleton for the Python -> Rust translation agent.

    python agent.py                  # run with defaults
    python agent.py --budget 40      # cap on model calls (graded: do not raise)

WHAT IS GIVEN
  * the loop
  * six working tools
  * trajectory logging

WHAT YOU FILL IN   (search for "TODO")
  1. call_model()      - talk to whichever model you have access to
  2. system_prompt()   - what the agent is told about its job
  3. build_context()   - CONTEXT MANAGEMENT. The hard one.
  4. should_stop()     - TERMINATION. The one everyone forgets.
  5. the tool set      - add, remove, or reshape tools. This matters more
                         than you expect; see README.

The skeleton runs as given and accomplishes nothing. That is intentional.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, time, sys, urllib.error, urllib.request

HERE   = pathlib.Path(__file__).parent
RUST   = HERE / "rust"
LIB    = RUST / "src" / "lib.rs"
PYSRC  = HERE / "reference" / "version.py"
LOGS   = HERE / "logs"
NOTES  = LOGS / "notes.md"
BEST   = LOGS / "best-lib.rs"
BIN    = RUST / "target" / "release" / "harness"

# Keep these small: the whole point of TODO 3 is not to grow the window.
RECENT_TURNS = 8
TOOL_CHARS = 3500
MEMORY_NOTES = 1200


# ============================================================== TODO 1
def call_model(messages: list[dict], tools: list[dict]) -> dict:
    """Send `messages` + `tools` to Gemini; return its reply.

    Return shape expected by the loop below:
        {"text": str | None,
         "tool_calls": [{"name": str, "arguments": dict}, ...]}
    """
    return _call_gemini(messages, tools)


def _gemini_key() -> str:
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY. Optional: GEMINI_MODEL (default gemini-3.5-flash-lite).")
    return key


def _http_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    """POST JSON. Retry 429 (quota) and 503/500 (high demand) with backoff."""
    body_bytes = json.dumps(payload).encode()
    last_err = None
    for attempt in range(8):
        req = urllib.request.Request(
            url, data=body_bytes,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            last_err = RuntimeError(f"HTTP {e.code} from {url}: {body[:2000]}")
            retryable = e.code in {429, 500, 503}
            if not retryable or attempt == 7:
                raise last_err from e
            wait = min(90.0, 8.0 * (2 ** attempt))
            m = re.search(r"retry in ([\d.]+)s", body, re.I)
            if m:
                wait = min(90.0, float(m.group(1)) + 1.0)
            print(f"      [gemini {e.code}] sleeping {wait:.0f}s (attempt {attempt + 1}/8)")
            time.sleep(wait)
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"network error calling {url}: {e}")
            if attempt == 7:
                raise last_err from e
            wait = min(60.0, 5.0 * (2 ** attempt))
            print(f"      [gemini net] sleeping {wait:.0f}s (attempt {attempt + 1}/8)")
            time.sleep(wait)
    raise last_err  # pragma: no cover


def _gemini_schema(params: dict | None) -> dict | None:
    """Gemini wants JSON-schema objects; drop empty parameter blocks."""
    if not params:
        return None
    props = params.get("properties") or {}
    required = params.get("required") or []
    if not props and not required:
        return None
    return params


def _to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    system, contents, pending = [], [], []

    def user_parts() -> list:
        if contents and contents[-1]["role"] == "user":
            return contents[-1]["parts"]
        parts: list = []
        contents.append({"role": "user", "parts": parts})
        return parts

    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system.append(m["content"])
        elif role == "user":
            user_parts().append({"text": m.get("content") or ""})
        elif role == "assistant":
            pending = []
            raw = m.get("_gemini_parts")
            if raw:
                contents.append({"role": "model", "parts": raw})
                for p in raw:
                    fc = p.get("functionCall") or {}
                    if fc.get("name"):
                        pending.append({"name": fc["name"], "id": fc.get("id")})
                continue
            parts = []
            if m.get("content"):
                parts.append({"text": m["content"]})
            for c in m.get("tool_calls") or []:
                fc = {"name": c["name"], "args": c.get("arguments") or {}}
                if c.get("id"):
                    fc["id"] = c["id"]
                part = {"functionCall": fc}
                if c.get("thoughtSignature"):
                    part["thoughtSignature"] = c["thoughtSignature"]
                parts.append(part)
                pending.append({"name": c["name"], "id": c.get("id")})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        elif role == "tool":
            meta = pending.pop(0) if pending else {}
            fr = {
                "name": m.get("name") or meta.get("name") or "tool",
                "response": {"output": m.get("content") or ""},
            }
            if meta.get("id"):
                fr["id"] = meta["id"]
            user_parts().append({"functionResponse": fr})
    return "\n\n".join(system), contents


def _call_gemini(messages: list[dict], tools: list[dict]) -> dict:
    key = _gemini_key()
    model = os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite"
    decls = []
    for t in tools:
        d = {"name": t["name"], "description": t.get("description") or ""}
        schema = _gemini_schema(t.get("parameters"))
        if schema:
            d["parameters"] = schema
        decls.append(d)
    system, contents = _to_gemini_contents(messages)
    payload = {
        "contents": contents,
        "tools": [{"functionDeclarations": decls}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _http_json(url, payload, {"x-goog-api-key": key})
    cand = (data.get("candidates") or [{}])[0]
    parts = ((cand.get("content") or {}).get("parts")) or []
    texts, calls = [], []
    for p in parts:
        if p.get("text") and not p.get("thought"):
            texts.append(p["text"])
        fc = p.get("functionCall")
        if fc:
            args = fc.get("args") or fc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            calls.append({
                "name": fc.get("name") or "",
                "arguments": args or {},
                "id": fc.get("id"),
                "thoughtSignature": p.get("thoughtSignature"),
            })
    return {
        "text": "\n".join(texts) or None,
        "tool_calls": calls,
        "_gemini_parts": parts,
    }


# ============================================================== TODO 2
def system_prompt() -> str:
    """What the agent is told about its job.

    Bake the *contract* and the known failure modes; make the agent read
    `reference/version.py` for the actual grammar and bump rules. A dumped
    spec goes stale the moment the source disagrees with your memory of it.
    """
    return """\
You translate `reference/version.py` (python-semver) into `rust/src/lib.rs`.
`evaluate.py` compares your crate to the real `semver` package. The grading
seed is not the practice seed — do not hard-code practice cases.

Required API (keep these signatures; you may add helpers):
  pub struct Version { major: u64, minor: u64, patch: u64,
                       prerelease: Option<String>, build: Option<String> }
  pub fn parse(s: &str) -> Result<Version, String>
  pub fn to_string(v: &Version) -> String
  pub fn compare(a: &Version, b: &Version) -> std::cmp::Ordering
  pub fn bump_major / bump_minor / bump_patch

Rules (violations fail the assignment even if tests pass):
  no unsafe, no extra crates, no calling Python, no todo!/unimplemented!/panic!

Do not guess. Read the Python with grep_python / read_python, then write Rust.
Port tests from reference/test_*.py into #[cfg(test)] — cargo test is scored.

Pitfalls first drafts always miss (verify each against the source, not memory):
  * parse is strict SemVer 2.0: require X.Y.Z. Reject leading zeros on
    major/minor/patch and on *numeric* prerelease ids. Build ids MAY have
    leading zeros. No 'v' prefix, no whitespace, no empty identifiers.
  * compare IGNORES build metadata entirely.
  * a version WITH a prerelease is lower than the same version without.
  * numeric prerelease identifiers rank below non-numeric ones.
  * bump_* always increment the field and DROP prerelease and build.
    bump_patch("1.0.0-alpha") is 1.0.1, not 1.0.0. Check Version.bump_patch,
    not next_version — they differ on prereleases.
  * own Strings on Version; do not .clone() or .unwrap() your way out of the
    borrow checker. Prefer &str locally.

Workflow (stay cheap — budget is 40 model calls):
  1. Read the Python parse / compare / bump functions and the test files.
  2. Write a complete lib.rs (or patch_rust for a surgical fix).
  3. cargo_build. Fix compile errors with patch_rust, not a full rewrite.
  4. cargo_test. Add failing cases you actually observed.
  5. evaluate (n=60 while iterating). Use probe for a single harness command.
  6. If the score drops, the scaffold rolls back to the best lib.rs — do not
     fight it; fix the regression and evaluate again.
  7. Stop when evaluate is ~100%, the semver.org chain passes, cargo test
     is green, and there are no rule violations. Call nothing else.

Persist short working notes with write_notes so later steps do not depend
on a huge chat transcript. Prefer patch_rust over rewrite. Never dump the
whole file into chat.
"""


# ============================================================== TODO 3
def build_context(history: list[dict], step: int) -> list[dict]:
    """Turn the full history into the messages you actually send.

    WRITE     notes + best-lib live on disk, not in the window
    SELECT    keep system, the task, a memory card, and the last few turns
    COMPRESS  older tool dumps become one-line stubs
    ISOLATE   cargo/evaluate output is trimmed to the signal
    """
    system, task, rest = _split_history(history)
    memory = _memory_card(history, step)
    compressed = [_compress_msg(m) for m in rest[:-RECENT_TURNS]] if len(rest) > RECENT_TURNS else []
    recent = [_trim_msg(m, TOOL_CHARS) for m in rest[-RECENT_TURNS:]]
    out = [system, task]
    if memory:
        out.append({"role": "user", "content": memory})
    if compressed:
        out.append({
            "role": "user",
            "content": "Earlier steps (compressed, do not re-do them):\n" + _summarise(compressed),
        })
    out.extend(recent)
    return out


def _split_history(history: list[dict]) -> tuple[dict, dict, list[dict]]:
    system = history[0] if history and history[0].get("role") == "system" else {"role": "system", "content": system_prompt()}
    task = next((m for m in history if m.get("role") == "user"), {"role": "user", "content": "Translate."})
    skip = {id(system), id(task)}
    rest = [m for m in history if id(m) not in skip]
    return system, task, rest


def _memory_card(history: list[dict], step: int) -> str:
    scores = _score_history(history)
    last = scores[-1] if scores else None
    best = max(scores) if scores else None
    notes = NOTES.read_text()[:MEMORY_NOTES] if NOTES.exists() else "(none)"
    lib_bytes = LIB.stat().st_size if LIB.exists() else 0
    last_eval = _last_tool(history, "evaluate")
    last_build = _last_tool(history, "cargo_build")
    bits = [
        f"[working memory — step {step}]",
        f"last_score={last}  best_score={best}  lib.rs={lib_bytes}B",
        f"notes:\n{notes}",
    ]
    if last_eval:
        bits.append("last evaluate (trimmed):\n" + _trim(last_eval, 900))
    if last_build and _looks_like_error(last_build):
        bits.append("last cargo_build errors:\n" + _cargo_signal(last_build))
    return "\n".join(bits)


def _compress_msg(m: dict) -> dict:
    role = m.get("role")
    if role == "tool":
        name = m.get("name") or "?"
        body = m.get("content") or ""
        if name in {"read_python", "read_rust", "read_python_slice"}:
            return {"role": "tool", "name": name, "content": f"[{name}: {len(body)} chars — reread if needed]"}
        if name in {"cargo_build", "cargo_test"}:
            return {"role": "tool", "name": name, "content": _cargo_signal(body)}
        if name == "evaluate":
            return {"role": "tool", "name": name, "content": _eval_signal(body)}
        return {"role": "tool", "name": name, "content": _trim(body, 240)}
    if role == "assistant":
        names = [c.get("name") for c in (m.get("tool_calls") or [])]
        text = (m.get("content") or "").strip()
        summary = (text.splitlines() or [""])[0][:160]
        return {"role": "assistant", "content": summary, "tool_calls": m.get("tool_calls") or [],
                "_stub": f"called {names or 'no tools'}"}
    return {"role": m.get("role") or "user", "content": _trim(m.get("content") or "", 240)}


def _trim_msg(m: dict, limit: int) -> dict:
    if m.get("role") != "tool":
        return m
    name = m.get("name") or ""
    body = m.get("content") or ""
    if name in {"cargo_build", "cargo_test"}:
        body = _cargo_signal(body)
    elif name == "evaluate":
        body = _eval_signal(body)
    elif name in {"read_python", "read_rust"} and len(body) > limit:
        body = body[:limit] + f"\n...[{len(body)} chars total; use a slice/grep]"
    else:
        body = _trim(body, limit)
    return {**m, "content": body}


def _summarise(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        role = m.get("role")
        if role == "tool":
            lines.append(f"- {m.get('name')}: {_one_line(m.get('content') or '')}")
        elif role == "assistant":
            calls = [c.get("name") for c in (m.get("tool_calls") or [])]
            lines.append(f"- assistant {calls or 'talk'}: {_one_line(m.get('content') or '')}")
    return "\n".join(lines[-24:])


def _trim(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + f"\n...[{len(s)} chars truncated]"


def _one_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()[:180]


def _looks_like_error(s: str) -> bool:
    return bool(re.search(r"error(\[|:)|FAILED|FAILED|panic", s))


def _cargo_signal(s: str) -> str:
    errs = [ln for ln in s.splitlines() if re.search(r"error(\[|:)|FAILED|failed|warning: unused", ln)]
    if errs:
        tail = "\n".join(s.splitlines()[-30:])
        return _trim("\n".join(errs[:40]) + "\n---\n" + tail, TOOL_CHARS)
    return _trim(s, 400)


def _eval_signal(s: str) -> str:
    keep = []
    for ln in s.splitlines():
        if re.search(r"\[(1|2|3|4)\]|OVERALL|correctness|precedence|VIOLATION|First failures|^  - ", ln):
            keep.append(ln)
    return _trim("\n".join(keep) if keep else s, TOOL_CHARS)


# ============================================================== TODO 4
def should_stop(history: list[dict], step: int, budget: int, last_score: float | None) -> tuple[bool, str]:
    """Return (stop?, why)."""
    if step >= budget:
        return True, f"budget exhausted ({budget} model calls)"

    scores = _score_history(history)
    score = last_score if last_score is not None else (scores[-1] if scores else None)
    last_eval = _last_tool(history, "evaluate") or ""
    chain_ok = "semver.org precedence chain: PASS" in last_eval
    violations = "RULE VIOLATIONS" in last_eval
    tests_ok = _cargo_tests_green(history)

    # Done means measured done, not a speech act.
    if score is not None and score >= 99.5 and chain_ok and tests_ok and not violations:
        return True, f"done: {score:.1f}% differential, spec chain pass, cargo test green"

    if _claimed_done(history) and score is not None and score >= 98.0 and chain_ok and not violations:
        return True, f"agent claimed done and evaluate agrees ({score:.1f}%)"

    if _no_tool_streak(history) >= 2:
        if score is not None and score >= 98.0 and chain_ok:
            return True, "no more tool calls after a high score — treating as done"
        return True, "stuck: two consecutive model turns with no tool call"

    if _repeat_streak(history) >= 3:
        return True, "stuck: identical tool call repeated 3 times"

    # Plateau only after we have actually evaluated a few times.
    if len(scores) >= 3 and score is not None:
        recent = scores[-3:]
        if max(recent) - min(recent) < 0.05 and score >= 90.0 and chain_ok:
            return True, f"plateau at {score:.1f}% for 3 evaluates (spec chain pass)"
        if max(recent) - min(recent) < 0.05 and score < 40.0 and step >= 20:
            return True, f"stuck at {score:.1f}% with no movement — stop wasting budget"

    return False, ""


def _score_history(history: list[dict]) -> list[float]:
    out = []
    for m in history:
        if m.get("role") == "tool" and m.get("name") == "evaluate":
            sc = _parse_score(m.get("content") or "")
            if sc is not None:
                out.append(sc)
    return out


def _parse_score(text: str) -> float | None:
    m = re.search(r"correctness\s+([\d.]+)%", text)
    if m:
        return float(m.group(1))
    m = re.search(r"OVERALL\s+\d+/\d+\s+([\d.]+)%", text)
    return float(m.group(1)) if m else None


def _last_tool(history: list[dict], name: str) -> str | None:
    for m in reversed(history):
        if m.get("role") == "tool" and m.get("name") == name:
            return m.get("content") or ""
    return None


def _cargo_tests_green(history: list[dict]) -> bool:
    text = _last_tool(history, "cargo_test") or ""
    m = re.search(r"(\d+) passed; (\d+) failed", text)
    return bool(m and int(m.group(1)) > 0 and int(m.group(2)) == 0)


def _claimed_done(history: list[dict]) -> bool:
    for m in reversed(history):
        if m.get("role") == "assistant":
            if m.get("tool_calls"):
                return False
            return bool(re.search(r"\b(done|finished|complete|ship it)\b", (m.get("content") or ""), re.I))
    return False


def _no_tool_streak(history: list[dict]) -> int:
    n = 0
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            break
        n += 1
    return n


def _repeat_streak(history: list[dict]) -> int:
    sigs = []
    for m in history:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            sigs.append((c.get("name"), json.dumps(c.get("arguments") or {}, sort_keys=True)[:200]))
    if not sigs:
        return 0
    last, n = sigs[-1], 1
    for s in reversed(sigs[:-1]):
        if s == last:
            n += 1
        else:
            break
    return n


# ================================================================== tools
def _run(cmd, cwd=None, timeout=180):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s"


def t_read_python(_args):
    """The source to translate. Prefer grep_python / read_python_slice after the first look."""
    return PYSRC.read_text() if PYSRC.exists() else "reference/version.py missing - run fetch_source.py"


def t_read_python_slice(args):
    start = max(1, int(args.get("start") or 1))
    end = int(args.get("end") or start + 80)
    if not PYSRC.exists():
        return "reference/version.py missing"
    lines = PYSRC.read_text().splitlines()
    end = min(end, len(lines))
    chunk = "\n".join(f"{i}:{lines[i-1]}" for i in range(start, end + 1))
    return f"{PYSRC.name} lines {start}-{end}/{len(lines)}\n{chunk}"


def t_grep_python(args):
    pat = args.get("pattern") or ""
    if not pat:
        return "pattern required"
    try:
        rx = re.compile(pat)
    except re.error as e:
        return f"invalid regex: {e}"
    if not PYSRC.exists():
        return "reference/version.py missing"
    hits = []
    for i, ln in enumerate(PYSRC.read_text().splitlines(), 1):
        if rx.search(ln):
            hits.append(f"{i}:{ln}")
            if len(hits) >= 40:
                hits.append("...truncated")
                break
    return "\n".join(hits) or "(no matches)"


def t_read_tests(args):
    name = args.get("file") or "test_parsing.py"
    path = HERE / "reference" / pathlib.Path(name).name
    if not path.exists():
        return f"missing {path.name}; try test_parsing.py, test_compare.py, test_bump.py"
    start = max(1, int(args.get("start") or 1))
    lines = path.read_text().splitlines()
    end = min(int(args.get("end") or start + 100), len(lines))
    return f"{path.name} {start}-{end}/{len(lines)}\n" + "\n".join(
        f"{i}:{lines[i-1]}" for i in range(start, end + 1)
    )


def t_read_rust(_args):
    return LIB.read_text()


def t_write_rust(args):
    """Overwrite rust/src/lib.rs. `content` must be the WHOLE file."""
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        return "write_rust: 'content' must be a non-empty string"
    LOGS.mkdir(exist_ok=True)
    if LIB.exists():
        (LOGS / "prev-lib.rs").write_text(LIB.read_text())
    LIB.write_text(content)
    return f"wrote {len(content)} bytes to rust/src/lib.rs"


def t_patch_rust(args):
    """Replace the first occurrence of `old` with `new` in lib.rs."""
    old, new = args.get("old"), args.get("new")
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        return "patch_rust: need string fields 'old' and 'new'"
    src = LIB.read_text()
    n = src.count(old)
    if n == 0:
        return "patch_rust: 'old' not found (read_rust and copy exact text)"
    if n > 1:
        return f"patch_rust: 'old' matches {n} times — make it unique"
    LOGS.mkdir(exist_ok=True)
    (LOGS / "prev-lib.rs").write_text(src)
    LIB.write_text(src.replace(old, new, 1))
    return f"patched 1 occurrence ({len(old)} -> {len(new)} chars)"


def t_cargo_build(_args):
    return _run(["cargo", "build", "--release"], cwd=RUST)


def t_cargo_test(_args):
    return _run(["cargo", "test", "--release"], cwd=RUST)


def t_evaluate(args):
    """Practice seed only. The grading seed is different - do not tune to this.

    Rolls back to the best-scoring lib.rs if this run is worse.
    """
    n = int(args.get("n") or 60)
    n = max(20, min(n, 120))
    LOGS.mkdir(exist_ok=True)
    current = LIB.read_text() if LIB.exists() else ""
    out = _run([sys.executable, str(HERE / "evaluate.py"), "--n", str(n)], cwd=HERE)
    score = _parse_score(out)
    best = None
    if (LOGS / "best-score.txt").exists():
        try:
            best = float((LOGS / "best-score.txt").read_text().strip())
        except ValueError:
            best = None
    extra = ""
    if score is not None:
        if best is None or score > best + 0.001:
            BEST.write_text(current)
            (LOGS / "best-score.txt").write_text(str(score))
            extra = f"\n[scaffold] new best score {score:.2f}% saved"
        elif best is not None and score + 0.05 < best and BEST.exists():
            LIB.write_text(BEST.read_text())
            extra = (
                f"\n[scaffold] score {score:.2f}% < best {best:.2f}% — "
                "rolled rust/src/lib.rs back to the best snapshot. "
                "Fix the regression; do not resume from the worse file."
            )
    return out + extra


def t_probe(args):
    """Run one harness command, e.g. 'parse 1.0.0-alpha' or 'compare 1.0.0-alpha 1.0.0'."""
    cmd = (args.get("command") or "").strip()
    if not cmd:
        return "probe: pass command like 'parse 1.0.0'"
    if not BIN.exists():
        build = _run(["cargo", "build", "--release"], cwd=RUST)
        if not BIN.exists():
            return "probe: harness missing\n" + build
    return _probe_once(cmd)


def _probe_once(cmd: str) -> str:
    try:
        p = subprocess.run([str(BIN)], input=cmd + "\n", capture_output=True,
                           text=True, timeout=20)
        return (p.stdout or p.stderr or "(no output)").strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def t_write_notes(args):
    text = args.get("content")
    if not isinstance(text, str):
        return "write_notes: 'content' string required"
    LOGS.mkdir(exist_ok=True)
    NOTES.write_text(text)
    return f"wrote {len(text)} chars of notes"


def t_read_notes(_args):
    return NOTES.read_text() if NOTES.exists() else "(no notes yet)"


# TODO 5: finer actions — slice/grep the Python, patch instead of rewrite,
# probe a single harness command, persist notes outside the window.
TOOLS = [
    dict(name="read_python", description="Read all of reference/version.py. Prefer grep_python or read_python_slice after the first full read.",
         parameters={"type": "object", "properties": {}}, fn=t_read_python),
    dict(name="read_python_slice", description="Read a line range of reference/version.py (1-based, inclusive).",
         parameters={"type": "object", "required": ["start", "end"],
                     "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}}}, fn=t_read_python_slice),
    dict(name="grep_python", description="Regex search over reference/version.py. Returns matching lines with numbers.",
         parameters={"type": "object", "required": ["pattern"],
                     "properties": {"pattern": {"type": "string"}}}, fn=t_grep_python),
    dict(name="read_tests", description="Read a slice of a reference test file (test_parsing.py, test_compare.py, test_bump.py).",
         parameters={"type": "object",
                     "properties": {"file": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}}, fn=t_read_tests),
    dict(name="read_rust", description="Read the current rust/src/lib.rs.",
         parameters={"type": "object", "properties": {}}, fn=t_read_rust),
    dict(name="write_rust", description="Overwrite rust/src/lib.rs with the complete file contents. Use patch_rust for small fixes.",
         parameters={"type": "object", "required": ["content"],
                     "properties": {"content": {"type": "string"}}}, fn=t_write_rust),
    dict(name="patch_rust", description="Replace exactly one occurrence of 'old' with 'new' in rust/src/lib.rs.",
         parameters={"type": "object", "required": ["old", "new"],
                     "properties": {"old": {"type": "string"}, "new": {"type": "string"}}}, fn=t_patch_rust),
    dict(name="cargo_build", description="Compile the crate. Returns compiler errors.",
         parameters={"type": "object", "properties": {}}, fn=t_cargo_build),
    dict(name="cargo_test", description="Run the crate's own tests.",
         parameters={"type": "object", "properties": {}}, fn=t_cargo_test),
    dict(name="evaluate", description="Differential evaluation on the practice seed. Optional n (20-120, default 60). Rolls back if the score drops.",
         parameters={"type": "object", "properties": {"n": {"type": "integer"}}}, fn=t_evaluate),
    dict(name="probe", description="Run one harness command (parse X / compare A B / bump major|minor|patch X / format X).",
         parameters={"type": "object", "required": ["command"],
                     "properties": {"command": {"type": "string"}}}, fn=t_probe),
    dict(name="write_notes", description="Persist short working memory to disk so later context can drop the chat history.",
         parameters={"type": "object", "required": ["content"],
                     "properties": {"content": {"type": "string"}}}, fn=t_write_notes),
    dict(name="read_notes", description="Read the persisted notes.",
         parameters={"type": "object", "properties": {}}, fn=t_read_notes),
]
BY_NAME = {t["name"]: t for t in TOOLS}
SCHEMAS = [{k: t[k] for k in ("name", "description", "parameters")} for t in TOOLS]


# =================================================================== loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40, help="max model calls (graded cap: 40)")
    ap.add_argument("--task", default="Translate reference/version.py into rust/src/lib.rs.")
    a = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    def rec(**kw):
        with log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **kw}) + "\n")

    history = [{"role": "system", "content": system_prompt()},
               {"role": "user",   "content": a.task}]
    rec(event="start", budget=a.budget, task=a.task)

    step, last_score = 0, None
    while True:
        stop, why = should_stop(history, step, a.budget, last_score)
        if stop:
            print(f"\n[stop] {why}")
            rec(event="stop", reason=why, steps=step, last_score=last_score)
            break

        step += 1
        reply = call_model(build_context(history, step), SCHEMAS)
        rec(event="model", step=step, reply=reply)

        if reply.get("text"):
            print(f"[{step}] {reply['text'][:200]}")
        hist_asst = {"role": "assistant", "content": reply.get("text") or "",
                     "tool_calls": reply.get("tool_calls", [])}
        if reply.get("_gemini_parts"):
            hist_asst["_gemini_parts"] = reply["_gemini_parts"]
        history.append(hist_asst)

        calls = reply.get("tool_calls") or []
        if not calls:
            # No tool call: should_stop decides if this is done or stuck.
            continue

        for c in calls:
            tool = BY_NAME.get(c["name"])
            if tool is None:
                out = f"unknown tool {c['name']!r}"
            else:
                try:
                    out = tool["fn"](c.get("arguments") or {})
                except Exception as e:
                    out = f"tool {c['name']} raised {type(e).__name__}: {e}"
            print(f"      -> {c['name']}: {str(out).splitlines()[0][:120] if out else ''}")
            rec(event="tool", step=step, name=c["name"], output=str(out)[:4000])
            history.append({"role": "tool", "name": c["name"], "content": str(out)})
            if c["name"] == "evaluate":
                sc = _parse_score(str(out))
                if sc is not None:
                    last_score = sc

    print(f"\ntrajectory: {log}")
    print("final score:")
    subprocess.run([sys.executable, str(HERE / "evaluate.py")], cwd=HERE)


if __name__ == "__main__":
    main()
