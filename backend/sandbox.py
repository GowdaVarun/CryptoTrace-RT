"""
sandbox.py — CryptoTrace dynamic analysis sandbox wrapper.

Runs an uploaded binary inside a locked-down Docker container and extracts
performance / syscall features.  Never executes uploads on the host directly.
"""

import os
import re
import shutil
import subprocess
import time

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DYNAMIC_FEATURES = {
    "dyn_exec_time": 0.0,
    "dyn_instructions": 0.0,
    "dyn_cycles": 0.0,
    "dyn_branches": 0.0,
    "dyn_branch_misses": 0.0,
    "dyn_total_syscalls": 0.0,
    "dyn_unique_syscalls": 0.0,
    "dyn_getrandom_calls": 0.0,
    "dyn_read_calls": 0.0,
    "dyn_write_calls": 0.0,
    "dyn_ipc": 0.0,
    "dyn_branch_miss_ratio": 0.0,
}

STATIC_DETECTION_FEATURES = (
    "n_crypto_imports",
    "n_crypto_import_categories",
    "crypto_import_ratio",
    "has_crypto_library",
    "n_crypto_libraries",
    "crypto_constant_hits",
    "rodata_crypto_hits",
    "n_crypto_strings",
    "crypto_string_ratio",
)

# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def has_static_detection(static_features: dict) -> bool:
    """Return True when explicit static crypto indicators are present."""
    return any(
        float(static_features.get(name, 0.0) or 0.0) > 0
        for name in STATIC_DETECTION_FEATURES
    )


def empty_dynamic_features() -> dict:
    return dict(DEFAULT_DYNAMIC_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Docker helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sandbox_image() -> str:
    return os.getenv("CRYPTOTRACE_SANDBOX_IMAGE", "cryptotrace-sandbox:latest")


def _docker_base_cmd(binary_path: str) -> list[str]:
    """Return the `docker run` prefix with all security constraints applied."""
    image        = _sandbox_image()
    cpu_limit    = os.getenv("CRYPTOTRACE_SANDBOX_CPUS",   "0.5")
    memory_limit = os.getenv("CRYPTOTRACE_SANDBOX_MEMORY", "128m")
    pids_limit   = os.getenv("CRYPTOTRACE_SANDBOX_PIDS",   "64")

    return [
        "docker", "run",
        "--rm",
        # ── Network ──────────────────────────────────────────────────────────
        "--network", "none",
        # ── Resource limits ──────────────────────────────────────────────────
        "--cpus",        cpu_limit,
        "--memory",      memory_limit,
        "--pids-limit",  pids_limit,
        # ── Capability hardening ─────────────────────────────────────────────
        "--cap-drop",    "ALL",
        "--cap-add",     "PERFMON",   # needed for perf stat
        "--security-opt", "no-new-privileges",
        # ── Filesystem hardening ─────────────────────────────────────────────
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        # ── User ─────────────────────────────────────────────────────────────
        "--user", "65534:65534",
        "--workdir", "/sandbox",
        # ── Sample mount (read-only) ──────────────────────────────────────────
        "-v", f"{os.path.abspath(binary_path)}:/sandbox/sample:ro",
        image,
    ]


def _docker_available(image: str) -> tuple[bool, str]:
    """
    Return (True, 'available') when Docker is reachable and the sandbox image
    exists locally.  Returns (False, reason) otherwise — never raises.
    """
    if not shutil.which("docker"):
        return False, "docker_not_found"

    try:
        subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
    except Exception:
        return False, "docker_unavailable"

    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
    except Exception:
        return False, f"image_missing:{image}"

    return True, "available"


def _run_container(binary_path: str, args: list[str], timeout: int | float) -> subprocess.CompletedProcess:
    """Spin up the sandbox container with `args` appended after the image name."""
    cmd = _docker_base_cmd(binary_path) + args
    return subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_perf(perf_output: str) -> dict:
    """
    Parse `perf stat -x, ...` CSV output (written to stderr by perf).

    Each line looks like:
        123456,,instructions,1234,100.00,,
    or (newer kernels):
        123456,<not counted>,instructions,...
    """
    metrics = {
        "dyn_instructions":   0.0,
        "dyn_cycles":         0.0,
        "dyn_branches":       0.0,
        "dyn_branch_misses":  0.0,
    }

    EVENT_MAP = {
        "instructions":  "dyn_instructions",
        "cycles":        "dyn_cycles",
        "branches":      "dyn_branches",
        "branch-misses": "dyn_branch_misses",
    }

    for line in perf_output.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue

        val_str = parts[0].strip()
        event   = parts[2].strip()

        if not val_str or val_str in ("<not counted>", "<not supported>"):
            continue

        # Strip any thousand-separators that some perf versions emit
        val_str = val_str.replace(".", "").replace(",", "")
        try:
            value = float(val_str)
        except ValueError:
            continue

        key = EVENT_MAP.get(event)
        if key:
            metrics[key] = value

    return metrics


# Pre-compiled pattern for the strace -c summary table body lines:
#   % time  seconds  usecs/call  calls  errors  syscall
#   ------  -------  ----------  -----  ------  -------
#    99.12   0.000123         12    456           read
_STRACE_ROW = re.compile(
    r"^\s*[\d.]+\s+[\d.]+\s+\d+\s+(?P<calls>\d+)\s+(?:\d+\s+)?(?P<syscall>\w+)\s*$"
)


def _parse_strace(strace_output: str) -> dict:
    """
    Parse `strace -c` summary output (written to stderr).

    The summary block is bracketed by lines of dashes; we use a regex on every
    line inside that block instead of relying on brittle column splitting.
    """
    metrics = {
        "dyn_total_syscalls":   0.0,
        "dyn_unique_syscalls":  0.0,
        "dyn_getrandom_calls":  0.0,
        "dyn_read_calls":       0.0,
        "dyn_write_calls":      0.0,
    }

    in_table = False
    for line in strace_output.splitlines():
        stripped = line.strip()

        # The summary block is delimited by two "------" ruler lines.
        if re.match(r"^-{6,}", stripped):
            in_table = not in_table
            continue

        if not in_table:
            continue

        # Skip the header row
        if stripped.startswith("%"):
            continue

        m = _STRACE_ROW.match(line)
        if not m:
            continue

        calls   = float(m.group("calls"))
        syscall = m.group("syscall")

        metrics["dyn_total_syscalls"]  += calls
        metrics["dyn_unique_syscalls"] += 1

        if syscall == "getrandom":
            metrics["dyn_getrandom_calls"] += calls
        elif syscall == "read":
            metrics["dyn_read_calls"] += calls
        elif syscall == "write":
            metrics["dyn_write_calls"] += calls

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_in_docker_sandbox(binary_path: str, timeout_seconds: int = 2) -> tuple[dict, dict]:
    """
    Execute *binary_path* inside a locked-down Docker container and return
    (dynamic_features, metadata).

    Three container invocations are performed in sequence:
      1. Plain execution — captures wall-clock time and return code.
      2. perf stat    — captures CPU counters (skipped if perf unavailable).
      3. strace -c    — captures syscall summary (skipped if strace unavailable).

    All three share the same Docker security constraints; only the entrypoint
    command differs.
    """
    image    = _sandbox_image()
    features = empty_dynamic_features()
    metadata = {
        "mode":         "docker",
        "image":        image,
        "status":       "not_started",
        "perf_status":  "not_run",
        "strace_status":"not_run",
        "return_code":  None,
    }

    # ── Pre-flight check ─────────────────────────────────────────────────────
    available, reason = _docker_available(image)
    if not available:
        metadata["status"] = reason
        return features, metadata

    tool_timeout = timeout_seconds + 3  # extra headroom for perf/strace wrapper

    # ── Step 1: plain execution (wall-clock time) ─────────────────────────────
    try:
        start  = time.monotonic()
        result = _run_container(binary_path, ["/sandbox/sample"], timeout_seconds)
        features["dyn_exec_time"] = time.monotonic() - start
        metadata["return_code"]   = result.returncode
        metadata["status"] = (
            "completed" if result.returncode == 0 else "completed_nonzero"
        )
    except subprocess.TimeoutExpired:
        features["dyn_exec_time"] = float(timeout_seconds)
        metadata["status"] = "timeout"
        return features, metadata   # no point running perf/strace after a timeout
    except Exception as exc:
        metadata["status"] = f"error:{exc.__class__.__name__}"
        return features, metadata

    # ── Step 2: perf stat ────────────────────────────────────────────────────
    # We check for the perf binary inside the container before invoking it so
    # the status accurately reflects whether perf is absent vs. failed.
    _PERF_CMD = (
        "perf stat -x, "
        "-e instructions,cycles,branches,branch-misses "
        "/sandbox/sample 2>&1"   # perf writes to stderr; redirect so we capture it
    )
    try:
        result = _run_container(
            binary_path,
            ["sh", "-c", f"command -v perf >/dev/null 2>&1 || exit 127; {_PERF_CMD}"],
            tool_timeout,
        )
        if result.returncode == 127:
            metadata["perf_status"] = "unavailable"
        elif result.returncode == 0:
            # perf stat -x, writes CSV to stderr; we redirected 2>&1 so it's in stderr too
            parsed = _parse_perf(result.stderr)
            if any(v > 0 for v in parsed.values()):
                features.update(parsed)
                metadata["perf_status"] = "completed"
            else:
                # Kernel may not expose perf counters (common in VMs / WSL)
                metadata["perf_status"] = "no_data"
        else:
            metadata["perf_status"] = f"failed_rc:{result.returncode}"
    except subprocess.TimeoutExpired:
        metadata["perf_status"] = "timeout"
    except Exception as exc:
        metadata["perf_status"] = f"error:{exc.__class__.__name__}"

    # ── Step 3: strace -c ────────────────────────────────────────────────────
    _STRACE_CMD = "strace -c /sandbox/sample"
    try:
        result = _run_container(
            binary_path,
            ["sh", "-c", f"command -v strace >/dev/null 2>&1 || exit 127; {_STRACE_CMD}"],
            tool_timeout,
        )
        if result.returncode == 127:
            metadata["strace_status"] = "unavailable"
        elif result.returncode in (0, 1):
            # strace -c exits with the tracee's exit code; 0 or 1 are both fine.
            parsed = _parse_strace(result.stderr)
            if parsed["dyn_total_syscalls"] > 0:
                features.update(parsed)
                metadata["strace_status"] = "completed"
            else:
                metadata["strace_status"] = "no_data"
        else:
            metadata["strace_status"] = f"failed_rc:{result.returncode}"
    except subprocess.TimeoutExpired:
        metadata["strace_status"] = "timeout"
    except Exception as exc:
        metadata["strace_status"] = f"error:{exc.__class__.__name__}"

    # ── Derived features ──────────────────────────────────────────────────────
    features["dyn_ipc"] = features["dyn_instructions"] / max(features["dyn_cycles"], 1)
    features["dyn_branch_miss_ratio"] = (
        features["dyn_branch_misses"] / max(features["dyn_branches"], 1)
    )

    return features, metadata