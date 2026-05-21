#!/usr/bin/env python3
"""Security hygiene scanner for tracked files.

Scans repository files for high-risk secrets and operational-metadata leaks
that must never land in a public repository (private keys, API tokens, raw
server IPs, SSH endpoints, private-key paths, provider pod details, etc.).

Usage:
    python scripts/security_scan_changed_files.py          # changed tracked files
    python scripts/security_scan_changed_files.py --all     # all tracked files

Exit status:
    0  no risky patterns found
    1  one or more risky patterns found
    2  invocation / environment error

Notes:
    - Standard library only.
    - Matched secret-looking values are redacted in output; the scanner never
      prints the sensitive value itself.
    - Binary and oversized files are skipped.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Files larger than this are skipped (likely data/binary, not source).
MAX_FILE_BYTES = 2_000_000

# Extensions that are almost always binary — skip outright.
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar",
    ".parquet", ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".so", ".pyc",
    ".ico", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mov",
}

# This scanner file legitimately contains the patterns it searches for.
SELF_NAME = "security_scan_changed_files.py"


# Each rule: (label, compiled_regex). Patterns are intentionally conservative;
# false positives are acceptable, leaked secrets are not.
def _build_rules() -> list[tuple[str, re.Pattern[str]]]:
    raw: list[tuple[str, str]] = [
        ("private-key-block", r"BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY"),
        ("github-token", r"gh[pousr]_[A-Za-z0-9_]{20,}"),
        ("openai-api-key", r"sk-[A-Za-z0-9_-]{20,}"),
        ("anthropic-api-key", r"sk-ant-[A-Za-z0-9_-]{20,}"),
        ("aws-access-key", r"AKIA[0-9A-Z]{16}"),
        ("aws-secret-env", r"AWS_SECRET_ACCESS_KEY"),
        ("openai-env", r"OPENAI_API_KEY"),
        ("anthropic-env", r"ANTHROPIC_API_KEY"),
        ("password-assignment", r"(?i)password\s*[=:]\s*\S+"),
        ("token-assignment", r"(?i)\btoken\s*[=:]\s*\S+"),
        ("api-key-assignment", r"(?i)api[_-]?key\s*[=:]\s*\S+"),
        ("apikey-assignment", r"(?i)\bapikey\s*[=:]\s*\S+"),
        ("secret-assignment", r"(?i)\bsecret\s*[=:]\s*\S+"),
        ("ssh-root-at-ip", r"root@(\d{1,3}\.){3}\d{1,3}"),
        ("scp-port", r"scp\s+-P\s+\d+"),
        ("sftp-url", r"sftp://"),
        ("ssh-port-flag", r"ssh\s+-p\s+\d+"),
        ("ssh-hostname-ip", r"HostName\s+(\d{1,3}\.){3}\d{1,3}"),
        ("ssh-identityfile", r"IdentityFile\s+\S+"),
        ("user-ssh-path", r"/Users/[^\s'\"]+/\.ssh/[^\s'\"]+"),
        # Flags non-standard / custom-named key paths under ~/.ssh (which reveal
        # operational layout) but not the default config / standard key names,
        # which are generic SSH guidance and leak nothing.
        (
            "home-ssh-custom-path",
            r"~/\.ssh/(?!(config|known_hosts|authorized_keys|"
            r"id_(ed25519|rsa|ecdsa|dsa)(\.pub)?)(\b|$))[A-Za-z0-9._-]+",
        ),
    ]
    return [(label, re.compile(pat)) for label, pat in raw]


# Detects a raw IPv4 with nearby SSH/SFTP/RunPod context on the same line.
_IP_RE = re.compile(r"(?<!\d)(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\d)")
_CONTEXT_RE = re.compile(r"(?i)(ssh|sftp|scp|runpod|hostname|pod|cyberduck|port|workspace)")

# Placeholder tokens — lines that only contain placeholders are not leaks.
_PLACEHOLDER_RE = re.compile(
    r"<(ssh-alias|remote-host|remote-port|remote-user|private-key-path|"
    r"workspace-path|provider-endpoint)>"
)

RULES = _build_rules()


def _is_valid_ipv4(match: re.Match[str]) -> bool:
    return all(0 <= int(g) <= 255 for g in match.groups())


def redact(value: str) -> str:
    """Redact a matched value so the scanner never echoes a secret."""
    value = value.strip()
    if len(value) <= 8:
        return "<redacted>"
    return f"{value[:4]}…<redacted:{len(value)} chars>"


def list_target_files(scan_all: bool) -> list[Path]:
    try:
        if scan_all:
            out = subprocess.run(
                ["git", "ls-files"],
                capture_output=True, text=True, check=True,
            ).stdout
        else:
            tracked_changed = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, check=True,
            ).stdout
            out = tracked_changed + "\n" + staged
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[security-scan] git invocation failed: {exc}", file=sys.stderr)
        sys.exit(2)

    seen: set[str] = set()
    files: list[Path] = []
    for name in out.splitlines():
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        files.append(Path(name))
    return files


def should_skip(path: Path) -> bool:
    if path.name == SELF_NAME:
        return True
    if path.suffix.lower() in BINARY_EXTS:
        return True
    if not path.is_file():
        return True
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return True
    except OSError:
        return True
    return False


def scan_line(line: str) -> list[tuple[str, str]]:
    """Return list of (label, redacted_value) findings for a single line."""
    if _PLACEHOLDER_RE.search(line) and not _IP_RE.search(line):
        # Pure placeholder line — nothing real to leak.
        return []

    findings: list[tuple[str, str]] = []
    for label, rx in RULES:
        m = rx.search(line)
        if m:
            findings.append((label, redact(m.group(0))))

    # IP + SSH/SFTP context heuristic.
    ip_match = _IP_RE.search(line)
    if ip_match and _is_valid_ipv4(ip_match) and _CONTEXT_RE.search(line):
        findings.append(("ip-with-ssh-context", redact(ip_match.group(0))))

    return findings


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    results: list[tuple[int, str, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                for label, redacted in scan_line(line):
                    results.append((lineno, label, redacted))
    except OSError as exc:
        print(f"[security-scan] could not read {path}: {exc}", file=sys.stderr)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan tracked files for secrets / operational-metadata leaks.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan all tracked files instead of only changed ones.",
    )
    args = parser.parse_args()

    files = list_target_files(args.all)
    total_findings = 0
    flagged_files = 0

    for path in files:
        if should_skip(path):
            continue
        findings = scan_file(path)
        if findings:
            flagged_files += 1
            for lineno, label, redacted in findings:
                total_findings += 1
                print(f"{path}:{lineno}: [{label}] {redacted}")

    if total_findings:
        scope = "all tracked" if args.all else "changed"
        print(
            f"\n[security-scan] FAIL — {total_findings} finding(s) across "
            f"{flagged_files} {scope} file(s).",
            file=sys.stderr,
        )
        print(
            "[security-scan] Redact the values above or move them to a "
            "local-only private runbook before committing.",
            file=sys.stderr,
        )
        return 1

    scope = "all tracked" if args.all else "changed"
    print(f"[security-scan] OK — no risky patterns in {scope} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
