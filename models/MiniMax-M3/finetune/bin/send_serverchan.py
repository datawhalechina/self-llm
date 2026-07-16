#!/usr/bin/env python3
"""Send a ServerChan notification using only Python standard libraries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ENV_KEYS = ("SERVERCHAN_SENDKEY", "SERVERCHAN_KEY", "SCT_SENDKEY", "SC_SENDKEY")
DEFAULT_ENDPOINT_BASE = "https://sctapi.ftqq.com"


def resolve_sendkey(value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    for name in ENV_KEYS:
        env_value = os.environ.get(name)
        if env_value and env_value.strip():
            return env_value.strip()
    return ""


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def read_body(args: argparse.Namespace) -> str:
    body = args.desp or ""
    if args.body_file:
        body_path = Path(args.body_file)
        body = body_path.read_text(encoding="utf-8")
    return body


def build_payload(args: argparse.Namespace) -> dict[str, str]:
    title = args.title or args.title_arg
    if not title:
        raise ValueError("title is required; pass --title or a positional title")

    payload = {"title": title, "desp": read_body(args)}
    optional_fields = {
        "short": args.short,
        "tags": args.tags,
        "channel": args.channel,
        "openid": args.openid,
    }
    for key, value in optional_fields.items():
        if value:
            payload[key] = value
    if args.noip:
        payload["noip"] = "1"
    return payload


def post_form(url: str, payload: dict[str, str], timeout: float) -> tuple[int, str]:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def parse_response(status: int, body: str) -> tuple[bool, str]:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return 200 <= status < 300, body.strip()

    code = parsed.get("code")
    message = parsed.get("message") or parsed.get("msg") or body.strip()
    if code in (0, "0", None) and 200 <= status < 300:
        return True, str(message)
    return False, str(message)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a ServerChan/Server酱 notification.",
    )
    parser.add_argument("title_arg", nargs="?", help="notification title")
    parser.add_argument("--title", help="notification title")
    parser.add_argument("--desp", "--body", dest="desp", help="message body")
    parser.add_argument("--body-file", help="read message body from a UTF-8 file")
    parser.add_argument("--sendkey", help="ServerChan SendKey/AppKey")
    parser.add_argument("--endpoint-base", default=DEFAULT_ENDPOINT_BASE)
    parser.add_argument("--short", help="card summary/short description")
    parser.add_argument("--tags", help="ServerChan3 tags, separated by |")
    parser.add_argument("--channel", help="Turbo channels, separated by |")
    parser.add_argument("--openid", help="Turbo copy recipients, separated by , or |")
    parser.add_argument("--noip", action="store_true", help="hide caller IP where supported")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true", help="print payload summary without sending")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    sendkey = resolve_sendkey(args.sendkey)
    if not sendkey:
        names = ", ".join(ENV_KEYS)
        print(f"Missing SendKey/AppKey. Pass --sendkey or set one of: {names}", file=sys.stderr)
        return 2

    try:
        payload = build_payload(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    endpoint = args.endpoint_base.rstrip("/")
    url = f"{endpoint}/{urllib.parse.quote(sendkey, safe='')}.send"

    if args.dry_run:
        visible_payload = {key: value for key, value in payload.items() if value}
        print(
            json.dumps(
                {
                    "sendkey": mask_secret(sendkey),
                    "url": f"{endpoint}/{mask_secret(sendkey)}.send",
                    "payload": visible_payload,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        status, body = post_form(url, payload, args.timeout)
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        print(f"ServerChan HTTP error {exc.code}: {response_body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"ServerChan request failed: {exc.reason}", file=sys.stderr)
        return 1
    except TimeoutError:
        print("ServerChan request timed out", file=sys.stderr)
        return 1

    ok, message = parse_response(status, body)
    if ok:
        print(f"ServerChan send succeeded: {message}")
        return 0
    print(f"ServerChan send failed: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
