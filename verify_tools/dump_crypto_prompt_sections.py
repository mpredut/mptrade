#!/usr/bin/env python3
"""Dumps formatted sections of auto_cryptotrade_executor prompt for quality evaluation."""

import sys
from verify_tools.check_crypto_prompt_quality import inspect_crypto_prompt


def main():
    res = inspect_crypto_prompt(sys.argv[1] if len(sys.argv) > 1 else ".")
    prompt = res["full_prompt"]
    lines = prompt.splitlines()

    cur_sec = "PROMPT_HEADER"
    sections = {cur_sec: []}

    for l in lines:
        if l.startswith("### "):
            cur_sec = l.strip()
            sections[cur_sec] = []
        elif l.startswith("INTERACTIVE THREAD RULES:"):
            cur_sec = "INTERACTIVE_RULES"
            sections[cur_sec] = []
        else:
            sections[cur_sec].append(l)

    for sec_name, sec_content in sections.items():
        print(f"\n{'=' * 80}")
        print(f"SECTION: {sec_name}")
        print(f"{'=' * 80}")
        for l in sec_content:
            print(l)


if __name__ == "__main__":
    main()
