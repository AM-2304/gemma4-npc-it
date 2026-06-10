#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PIPPA Malformed Placeholder Fix — Step 03
Handles ALL known malformed placeholder variants from PIPPA.

Usage:
    python data/process_pippa/03_replace_weird_names.py \
        [--input data/processed/pippa_placeholders_replaced.jsonl] \
        [--output data/processed/pippa_clean.jsonl]
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Catch-all regex patterns
CHAR_REGEX = re.compile(r'\{+\[?[Cc][Hh]?[Aa][Rr]\]?\}+')
USER_REGEX = re.compile(r'\{+\[?[Uu][Ss][Ee][Rr][_\d]*\]?\}+')
CHAR_BRACKET = re.compile(r'\[+[Cc][Hh]?[Aa][Rr]\]+')
USER_BRACKET = re.compile(r'\[+[Uu][Ss][Ee][Rr][_\d]*\]+')
RESIDUAL = re.compile(r'\{\{[^}]*\}\}')

CHAR_LITERALS = [
    '{char}}', '{{char}', '{[Char}}', '{{Char}}', '[char]', '[Char]',
]
USER_LITERALS = [
    '{user}}', '{{user}', '{{User}}', '[user]', '[User]',
    '{{random_user4}}',
] + [f'{{{{user_{i}}}}}' for i in range(1, 10)]


def fix_text(text, bot_name, user_name, counters):
    if text is None:
        return text
    for p in CHAR_LITERALS:
        if p in text:
            counters[f"char:{p}"] = counters.get(f"char:{p}", 0) + text.count(p)
            text = text.replace(p, bot_name)
    for p in USER_LITERALS:
        if p in text:
            counters[f"user:{p}"] = counters.get(f"user:{p}", 0) + text.count(p)
            text = text.replace(p, user_name)
    for rx, name, key in [
        (CHAR_REGEX, bot_name, "char:regex"),
        (USER_REGEX, user_name, "user:regex"),
        (CHAR_BRACKET, bot_name, "char:bracket"),
        (USER_BRACKET, user_name, "user:bracket"),
    ]:
        for m in rx.findall(text):
            text = text.replace(m, name)
            counters[key] = counters.get(key, 0) + 1
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/pippa_placeholders_replaced.jsonl")
    parser.add_argument("--output", default="data/processed/pippa_clean.jsonl")
    parser.add_argument("--names", default="data/process_pippa/first_names.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.names, "r") as f:
        names = json.load(f)

    with open(args.input, "r") as f:
        total = sum(1 for _ in f)

    counters = {}
    residuals = {}
    written = 0

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for idx, line in enumerate(tqdm(fin, total=total, desc="Fixing")):
            row = json.loads(line.strip())
            bn = row.get("bot_name", "Character")
            un = rng.choice(names)
            for f in ["bot_description", "bot_definitions"]:
                if row.get(f):
                    row[f] = fix_text(row[f], bn, un, counters)
            if row.get("conversation") and isinstance(row["conversation"], list):
                for turn in row["conversation"]:
                    if "message" in turn and isinstance(turn["message"], str):
                        turn["message"] = fix_text(turn["message"], bn, un, counters)
            # Check residuals
            row_res = []
            for f in ["bot_description", "bot_definitions"]:
                if row.get(f):
                    row_res.extend(RESIDUAL.findall(row[f]))
            if row.get("conversation"):
                for turn in row["conversation"]:
                    if "message" in turn:
                        row_res.extend(RESIDUAL.findall(turn.get("message", "")))
            if row_res:
                residuals[idx] = list(set(row_res))
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"Processed {written} rows")
    for p, c in sorted(counters.items(), key=lambda x: -x[1]):
        logger.info(f"  {p:35s}: {c:6d}")
    if residuals:
        logger.warning(f"{len(residuals)} rows have unresolved patterns")
        for idx, pats in list(residuals.items())[:10]:
            logger.warning(f"  Row {idx}: {pats}")
    else:
        logger.info("✅ No unresolved placeholder patterns remain")
    logger.info(f"✅ Output: {args.output}")


if __name__ == "__main__":
    main()
