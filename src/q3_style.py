#!/usr/bin/env python3
"""Style features, copied verbatim from vendor/persona-mech/src/m7_control_analyse.py:style (PREREG Q3 'persona-mech features')."""
import math
import re
HEDGE = re.compile(r"\b(perhaps|maybe|might|could|possibly|it seems|i think|arguably|somewhat|likely|generally|often|tend to)\b", re.I)
VERIFY = re.compile(r"\b(let me (?:verify|double[- ]check|check|confirm)|to verify|double[- ]check)\b", re.I)
FLATTER = re.compile(r"\b(great question|excellent question|good question|what a (?:great|wonderful)|you're absolutely right|i love that)\b", re.I)


def style(text):
    words = max(1, len(text.split())); lines = text.splitlines() or [""]
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    return {"log_len": math.log(words), "headers": sum(l.lstrip().startswith("#") for l in lines) / words * 100,
            "bullets": sum(bool(re.match(r"\s*([-*•]|\d+\.)\s", l)) for l in lines) / words * 100,
            "emdash": text.count("—") / words * 100, "hedges": len(HEDGE.findall(text)) / words * 100,
            "verify": len(VERIFY.findall(text)) / words * 100, "flattery": len(FLATTER.findall(text)) / words * 100,
            "exclaim": text.count("!") / words * 100, "questions": text.count("?") / words * 100,
            "bold": text.count("**") / 2 / words * 100, "sent_len": words / max(1, len(sents))}
