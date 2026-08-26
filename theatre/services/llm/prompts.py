"""Prompts for grounded Bengali theatre-production generation."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """আপনি একজন অভিজ্ঞ বাংলা নাট্যকার, মঞ্চ-পরিচালক ও আলোক-পরিকল্পনাবিদ।
শুধু বৈধ JSON অবজেক্ট ফেরত দিন। Markdown code fence, ব্যাখ্যা বা JSON-এর বাইরের কোনো লেখা দেবেন না।
উদ্ধার করা উদাহরণগুলো কেবল রেফারেন্স; তাদের সংলাপ হুবহু কপি করবেন না।"""


def build_generation_prompt(context: str, response_schema: dict[str, Any]) -> str:
    schema = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
    return f"""নিচের RAG প্রসঙ্গ ব্যবহার করে একটি সম্পূর্ণ নতুন বাংলা থিয়েটার প্রযোজনা তৈরি করুন।

প্রতিটি দৃশ্যে সংলাপ, stage_directions, blocking এবং lighting থাকতে হবে।
Blocking-এর from ও to কেবল USL, USC, USR, CSL, CSC, CSR, DSL, DSC, DSR থেকে নিন।
Lighting-এর focus_zone-ও একই তালিকা থেকে নিন; RGB-এর প্রতিটি মান 0-255 এবং intensity 0-100 রাখুন।
প্রদত্ত JSON schema অক্ষরে অক্ষরে অনুসরণ করুন। শুধু JSON দিন।

RAG CONTEXT
{context}

REQUIRED JSON SCHEMA
{schema}"""
