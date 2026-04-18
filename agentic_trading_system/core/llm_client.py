from __future__ import annotations

import json
import os
from typing import Type

from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel


class LLMClient:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider.lower()
        self.model_name = model_name
        if self.provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
        else:
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_json(self, prompt: str, schema: Type[BaseModel], temperature: float = 0.1) -> dict:
        schema_fields = ", ".join(schema.model_fields.keys())
        guided_prompt = (
            f"{prompt}\n\n"
            "Return exactly one valid JSON object.\n"
            f"Allowed top-level keys: {schema_fields}.\n"
            "Do not include markdown fences or any extra keys."
        )
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Do not wrap it in markdown fences.",
                    },
                    {"role": "user", "content": guided_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            content = response.choices[0].message.content or "{}"
            return schema(**json.loads(content)).model_dump()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=guided_prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )
        return schema(**json.loads(response.text)).model_dump()
