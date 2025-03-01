"""
title: AWS Bedrock Anthropic Pipeline
author: lentil32
date: 2025-02-25
version: 1.2
license: MIT
description: A pipeline for generating text and processing images using AWS Bedrock for Anthropic Claude models with multiple reasoning efforts.
requirements: boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, MAX_TOKENS
"""

import os
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
import boto3

# Import helper function - adjust the import path according to your project structure
try:
    from utils.pipelines.main import pop_system_message
except ImportError:
    # Fallback function if import fails
    def pop_system_message(messages):
        system_message = None
        filtered_messages = []

        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content", "")
            else:
                filtered_messages.append(message)

        return system_message, filtered_messages

# Mapping for reasoning effort budget tokens
REASONING_EFFORT_BUDGET_TOKEN_MAP = {
    "none": None,
    "low": 1_000,
    "medium": 8_000,
    "high": 32_000,
    "max": 64_000,
}

class Pipeline:
    """Pipeline class for AWS Bedrock Anthropic models"""

    class Valves(BaseModel):
        """Configuration for AWS Bedrock"""
        AWS_REGION: str = Field(default_factory=lambda: os.getenv('AWS_REGION', 'us-east-1'),
                             description="AWS region for Bedrock services")
        AWS_ACCESS_KEY: str = Field(default_factory=lambda: os.getenv('AWS_ACCESS_KEY', ''),
                                 description="AWS access key for authentication")
        AWS_SECRET_KEY: str = Field(default_factory=lambda: os.getenv('AWS_SECRET_KEY', ''),
                                 description="AWS secret key for authentication", exclude=True)
        MAX_TOKENS: int = Field(default_factory=lambda: int(os.getenv('MAX_TOKENS', '128_000')),
                                     description="Maximum tokens for response generation")
        REASONING_EFFORT: str = Field(default="none",
                                          description="Reasoning effort (none, low, medium, high, max)")
        TEMPERATURE: float = Field(default=1.0,
                                        description="Temperature for response generation")

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic_bedrock"
        self.name = "anthropic_bedrock/"

        # Initialize valves with AWS region from environment variable or default
        self.valves = self.Valves()

        # Set up the Bedrock runtime client
        self._initialize_bedrock_client()

    def _initialize_bedrock_client(self):
        """Initialize the Bedrock client with current credentials"""
        kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': self.valves.AWS_REGION,
        }

        # Only add credentials if both are provided
        if self.valves.AWS_ACCESS_KEY and self.valves.AWS_SECRET_KEY:
            kwargs['aws_access_key_id'] = self.valves.AWS_ACCESS_KEY
            kwargs['aws_secret_access_key'] = self.valves.AWS_SECRET_KEY

        self.bedrock = boto3.client(**kwargs)

    def get_anthropic_models(self):
        """Return available Anthropic models on AWS Bedrock"""
        return [
            {"id": "anthropic.claude-3-haiku-20240307-v1:0", "name": "claude-3-haiku"},
            {"id": "anthropic.claude-3-opus-20240229-v1:0", "name": "claude-3-opus"},
            {"id": "anthropic.claude-3-sonnet-20240229-v1:0", "name": "claude-3-sonnet"},
            {"id": "anthropic.claude-3-5-haiku-20241022-v1:0", "name": "claude-3.5-haiku"},
            {"id": "anthropic.claude-3-5-sonnet-20241022-v1:0", "name": "claude-3.5-sonnet"},
            {"id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0", "name": "claude-3.7-sonnet (inference profile)"},
        ]

    async def on_startup(self):
        """Lifecycle method called on pipeline startup"""
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """Lifecycle method called on pipeline shutdown"""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        """Called when pipeline valves (configuration) are updated"""
        # Recreate the Bedrock client when configuration changes
        self._initialize_bedrock_client()

    def pipelines(self) -> List[dict]:
        """Return available pipelines (models)"""
        return self.get_anthropic_models()

    def process_image(self, image_data):
        """Process image data for inclusion in messages"""
        if image_data["url"].startswith("data:image"):
            # Handle base64-encoded images
            mime_type, base64_data = image_data["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # Handle image URLs
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["url"]},
            }

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline method for processing requests"""
        try:
            # Remove unnecessary keys from body
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            # Extract system message if present
            system_message, messages = pop_system_message(messages)

            # Process messages, including text and images
            processed_messages = []
            image_count = 0
            total_image_size = 0

            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image_url":
                            if image_count >= 5:
                                raise ValueError("Maximum of 5 images per API call exceeded")

                            processed_image = self.process_image(item["image_url"])
                            processed_content.append(processed_image)

                            # Calculate image size for base64 data
                            if processed_image["source"]["type"] == "base64":
                                image_size = len(processed_image["source"]["data"]) * 3 / 4
                            else:
                                image_size = 0

                            total_image_size += image_size
                            if total_image_size > 100 * 1024 * 1024:
                                raise ValueError("Total size of images exceeds 100 MB limit")

                            image_count += 1
                else:
                    processed_content = [{"type": "text", "text": message.get("content", "")}]

                processed_messages.append({"role": message["role"], "content": processed_content})

            # Prepare the payload for Bedrock with configured defaults
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", self.valves.MAX_TOKENS),
                "temperature": body.get("temperature", self.valves.TEMPERATURE),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.9),
                "stop_sequences": body.get("stop", []),
            }
            if system_message:
                payload["system"] = str(system_message)

            # Handle reasoning effort for models supporting thinking (e.g., Claude 3.7)
            supports_thinking = "claude-3-7" in model_id
            reasoning_effort = body.get("reasoning_effort", self.valves.REASONING_EFFORT)
            budget_tokens = REASONING_EFFORT_BUDGET_TOKEN_MAP.get(reasoning_effort)

            if not budget_tokens and reasoning_effort not in REASONING_EFFORT_BUDGET_TOKEN_MAP.keys():
                try:
                    budget_tokens = int(reasoning_effort)
                except ValueError:
                    budget_tokens = None

            if supports_thinking and budget_tokens:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                payload["temperature"] = 1.0
                if "top_k" in payload:
                    del payload["top_k"]
                if "top_p" in payload:
                    del payload["top_p"]

            # Handle streaming vs non-streaming requests
            if body.get("stream", False):
                return self.stream_response(model_id, payload)
            else:
                response = self.bedrock.invoke_model(
                    modelId=model_id,
                    body=json.dumps(payload),
                )
                response_body = json.loads(response['body'].read())
                for content in response_body['content']:
                    if content.get("type") == "text":
                        return content["text"]
                return ""

        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, model_id: str, payload: dict) -> Generator:
        """Handle streaming responses from Bedrock"""
        response = self.bedrock.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(payload),
        )
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'].decode('utf-8'))
            chunk_type = chunk.get("type")
            if chunk_type == "content_block_start":
                content_block = chunk.get("content_block", {})
                if content_block.get("type") == "thinking":
                    yield "<think>"
                elif content_block.get("type") == "text":
                    yield content_block.get("text", "")
            elif chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "thinking_delta":
                    yield delta.get("thinking", "")
                elif delta.get("type") == "text_delta":
                    yield delta.get("text", "")
                elif delta.get("type") == "signature_delta":
                    yield "\n </think> \n\n"
            elif chunk_type == "message_stop":
                break
