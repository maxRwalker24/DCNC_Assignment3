# app/bedrock_invoke.py
"""
This module provides a small helper function for calling an AWS Bedrock model
(Claude 3) using the 'converse' API. It sends one prompt and returns the model's
response text.

This module wraps the Bedrock API call so I can send a prompt to Claude 3 and retrieve the assistant's text. It handles errors simply and returns the first generated text block.

"""

import os
import json
import boto3

# Read region and model ID from environment variables, with safe defaults.
REGION = os.getenv("BEDROCK_REGION", "ap-southeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240229-v1:0")


def invoke_bedrock(
    prompt_text: str,
    max_tokens: int = 200,
    temperature: float = 0.3,
    top_p: float = 0.9,
    debug: bool = False,
) -> str:
    """
    Sends a prompt to an AWS Bedrock Claude model and returns the first text response.

    Parameters
    ----------
    prompt_text : str
        The text you want to send to the model. Usually includes system instructions,
        context, and the user's question.

    max_tokens : int
        Maximum tokens the model can generate in its reply.

    temperature : float
        Controls randomness. 0.0 = deterministic, 1.0 = more creative.

    top_p : float
        Controls nucleus sampling. Usually left at 0.9.

    debug : bool
        If True, exceptions are raised normally (useful for debugging).
        If False, errors are captured and returned as a readable message.

    Returns
    -------
    str
        The assistant's reply text.
        If something goes wrong, returns an informative error message.
    """

    # Create a Bedrock "runtime" client. This must have AWS credentials available.
    client = boto3.client("bedrock-runtime", region_name=REGION)

    try:
        # Call the Bedrock "converse" API used for Claude 3
        response = client.converse(
            modelId=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [{"text": prompt_text}],
            }],
            inferenceConfig={
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": float(top_p),
            }
        )

        # The Claude 3 API returns a list of "content" blocks.
        # We extract the first one that contains text.
        message = response.get("output", {}).get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            if "text" in block:
                return block["text"]

        # Fallback: return a readable version of the full response
        return json.dumps(response, indent=2)

    except Exception as e:
        # If debug=True, allow errors to crash loudly (useful while coding).
        if debug:
            raise

        # Otherwise, return a simple error message for the UI.
        return f"[Bedrock error] {e}"
