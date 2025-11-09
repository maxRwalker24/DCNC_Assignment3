import os
import json
import boto3

REGION = os.getenv("BEDROCK_REGION", "ap-southeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

def invoke_bedrock(prompt_text, max_tokens=640, temperature=0.2, top_p=0.9):
    client = boto3.client("bedrock-runtime", region_name=REGION)
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    resp = client.invoke_model(
        body=json.dumps(payload),
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
    )
    data = json.loads(resp["body"].read())
    return data["content"][0]["text"]
