import os, json, boto3

REGION = os.getenv("BEDROCK_REGION", "ap-southeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240229-v1:0")  # or haiku 20240307

def invoke_bedrock(prompt_text, max_tokens=200, temperature=0.3, top_p=0.9, debug=False):
    client = boto3.client("bedrock-runtime", region_name=REGION)
    try:
        resp = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt_text}]}],
            inferenceConfig={
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": float(top_p),
            },
        )
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        for b in blocks:
            if "text" in b:
                return b["text"]
        return json.dumps(resp, indent=2)
    except Exception as e:
        if debug:
            raise
        return f"[Bedrock error] {e}"
