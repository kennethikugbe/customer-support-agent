import boto3

REGION = "us-east-1"
MODEL_ID = "global.amazon.nova-2-lite-v1:0"

client = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
)

response = client.converse(
    modelId=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "Reply with only MODEL_ACCESS_OK"}
            ],
        }
    ],
    inferenceConfig={
        "maxTokens": 64,
        "temperature": 0.0,
    },
)

text = "\n".join(
    block["text"]
    for block in response["output"]["message"]["content"]
    if "text" in block
)

print(text)
print("Stop reason:", response.get("stopReason"))
