"""Run local setup checks against the two deployed project Lambda functions.

Run from starter/:
    uv run python check_lambdas.py

Uses the local 'udacity' AWS profile and us-east-1. Does not create resources.
These direct Lambda checks do not replace the six final deployed-agent tests.
"""

import base64
import json

import boto3


REGION = "us-east-1"
PROFILE = "udacity"


def invoke_json(client, function_name, event, tool_name=None):
    """Invoke synchronously, check function errors, and decode the JSON body."""
    options = {}
    if tool_name is not None:
        # AgentCore Gateway supplies this routing value to the refund Lambda.
        context = {"custom": {"bedrockAgentCoreToolName": tool_name}}
        options["ClientContext"] = base64.b64encode(
            json.dumps(context).encode("utf-8")
        ).decode("ascii")

    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(event).encode("utf-8"),
        **options,
    )
    result = json.loads(response["Payload"].read())

    if response.get("FunctionError"):
        raise RuntimeError(f"{function_name} raised an error: {json.dumps(result)}")
    if result.get("statusCode") != 200:
        raise RuntimeError(f"{function_name} returned an error: {json.dumps(result)}")

    return json.loads(result["body"])


def main():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    client = session.client("lambda")

    order = invoke_json(
        client,
        "order-tracker",
        {
            "resource": "/orders/{order_id}",
            "httpMethod": "GET",
            "pathParameters": {"order_id": "ORD-001"},
        },
    )
    expected_order = {
        "order_id": "ORD-001",
        "status": "SHIPPED",
        "tracking_number": "TRK987654321",
        "carrier": "UPS",
    }
    for key, expected in expected_order.items():
        if order.get(key) != expected:
            raise RuntimeError(f"Unexpected order field {key}: {order.get(key)!r}")
    if not order.get("estimated_delivery"):
        raise RuntimeError("Order response has no estimated delivery date.")

    print("ORDER CHECK PASSED")
    print(json.dumps(order, indent=2))

    refund = invoke_json(
        client,
        "refund-processor",
        {
            "order_id": "ORD-002",
            "reason": "Customer requested a return",
            "amount": 139.99,
        },
        tool_name="refund_processor___initiate_refund",
    )
    if (
        refund.get("status") != "APPROVED"
        or refund.get("order_id") != "ORD-002"
        or refund.get("amount") != 139.99
        or not str(refund.get("refund_id", "")).startswith("REF-")
        or "3-5 business days" not in refund.get("message", "")
    ):
        raise RuntimeError(f"Unexpected refund response: {json.dumps(refund)}")

    print("\nREFUND CHECK PASSED")
    print(json.dumps(refund, indent=2))
    print("\nBoth Lambda setup checks passed.")


if __name__ == "__main__":
    main()
