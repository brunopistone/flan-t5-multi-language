import boto3
import json
import logging
import os
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

comprehend_client = boto3.client("comprehend")
sagemaker_runtime_client = boto3.client('sagemaker-runtime')
translate_client = boto3.client("translate")

endpoint_name = os.getenv("SAGEMAKER_ENDPOINT", default=None)

def detect_language(body):
    try:
        results = comprehend_client.detect_dominant_language(Text=body)

        max_result = max(results["Languages"], key=lambda x: x['Score'])

        return max_result["LanguageCode"]
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def translate_string(row, start_lan="it", end_lan="en"):
    try:
        logger.info("Translating {} from {} to {}".format(row, start_lan, end_lan))

        response = translate_client.translate_text(
            Text=row,
            SourceLanguageCode=start_lan,
            TargetLanguageCode=end_lan
        )

        return response["TranslatedText"]

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e    

def lambda_handler(event, context):
    try:
        payload = event["payload"]
        parameters = event["parameters"]

        payload = payload.replace("\n", "")

        start_lan = detect_language(payload)

        if start_lan != "en":
            payload = translate_string(payload, start_lan, "en")

            logger.info("Translated sentence: {}".format(payload))
        else:
            logger.info("Detected en language")

        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                "inputs": payload,
                "parameters": parameters
            }))

        results = json.loads(response['Body'].read().decode())

        if start_lan != "en":
            results[0]["generated_text"] = translate_string(results[0]["generated_text"], "en", start_lan)
        else:
            logger.info("Detected en language")

        return {
            'statusCode': 200,
            'body': json.dumps(results)
        }
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        return {
            'statusCode': 500,
            'body': json.dumps(stacktrace)
        }