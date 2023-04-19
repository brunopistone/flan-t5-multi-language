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

        if len(payload) > 255:
            n = 255
            chunks = [payload[i:i + n] for i in range(0, len(payload), n)]
        else:
            chunks = [payload]

        translated_chunks = []
        start_langs = []

        for chunk in chunks:
            start_lan = detect_language(chunk)
            start_langs.append(start_lan)

            if start_lan != "en":
                chunk = translate_string(chunk, start_lan, "en")

                logger.info("Translated sentence: {}".format(chunk))
            else:
                logger.info("Detected en language")

            translated_chunks.append(chunk)

        payload = "".join(translated_chunks)

        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                "inputs": payload,
                "parameters": parameters
            }))

        results = json.loads(response['Body'].read().decode())

        start_lan = max(set(start_langs), key=start_langs.count)

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
            'body': json.dumps(e)
        }