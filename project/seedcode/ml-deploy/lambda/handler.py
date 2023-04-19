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

def detect_prompt(payload):
    first_line = payload.splitlines()[0]

    start_lan = detect_language(first_line)

    if start_lan != "en":
        first_line = translate_string(first_line, start_lan, "en")

    if is_prompt(first_line):
        prompt = first_line
        payload = "".join(payload.splitlines()[1:])
    else:
        prompt = payload.splitlines()[-1]
        payload = "".join(payload.splitlines()[:-1])

    return prompt, payload

def get_chunks(prompt, payload):
    cleaned_lines = []

    for line in payload.splitlines():
        line = line.replace('\n', '')
        line = line.replace('\t', '')
        line = line.replace('  ', ' ')

        cleaned_lines.append(line)

    payload = " ".join(cleaned_lines)

    chunks = payload.split(".")

    if len(chunks[-1]) < 5:
        chunks = chunks[:-1]

    for i in range(len(chunks)):
        chunks[i] = prompt + "\n" + chunks[i]

    return chunks

def is_prompt(text):
    detect_words = ['why', 'how', 'what', 'who', 'where', 'is', 'when', 'which', 'whose', 'are', 'do', 'does',
                    'can', 'could', 'should', 'will', 'have', 'has', "summary", "summarize", "rephrase", "rewrite"]

    first_word = text.split()[0]
    if first_word.lower() in detect_words:
        return True
    return False

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

        prompt, payload = detect_prompt(payload)

        chunks = get_chunks(prompt, payload)

        start_langs = []
        chunks_model = []

        for chunk in chunks:
            start_lan = detect_language(chunk)
            start_langs.append(start_lan)

            if start_lan != "en":
                chunk = translate_string(chunk, start_lan, "en")

                logger.info("Translated sentence: {}".format(chunk))
            else:
                logger.info("Detected en language")

            chunks_model.append(chunk)

        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                "inputs": chunks_model,
                "parameters": parameters
            }))

        result = json.loads(response['Body'].read().decode())

        start_lan = max(set(start_langs), key=start_langs.count)

        if start_lan != "en":
            return_response = translate_string(result[0]["generated_text"], "en", start_lan)
        else:
            logger.info("Detected en language")

            return_response = result[0]["generated_text"]

        logger.info("Generated text: {}".format(return_response))

        return {
            'statusCode': 200,
            'body': json.dumps([{
                "generated_text": return_response
            }])
        }
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        return {
            'statusCode': 500,
            'body': json.dumps(stacktrace)
        }
