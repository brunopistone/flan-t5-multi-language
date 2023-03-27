import boto3
import streamlit as st
import json

# Create a low-level client representing Amazon SageMaker Runtime
session = boto3.Session()
lambda_client = boto3.client("lambda")

# The name of the endpoint. The name must be unique within an AWS Region in your AWS account.
lambda_function_name = "Multi-Language-GenAI"

st.sidebar.title("Flan-T5 Parameters")

stop_word = st.sidebar.text_input("Stop word")
length_penalty = st.sidebar.slider("Length Penalty", min_value=0.0, max_value=10.0, value=2.0)
min_length, max_length = st.sidebar.slider("Min/Max length", 0, 500, (10, 50))
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3)
rep_penalty = st.sidebar.slider("Repetition Penalty", min_value=0.9, max_value=1.2, value=1.0)

def generate_text(prompt):
    parameters = {
        "early_stopping": True,
        "length_penalty": length_penalty,
        "min_length": min_length,
        "max_length": max_length,
        "temperature": temperature,
        "repetition_penalty": rep_penalty
    }

    print("Payload: {}".format(prompt))
    print("Parameters: {}".format(parameters))

    body = {
        "payload": prompt,
        "parameters": parameters
    }

    response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        Payload=json.dumps(body)
    )

    results = json.loads(response['Payload'].read().decode("utf-8"))

    if results["statusCode"] == 200:
        if len(json.loads(results["body"])) > 0 and "generated_text" in json.loads(results["body"])[0]:
            return json.loads(results["body"])[0]["generated_text"]

    return results

st.header("Flan-T5-XXL Playground")
prompt = st.text_area("Enter your prompt here:")

if st.button("Run"):
    generated_text = generate_text(prompt)
    if len(stop_word) > 0:
        generated_text = generated_text[:generated_text.rfind(stop_word)]
    st.write(generated_text)