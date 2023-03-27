import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    # load model and processor from model_dir
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    # unpack model and tokenizer
    model, tokenizer = model_and_tokenizer

    # process input
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", None)

    logger.info("inputs: {}".format(inputs))

    # preprocess
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids

    # pass inputs with all kwargs in data
    if parameters is not None:
        outputs = model.generate(input_ids, **parameters)
    else:
        outputs = model.generate(input_ids)

    # postprocess the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return [{"generated_text": prediction}]
