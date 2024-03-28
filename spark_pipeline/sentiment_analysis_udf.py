from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType


# Load tokenizer and model
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def sentiment_analysis(text) -> int:
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)

    probabilities = F.softmax(output.logits, dim=1)
    probabilities = probabilities.tolist()[0]
    result = float(probabilities[1] - probabilities[0])
    # We are going to translate the results from a vector [N, P] to
    # a single score where -1 means negative 100%, 0 means mixed, 1 means
    # positive 100% confidence. This will allow us to take into account
    # more mixed reviews
    return result



