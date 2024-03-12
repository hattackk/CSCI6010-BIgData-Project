from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# Load tokenizer and model
model_name = "/bert-tiny-finetuned-sms-spam-detection/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define a function for spam filtering
def spam_filter(text):
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        output = model(**encoded_input)
    
    probabilities = F.softmax(output.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    if predicted_class == 1:
        return 1  # Mark as spam
    else:
        return 0  # Not spam

