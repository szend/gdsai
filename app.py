import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


input_text = "3, create field called new_avg from age * avg lenght, fields: age, avgLenght, name, kapid"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)


outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=False)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


print("result:" + generated_text)
