import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


MODEL_NAME = "t5-small" 

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
inputs = tokenizer(data.input_text, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)



st.write('Text: ' + generated_text)
