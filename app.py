import streamlit as st  
from transformers import T5Tokenizer, T5ForConditionalGeneration  

# Load model and tokenizer  
MODEL_NAME = 't5-base'  # Change to your specific model if needed  
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)  
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)  

# Create Streamlit interface  
st.title("Text Translation with T5")  
st.write("Enter text to translate from English to French:")  

# Text input  
input_text = st.text_area("Input Text:", "Hello, how are you?")  

if st.button("Translate"):  
    # Prepare the input  
    input_ids = tokenizer.encode("translate English to French: " + input_text, return_tensors="pt")  
    
    # Generate output  
    with torch.no_grad():  
        outputs = model.generate(input_ids)  
    
    # Decode and display the output  
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    st.success(f"Translated Text: {translated_text}")
