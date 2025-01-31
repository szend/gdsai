import streamlit as st
from transformers import pipeline,  T5Tokenizer, T5ForConditionalGeneration,  AutoTokenizer, AutoModelForCausalLM

def main():
    st.title("Hugging Face Model Demo")

    # Create an input text box
    input_text = st.text_input("Enter your text", "")



    tokenizer = AutoTokenizer.from_pretrained("gpt2") 
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    inputs = tokenizer(data.input_text, return_tensors="pt", padding=True, truncation=True)
    
    
    outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    
    if st.button("Analyze"):
        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)    
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Prediction:",generated_text)

if __name__ == "__main__":
    main()
