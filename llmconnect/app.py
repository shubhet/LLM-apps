import os
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient

load_dotenv()
hf_token = os.getenv("HG_KEY") or os.getenv("HUGGINGFACEHUB_API_KEY")

if not hf_token:
    st.error("Token not found in .env file")
    st.stop()

st.set_page_config(page_title="HF Chat Demo")
st.title("🤖 Hugging Face Chat")

question = st.text_input("Ask a question")

# Model:provider pairs to try in order
MODELS = [
    ("meta-llama/Llama-3.2-3B-Instruct", "auto"),
    ("Qwen/Qwen2.5-72B-Instruct",         "nebius"),
    ("Qwen/Qwen2.5-72B-Instruct",         "together"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "auto"),
]

if question:
    with st.spinner("Generating response..."):
        last_error = None
        for model, provider in MODELS:
            try:
                client = InferenceClient(api_key=hf_token, provider=provider)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user",   "content": question}
                    ],
                    max_tokens=512,
                    temperature=0.7,
                )
                response = completion.choices[0].message.content
                st.success(f"Response *(model: {model} | provider: {provider})*")
                st.write(response)
                break
            except Exception as e:
                last_error = str(e)
                continue
        else:
            st.error(f"All options failed. Last error: {last_error}")