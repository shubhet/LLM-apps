import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
from typing import Optional, List, Any

load_dotenv()

HF_TOKEN = os.getenv("AH_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class HFInferenceLLM(LLM):
    model_id: str = "Qwen/Qwen2.5-72B-Instruct"  
    hf_token: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        client = InferenceClient(
            provider="novita",        
            api_key=self.hf_token,
        )
        response = client.chat_completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked"),
    ("user", "Question:{question}")
])

# Streamlit UI
st.title("Langchain Demo With Qwen Model")
input_text = st.chat_input("What question you have in mind?")

# LLM
llm = HFInferenceLLM(model_id="Qwen/Qwen2.5-72B-Instruct", hf_token=HF_TOKEN)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    with st.spinner("Thinking..."):
        st.write(chain.invoke({"question": input_text}))