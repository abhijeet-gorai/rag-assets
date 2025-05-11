import os
from langchain_groq import ChatGroq

def init_chat_model(
    model_id="meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens=2000,
    temperature=0,
    top_p=0.1,
    stop_sequences=[],
    groq_apikey=os.getenv("GROQ_APIKEY"),
):
    if not groq_apikey:
        raise ValueError(
            "GROQ_APIKEY is required for connecting to GROQ. "
            "Either set it as environment variables or provide it during initialization."
        )
    llm = ChatGroq(
        model=model_id,
        temperature=temperature,
        stop_sequences=stop_sequences,
        max_tokens=max_tokens,
        model_kwargs={"top_p": top_p},
        api_key=groq_apikey,
        max_retries=2
    )
    return llm
