from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_perplexity.chat_models import ChatPerplexity
from dotenv import load_dotenv
from langserve import add_routes
import os
from typing import Annotated
from pydantic import BaseModel, SkipValidation

load_dotenv()
api_key = os.getenv("PPLX_API_KEY")
if not api_key:
    raise ValueError("PPLX_API_KEY not found in .env")

prompt_str = """
You are a research assistant powered by Perplexity AI.
Given a question, provide a concise and accurate answer based on reliable sources.
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_str),
    HumanMessagePromptTemplate.from_template("{question}")
])

llm = ChatPerplexity(
    model="sonar",          # e.g. "sonar", "sonar-reasoning", "sonar-deep-research", "perplexity-advanced"
    temperature=0.7,
    api_key=api_key
)

parser = StrOutputParser()

chain = prompt | llm | parser
# print(chain.invoke({"topic": "Transformers in NLP"}))
# Sreaming response to console
# for chunk in chain.stream({"topic": "LLM evaluation best practices"}):
#     print(chunk, end="", flush=True)

# Create FastAPI app
app = FastAPI(
    title="My LangChain API",
    version="1.0",
    description="API server using LangChain's Runnable interfaces"
)

# Add the chain as a route
add_routes(app, chain, path="/sonar")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)