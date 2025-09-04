from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_perplexity.chat_models import ChatPerplexity
from dotenv import load_dotenv
from langserve import add_routes
import os
from typing import Annotated
from pydantic import BaseModel, SkipValidation

class Foo(BaseModel): # <-- BaseModel is from Pydantic v2
    model: Annotated[ChatPerplexity, SkipValidation()]

load_dotenv()
api_key = os.getenv("PPLX_API_KEY")
if not api_key:
    raise ValueError("PPLX_API_KEY not found in .env")

# 1) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant powered by Perplexity AI."),
    ("human", "{question}")
])

# 2) LLM (Perplexity Sonar)
llm = ChatPerplexity(
    model="sonar",          # e.g. "sonar", "sonar-reasoning", "sonar-deep-research", "perplexity-advanced"
    temperature=0.7,
    api_key=api_key
)

# 3) Output parser (string)
parser = StrOutputParser()

# 4) LCEL chain
chain = prompt | llm | parser

# Run
# print(chain.invoke({"topic": "Transformers in NLP"}))
# Basic example
# for chunk in (prompt | llm | parser).stream({"topic": "LLM evaluation best practices"}):
#     print(chunk, end="", flush=True)


# app = FastAPI(title="Perplexity Sonar LCEL API")

# @app.get("/")
# def root():
#     return RedirectResponse("/docs")

# add_routes(app, chain, path="/sonar")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


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