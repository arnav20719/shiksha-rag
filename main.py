import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Verify your API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
print(f"API key loaded: {bool(api_key)}")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Test the LLM with a simple call
response = llm.invoke("Say 'setup complete' in exactly two words.")
print(f"LLM response: {response.content}")

# Test embeddings with a simple call
vector = embeddings.embed_query("test")
print(f"Embedding vector length: {len(vector)}")