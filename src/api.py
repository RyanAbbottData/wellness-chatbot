from fastapi import FastAPI

from chatbot import get_query_from_prompt


app = FastAPI()

@app.get("/query/{q}")
def get_query(q: str):
    query = get_query_from_prompt(q)
    return query