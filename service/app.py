from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

from core import get_reference, make_prompt, llm_request, llm_model

app = FastAPI()

class RagRequest(BaseModel):
    message: str

class RagResponse(BaseModel):
    response: str

@app.post("/message", response_model=RagResponse)
async def rag_endpoint(request: RagRequest):
    try:
        question = request.message
        ref_docs = get_reference(question)
        if len(ref_docs) < 3:
            raise HTTPException(status_code=500, detail="Not enough reference documents found.")
        prompt = make_prompt(question, ref_docs)
        response = llm_request(llm_model[0], prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)