from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat_service import chat_stream

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    state: str | None = None
    disaster_type: str | None = None
    job_title: str | None = None
    fips_code: str | None = None


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    RAG-powered advisor endpoint.

    Returns a Server-Sent Events (SSE) stream of tokens.
    Each event: "data: <token>\\n\\n"
    Final event: "data: [DONE]\\n\\n"
    """
    async def event_generator():
        try:
            async for token in chat_stream(
                message=req.message,
                state=req.state,
                disaster_type=req.disaster_type,
                job_title=req.job_title,
                fips_code=req.fips_code,
            ):
                # Escape newlines inside token so SSE framing isn't broken
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )
