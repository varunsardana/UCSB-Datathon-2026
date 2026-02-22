from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import analytics, chat, disasters, forecast, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all heavy resources at startup so requests are fast
    from rag.embeddings import load_embedding_model
    from rag.vectorstore import init_vectorstore
    from services.model_service import load_model

    print("Starting DisasterShift API...")
    load_embedding_model()
    init_vectorstore()
    load_model()
    print("All services ready.")
    yield


app = FastAPI(title="DisasterShift API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(disasters.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "DisasterShift API"}
