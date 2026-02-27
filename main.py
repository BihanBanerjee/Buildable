from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth.router import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Buildable")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router=router)


@app.get("/health")
def health():
    return {"status": "ok"}

