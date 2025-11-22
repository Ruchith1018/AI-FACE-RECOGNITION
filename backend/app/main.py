from fastapi import FastAPI
from app.auth_router import router
from app.database import init_db


app = FastAPI(title="AI Face Auth")

# create tables
init_db()

app.include_router(router)
