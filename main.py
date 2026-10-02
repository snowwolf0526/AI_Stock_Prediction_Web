from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="AI 量化股市預測 API", 
    description="專為前端分離架構設計的企業級量化預測微服務",
    version="1.0.0"
)

# 將我們剛剛寫好的 API 路由註冊進來，並加上 /api/v1 的網址前綴
app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "伺服器運作正常！請訪問 http://127.0.0.1:8000/docs 查看 API 文件。"}