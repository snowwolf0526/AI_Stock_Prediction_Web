# 使用 Python 3.12 作為基礎環境
FROM python:3.12

# 設定工作目錄
WORKDIR /code

# 複製環境套件清單並安裝
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# 複製所有專案原始碼到容器內
COPY . .

# 啟動 FastAPI 伺服器 (Hugging Face 規定必須使用 7860 通訊埠)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]