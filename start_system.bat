@echo off
echo =========================================
echo   AI 股市預測系統 - 啟動程序 (微服務版)
echo =========================================

echo [1/3] 正在啟動 FastAPI 後端伺服器...
start cmd /k "uvicorn main:app --reload"

echo [2/3] 正在啟動 MLflow 版控儀表板...
start cmd /k "mlflow ui --backend-store-uri sqlite:///mlflow.db"

echo [3/3] 等待後端就緒，準備啟動 Streamlit 前端...
timeout /t 5 /nobreak >nul
start cmd /k "streamlit run app.py"

echo 所有系統皆已觸發啟動！