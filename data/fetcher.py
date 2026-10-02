import yfinance as yf
import pandas as pd
import requests
import os
import google.generativeai as genai
from fastapi import HTTPException

# ==========================================
# 🛡️ 防線一：建立偽裝成真人的 Session
# ==========================================
# 讓 yfinance 使用這個帶有正常瀏覽器 Header 的 session，極大幅度降低被判定為爬蟲的機率
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})

# 初始化 Gemini (你原本已經做好的部分)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_company_name(ticker: str) -> str:
    try:
        # 套用偽裝 session
        stock = yf.Ticker(f"{ticker}.TW", session=session)
        info = stock.info
        return info.get('shortName', ticker)
    except Exception:
        return ticker

def fetch_stock_and_macro_data(ticker: str):
    try:
        # 套用偽裝 session
        stock = yf.Ticker(f"{ticker}.TW", session=session)
        df = stock.history(period="1y")
        
        # ==========================================
        # 🛡️ 防線二：攔截空資料，避免伺服器 500 當機
        # ==========================================
        if df.empty:
            raise HTTPException(
                status_code=429, 
                detail=f"Yahoo Finance 暫時阻擋了 {ticker} 的資料請求 (Rate Limit)，請稍後再試。"
            )
            
        # 確保時區與格式正確
        df.index = df.index.tz_localize(None)
        
        # 基礎技術指標計算 (保留你原本的邏輯)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        
        df.dropna(inplace=True)
        
        if df.empty:
            raise HTTPException(status_code=400, detail=f"標的 {ticker} 的有效交易數據不足。")
            
        return df
        
    except HTTPException:
        # 如果是我們自己拋出的 429 或 400，直接往上傳遞
        raise
    except Exception as e:
        # 捕捉其他未知的套件錯誤，回傳 500 並附帶原因
        raise HTTPException(status_code=500, detail=f"抓取股價資料時發生錯誤: {str(e)}")

def fetch_news_sentiment(ticker: str):
    try:
        # 這裡是你成功實作的 Gemini 卸載邏輯
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"請分析關於台灣股票 {ticker} 的近期市場情緒，並只回傳 0.0 到 1.0 之間的數字（0代表極度悲觀，1代表極度樂觀，0.5代表中立）。若無特別消息請回傳 0.5。"
        
        response = model.generate_content(prompt)
        score = float(response.text.strip())
        # 確保分數在安全範圍內
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Gemini API 呼叫失敗: {e}")
        return 0.5 # 發生異常時的安全退路