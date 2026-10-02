import yfinance as yf
import pandas as pd
import os
import google.generativeai as genai
from fastapi import HTTPException

# 初始化 Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_company_name(ticker: str) -> str:
    try:
        tw_ticker = ticker if ticker.endswith(".TW") else f"{ticker}.TW"
        stock = yf.Ticker(tw_ticker)
        info = stock.info
        return info.get('shortName', ticker)
    except Exception:
        return ticker

def fetch_stock_and_macro_data(ticker: str):
    try:
        tw_ticker = ticker if ticker.endswith(".TW") else f"{ticker}.TW"
        stock = yf.Ticker(tw_ticker)
        df = stock.history(period="2y")
        
        if df.empty:
            raise HTTPException(
                status_code=400, 
                detail=f"Yahoo Finance 無法取得 {tw_ticker} 的歷史資料，請確認代號是否正確或稍後再試。"
            )
            
        df.index = df.index.tz_localize(None)
        
        # ==========================================
        # 📈 補回總體經濟指標 (大盤、費半、美元匯率)
        # ==========================================
        try:
            macro_df = pd.DataFrame(index=df.index)
            macro_tickers = {"TWII": "^TWII", "SOX": "^SOX", "USDTWD": "TWD=X"}
            
            for name, m_tick in macro_tickers.items():
                m_data = yf.Ticker(m_tick).history(period="2y")
                if not m_data.empty:
                    m_data.index = m_data.index.tz_localize(None)
                    macro_df[f'{name}_Return'] = m_data['Close'].pct_change()
                else:
                    macro_df[f'{name}_Return'] = 0.0
                    
            # 將總經資料合併回主表
            df = df.join(macro_df)
        except Exception as e:
            print(f"總經資料抓取異常，啟動備用防護: {e}")
            
        # 防護機制：確保欄位一定存在，避免 KeyError
        for col in ['TWII_Return', 'SOX_Return', 'USDTWD_Return']:
            if col not in df.columns:
                df[col] = 0.0
                
        # 填補計算報酬率產生的空值
        df.fillna(0, inplace=True)
        # ==========================================
        
        # 降級 Date 變成一般欄位
        df.reset_index(inplace=True)
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
        # 基礎均線特徵
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        
        df.dropna(inplace=True)
        
        if df.empty:
            raise HTTPException(status_code=400, detail=f"標的 {ticker} 計算特徵後的有效數據不足。")
            
        return df
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抓取股價資料時發生未預期錯誤: {str(e)}")

def fetch_news_sentiment(ticker: str):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"請分析關於台灣股票 {ticker} 的近期市場情緒。請回傳兩行內容：第一行只需填寫 0.0 到 1.0 之間的數字（0為極度悲觀，1為極度樂觀，0.5為中立）。第二行請用 30 個字以內總結原因。"
        
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        
        try:
            score = float(lines[0].strip())
        except ValueError:
            score = 0.5
            
        reason = "\n".join(lines[1:]).strip() if len(lines) > 1 else "依據近期市場綜合表現進行評估。"
        return max(0.0, min(1.0, score)), reason
        
    except Exception as e:
        print(f"Gemini API 呼叫失敗: {e}")
        return 0.5, "目前無法連線至 AI 引擎取得診斷報告，預設為中立情緒。"