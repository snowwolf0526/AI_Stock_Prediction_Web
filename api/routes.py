from fastapi import HTTPException
import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas_ta as ta
import time

from data.fetcher import get_company_name, fetch_stock_and_macro_data, fetch_news_sentiment
from core.model import train_and_predict
from core.llm_agent import generate_ai_report

router = APIRouter()
RESPONSE_CACHE = {}
CACHE_TTL = 600 

class PredictResponse(BaseModel):
    ticker: str
    company_name: str
    prediction: str
    confidence: float
    ai_report: str
    top_features: list
    strategy_roi: float
    market_roi: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    recommended_position: float  # 🆕 新增部位控管回傳值
    history_data: dict
    importance_data: dict
    backtest_data: dict

@router.get("/predict", response_model=PredictResponse)
async def predict_stock(ticker: str):
    clean_ticker = ticker.strip().upper()
    current_time = time.time()
    
    if clean_ticker in RESPONSE_CACHE:
        cached_data, timestamp = RESPONSE_CACHE[clean_ticker]
        if current_time - timestamp < CACHE_TTL:
            return cached_data

    try:
        pure_code = clean_ticker.split('.')[0] if clean_ticker.endswith(".TW") or clean_ticker.endswith(".TWO") else clean_ticker
        yf_ticker = f"{pure_code}.TW"
        company_name = get_company_name(pure_code)
        
        df = fetch_stock_and_macro_data(yf_ticker)
        if df.empty or len(df) < 300:
            raise HTTPException(status_code=400, detail="歷史資料不足，無法預測")

        avg_sentiment, ai_reason = fetch_news_sentiment(company_name)
        df['Sentiment'] = avg_sentiment
        
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df['Regime_Bull'] = (df['SMA_50'] > df['SMA_200']).astype(int)
        
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.obv(append=True)
        df.ta.mfi(length=14, append=True)
        
        df['Next_Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 
                    'TWII_Return', 'SOX_Return', 'USDTWD_Return',
                    'SMA_5', 'SMA_10', 'RSI_14', 'MACD_12_26_9', 'OBV', 'MFI_14',
                    'DayOfWeek', 'Month', 'Regime_Bull']

        # 🆕 接收 recommended_position
        prediction, confidence, top_features, strategy_roi, market_roi, sharpe, mdd, win, rec_pos, importance_df, bt_test = train_and_predict(df, features)
        
        report = generate_ai_report(yf_ticker, company_name, prediction, confidence * 100, ai_reason, top_features, strategy_roi, market_roi)
        
        history_df = df.tail(90).copy()
        history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d')
        history_data = history_df[['Date', 'Open', 'High', 'Low', 'Close', 'SMA_5', 'SMA_10']].to_dict(orient='list')
        
        importance_data = importance_df.to_dict(orient='list')
        
        bt_data = {}
        if not bt_test.empty:
            bt_test_copy = bt_test.copy()
            bt_test_copy['Date'] = bt_test_copy['Date'].dt.strftime('%Y-%m-%d')
            # 🆕 將歷史每日部位比例 (Position_Size) 也傳給前端
            bt_data = bt_test_copy[['Date', 'Cum_Strategy', 'Cum_Market', 'Position_Size']].to_dict(orient='list')

        final_response = PredictResponse(
            ticker=yf_ticker,
            company_name=company_name,
            prediction="上漲" if prediction == 1 else "下跌",
            confidence=confidence * 100,
            ai_report=report,
            top_features=top_features,
            strategy_roi=strategy_roi,
            market_roi=market_roi,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=win,
            recommended_position=rec_pos,  # 🆕
            history_data=history_data,
            importance_data=importance_data,
            backtest_data=bt_data
        )
        
        RESPONSE_CACHE[clean_ticker] = (final_response, current_time)
        return final_response
        
    except HTTPException as he:
        # 新增這兩行：如果是我們自己拋出的 400 錯誤，直接放行，不要變成 500
        raise he 
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {str(e)}")