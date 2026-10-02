import yfinance as yf
import feedparser
from snownlp import SnowNLP
import pandas as pd
import pandas_ta as ta  # 載入技術指標套件
from datetime import datetime, timedelta
import warnings

# 忽略未來版本警告，保持終端機畫面乾淨
warnings.filterwarnings('ignore')

def get_news_sentiment(keyword, days=730):
    """
    從 Google News RSS 爬取新聞，並計算情緒分數
    """
    print(f"正在爬取 {keyword} 的新聞 (這可能需要一點時間)...")
    url = f"https://news.google.com/rss/search?q={keyword}+when:{days}d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(url)
    
    news_list = []
    for entry in feed.entries:
        title = entry.title
        try:
            published_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S GMT").strftime("%Y-%m-%d")
        except Exception:
            continue
        
        try:
            sentiment_score = SnowNLP(title).sentiments
        except:
            sentiment_score = 0.5 
            
        news_list.append({
            "Date": published_date,
            "Title": title,
            "Sentiment": sentiment_score
        })
        
    df_news = pd.DataFrame(news_list)
    
    if not df_news.empty:
        df_sentiment = df_news.groupby("Date")["Sentiment"].mean().reset_index()
        return df_sentiment
    return pd.DataFrame(columns=["Date", "Sentiment"])

def get_stock_data(ticker, days=730):
    """
    使用 yfinance 抓取股價 (已修正 yfinance 新版 MultiIndex 問題)
    """
    print(f"正在抓取 {ticker} 的股價資料...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    stock = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
    
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.droplevel(1)
        
    stock.reset_index(inplace=True)
    stock['Date'] = stock['Date'].dt.strftime('%Y-%m-%d')
    return stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# ======== 測試執行區 ========
if __name__ == "__main__":
    stock_name = "台積電"
    stock_ticker = "2330.TW"
    
    # 1. 獲取資料 (改為抓取 730 天，讓指標有足夠資料計算)
    df_sentiment = get_news_sentiment(stock_name, days=730)
    df_stock = get_stock_data(stock_ticker, days=730)
    
    # 2. 資料對齊
    final_data = pd.merge(df_stock, df_sentiment, on="Date", how="left")
    final_data['Sentiment'] = final_data['Sentiment'].fillna(0.5)
    
    # 3. 特徵工程 (加入技術指標)
    print("正在計算技術指標與預測目標...")
    final_data.ta.sma(length=5, append=True)   # 5日均線
    final_data.ta.sma(length=10, append=True)  # 10日均線
    final_data.ta.rsi(length=14, append=True)  # 14日 RSI
    final_data.ta.macd(append=True)            # MACD
    
    # 4. 定義預測目標 (Y)
    final_data['Next_Close'] = final_data['Close'].shift(-1)
    # 明天收盤 > 今天收盤 = 1 (漲)，否則 = 0 (跌)
    final_data['Target'] = (final_data['Next_Close'] > final_data['Close']).astype(int)
    
    # 5. 清理因計算指標或位移而產生的空值 (NaN)
    final_data.dropna(inplace=True)
    
    print("\n======== 最終完整訓練資料表 (X 與 Y) ========")
    pd.set_option('display.max_columns', None) # 讓終端機顯示所有欄位不要折疊
    print(final_data.tail())

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib  # 用來把模型存成檔案的神器

def train_xgboost_model(df):
    print("\n======== 開始訓練 XGBoost 模型 ========")
    
    # 1. 定義特徵 (X) 與 目標 (Y)
    # 小心：絕對不能把 Date, Next_Close 和 Target 放到 X 裡面，不然模型會作弊！
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Sentiment', 'SMA_5', 'SMA_10', 'RSI_14', 
                'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    
    X = df[features]
    y = df['Target']
    
    # 2. 切分訓練集與測試集 (Time-Series Split)
    # 股票預測「絕對不能」隨機切分 (Random Split)！我們必須用「過去」預測「未來」。
    # 這裡我們拿前 80% 的時間當訓練，最後 20% 當作考試(測試)。
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"訓練資料筆數: {len(X_train)} 筆 (過去)")
    print(f"測試資料筆數: {len(X_test)} 筆 (未來)")
    
    # 3. 建立 XGBoost 分類器並訓練
    # 這裡先設定一些基礎參數 (專題報告時可以說你有做 Hyperparameter tuning)
    model = xgb.XGBClassifier(
        n_estimators=100,      # 長出 100 棵決策樹
        learning_rate=0.05,    # 學習率
        max_depth=4,           # 樹的最大深度 (避免過度擬合)
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 4. 讓模型在測試集上考試，並評估準確率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 模型預測準確率 (Accuracy): {accuracy * 100:.2f}%")
    print("\n詳細分類報告:")
    print(classification_report(y_test, y_pred))
    
    # 5. 【最重要的一步】把訓練好的模型打包存檔！
    joblib.dump(model, 'stock_xgboost_model.pkl')
    print("\n✅ 模型已成功儲存為 'stock_xgboost_model.pkl'！")
    print("有了這個檔案，我們的網頁就可以直接載入它來做即時預測了。")

# 把這行加在程式碼最後面 (在 print(final_data.tail()) 的下一行)
train_xgboost_model(final_data)