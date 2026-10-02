import streamlit as st
import requests
import plotly.graph_objects as go
import time
import pandas as pd

st.set_page_config(page_title="AI 股市預測系統", layout="wide")

st.title("📈 畢業專題：AI 股市預測系統 (微服務架構版)")
st.markdown("本系統已全面重構為 **前後端分離架構**。前端只負責介面展示，核心運算與 AI 報告皆由 FastAPI 後端微服務即時運算並回傳。")

raw_ticker = st.text_input("🎯 輸入台股代號 (如: 2330, 2454, 0050)", value="2330", max_chars=8)

# ==========================================
# 區塊一：單一標的深度預測
# ==========================================
if st.button("🚀 呼叫後端 API 進行預測", type="primary"):
    with st.spinner('正在呼叫 FastAPI 後端進行深度運算，請稍候...'):
        try:
            # 關鍵修改：將 API 網址指向你筆電的本地端 FastAPI (127.0.0.1:8000)
            api_url = f"http://127.0.0.1:8000/api/v1/predict?ticker={raw_ticker.strip()}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ API 呼叫成功！標的：{data['company_name']} ({data['ticker']})")
                
                # --- 資金部位控管儀表板 ---
                st.markdown("### 💰 AI 動態資金控管建議 (Position Sizing)")
                rec_pos = data['recommended_position']
                
                # 依據水位給予不同顏色提示
                if rec_pos >= 75:
                    pos_color, pos_text = "green", "積極建倉 (模型高度確信)"
                elif rec_pos >= 30:
                    pos_color, pos_text = "orange", "部分試單 (模型中度確信)"
                else:
                    pos_color, pos_text = "red", "空手觀望或極輕倉 (模型無信心或看跌)"
                    
                st.markdown(f"**明日建議持倉比例：** <span style='color:{pos_color}; font-size:20px; font-weight:bold;'>{rec_pos:.1f}%</span> ── {pos_text}", unsafe_allow_html=True)
                st.progress(int(rec_pos))
                st.markdown("---")
                
                # --- 頂部數據儀表板 ---
                st.markdown("### 🔮 預測與核心績效")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="預測方向", value=f"{data['prediction']} " + ("📈" if data['prediction']=="上漲" else "📉"))
                with col2:
                    st.metric(label="聯合模型信心", value=f"{data['confidence']:.1f}%")
                with col3:
                    st.metric(label="AI 策略報酬", value=f"{data['strategy_roi']:.2f}%")
                with col4:
                    st.metric(label="大盤持有報酬", value=f"{data['market_roi']:.2f}%")
                
                # --- 進階風險控管指標 ---
                st.markdown("### 🛡️ 量化風險評估指標")
                r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                with r_col1:
                    st.metric(label="策略勝率 (Win Rate)", value=f"{data['win_rate']:.1f}%")
                with r_col2:
                    st.metric(label="夏普比率 (Sharpe Ratio)", value=f"{data['sharpe_ratio']:.2f}", help="衡量承受每單位風險所獲得的超額報酬，>1 為佳")
                with r_col3:
                    st.metric(label="最大回撤 (Max Drawdown)", value=f"{data['max_drawdown']:.2f}%", help="策略在回測期間發生的最大帳面虧損幅度")
                with r_col4:
                    st.caption("備註：以上指標已包含依據『動態資金控管』計算之滑價與手續費模型。")
                    
                # --- AI 報告 ---
                st.markdown("### 🤖 首席 AI 總體診斷報告")
                st.info(data['ai_report'])
                
                st.markdown("---")
                
                # --- 畫圖區塊 ---
                hist = data['history_data']
                fig_k = go.Figure(data=[go.Candlestick(x=hist['Date'], open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K線')])
                fig_k.add_trace(go.Scatter(x=hist['Date'], y=hist['SMA_5'], line=dict(color='orange', width=1.5), name='5日均線'))
                fig_k.add_trace(go.Scatter(x=hist['Date'], y=hist['SMA_10'], line=dict(color='blue', width=1.5), name='10日均線'))
                fig_k.update_layout(title=f"{data['company_name']} 近 90 日走勢與均線", yaxis_title='股價 (TWD)', template='plotly_white', height=450)
                st.plotly_chart(fig_k, use_container_width=True)
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    imp = data['importance_data']
                    fig_imp = go.Figure(go.Bar(x=imp['重要性'], y=imp['特徵'], orientation='h', marker=dict(color='teal')))
                    fig_imp.update_layout(title='XGBoost 決策關鍵特徵權重', template='plotly_white', height=350)
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                with col_chart2:
                    bt = data['backtest_data']
                    if bt:
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(x=bt['Date'], y=bt['Cum_Strategy'], line=dict(color='red', width=2.5), name='AI 策略 (含動態部位)'))
                        fig_bt.add_trace(go.Scatter(x=bt['Date'], y=bt['Cum_Market'], line=dict(color='gray', width=1.5, dash='dash'), name='單純持有'))
                        fig_bt.update_layout(title='近半年歷史回測：動態資金 AI 策略 vs 大盤', yaxis_title='累積資產倍數', template='plotly_white', height=350, hovermode='x unified')
                        st.plotly_chart(fig_bt, use_container_width=True)
                
            else:
                st.error(f"❌ API 呼叫失敗！錯誤代碼：{response.status_code}")
                st.error(response.json().get('detail', '未知錯誤'))
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ 連線不到後端 API！請確認你的 FastAPI 伺服器 (uvicorn) 是否有在另一個終端機開啟。")
        except Exception as e:
            st.error(f"發生未預期的錯誤：{e}")


# ==========================================
# 區塊二：動態選股池 (MVP 掃描版)
# ==========================================
st.markdown("---")
st.subheader("🔥 AI 動態選股池 (MVP 掃描版)")
st.write("一鍵掃描台股重點權值股，尋找今日最具潛力的強勢標的。")

if st.button("🚀 啟動 AI 策略掃描", type="secondary"):
    target_stocks = ["2330", "2317", "2454", "2308", "2881"] 
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(target_stocks):
        status_text.text(f"🔍 正在呼叫本地端微服務分析標的：{ticker}...")
        
        # 關鍵修改：將 API 網址指向你筆電的本地端 FastAPI (127.0.0.1:8000)
        api_url = f"http://127.0.0.1:8000/api/v1/predict?ticker={ticker}"
        
        try:
            response = requests.get(api_url, timeout=40)
            if response.status_code == 200:
                data = response.json()
                
                results.append({
                    "股票代號": f"{data.get('company_name', ticker)} ({ticker})",
                    "AI 預測方向": data.get("prediction", "未知"),
                    "模型信心度": f"{data.get('confidence', 0):.1f}%",
                    "建議持倉水位": f"{data.get('recommended_position', 0):.1f}%"
                })
            else:
                st.warning(f"標的 {ticker} 分析失敗 (HTTP {response.status_code})")
        except Exception as e:
            st.warning(f"標的 {ticker} 連線超時或異常")
            
        progress_bar.progress((i + 1) / len(target_stocks))
        time.sleep(5)
        
    status_text.text("✅ 策略掃描完成！")
    
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)