import os
from dotenv import load_dotenv

load_dotenv()

def generate_ai_report(ticker: str, company_name: str, prediction: int, confidence: float, ai_reason: str, top_features: list, strategy_roi: float, market_roi: float) -> str:
    """呼叫 Gemini，若失敗則啟動優雅降級備案"""
    pred_str = '上漲' if prediction == 1 else '下跌'
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-3.6-flash')
        
        prompt = f"""你是頂級量化交易首席分析師。根據以下數據撰寫150字專業診斷報告。
        標的：{ticker} ({company_name})
        預測：{pred_str} (信心：{confidence:.1f}%)
        消息面：{ai_reason}
        關鍵特徵：{', '.join(top_features)}
        近半年報酬 (已扣手續費)：AI策略 {strategy_roi:.2f}% vs 大盤 {market_roi:.2f}%
        請給出客觀的投資與風險控管建議。"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        # 當 API 發生 429 限制或斷線時，回傳備用罐頭報告
        return (f"【系統備用生成模組】因分析連線擁擠，啟用量化數據自動解讀：\n\n"
                f"根據最新 XGBoost 模型運算，{company_name} ({ticker}) 短期走勢預測為「{pred_str}」，"
                f"聯合模型信心水準達 {confidence:.1f}%。本次決策核心特徵包含 {', '.join(top_features)}。\n\n"
                f"歷史回測績效顯示，本 AI 策略近半年報酬率為 {strategy_roi:.2f}%，對比大盤持有報酬為 {market_roi:.2f}%。"
                f"建議投資人參考上述客觀數據，切勿盲目追高，並嚴格控管停損風險。")