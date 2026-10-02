import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
import sqlite3
import mlflow
import mlflow.sklearn
import os

# 🆕 1. 設定 MLflow 儲存路徑 (會在專案底下產生 mlruns 資料夾)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Stock_Prediction_Microservice")

def train_and_predict(df: pd.DataFrame, features: list):
    train_df = df.dropna(subset=features + ['Target'])
    X_today = df.iloc[-1:][features]
    ticker_name = df['Date'].iloc[-1].strftime('%Y-%m-%d') # 抓個日期當識別
    
    # 🆕 2. 啟動 MLflow 追蹤
    with mlflow.start_run(run_name=f"Predict_{ticker_name}"):
        
        # 定義模型參數
        xgb_params = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4, 'random_state': 42}
        rf_params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 42}
        
        # 紀錄參數到 MLflow
        mlflow.log_params({"xgb_" + k: v for k, v in xgb_params.items()})
        mlflow.log_params({"rf_" + k: v for k, v in rf_params.items()})
        mlflow.log_param("features_count", len(features))
        
        clf1 = XGBClassifier(**xgb_params)
        clf2 = RandomForestClassifier(**rf_params)
        ensemble_model = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2)], voting='soft', n_jobs=-1)
        
        ensemble_model.fit(train_df[features], train_df['Target'])
        prediction = ensemble_model.predict(X_today)[0]
        probability = ensemble_model.predict_proba(X_today)[0]
        
        prob_up = probability[1]
        recommended_position = max(0.0, min(1.0, (prob_up - 0.5) * 2)) * 100
        
        clf1.fit(train_df[features], train_df['Target'])
        importance_df = pd.DataFrame({'特徵': features, '重要性': clf1.feature_importances_}).sort_values(by='重要性', ascending=True)
        top_features = importance_df['特徵'].iloc[-3:].tolist()
        
        backtest_days = 126
        strategy_roi, market_roi, sharpe_ratio, max_drawdown, win_rate = 0, 0, 0, 0, 0
        bt_test = pd.DataFrame()
        
        if len(train_df) > backtest_days * 1.5:
            bt_train = train_df.iloc[:-backtest_days]
            bt_test = train_df.iloc[-backtest_days:].copy()
            
            bt_model = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2)], voting='soft')
            bt_model.fit(bt_train[features], bt_train['Target'])
            
            bt_test['Prob_Up'] = bt_model.predict_proba(bt_test[features])[:, 1]
            bt_test['Daily_Return'] = bt_test['Close'].pct_change().fillna(0)
            
            target_position = np.clip((bt_test['Prob_Up'] - 0.5) * 2, 0, 1)
            is_limit_locked = (bt_test['Daily_Return'].abs() >= 0.095) & (bt_test['High'] == bt_test['Low'])
            is_illiquid = bt_test['Volume'] < 200000
            can_trade = ~(is_limit_locked | is_illiquid)
            
            bt_test['Position_Size'] = np.where(can_trade, target_position, np.nan)
            bt_test['Position_Size'] = bt_test['Position_Size'].ffill().fillna(0)
            
            bt_test['Strategy_Return'] = bt_test['Position_Size'].shift(1).fillna(0) * bt_test['Daily_Return']
            bt_test['Position_Change'] = bt_test['Position_Size'].diff().fillna(0).abs()
            bt_test['Strategy_Return'] = bt_test['Strategy_Return'] - (bt_test['Position_Change'] * 0.003)
            bt_test['Strategy_Return'] = bt_test['Strategy_Return'].clip(lower=-0.05)
            
            bt_test['Cum_Market'] = (1 + bt_test['Daily_Return']).cumprod()
            bt_test['Cum_Strategy'] = (1 + bt_test['Strategy_Return']).cumprod()
            market_roi = (bt_test['Cum_Market'].iloc[-1] - 1) * 100
            strategy_roi = (bt_test['Cum_Strategy'].iloc[-1] - 1) * 100
            
            winning_days = len(bt_test[bt_test['Strategy_Return'] > 0])
            total_trade_days = len(bt_test[bt_test['Strategy_Return'] != 0])
            win_rate = (winning_days / total_trade_days * 100) if total_trade_days > 0 else 0
            
            daily_rf = 0.02 / 252
            excess_return = bt_test['Strategy_Return'] - daily_rf
            sharpe_ratio = (excess_return.mean() / excess_return.std()) * np.sqrt(252) if excess_return.std() != 0 else 0
            
            bt_test['Drawdown'] = (bt_test['Cum_Strategy'] - bt_test['Cum_Strategy'].cummax()) / bt_test['Cum_Strategy'].cummax()
            max_drawdown = bt_test['Drawdown'].min() * 100
            
            # 🆕 3. 紀錄績效指標到 MLflow
            mlflow.log_metrics({
                "strategy_roi": strategy_roi,
                "market_roi": market_roi,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate
            })
            
            # 🆕 4. 資料庫落地：將回測明細存入 SQLite
            try:
                # 連線到本地資料庫檔案 (若不存在會自動建立)
                conn = sqlite3.connect('quant_data.db')
                
                # 整理要存入資料庫的欄位
                db_df = bt_test[['Date', 'Close', 'Volume', 'Prob_Up', 'Position_Size', 'Strategy_Return', 'Cum_Strategy']].copy()
                db_df['Run_ID'] = mlflow.active_run().info.run_id # 關聯 MLflow 的 ID
                
                # 將 DataFrame 寫入名為 backtest_history 的資料表
                db_df.to_sql('backtest_history', con=conn, if_exists='append', index=False)
                conn.close()
                print("✅ 回測資料已成功寫入 SQLite 資料庫")
            except Exception as e:
                print(f"⚠️ 資料庫寫入失敗: {e}")

    return prediction, probability[prediction], top_features, strategy_roi, market_roi, sharpe_ratio, max_drawdown, win_rate, recommended_position, importance_df, bt_test