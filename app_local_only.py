import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from arch import arch_model
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

# ==========================================
# 0. 頁面設定與參數
# ==========================================
st.set_page_config(page_title="Dynamic Momentum Strategy (Final Audited)", layout="wide")
warnings.simplefilter(action='ignore')

# CSS 美化
st.markdown("""
<style>
    .metric-card {
        background-color: #eef2f5; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #d1d5db;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label {font-size: 14px; color: #555555; margin-bottom: 0; font-weight: 500;}
    .metric-value {font-size: 24px; font-weight: bold; color: #000000 !important; margin: 5px 0;}
    .metric-sub {font-size: 12px; color: #666666; margin-bottom: 0;}
    .buy-text {color: #28a745; font-weight: bold;}
    .sell-text {color: #dc3545; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# === 核心參數 ===
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"} 
SAFE_POOL = ["GLD", "TLT"] 

# [修正 2] 統一參數為 Q80 / Q65
RISK_CONFIG = {
    "UPRO": {"exit_q": 0.80, "entry_q": 0.65},
    "EURL": {"exit_q": 0.80, "entry_q": 0.65},
    "EDC":  {"exit_q": 0.80, "entry_q": 0.65}
}

ROLLING_WINDOW_SIZE = 1260 
SMA_WINDOW = 200
MOM_PERIODS = [3, 6, 9, 12]
TRANSACTION_COST = 0.001 
RF_RATE = 0.04 

# === 合成數據參數 ===
LEVERAGE_RATIO = 3.0
def get_daily_leverage_cost(date):
    year = date.year
    if year <= 2007 or year >= 2022: return 0.05 / 252 
    else: return 0.02 / 252

# ==========================================
# 1. 核心邏輯函數 (Live Dashboard)
# ==========================================

@st.cache_data(ttl=3600, show_spinner="正在下載市場數據...")
def get_market_data():
    tickers = list(MAPPING.keys()) + list(MAPPING.values()) + SAFE_POOL
    try:
        data = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]: data = data['Close']
            else: data = data['Close'] if 'Close' in data else data
        
        start_filter = pd.Timestamp("2010-01-01")
        return data.loc[start_filter:].ffill().dropna()
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="正在計算 GARCH 風控模型...")
def calculate_risk_metrics(data):
    if data.empty: return {}
    risk_details = {}
    
    for trade_t, signal_t in MAPPING.items():
        if signal_t not in data.columns: continue

        series = data[signal_t]
        ret = series.pct_change() * 100
        sma = series.rolling(SMA_WINDOW).mean()
        
        window = ret.dropna().tail(1260*2) 
        if len(window) < 100: continue

        try:
            am = arch_model(window, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            cond_vol = res.conditional_volatility * np.sqrt(252)
            
            df = pd.DataFrame({'Price': series, 'Ret': ret, 'SMA': sma})
            df['Vol'] = cond_vol
            df = df.dropna()

            cfg = RISK_CONFIG[trade_t]
            # [修正 1] 避免未來視角: 使用 shift(1)
            # 今天的閾值是由昨天收盤算出的分布決定的
            df['Exit_Th'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
            df['Entry_Th'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
            
            df['GARCH_State'] = np.nan
            valid = df['Exit_Th'].notna()
            # 訊號判斷: 
            # 若今日Vol > 今日閾值(昨天算的)，則轉為避險
            # 這裡邏輯是: 盤中若波動率飆升超過警戒線，收盤確認後，明日執行避險
            df.loc[valid & (df['Vol'] > df['Exit_Th']), 'GARCH_State'] = 0.0 
            df.loc[valid & (df['Vol'] < df['Entry_Th']), 'GARCH_State'] = 1.0 
            df['GARCH_State'] = df['GARCH_State'].ffill().fillna(1.0)
            
            df['SMA_State'] = (df['Price'] > df['SMA']).astype(float)
            df['Weight'] = (0.5 * df['GARCH_State']) + (0.5 * df['SMA_State'])
            
            risk_details[trade_t] = df
        except: continue
        
    return risk_details

@st.cache_data(ttl=3600)
def calculate_selection_metrics(data):
    if data.empty: return pd.DataFrame()
    prices = data[list(MAPPING.keys())]
    metrics = []
    
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        p_now = prices[ticker].iloc[-1]
        for m in MOM_PERIODS:
            lookback = m * 21
            if len(prices) > lookback:
                p_prev = prices[ticker].iloc[-1-lookback]
                ret = (p_now - p_prev) / p_prev
                row[f'Ret_{m}M'] = ret
            else: row[f'Ret_{m}M'] = np.nan
        
        vol_window = 126
        daily_ret = prices[ticker].pct_change().tail(vol_window)
        vol = daily_ret.std() * np.sqrt(252)
        row['Vol_Ann'] = vol
        metrics.append(row)
        
    df = pd.DataFrame(metrics).set_index('Ticker')
    z_score_sum = pd.Series(0.0, index=df.index)
    for m in MOM_PERIODS:
        col = f'Ret_{m}M'
        risk_adj = df[col] / (df['Vol_Ann'] + 1e-6)
        z = (risk_adj - risk_adj.mean()) / (risk_adj.std() + 1e-6)
        df[f'Z_{m}M'] = z
        z_score_sum += z
    df['Total_Z'] = z_score_sum
    df['Rank'] = df['Total_Z'].rank(ascending=False)
    return df.sort_values('Total_Z', ascending=False)

@st.cache_data(ttl=3600)
def get_safe_asset_status(data):
    """
    [修正 3] 每月調整一次 GLD/TLT
    邏輯：比較上個月底 (Monthly Resample) 的 12M 報酬
    """
    if data.empty: return "TLT", {}
    
    # 取月度數據
    monthly = data[SAFE_POOL].resample('M').last()
    
    # 確保有足夠歷史
    if len(monthly) > 12:
        # 比較上個月底的數據 (iloc[-1] 是本月還沒走完的，iloc[-2] 是上個月底)
        # 實際上 Live Dashboard 應該看「最新已完成的月份」或「當下即時狀態」
        # 為了符合「每月調整一次」的邏輯，我們只取最近一個「月底」的訊號
        
        # 這裡我們取 monthly 的最後一筆 (即最新數據，可能是月中也可能是月底)
        # 但為了嚴謹，回測邏輯是月初看上個月底。Dashboard 顯示 "當前狀態"
        p_now = monthly.iloc[-1]
        p_prev = monthly.iloc[-13] # 12個月前
        ret_12m = (p_now / p_prev) - 1
    else:
        ret_12m = pd.Series(0.0, index=SAFE_POOL)
    
    winner = ret_12m.idxmax()
    
    details = pd.DataFrame({
        "Ticker": SAFE_POOL, 
        "12M Return": ret_12m.values
    }).set_index("Ticker")
    
    return winner, details

# ==========================================
# 2. 回測專用邏輯 (合成數據 + 長回測)
# ==========================================

@st.cache_data(ttl=3600, show_spinner="生成長歷史合成數據中 (2005~)...")
def get_synthetic_backtest_data():
    tickers = list(MAPPING.values()) + SAFE_POOL + ['VT']
    try:
        data_raw = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data_raw.columns, pd.MultiIndex):
            if 'Close' in data_raw.columns.levels[0]: data_raw = data_raw['Close']
            else: data_raw = data_raw['Close'] if 'Close' in data_raw else data_raw
        
        data_raw = data_raw.ffill().dropna(subset=['VGK', 'EEM', 'SPY', 'GLD', 'TLT'])
        
        synthetic_data = pd.DataFrame(index=data_raw.index)
        for t in SAFE_POOL + ['VT']:
            if t in data_raw.columns: synthetic_data[t] = data_raw[t]
            
        REVERSE_MAP = {v: k for k, v in MAPPING.items()} 
        for ticker_1x in MAPPING.values():
            ticker_3x = REVERSE_MAP[ticker_1x]
            ret_1x = data_raw[ticker_1x].pct_change().fillna(0)
            costs = pd.Series([get_daily_leverage_cost(d) for d in ret_1x.index], index=ret_1x.index)
            ret_3x = (ret_1x * 3.0) - costs
            syn_price = (1 + ret_3x).cumprod() * 100
            synthetic_data[ticker_3x] = syn_price
            synthetic_data[f"RAW_{ticker_3x}"] = data_raw[ticker_1x] 
            
        return synthetic_data
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 3. 應用程式主邏輯
# ==========================================

data = get_market_data()

if data.empty:
    st.error("❌ 無法下載數據，請稍後再試。")
    st.stop()

risk_data = calculate_risk_metrics(data)
selection_df = calculate_selection_metrics(data)
safe_winner, safe_details_df = get_safe_asset_status(data)

latest_date = data.index[-1]
winner_ticker = selection_df.index[0] 

if winner_ticker not in risk_data:
    st.error(f"❌ 缺少 {winner_ticker} 的風控數據。")
    st.stop()

winner_risk_df = risk_data[winner_ticker]
latest_risk_row = winner_risk_df.iloc[-1]
final_weight = latest_risk_row['Weight']

# ==========================================
# 4. Dashboard 前端顯示
# ==========================================

st.title("🛡️ 雙重動能與動態風控策略")
st.caption(f"數據基準日: {latest_date.strftime('%Y-%m-%d')}")

# 白皮書區塊
with st.expander("📖 策略白皮書 (Strategy Whitepaper)", expanded=False):
    st.markdown("""
    ### 策略邏輯摘要
    本策略採用 **訊號與執行分離 (Decoupled Signal)** 架構，利用 1x 原型預測風險，操作 3x 槓桿獲利。
    
    #### 1. 選股引擎 (Selection Engine)
    * **對象**: UPRO, EURL, EDC (3x 槓桿)。
    * **邏輯**: 計算 3M, 6M, 9M, 12M 的 **風險調整後報酬**，並進行 **Z-Score** 排序。
    * **決策**: 選出總分最高的標的作為本月 Winner。
    
    #### 2. 風控引擎 (Risk Engine)
    * **對象**: SPY, VGK, EEM (1x 原型)。
    * **A 軌 (GARCH)**: 每日滾動預測波動率。若 `Vol > Exit(Q80)` 避險；若 `Vol < Entry(Q65)` 持有。
    * **B 軌 (SMA)**: 若價格 > 200MA 持有；否則避險。
    * **權重**: 0.5 * GARCH + 0.5 * SMA。
    
    #### 3. 避險輪動 (Safe Asset Rotation)
    * 當風控建議空倉時，持有 **GLD** 或 **TLT**。
    * **規則**: **每月初** 比較兩者過去 12 個月績效，持有較強者。
    """)

# Summary Metrics
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🏆 本月進攻贏家", winner_ticker, "Rank #1")
with c2:
    if final_weight == 1.0: st.markdown(f"### 🎯 權重: :green[100%]")
    elif final_weight == 0.5: st.markdown(f"### 🎯 權重: :orange[50%]")
    else: st.markdown(f"### 🎯 權重: :red[0%]")
with c3:
    g_state = latest_risk_row['GARCH_State']
    st.metric("波動率風控 (GARCH)", "安全" if g_state == 1.0 else "危險", delta="✅" if g_state == 1.0 else "🔻")
with c4:
    safe_ret = safe_details_df.loc[safe_winner, '12M Return']
    st.metric("🛡️ 當前最佳避險", safe_winner, f"12M Ret: {safe_ret:.1%}")

st.divider()

# Strategy Tabs
st.subheader("📊 策略透視")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1️⃣ 數據層", "2️⃣ 風控層", "3️⃣ 權重層", "4️⃣ 選股層", "5️⃣ 避險資產層", "6️⃣ 執行層"])

with tab1:
    st.caption("最新市場價格 (含 1x 原型)")
    cols = list(MAPPING.keys()) + list(MAPPING.values()) + SAFE_POOL
    st.dataframe(data[cols].tail(5).sort_index(ascending=False).style.format("{:.2f}"), use_container_width=True)

with tab2:
    st.caption("風控指標詳情 (Q80 Exit / Q65 Entry)")
    risk_summary = []
    for ticker, signal_t in MAPPING.items():
        if ticker in risk_data:
            row = risk_data[ticker].iloc[-1]
            risk_summary.append({
                "標的": ticker, "Vol": f"{row['Vol']:.2f}%", 
                "Exit(Q80)": f"{row['Exit_Th']:.2f}%", "Entry(Q65)": f"{row['Entry_Th']:.2f}%",
                "GARCH": "🟢" if row['GARCH_State']==1 else "🔴",
                "SMA": "🟢" if row['SMA_State']==1 else "🔴"
            })
    st.dataframe(pd.DataFrame(risk_summary), use_container_width=True)

with tab3:
    st.caption("權重計算：0.5*GARCH + 0.5*SMA")
    w_summary = []
    for ticker in MAPPING.keys():
        if ticker in risk_data:
            row = risk_data[ticker].iloc[-1]
            w_summary.append({
                "標的": ticker, "GARCH(0/1)": int(row['GARCH_State']), "SMA(0/1)": int(row['SMA_State']), "總權重": row['Weight']
            })
    st.dataframe(pd.DataFrame(w_summary), use_container_width=True)

with tab4:
    st.caption("動能排名 (Risk-Adjusted Z-Score)")
    st.dataframe(selection_df.style.format("{:.2f}"), use_container_width=True)

with tab5:
    st.caption("避險資產輪動 (Safe Asset Rotation)")
    safe_display = safe_details_df.copy()
    safe_display['Selected'] = safe_display.index.map(lambda x: '✅' if x == safe_winner else '')
    st.dataframe(safe_display.style.format({"12M Return": "{:.2%}"}).map(lambda x: 'color: green' if x == '✅' else '', subset=['Selected']), use_container_width=True)

with tab6:
    st.markdown("#### 🚀 最終執行指令")
    holdings = []
    if final_weight > 0: holdings.append(f"**{final_weight*100:.0f}% {winner_ticker}** (進攻)")
    safe_weight = 1.0 - final_weight
    if safe_weight > 0: holdings.append(f"**{safe_weight*100:.0f}% {safe_winner}** (避險)")
    st.success(f"建議組合: {' + '.join(holdings)}")

# ==========================================
# 5. 歷史回測分析 (Synthetic Backtest)
# ==========================================
st.markdown("---")
st.header("⏳ 歷史回測分析 (Synthetic)")

syn_data = get_synthetic_backtest_data()

if syn_data.empty:
    st.warning("合成數據生成失敗。")
else:
    BACKTEST_GARCH_WINDOW = 504 
    est_start_date = syn_data.index[0] + timedelta(days=(BACKTEST_GARCH_WINDOW + 252) * 1.45) 
    start_date_str = est_start_date.strftime('%Y-%m-%d')

    st.caption(f"""
    **回測設定說明：**
    1.  **數據源**：使用 1x 原型合成 3x 數據 (含動態損耗)。
    2.  **回測起點**：約 {start_date_str} (確保覆蓋 2008)。
    3.  **交易成本**：0.1% | **GARCH 暖機**：2 年 (504天)。
    4.  **避險**：GLD/TLT (每月切換一次)。
    5.  **基準 (Benchmark)**：UPRO + EURL + EDC (每季等權重)。
    """)

    if st.button("🚀 開始執行回測 (Synthetic)"):
        with st.spinner("正在進行歷史運算..."):
            # 1. 計算歷史風控權重
            h_risk_weights = pd.DataFrame(index=syn_data.index, columns=MAPPING.keys())
            
            for ticker_3x in MAPPING.keys():
                col_1x = f"RAW_{ticker_3x}"
                if col_1x not in syn_data.columns: continue
                s = syn_data[col_1x]
                r = s.pct_change() * 100
                sma = s.rolling(SMA_WINDOW).mean()
                
                win = r.dropna()
                am = arch_model(win, vol='Garch', p=1, q=1, dist='t', rescale=False)
                res = am.fit(disp='off', show_warning=False)
                vol = res.conditional_volatility * np.sqrt(252)
                
                df = pd.DataFrame({'Vol': vol, 'Price': s, 'SMA': sma})
                cfg = RISK_CONFIG[ticker_3x]
                
                # [修正 1] 應用 Shift(1) 避免未來視角
                roll_ex = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
                roll_en = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
                
                g_sig = pd.Series(np.nan, index=df.index)
                valid = roll_ex.notna()
                g_sig.loc[valid & (df['Vol'] > roll_ex)] = 0.0
                g_sig.loc[valid & (df['Vol'] < roll_en)] = 1.0
                g_sig = g_sig.ffill().fillna(0.0)
                
                s_sig = (df['Price'] > df['SMA']).astype(float)
                h_risk_weights[ticker_3x] = (0.5 * g_sig) + (0.5 * s_sig)
                
            h_risk_weights = h_risk_weights.dropna()
            
            # 2. 歷史動能 (Selection) - 月頻
            monthly_prices = syn_data[list(MAPPING.keys())].resample('M').last()
            mom_score = pd.DataFrame(0.0, index=monthly_prices.index, columns=monthly_prices.columns)
            for m in MOM_PERIODS: mom_score += monthly_prices.pct_change(m)
            hist_winners = mom_score.idxmax(axis=1)
            
            # 3. 歷史避險 (Rotation) - [修正 3] 月頻
            safe_monthly = syn_data[SAFE_POOL].resample('M').last()
            safe_mom = safe_monthly.pct_change(12) # 12個月
            hist_safe = safe_mom.idxmax(axis=1).fillna('TLT')
            
            # 4. 逐日回測
            dates = syn_data.index
            start_idx = BACKTEST_GARCH_WINDOW + 252 
            
            strategy_ret = []
            valid_dates = []
            hold_counts = defaultdict(float)
            prev_pos = {} 
            
            progress = st.progress(0)
            
            for i in range(start_idx, len(dates)):
                if i % 100 == 0: progress.progress((i - start_idx) / (len(dates)-start_idx))
                today = dates[i]
                
                # 取得"昨天"的日期 (或上次訊號更新日)
                yesterday = dates[i-1]
                
                # [關鍵修正] 使用昨天以前的數據決定今日持倉
                
                # A. 決定進攻標的 (每月初更新)
                # 找到 yesterday 之前最近的一個月底
                past_wins = hist_winners[hist_winners.index <= yesterday]
                if past_wins.empty: continue
                target_risky = past_wins.iloc[-1]
                
                # B. 決定避險標的 (每月初更新) [修正 3]
                past_safe = hist_safe[hist_safe.index <= yesterday]
                if past_safe.empty: target_safe = 'TLT'
                else: target_safe = past_safe.iloc[-1]
                
                # C. 決定權重 (每日更新)
                if target_risky in h_risk_weights.columns and yesterday in h_risk_weights.index:
                    w_risk = h_risk_weights.loc[yesterday, target_risky]
                    if pd.isna(w_risk): w_risk = 0.0
                else: w_risk = 0.0
                w_safe = 1.0 - w_risk
                
                # D. 構建持倉
                curr_pos = {}
                if w_risk > 0: curr_pos[target_risky] = w_risk
                if w_safe > 0: curr_pos[target_safe] = w_safe
                
                # E. 計算成本
                cost = 0.0
                all_assets = set(list(prev_pos.keys()) + list(curr_pos.keys()))
                for asset in all_assets:
                    w_prev = prev_pos.get(asset, 0.0)
                    w_curr = curr_pos.get(asset, 0.0)
                    if w_prev != w_curr:
                        cost += abs(w_curr - w_prev) * TRANSACTION_COST
                
                # F. 計算損益 (今日漲跌)
                day_ret = 0.0
                if w_risk > 0:
                    r = syn_data[target_risky].pct_change().iloc[i]
                    if np.isnan(r): r=0
                    day_ret += w_risk * r
                if w_safe > 0:
                    r = syn_data[target_safe].pct_change().iloc[i]
                    if np.isnan(r): r=0
                    day_ret += w_safe * r
                    
                strategy_ret.append(day_ret - cost)
                valid_dates.append(today)
                hold_counts[target_risky] += w_risk
                hold_counts[target_safe] += w_safe
                prev_pos = curr_pos
                
            progress.empty()
            
            # --- Result ---
            eq = pd.Series(strategy_ret, index=valid_dates)
            cum_eq = (1 + eq).cumprod()
            dd = cum_eq / cum_eq.cummax() - 1
            
            # Benchmark (Qtly EqW)
            b_subset = syn_data[list(MAPPING.keys())].loc[valid_dates].copy()
            b_equity_series = pd.Series(1.0, index=b_subset.index)
            curr_cap = 1.0
            q_ends = b_subset.groupby(pd.Grouper(freq='QE')).apply(lambda x: x.index[-1] if len(x)>0 else None).dropna()
            cps = sorted(list(set([b_subset.index[0]] + list(q_ends) + [b_subset.index[-1]])))
            
            for i in range(len(cps)-1):
                t_s = cps[i]
                t_e = cps[i+1]
                if t_s >= t_e: continue
                seg = b_subset.loc[t_s:t_e]
                if len(seg) < 2: continue
                rel = seg.div(seg.iloc[0])
                val = rel.mean(axis=1) * curr_cap
                b_equity_series.loc[t_s:t_e] = val
                curr_cap = val.iloc[-1]
            
            bench_eq = b_equity_series
            bench_dd = bench_eq / bench_eq.cummax() - 1
            
            # Benchmark 2 (VT)
            vt_eq = pd.Series(1.0, index=valid_dates)
            if 'VT' in syn_data.columns:
                vt_ret = syn_data['VT'].loc[valid_dates].pct_change().fillna(0)
                vt_eq = (1 + vt_ret).cumprod()
            vt_dd = vt_eq / vt_eq.cummax() - 1
            
            # Stats
            def calc_stats(equity, daily_r):
                if len(equity) < 1: return 0,0,0,0
                d = (equity.index[-1] - equity.index[0]).days
                y = d / 365.25
                cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/y) - 1
                mdd = (equity / equity.cummax() - 1).min()
                neg = daily_r[daily_r < 0]
                sortino = (cagr - RF_RATE) / (neg.std() * np.sqrt(252) + 1e-6)
                roll5 = equity.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260) - 1).mean()
                return cagr, sortino, roll5, mdd

            s_cagr, s_sort, s_roll, s_mdd = calc_stats(cum_eq, eq)
            b3_cagr, b3_sort, b3_roll, b3_mdd = calc_stats(bench_eq, bench_eq.pct_change().fillna(0))
            vt_cagr, vt_sort, vt_roll, vt_mdd = calc_stats(vt_eq, vt_eq.pct_change().fillna(0))
            
            total_d = len(valid_dates)
            time_in_mkt = (hold_counts['UPRO'] + hold_counts['EURL'] + hold_counts['EDC']) / total_d
            
            alloc_str = " | ".join([f"{k.replace('Syn_','')}:{v/total_d:.0%}" for k, v in hold_counts.items() if v/total_d > 0.01])
            
            # --- Display ---
            st.write("### 📈 回測績效指標")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            def metric_box(label, value, b3_val=None, vt_val=None, fmt="{:.2%}"):
                b3_str = f"3x: {fmt.format(b3_val)}" if b3_val is not None else ""
                vt_str = f"VT: {fmt.format(vt_val)}" if vt_val is not None else ""
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{label}</p>
                    <p class="metric-value">{fmt.format(value)}</p>
                    <p class="metric-sub">{b3_str} | {vt_str}</p>
                </div>
                """, unsafe_allow_html=True)

            with m1: metric_box("CAGR", s_cagr, b3_cagr, vt_cagr)
            with m2: metric_box("Sortino", s_sort, b3_sort, vt_sort, "{:.2f}")
            with m3: metric_box("Avg 5Y Roll", s_roll, b3_roll, vt_roll)
            with m4: metric_box("Max DD", s_mdd, b3_mdd, vt_mdd)
            with m5: metric_box("Time in 3x", time_in_mkt, None, None) 
            
            st.markdown(f"**平均資產分佈:** {alloc_str}")
            
            # Charts
            st.write("### 📊 權益曲線與回撤")
            
            df_chart = pd.DataFrame({
                'Date': cum_eq.index,
                'Strategy': cum_eq,
                'Bench (3x EqW)': bench_eq,
                'Bench (VT)': vt_eq
            }).melt('Date', var_name='Asset', value_name='NAV')
            
            chart = alt.Chart(df_chart).mark_line().encode(
                x='Date',
                y=alt.Y('NAV', axis=alt.Axis(title='NAV (Log)'), scale=alt.Scale(type='log')),
                color=alt.Color('Asset', scale=alt.Scale(domain=['Strategy', 'Bench (3x EqW)', 'Bench (VT)'], range=['#1f77b4', '#999999', '#2ca02c'])),
                tooltip=['Date', 'Asset', alt.Tooltip('NAV', format='.2f')]
            ).properties(height=350, title="權益曲線 (Log Scale)").interactive()
            st.altair_chart(chart, use_container_width=True)
            
            df_dd = pd.DataFrame({
                'Date': cum_eq.index,
                'Strategy': dd,
                'Bench (3x EqW)': bench_dd,
                'Bench (VT)': vt_dd
            }).melt('Date', var_name='Asset', value_name='Drawdown')
            
            chart_dd = alt.Chart(df_dd).mark_line().encode(
                x='Date', y=alt.Y('Drawdown', axis=alt.Axis(format='%')),
                color='Asset', tooltip=['Date', 'Asset', alt.Tooltip('Drawdown', format='.2%')]
            ).properties(height=200, title="回撤幅度").interactive()
            st.altair_chart(chart_dd, use_container_width=True)
            
            roll5_s = cum_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260) - 1)
            roll5_b = bench_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260) - 1)
            roll5_v = vt_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260) - 1)
            
            df_roll = pd.DataFrame({
                'Date': cum_eq.index, 'Strategy': roll5_s, 'Bench (3x)': roll5_b, 'Bench (VT)': roll5_v
            }).melt('Date', var_name='Asset', value_name='CAGR')
            
            chart_roll = alt.Chart(df_roll.dropna()).mark_line().encode(
                x='Date', y=alt.Y('CAGR', axis=alt.Axis(format='%')),
                color='Asset', tooltip=['Date', 'Asset', alt.Tooltip('CAGR', format='.2%')]
            ).properties(height=250, title="滾動 5 年年化報酬率 (Rolling 5Y CAGR)").interactive()
            st.altair_chart(chart_roll, use_container_width=True)
