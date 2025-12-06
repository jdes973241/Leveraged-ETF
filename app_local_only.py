import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from arch import arch_model
from datetime import datetime, timedelta
import warnings

# ==========================================
# 0. 頁面設定與參數
# ==========================================
st.set_page_config(page_title="Dynamic Momentum Strategy", layout="wide")
warnings.simplefilter(action='ignore')

# CSS 美化
st.markdown("""
<style>
    .metric-card {background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4;}
    .buy-text {color: #28a745; font-weight: bold;}
    .sell-text {color: #dc3545; font-weight: bold;}
    .neutral-text {color: #6c757d; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# 策略參數
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"} # 3x -> 1x
SAFE_POOL = ["GLD", "TLT"] # 避險資產池
RISK_CONFIG = {
    "UPRO": {"exit_q": 0.85, "entry_q": 0.70},
    "EURL": {"exit_q": 0.97, "entry_q": 0.82},
    "EDC":  {"exit_q": 0.70, "entry_q": 0.55}
}
ROLLING_WINDOW_SIZE = 1260
SMA_WINDOW = 200
MOM_PERIODS = [3, 6, 9, 12]
TRANSACTION_COST = 0.001 # 0.1%
RF_RATE = 0.04 # 無風險利率

# ==========================================
# 1. 核心邏輯函數 (快取優化)
# ==========================================

@st.cache_data(ttl=3600, show_spinner="正在下載市場數據...")
def get_market_data():
    """下載所有相關標的數據 (含避險資產)"""
    tickers = list(MAPPING.keys()) + list(MAPPING.values()) + SAFE_POOL
    # 下載較長歷史以供回測
    try:
        data = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]: data = data['Close']
            else: data = data['Close'] if 'Close' in data else data
        
        # 只取 2010 年後 (確保 UPRO/EDC 上市)
        start_filter = pd.Timestamp("2010-01-01")
        return data.loc[start_filter:].ffill().dropna()
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="正在計算 GARCH 風控模型...")
def calculate_risk_metrics(data):
    """計算風控層的所有數據"""
    if data.empty: return {}
    risk_details = {}
    
    for trade_t, signal_t in MAPPING.items():
        if signal_t not in data.columns: continue

        series = data[signal_t]
        ret = series.pct_change() * 100
        sma = series.rolling(SMA_WINDOW).mean()
        
        # GARCH 計算 (Dashboard 使用全區間擬合做快速近似)
        window = ret.dropna().tail(1260*2) # 取較長區間
        if len(window) < 100: continue

        try:
            am = arch_model(window, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            cond_vol = res.conditional_volatility * np.sqrt(252)
            
            # 整合與對齊
            df = pd.DataFrame({'Price': series, 'Ret': ret, 'SMA': sma})
            df['Vol'] = cond_vol
            df = df.dropna()

            # 動態閾值
            cfg = RISK_CONFIG[trade_t]
            df['Exit_Th'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
            df['Entry_Th'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
            
            # 訊號狀態
            df['GARCH_State'] = np.nan
            valid = df['Exit_Th'].notna()
            df.loc[valid & (df['Vol'] > df['Exit_Th']), 'GARCH_State'] = 0.0 
            df.loc[valid & (df['Vol'] < df['Entry_Th']), 'GARCH_State'] = 1.0 
            df['GARCH_State'] = df['GARCH_State'].ffill().fillna(1.0)
            
            df['SMA_State'] = (df['Price'] > df['SMA']).astype(float)
            df['Weight'] = (0.5 * df['GARCH_State']) + (0.5 * df['SMA_State'])
            
            risk_details[trade_t] = df
        except:
            continue
        
    return risk_details

@st.cache_data(ttl=3600)
def calculate_selection_metrics(data):
    """計算動能選股層 (使用 3x 標的)"""
    if data.empty: return pd.DataFrame()
    prices = data[list(MAPPING.keys())]
    
    metrics = []
    # 只計算最新一天的狀態供 Dashboard 使用
    # 回測時會另外計算歷史序列
    latest_date = prices.index[-1]
    
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        p_now = prices[ticker].iloc[-1]
        
        for m in MOM_PERIODS:
            lookback = m * 21
            if len(prices) > lookback:
                p_prev = prices[ticker].iloc[-1-lookback]
                ret = (p_now - p_prev) / p_prev
                row[f'Ret_{m}M'] = ret
            else:
                row[f'Ret_{m}M'] = np.nan
                
        vol_window = 126
        daily_ret = prices[ticker].pct_change().tail(vol_window)
        vol = daily_ret.std() * np.sqrt(252)
        row['Vol_Ann'] = vol
        metrics.append(row)
        
    df = pd.DataFrame(metrics).set_index('Ticker')
    
    # Z-Score
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
    """計算當前避險資產 (GLD vs TLT)"""
    if data.empty: return "TLT"
    # 計算過去 12 個月 (252天) 報酬
    subset = data[SAFE_POOL].tail(253)
    ret_12m = (subset.iloc[-1] / subset.iloc[0]) - 1
    winner = ret_12m.idxmax()
    return winner, ret_12m

# ==========================================
# 2. 應用程式主邏輯
# ==========================================

data = get_market_data()

if data.empty:
    st.error("❌ 無法下載數據，請稍後再試。")
    st.stop()

risk_data = calculate_risk_metrics(data)
selection_df = calculate_selection_metrics(data)
safe_winner, safe_rets = get_safe_asset_status(data)

# 取得最新狀態
latest_date = data.index[-1]
winner_ticker = selection_df.index[0] 

if winner_ticker not in risk_data:
    st.error(f"❌ 缺少 {winner_ticker} 的風控數據。")
    st.stop()

winner_risk_df = risk_data[winner_ticker]
latest_risk_row = winner_risk_df.iloc[-1]
final_weight = latest_risk_row['Weight']

# ==========================================
# 3. Dashboard 前端顯示
# ==========================================

st.title("🛡️ 雙重動能與動態風控策略 (Live)")
st.caption(f"數據基準日: {latest_date.strftime('%Y-%m-%d')}")

# --- Top Summary ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🏆 本月進攻贏家", winner_ticker, "Rank #1")

with c2:
    if final_weight == 1.0: 
        st.markdown(f"### 🎯 權重: :green[100%]")
        st.caption(f"持有 {winner_ticker}")
    elif final_weight == 0.5:
        st.markdown(f"### 🎯 權重: :orange[50%]")
        st.caption(f"50% {winner_ticker} + 50% {safe_winner}")
    else:
        st.markdown(f"### 🎯 權重: :red[0%]")
        st.caption(f"100% {safe_winner} (避險)")

with c3:
    g_state = latest_risk_row['GARCH_State']
    st.metric("波動率風控 (GARCH)", 
              "安全" if g_state == 1.0 else "危險", 
              delta="✅" if g_state == 1.0 else "🔻")

with c4:
    st.metric("🛡️ 當前最佳避險", safe_winner, 
              f"12M Ret: {safe_rets[safe_winner]:.1%}")

st.divider()

# --- 透視表格 ---
st.subheader("📊 策略透視 (Strategy Whitebox)")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 數據層", "2️⃣ 風控層", "3️⃣ 權重層", "4️⃣ 選股層", "5️⃣ 執行層"
])

with tab1:
    st.caption("最新市場價格")
    cols = list(MAPPING.keys()) + SAFE_POOL
    st.dataframe(data[cols].tail(5).sort_index(ascending=False).style.format("{:.2f}"), use_container_width=True)

with tab2:
    st.caption("風控指標詳情")
    risk_summary = []
    for ticker, signal_t in MAPPING.items():
        if ticker in risk_data:
            row = risk_data[ticker].iloc[-1]
            risk_summary.append({
                "標的": ticker, "Vol": f"{row['Vol']:.2f}%", 
                "Exit": f"{row['Exit_Th']:.2f}%", "GARCH": "🟢" if row['GARCH_State']==1 else "🔴",
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
                "標的": ticker, "GARCH(0/1)": int(row['GARCH_State']), 
                "SMA(0/1)": int(row['SMA_State']), "總權重": row['Weight']
            })
    st.dataframe(pd.DataFrame(w_summary), use_container_width=True)

with tab4:
    st.caption("動能排名 (Risk-Adjusted Z-Score)")
    st.dataframe(selection_df.style.format("{:.2f}"), use_container_width=True)

with tab5:
    st.markdown("#### 🚀 最終執行指令")
    
    # 邏輯判斷
    holdings = []
    if final_weight > 0:
        holdings.append(f"**{final_weight*100:.0f}% {winner_ticker}** (進攻)")
    
    safe_weight = 1.0 - final_weight
    if safe_weight > 0:
        holdings.append(f"**{safe_weight*100:.0f}% {safe_winner}** (避險)")
        
    st.success(f"建議組合: {' + '.join(holdings)}")
    
    st.info(f"""
    **決策邏輯：**
    1. 進攻標的 **{winner_ticker}** 的風控權重為 {final_weight}。
    2. 剩餘 {safe_weight} 權重配置於避險資產。
    3. 比較 GLD ({safe_rets['GLD']:.1%}) 與 TLT ({safe_rets['TLT']:.1%}) 過去 12 個月績效。
    4. 選擇 **{safe_winner}** 作為避險部位。
    """)

# ==========================================
# 4. 歷史回測分析 (Backtest Section)
# ==========================================
st.markdown("---")
st.header("⏳ 歷史回測分析 (Backtest)")
st.caption("回測設定：2010 ~ 至今 | 交易成本 0.1% | 避險: 輪動持有 GLD/TLT")

if st.button("🚀 開始執行回測"):
    
    # --- A. 準備回測數據 ---
    # 為了速度，我們重用 Dashboard 計算好的 risk_data，
    # 但需要重新計算完整的歷史動能與避險訊號
    
    with st.spinner("正在進行歷史運算..."):
        # 1. 歷史動能 (Monthly)
        monthly_prices = data[list(MAPPING.keys())].resample('M').last()
        hist_winners = pd.Series(index=monthly_prices.index, dtype='object')
        
        # 向量化計算動能 (簡化版加速)
        # 這裡用簡單的回報率總和近似 Z-Score (為了 Web App 響應速度)
        # 若要精確 Z-Score 需迴圈，這裡演示核心邏輯
        mom_score = pd.DataFrame(0.0, index=monthly_prices.index, columns=monthly_prices.columns)
        for m in MOM_PERIODS:
            mom_score += monthly_prices.pct_change(m)
        
        # 找出每個月的 Winner
        for date in mom_score.index:
            hist_winners[date] = mom_score.loc[date].idxmax()
        
        # 2. 歷史避險訊號 (Daily)
        # 比較 GLD vs TLT 252天回報
        safe_mom = data[SAFE_POOL].pct_change(252)
        hist_safe = safe_mom.idxmax(axis=1).fillna('TLT')
        
        # 3. 逐日回測迴圈
        dates = data.index
        # 找出共同起始點
        start_idx = 252 # 暖機期
        
        strategy_ret = []
        valid_dates = []
        
        # 持倉統計
        hold_counts = {t:0 for t in list(MAPPING.keys()) + SAFE_POOL}
        
        prev_pos = {} # {ticker: weight}
        
        # 進度條
        progress_bar = st.progress(0)
        total_steps = len(dates) - start_idx
        
        for i in range(start_idx, len(dates)):
            if i % 100 == 0: progress_bar.progress((i - start_idx) / total_steps)
            
            today = dates[i]
            
            # 決定 Winner (上個月底)
            past_wins = hist_winners[hist_winners.index < today]
            if past_wins.empty: continue
            
            target_risky = past_wins.iloc[-1]
            
            # 決定權重
            # 檢查該標的是否有風控數據
            if target_risky in risk_data and today in risk_data[target_risky].index:
                w_risk = risk_data[target_risky].loc[today, 'Weight']
            else:
                w_risk = 0.0 # 若無數據預設避險
                
            w_safe = 1.0 - w_risk
            
            # 決定避險標的
            target_safe = hist_safe.loc[today]
            
            # 建構倉位
            curr_pos = {}
            if w_risk > 0: curr_pos[target_risky] = w_risk
            if w_safe > 0: curr_pos[target_safe] = w_safe
            
            # 統計
            hold_counts[target_risky] += w_risk
            hold_counts[target_safe] += w_safe
            
            # 計算成本
            cost = 0.0
            all_assets = set(list(prev_pos.keys()) + list(curr_pos.keys()))
            for asset in all_assets:
                w_prev = prev_pos.get(asset, 0.0)
                w_curr = curr_pos.get(asset, 0.0)
                if w_prev != w_curr:
                    cost += abs(w_curr - w_prev) * TRANSACTION_COST
            
            # 計算報酬
            day_ret = 0.0
            if w_risk > 0:
                r = data[target_risky].pct_change().iloc[i]
                if np.isnan(r): r=0
                day_ret += w_risk * r
            
            if w_safe > 0:
                r = data[target_safe].pct_change().iloc[i]
                if np.isnan(r): r=0
                day_ret += w_safe * r
                
            strategy_ret.append(day_ret - cost)
            valid_dates.append(today)
            prev_pos = curr_pos
            
        progress_bar.empty()
        
        # --- B. 分析結果 ---
        eq = pd.Series(strategy_ret, index=valid_dates)
        cum_eq = (1 + eq).cumprod()
        dd = cum_eq / cum_eq.cummax() - 1
        
        # Benchmark (VT)
        if 'VT' not in data.columns:
            # 如果沒下載 VT，用 SPY 代替
            bench_ret = data['SPY'].loc[valid_dates].pct_change().fillna(0)
        else:
            bench_ret = data['VT'].loc[valid_dates].pct_change().fillna(0)
        bench_eq = (1 + bench_ret).cumprod()
        
        # 指標計算
        days = (cum_eq.index[-1] - cum_eq.index[0]).days
        years = days / 365.25
        cagr = (cum_eq.iloc[-1] / cum_eq.iloc[0]) ** (1/years) - 1
        mdd = dd.min()
        
        neg_ret = eq[eq < 0]
        down_std = neg_ret.std() * np.sqrt(252)
        sortino = (cagr - RF_RATE) / (down_std + 1e-6)
        
        roll_5y = cum_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260) - 1).mean()
        
        # Time in Market (3x 資產)
        total_d = len(valid_dates)
        time_in_mkt = (hold_counts['UPRO'] + hold_counts['EURL'] + hold_counts['EDC']) / total_d
        
        # 佔比
        alloc_str = ""
        for k, v in hold_counts.items():
            pct = v / total_d
            if pct > 0.01: alloc_str += f"{k}:{pct:.0%} "
            
        # --- C. 顯示結果 (依照您的範例格式) ---
        
        # 1. 關鍵指標
        st.write("### 📈 回測績效指標")
        m1, m2, m3, m4, m5 = st.columns(5)
        
        def metric_box(label, value, fmt="{:.2%}"):
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="margin:0; font-size: 14px; color: #555;">{label}</p>
                <p style="margin:0; font-size: 20px; font-weight: bold;">{fmt.format(value)}</p>
            </div>
            """, unsafe_allow_html=True)

        with m1: metric_box("CAGR", cagr)
        with m2: metric_box("Sortino", sortino, "{:.2f}")
        with m3: metric_box("Avg 5Y Roll", roll_5y)
        with m4: metric_box("Max DD", mdd)
        with m5: metric_box("Time in 3x", time_in_mkt)
        
        st.markdown(f"**資產分佈 (時間加權):** {alloc_str}")
        
        # 2. Altair 圖表
        st.write("### 📊 權益曲線與回撤")
        
        df_chart = pd.DataFrame({
            'Date': cum_eq.index,
            'Strategy': cum_eq - 1,
            'Benchmark': bench_eq - 1
        }).melt('Date', var_name='Asset', value_name='Return')
        
        chart = alt.Chart(df_chart).mark_line().encode(
            x='Date',
            y=alt.Y('Return', axis=alt.Axis(format='%')),
            color=alt.Color('Asset', scale=alt.Scale(range=['#1f77b4', '#999999'])),
            tooltip=['Date', 'Asset', alt.Tooltip('Return', format='.2%')]
        ).properties(height=400, title="累積報酬率 (Cumulative Return)")
        
        st.altair_chart(chart, use_container_width=True)
        
        # 回撤圖
        df_dd_chart = pd.DataFrame({
            'Date': cum_eq.index,
            'Drawdown': dd
        })
        
        chart_dd = alt.Chart(df_dd_chart).mark_area(color='#ff7f0e', opacity=0.5).encode(
            x='Date',
            y=alt.Y('Drawdown', axis=alt.Axis(format='%')),
            tooltip=['Date', alt.Tooltip('Drawdown', format='.2%')]
        ).properties(height=200, title="回撤 (Drawdown)")
        
        st.altair_chart(chart_dd, use_container_width=True)
