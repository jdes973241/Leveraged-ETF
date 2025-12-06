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
st.set_page_config(page_title="Dynamic Momentum Strategy", layout="wide")
warnings.simplefilter(action='ignore')

# CSS 美化 (含修正後的 metric 卡片樣式)
st.markdown("""
<style>
    /* 修正 metric card，強制深色文字以適配淺底色 */
    .metric-card {
        background-color: #eef2f5; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #d1d5db;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 14px; 
        color: #555555; 
        margin-bottom: 0;
        font-weight: 500;
    }
    .metric-value {
        font-size: 24px; 
        font-weight: bold; 
        color: #000000 !important; /* 強制黑色 */
        margin: 5px 0;
    }
    .metric-sub {
        font-size: 12px; 
        color: #666666; 
        margin-bottom: 0;
    }
    
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
ROLLING_WINDOW_SIZE = 1260 # GARCH 訓練視窗 (約5年)
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
    if data.empty: return "TLT", {}
    
    # 計算過去 12 個月 (252天) 報酬
    p_now = data[SAFE_POOL].iloc[-1]
    if len(data) > 252:
        p_prev = data[SAFE_POOL].iloc[-253]
        ret_12m = (p_now / p_prev) - 1
    else:
        ret_12m = pd.Series(0.0, index=SAFE_POOL)
        
    winner = ret_12m.idxmax()
    
    details = pd.DataFrame({
        "Ticker": SAFE_POOL,
        "Current Price": p_now.values,
        "12M Ago Price": p_prev.values if len(data) > 252 else [np.nan]*2,
        "12M Return": ret_12m.values
    }).set_index("Ticker")
    
    return winner, details

# ==========================================
# 2. 應用程式主邏輯
# ==========================================

data = get_market_data()

if data.empty:
    st.error("❌ 無法下載數據，請稍後再試。")
    st.stop()

risk_data = calculate_risk_metrics(data)
selection_df = calculate_selection_metrics(data)
safe_winner, safe_details_df = get_safe_asset_status(data)

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
    safe_ret = safe_details_df.loc[safe_winner, '12M Return']
    st.metric("🛡️ 當前最佳避險", safe_winner, 
              f"12M Ret: {safe_ret:.1%}")

st.divider()

# --- 透視表格 ---
st.subheader("📊 策略透視 (Strategy Whitebox)")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ 數據層", "2️⃣ 風控層", "3️⃣ 權重層", "4️⃣ 選股層", "5️⃣ 避險資產層", "6️⃣ 執行層"
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
    st.caption("避險資產輪動 (Safe Asset Rotation)")
    st.info("規則：若需要避險 (權重 < 1.0)，則比較 GLD 與 TLT 過去 12 個月的報酬，持有較強者。")
    
    safe_display = safe_details_df.copy()
    safe_display['Selected'] = safe_display.index.map(lambda x: '✅' if x == safe_winner else '')
    
    st.dataframe(
        safe_display.style.format({
            "Current Price": "{:.2f}",
            "12M Ago Price": "{:.2f}",
            "12M Return": "{:.2%}"
        }).map(lambda x: 'color: green' if x == '✅' else '', subset=['Selected']),
        use_container_width=True
    )

with tab6:
    st.markdown("#### 🚀 最終執行指令")
    
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
    3. 比較 GLD ({safe_details_df.loc['GLD', '12M Return']:.1%}) 與 TLT ({safe_details_df.loc['TLT', '12M Return']:.1%})。
    4. 選擇 **{safe_winner}** 作為避險部位。
    """)

# ==========================================
# 4. 歷史回測分析 (Backtest Section)
# ==========================================
st.markdown("---")
st.header("⏳ 歷史回測分析 (Backtest)")

# 估算回測起始日
if not data.empty:
    est_start_date = data.index[0] + timedelta(days=ROLLING_WINDOW_SIZE * 1.45) 
    start_date_str = est_start_date.strftime('%Y-%m-%d')
else:
    start_date_str = "N/A"

st.caption(f"""
設定說明：
- **回測起點**：約 {start_date_str} (扣除 {ROLLING_WINDOW_SIZE} 天 GARCH 暖機期)
- **交易成本**：{TRANSACTION_COST:.1%} | **避險**：GLD/TLT 輪動
- **基準 (Benchmark)**：UPRO + EURL + EDC (每季等權重再平衡)
""")

if st.button("🚀 開始執行回測"):
    
    with st.spinner("正在進行歷史運算 (History Calculation)..."):
        # 1. 歷史動能 (Monthly)
        monthly_prices = data[list(MAPPING.keys())].resample('M').last()
        hist_winners = pd.Series(index=monthly_prices.index, dtype='object')
        
        # 向量化計算動能
        mom_score = pd.DataFrame(0.0, index=monthly_prices.index, columns=monthly_prices.columns)
        for m in MOM_PERIODS:
            mom_score += monthly_prices.pct_change(m)
        
        for date in mom_score.index:
            hist_winners[date] = mom_score.loc[date].idxmax()
        
        # 2. 歷史避險訊號 (Daily)
        safe_mom = data[SAFE_POOL].pct_change(252)
        hist_safe = safe_mom.idxmax(axis=1).fillna('TLT')
        
        # 3. 逐日回測迴圈
        dates = data.index
        # [關鍵修正] 跳過 GARCH 暖機期
        start_idx = ROLLING_WINDOW_SIZE 
        
        if start_idx >= len(dates):
            st.error("數據長度不足以進行回測。")
            st.stop()
        
        strategy_ret = []
        valid_dates = []
        hold_counts = defaultdict(float)
        prev_pos = {} 
        
        progress_bar = st.progress(0)
        total_steps = len(dates) - start_idx
        
        for i in range(start_idx, len(dates)):
            if i % 100 == 0: progress_bar.progress((i - start_idx) / total_steps)
            today = dates[i]
            
            # Winner Logic
            past_wins = hist_winners[hist_winners.index < today]
            if past_wins.empty: continue
            target_risky = past_wins.iloc[-1]
            if pd.isna(target_risky) or target_risky not in MAPPING: continue

            # Weight Logic
            if target_risky in risk_data and today in risk_data[target_risky].index:
                w_risk = risk_data[target_risky].loc[today, 'Weight']
                if pd.isna(w_risk): w_risk = 0.0
            else:
                w_risk = 0.0 
            w_safe = 1.0 - w_risk
            
            # Safe Asset Logic
            target_safe = hist_safe.loc[today]
            if pd.isna(target_safe): target_safe = 'TLT' 
            
            # Cost Logic
            curr_pos = {}
            if w_risk > 0: curr_pos[target_risky] = w_risk
            if w_safe > 0: curr_pos[target_safe] = w_safe
            
            cost = 0.0
            all_assets = set(list(prev_pos.keys()) + list(curr_pos.keys()))
            for asset in all_assets:
                w_prev = prev_pos.get(asset, 0.0)
                w_curr = curr_pos.get(asset, 0.0)
                if w_prev != w_curr:
                    cost += abs(w_curr - w_prev) * TRANSACTION_COST
            
            # Return Logic
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
            hold_counts[target_risky] += w_risk
            hold_counts[target_safe] += w_safe
            prev_pos = curr_pos
            
        progress_bar.empty()
        
        # --- B. 分析結果 ---
        eq = pd.Series(strategy_ret, index=valid_dates)
        cum_eq = (1 + eq).cumprod()
        dd = cum_eq / cum_eq.cummax() - 1
        
        # [Benchmark 修正] Quarterly Rebalance (Equal Weight)
        bench_subset = data[list(MAPPING.keys())].loc[valid_dates].copy()
        b_equity_series = pd.Series(1.0, index=bench_subset.index)
        current_capital = 1.0
        
        # 找出季末日期
        quarter_ends = bench_subset.groupby(pd.Grouper(freq='QE')).apply(lambda x: x.index[-1] if len(x)>0 else None).dropna()
        check_points = sorted(list(set([bench_subset.index[0]] + list(quarter_ends) + [bench_subset.index[-1]])))
        
        for i in range(len(check_points)-1):
            t_start = check_points[i]
            t_end = check_points[i+1]
            if t_start >= t_end: continue
            
            # 區間計算 (以區間起點為基期歸一化)
            segment = bench_subset.loc[t_start:t_end]
            if len(segment) < 2: continue
            
            rel_price = segment.div(segment.iloc[0])
            # 等權重持有
            segment_val = rel_price.mean(axis=1) * current_capital
            
            b_equity_series.loc[t_start:t_end] = segment_val
            current_capital = segment_val.iloc[-1]
            
        bench_eq = b_equity_series
        bench_ret = bench_eq.pct_change().fillna(0)
        bench_dd = bench_eq / bench_eq.cummax() - 1
        
        # Metrics Helper
        def calc_stats(equity, daily_r):
            if len(equity) < 1: return 0,0,0,0,0
            d = (equity.index[-1] - equity.index[0]).days
            y = d / 365.25
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/y) - 1
            mdd = (equity / equity.cummax() - 1).min()
            neg = daily_r[daily_r < 0]
            down_std = neg.std() * np.sqrt(252)
            sortino = (cagr - RF_RATE) / (down_std + 1e-6)
            roll5 = equity.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260) - 1).mean()
            return cagr, sortino, roll5, mdd

        s_cagr, s_sort, s_roll, s_mdd = calc_stats(cum_eq, eq)
        b_cagr, b_sort, b_roll, b_mdd = calc_stats(bench_eq, bench_ret)
        
        # Time in Market
        total_d = len(valid_dates)
        time_in_mkt = (hold_counts['UPRO'] + hold_counts['EURL'] + hold_counts['EDC']) / total_d
        
        alloc_str = ""
        for k, v in hold_counts.items():
            pct = v / total_d
            if pct > 0.01: alloc_str += f"{k}:{pct:.0%} "
            
        # --- C. 顯示結果 ---
        st.write("### 📈 回測績效指標")
        m1, m2, m3, m4, m5 = st.columns(5)
        
        def metric_box(label, value, bench_val=None, fmt="{:.2%}"):
            bench_str = f"(Bench: {fmt.format(bench_val)})" if bench_val is not None else ""
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">{label}</p>
                <p class="metric-value">{fmt.format(value)}</p>
                <p class="metric-sub">{bench_str}</p>
            </div>
            """, unsafe_allow_html=True)

        with m1: metric_box("CAGR", s_cagr, b_cagr)
        with m2: metric_box("Sortino", s_sort, b_sort, "{:.2f}")
        with m3: metric_box("Avg 5Y Roll", s_roll, b_roll)
        with m4: metric_box("Max DD", s_mdd, b_mdd)
        with m5: metric_box("Time in 3x", time_in_mkt, None) 
        
        st.markdown(f"**資產分佈 (時間加權):** {alloc_str}")
        
        # Charts
        st.write("### 📊 權益曲線與回撤")
        
        df_chart = pd.DataFrame({
            'Date': cum_eq.index,
            'Strategy': cum_eq - 1,
            'Benchmark (EqW Qtly)': bench_eq - 1
        }).melt('Date', var_name='Asset', value_name='Return')
        
        chart = alt.Chart(df_chart).mark_line().encode(
            x='Date',
            y=alt.Y('Return', axis=alt.Axis(format='%')),
            color=alt.Color('Asset', scale=alt.Scale(domain=['Strategy', 'Benchmark (EqW Qtly)'], range=['#1f77b4', '#999999'])),
            tooltip=['Date', 'Asset', alt.Tooltip('Return', format='.2%')]
        ).properties(height=400, title="累積報酬率 (Cumulative Return)")
        
        st.altair_chart(chart, use_container_width=True)
        
        df_dd_chart = pd.DataFrame({
            'Date': cum_eq.index,
            'Strategy': dd,
            'Benchmark (EqW Qtly)': bench_dd
        }).melt('Date', var_name='Asset', value_name='Drawdown')
        
        chart_dd = alt.Chart(df_dd_chart).mark_line().encode(
            x='Date',
            y=alt.Y('Drawdown', axis=alt.Axis(format='%')),
            color=alt.Color('Asset', scale=alt.Scale(domain=['Strategy', 'Benchmark (EqW Qtly)'], range=['#ff7f0e', '#999999'])),
            tooltip=['Date', 'Asset', alt.Tooltip('Drawdown', format='.2%')]
        ).properties(height=200, title="回撤 (Drawdown)")
        
        st.altair_chart(chart_dd, use_container_width=True)
