import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from arch import arch_model
from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm
from tabulate import tabulate
import warnings

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="Dynamic Momentum (Monthly Lock)", layout="wide")
warnings.simplefilter(action='ignore')
alt.data_transformers.disable_max_rows()

# CSS
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 策略參數 (白皮書規格)
# ==========================================
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"} 
SAFE_POOL = ["GLD", "TLT"] 

# 風控參數 (Q80/Q65)
RISK_CONFIG = {
    "UPRO": {"exit_q": 0.80, "entry_q": 0.65},
    "EURL": {"exit_q": 0.80, "entry_q": 0.65},
    "EDC":  {"exit_q": 0.80, "entry_q": 0.65}
}

# 回測設定
BACKTEST_GARCH_WINDOW = 504 
SMA_WINDOW = 200
MOM_PERIODS = [3, 6, 9, 12]
TRANSACTION_COST = 0.001 
RF_RATE = 0.02 

# === 合成數據參數 ===
def get_daily_leverage_cost(date):
    year = date.year
    if year <= 2007 or year >= 2022: return 0.05 / 252 
    else: return 0.02 / 252

# ==========================================
# 2. 輔助函數 (日期處理)
# ==========================================
def get_monthly_data(df):
    """
    獲取每個月'實際最後交易日'的數據，避免 resample('M') 產生非交易日的 Bug。
    """
    if df.empty: return df
    period_idx = df.index.to_period('M')
    month_end_dates = df.index.to_series().groupby(period_idx).max()
    return df.loc[month_end_dates]

# ==========================================
# 3. 數據合成 (Synthetic Backtest Data)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="正在下載與合成數據...")
def get_synthetic_backtest_data():
    tickers = list(MAPPING.values()) + SAFE_POOL + ['VT']
    try:
        data_raw = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data_raw.columns, pd.MultiIndex):
            if 'Close' in data_raw.columns.levels[0]: data_raw = data_raw['Close']
            else: data_raw = data_raw['Close'] if 'Close' in data_raw else data_raw
        
        # 保留核心資產數據
        data_raw = data_raw.ffill().dropna(subset=['VGK', 'EEM', 'SPY', 'GLD', 'TLT'])
        
        synthetic_data = pd.DataFrame(index=data_raw.index)
        if 'VT' in data_raw.columns: synthetic_data['VT'] = data_raw['VT']
        for t in SAFE_POOL: 
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
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 4. Live 面板計算 (鎖定上個月底)
# ==========================================
@st.cache_data(ttl=3600)
def get_live_market_data():
    # 下載真實 ETF 數據
    real_tickers = list(MAPPING.keys()) + list(MAPPING.values()) + SAFE_POOL
    try:
        data = yf.download(real_tickers, period="5y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]: data = data['Close']
            else: data = data['Close'] if 'Close' in data else data
        return data.ffill().dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def calculate_live_selection(data):
    if data.empty: return pd.DataFrame(), None
    
    # 1. 取得真實交易日的月資料
    prices = data[list(MAPPING.keys())]
    monthly = get_monthly_data(prices)
    
    # 2. 鎖定「上個月底」
    # 邏輯：今天(最新數據日) 所在的月份不該被納入計算，因為還沒結束
    last_date = data.index[-1]
    current_period = last_date.to_period('M')
    
    # 篩選出所有小於當前月份的數據
    prev_months = monthly[monthly.index.to_period('M') < current_period]
    
    if prev_months.empty: return pd.DataFrame(), None
    
    ref_date = prev_months.index[-1] # 這就是「上個月最後一個交易日」
    
    # 3. 基於 Ref Date 計算動能
    metrics = []
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        p_now = monthly.loc[ref_date, ticker]
        
        for m in MOM_PERIODS:
            # 在 monthly 中回推 m 個月
            loc = monthly.index.get_loc(ref_date)
            if loc >= m:
                p_prev = monthly.iloc[loc-m][ticker]
                ret = (p_now - p_prev) / p_prev
                row[f'Ret_{m}M'] = ret
            else:
                row[f'Ret_{m}M'] = np.nan
        
        # Volatility (取 Ref Date 之前的日資料)
        d_loc = data.index.get_indexer([ref_date], method='pad')[0]
        if d_loc >= 126:
            subset = prices[ticker].iloc[d_loc-126 : d_loc]
            vol = subset.pct_change().std() * np.sqrt(252)
            row['Vol_Ann'] = vol
        else:
            row['Vol_Ann'] = np.nan
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
    
    return df.sort_values('Total_Z', ascending=False), ref_date

@st.cache_data(ttl=3600)
def calculate_live_safe(data):
    if data.empty: return "TLT", {}, None
    
    monthly = get_monthly_data(data[SAFE_POOL])
    
    last_date = data.index[-1]
    current_period = last_date.to_period('M')
    prev_months = monthly[monthly.index.to_period('M') < current_period]
    
    if prev_months.empty: return "TLT", pd.DataFrame(), None
    
    ref_date = prev_months.index[-1]
    loc = monthly.index.get_loc(ref_date)
    
    if loc >= 12:
        p_now = monthly.iloc[loc]
        p_prev = monthly.iloc[loc-12]
        ret_12m = (p_now / p_prev) - 1
    else:
        ret_12m = pd.Series(0.0, index=SAFE_POOL)
        
    winner = ret_12m.idxmax()
    details = pd.DataFrame({
        "Ticker": SAFE_POOL, "12M Return": ret_12m.values
    }).set_index("Ticker")
    
    return winner, details, ref_date

# ==========================================
# 5. 嚴格滾動訊號計算 (Backtest)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="計算滾動 GARCH 風控訊號 (需時較長)...")
def calculate_backtest_signals(data):
    # A. Risk Weights (Rolling GARCH Refit)
    h_risk_weights = pd.DataFrame(index=data.index, columns=MAPPING.keys())
    
    REFIT_STEP = 5 
    progress_bar = st.progress(0)
    step = 0
    total = len(MAPPING)
    
    for ticker_3x in MAPPING.keys():
        col_1x = f"RAW_{ticker_3x}"
        if col_1x not in data.columns: continue
        
        s_price = data[col_1x]
        s_ret = s_price.pct_change() * 100
        s_sma = s_price.rolling(SMA_WINDOW).mean()
        
        forecasts = {}
        model_res = None
        
        loop_start = BACKTEST_GARCH_WINDOW
        for i in range(loop_start, len(s_ret)):
            target_date = s_ret.index[i]
            
            if (i - loop_start) % REFIT_STEP == 0 or model_res is None:
                train_window = s_ret.iloc[i-BACKTEST_GARCH_WINDOW : i]
                if train_window.std() == 0: continue
                try:
                    am = arch_model(train_window, vol='Garch', p=1, q=1, dist='t', rescale=False)
                    model_res = am.fit(disp='off', show_warning=False)
                except: pass 
            
            if model_res:
                try:
                    fc = model_res.forecast(horizon=1, reindex=False)
                    vol = np.sqrt(fc.variance.iloc[-1].values[0]) * np.sqrt(252)
                    forecasts[target_date] = vol
                except:
                    if len(forecasts) > 0: forecasts[target_date] = list(forecasts.values())[-1]
        
        vol_series = pd.Series(forecasts).reindex(data.index)
        
        df_ind = pd.DataFrame({'Vol': vol_series, 'Price': s_price, 'SMA': s_sma})
        cfg = RISK_CONFIG[ticker_3x]
        
        # [Strict Lag]
        df_ind['Exit'] = df_ind['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
        df_ind['Entry'] = df_ind['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
        
        g_sig = pd.Series(np.nan, index=df_ind.index)
        valid = df_ind['Exit'].notna()
        g_sig.loc[valid & (df_ind['Vol'] > df_ind['Exit'])] = 0.0
        g_sig.loc[valid & (df_ind['Vol'] < df_ind['Entry'])] = 1.0
        g_sig = g_sig.ffill().fillna(0.0)
        
        s_sig = (df_ind['Price'] > df_ind['SMA']).astype(float)
        h_risk_weights[ticker_3x] = (0.5 * g_sig) + (0.5 * s_sig)
        
        step += 1
        progress_bar.progress(step / total)
        
    progress_bar.empty()
    h_risk_weights = h_risk_weights.dropna()
    
    # B. 歷史動能 (Selection) - 使用 get_monthly_data (真實交易日)
    monthly_prices = get_monthly_data(data[list(MAPPING.keys())])
    daily_vol = data[list(MAPPING.keys())].pct_change().rolling(126).std() * np.sqrt(252)
    monthly_vol = get_monthly_data(daily_vol)
    
    scores_df = pd.DataFrame(0.0, index=monthly_prices.index, columns=monthly_prices.columns)
    for m in MOM_PERIODS:
        ret_m = monthly_prices.pct_change(m)
        risk_adj = ret_m / (monthly_vol + 1e-6)
        mean = risk_adj.mean(axis=1)
        std = risk_adj.std(axis=1)
        z = risk_adj.sub(mean, axis=0).div(std + 1e-6, axis=0)
        scores_df += z
        
    hist_winners = scores_df.idxmax(axis=1)
    
    # C. 歷史避險 (Rotation) - 使用 get_monthly_data
    safe_monthly = get_monthly_data(data[SAFE_POOL])
    safe_mom = safe_monthly.pct_change(12)
    hist_safe = safe_mom.idxmax(axis=1).fillna('TLT')
    
    return h_risk_weights, hist_winners, hist_safe

# ==========================================
# 6. 嚴格回測迴圈
# ==========================================
def run_strict_backtest(data, risk_weights, winners_series, safe_signals):
    dates = data.index
    # 起點對齊
    vt_start = data['VT'].first_valid_index()
    warmup_done = data.index[0] + timedelta(days=BACKTEST_GARCH_WINDOW + 252 + 50) 
    
    start_date = max(vt_start, warmup_done)
    if start_date not in dates:
        start_idx = dates.searchsorted(start_date)
    else:
        start_idx = dates.get_loc(start_date)
        
    if start_idx >= len(dates): return None, None, None, None
    
    strategy_ret = []
    valid_dates = []
    hold_counts = defaultdict(float)
    prev_pos = {} 
    
    p_bar = st.progress(0)
    total_len = len(dates) - start_idx
    
    for idx, i in enumerate(range(start_idx, len(dates))):
        if idx % 100 == 0: p_bar.progress(idx / total_len)
        today = dates[i]
        yesterday = dates[i-1]
        
        # A. 訊號讀取 (Yesterday)
        # winners_series 的 index 是 "月底交易日"
        # 使用 <= yesterday 確保我們只讀取到 "已經過去的月底" 的訊號
        # 這嚴格保證了每月初執行，且整月不變
        past_wins = winners_series[winners_series.index <= yesterday]
        if past_wins.empty: continue
        target_risky = past_wins.iloc[-1]
        
        past_safe = safe_signals[safe_signals.index <= yesterday]
        if past_safe.empty: target_safe = 'TLT'
        else: target_safe = past_safe.iloc[-1]
        
        # B. 風控權重 (每日變動)
        if target_risky in risk_weights.columns and yesterday in risk_weights.index:
            w_risk = risk_weights.loc[yesterday, target_risky]
            if pd.isna(w_risk): w_risk = 0.0
        else: w_risk = 0.0
        w_safe = 1.0 - w_risk
        
        # C. 構建持倉
        curr_pos = {}
        if w_risk > 0: curr_pos[target_risky] = w_risk
        if w_safe > 0: curr_pos[target_safe] = w_safe
        
        # D. 成本
        cost = 0.0
        all_assets = set(list(prev_pos.keys()) + list(curr_pos.keys()))
        for asset in all_assets:
            w_prev = prev_pos.get(asset, 0.0)
            w_curr = curr_pos.get(asset, 0.0)
            if w_prev != w_curr:
                cost += abs(w_curr - w_prev) * TRANSACTION_COST
        
        # E. 損益
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
        
    p_bar.empty()
    
    # --- 結果 ---
    eq = pd.Series(strategy_ret, index=valid_dates)
    cum_eq = (1 + eq).cumprod()
    
    # Benchmarks
    b_subset = data[list(MAPPING.keys())].loc[valid_dates].copy()
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
    
    vt_eq = (1 + data['VT'].loc[valid_dates].pct_change().fillna(0)).cumprod()
    
    return cum_eq, b_equity_series, vt_eq, hold_counts

# ==========================================
# 7. 主程式入口
# ==========================================
st.title("🛡️ 雙重動能與動態風控 (Monthly Locked)")
st.caption("架構: 嚴格 T+1 / 滾動 GARCH / 月初換股 / 學術指標")

live_data = get_live_market_data()
if not live_data.empty:
    risk_data = calculate_risk_metrics(live_data) # 注意：這是用來顯示 Live GARCH 狀態
    selection_df, sel_ref_date = calculate_live_selection(live_data)
    safe_winner, safe_details_df, safe_ref_date = calculate_live_safe(live_data)
    
    winner_ticker = selection_df.index[0] if not selection_df.empty else "N/A"
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("本月贏家", winner_ticker, f"基準日: {sel_ref_date.strftime('%Y-%m-%d')}")
    with c2: st.metric("避險資產", safe_winner, f"基準日: {safe_ref_date.strftime('%Y-%m-%d')}")
    
    # 顯示詳細排名
    st.write("#### 📊 動能排名 (Live)")
    st.dataframe(selection_df.style.format("{:.2f}"))

st.divider()
st.header("⏳ 歷史回測")

syn_data = get_synthetic_backtest_data()

if not syn_data.empty:
    if st.button("🚀 執行嚴格回測"):
        h_risk_weights, hist_winners, hist_safe = calculate_backtest_signals(syn_data)
        s_eq, b_eq, v_eq, holds = run_strict_backtest(syn_data, h_risk_weights, hist_winners, hist_safe)
        
        if s_eq is not None:
            # Stats
            def calc_stats(equity, daily_r):
                days = (equity.index[-1] - equity.index[0]).days
                cagr = (equity.iloc[-1]) ** (365.25/days) - 1
                mdd = (equity / equity.cummax() - 1).min()
                rf_daily = RF_RATE / 252
                excess = daily_r - rf_daily
                sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
                downside = excess.copy()
                downside[downside > 0] = 0
                down_std = np.sqrt((downside**2).mean()) * np.sqrt(252)
                sortino = (excess.mean() * 252) / (down_std + 1e-6)
                return cagr, sortino, sharpe, mdd

            s_stat = calc_stats(s_eq, s_eq.pct_change().fillna(0))
            b_stat = calc_stats(b_eq, b_eq.pct_change().fillna(0))
            v_stat = calc_stats(v_eq, v_eq.pct_change().fillna(0))
            
            st.write("### 📊 最終回測結果")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("CAGR", f"{s_stat[0]:.2%}", f"vs VT: {v_stat[0]:.2%}")
            with m2: st.metric("Sortino", f"{s_stat[1]:.2f}", f"vs VT: {v_stat[1]:.2f}")
            with m3: st.metric("Sharpe", f"{s_stat[2]:.2f}", f"vs VT: {v_stat[2]:.2f}")
            with m4: st.metric("MaxDD", f"{s_stat[3]:.2%}", f"vs VT: {v_stat[3]:.2%}")
            
            # Plot
            df_plot = pd.DataFrame({
                'Date': s_eq.index,
                'Strategy': s_eq,
                'Bench (3x)': b_eq,
                'Bench (VT)': v_eq
            }).melt('Date', var_name='Asset', value_name='NAV')
            
            chart = alt.Chart(df_plot).mark_line().encode(
                x='Date', y=alt.Y('NAV', scale=alt.Scale(type='log')),
                color='Asset', tooltip=['Date', 'Asset', alt.Tooltip('NAV', format='.2f')]
            ).properties(width=800, height=400).interactive()
            st.altair_chart(chart, use_container_width=True)
