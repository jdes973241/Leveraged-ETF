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
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="Dynamic Momentum Dashboard", layout="wide")
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
    .buy-text {color: #28a745; font-weight: bold;}
    .sell-text {color: #dc3545; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心參數 (已更新為最新策略)
# ==========================================
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"} 
SAFE_POOL = ["GLD", "TLT"] 

# [修改 1] 更新風控閾值 Exit 0.95 / Entry 0.65
RISK_CONFIG = {
    "UPRO": {"exit_q": 0.95, "entry_q": 0.65},
    "EURL": {"exit_q": 0.95, "entry_q": 0.65},
    "EDC":  {"exit_q": 0.95, "entry_q": 0.65}
}

# [修改 2] 參數調整
LIVE_GARCH_WINDOW = 504  # 改為 504 天 (約2年)
SMA_MONTHS = 6           # 改為 6 個月月均線
MOM_PERIODS = [3, 6, 9, 12]
TRANSACTION_COST = 0.001 
RF_RATE = 0.02 

def get_daily_leverage_cost(date):
    year = date.year
    if year <= 2007 or year >= 2022: return 0.05 / 252 
    else: return 0.02 / 252

# [Live 用] 確保交易日正確
def get_monthly_data(df):
    if df.empty: return df
    period_idx = df.index.to_period('M')
    month_end_dates = df.index.to_series().groupby(period_idx).max()
    return df.loc[month_end_dates]

# ==========================================
# 2. Live 面板邏輯
# ==========================================
@st.cache_data(ttl=3600)
def get_live_data():
    tickers = list(MAPPING.keys()) + list(MAPPING.values()) + SAFE_POOL
    try:
        data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]: data = data['Close']
            else: data = data['Close'] if 'Close' in data else data
        return data.ffill().dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def calculate_live_risk(data):
    if data.empty: return {}
    
    # [修改 3] Live 計算邏輯加入 月均線 判斷
    # 先計算月均線訊號
    monthly_prices = get_monthly_data(data[list(MAPPING.keys())])
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    # 月底訊號: 價格 > 均線 = 1
    monthly_sig = (monthly_prices > monthly_sma).astype(float)
    # 擴展回日頻率 (ffill)
    daily_sma_sig = monthly_sig.reindex(data.index).ffill() 
    
    risk_details = {}
    for trade_t, signal_t in MAPPING.items():
        if signal_t not in data.columns: continue
        series = data[trade_t] # 這裡用實際交易標的(3x)看價格，或用訊號源亦可，策略定義 SMA 看 UPRO 本身或 SPY 皆可，這裡通常看訊號源較穩，但為了對齊上方參數 mapping，這裡我們統一用 MAPPING key (UPRO)
        # 修正：根據策略，SMA 是看 1x 源頭比較準 (SPY)，但為了 Dashboard 顯示方便，若 data 有 UPRO 就用 UPRO。
        # 為了跟回測一致，我們看 trade_t (UPRO) 的月均線狀態 (上方的 monthly_prices 包含 UPRO)
        
        ret = data[signal_t].pct_change() * 100 # GARCH 用訊號源 (SPY)
        
        # GARCH Window
        window = ret.dropna().tail(LIVE_GARCH_WINDOW * 2) 
        if len(window) < 100: continue
        
        try:
            am = arch_model(window, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            cond_vol = res.conditional_volatility * np.sqrt(252)
            
            # 組合 DataFrame
            df = pd.DataFrame({'Price': series, 'Ret': ret})
            df['Vol'] = cond_vol
            df = df.dropna()
            
            # SMA 狀態 (從上面算好的月均線擴展抓過來)
            if trade_t in daily_sma_sig.columns:
                df['SMA_State'] = daily_sma_sig[trade_t]
            else:
                df['SMA_State'] = 0.0
            
            cfg = RISK_CONFIG[trade_t]
            df['Exit_Th'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
            df['Entry_Th'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
            
            df['GARCH_State'] = np.nan
            valid = df['Exit_Th'].notna()
            df.loc[valid & (df['Vol'] > df['Exit_Th']), 'GARCH_State'] = 0.0 
            df.loc[valid & (df['Vol'] < df['Entry_Th']), 'GARCH_State'] = 1.0 
            df['GARCH_State'] = df['GARCH_State'].ffill().fillna(1.0)
            
            # 最終權重
            df['Weight'] = (0.5 * df['GARCH_State']) + (0.5 * df['SMA_State'])
            risk_details[trade_t] = df
        except: continue
    return risk_details

@st.cache_data(ttl=3600)
def calculate_live_selection(data):
    if data.empty: return pd.DataFrame(), None
    prices = data[list(MAPPING.keys())]
    monthly = get_monthly_data(prices)
    
    last_date = data.index[-1]
    current_period = last_date.to_period('M')
    
    # 確保只看"上個月底"的數據，避免用到本月未完成的數據
    prev_months = monthly[monthly.index.to_period('M') < current_period]
    if prev_months.empty: return pd.DataFrame(), None
    
    ref_date = prev_months.index[-1]
    
    metrics = []
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        p_now = monthly.loc[ref_date, ticker]
        
        for m in MOM_PERIODS:
            loc = monthly.index.get_loc(ref_date)
            if loc >= m:
                p_prev = monthly.iloc[loc-m][ticker]
                ret = (p_now - p_prev) / p_prev
                row[f'Ret_{m}M'] = ret
            else: row[f'Ret_{m}M'] = np.nan
            
        d_loc = data.index.get_indexer([ref_date], method='pad')[0]
        if d_loc >= 126:
            subset = prices[ticker].iloc[d_loc-126 : d_loc]
            vol = subset.pct_change().std() * np.sqrt(252)
            row['Vol_Ann'] = vol
        else: row['Vol_Ann'] = np.nan
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
    else: ret_12m = pd.Series(0.0, index=SAFE_POOL)
    
    winner = ret_12m.idxmax()
    details = pd.DataFrame({"Ticker": SAFE_POOL, "12M Return": ret_12m.values}).set_index("Ticker")
    return winner, details, ref_date

# ==========================================
# 3. 回測邏輯 (Updated Fast Version)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="生成回測數據...")
def get_synthetic_backtest_data():
    tickers = list(MAPPING.values()) + SAFE_POOL + ['VT']
    try:
        data_raw = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data_raw.columns, pd.MultiIndex):
            if 'Close' in data_raw.columns.levels[0]: data_raw = data_raw['Close']
            else: data_raw = data_raw['Close'] if 'Close' in data_raw else data_raw
        
        # 確保核心數據存在
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
    except: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="計算快速回測訊號 (Fast Fit)...")
def calculate_backtest_signals_fast(data):
    # [修改 4] 回測邏輯：改為 6 個月月均線
    # 1. 計算月均線訊號
    raw_cols = [f"RAW_{k}" for k in MAPPING.keys()]
    monthly_prices = get_monthly_data(data[raw_cols])
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sma_sig = (monthly_prices > monthly_sma).astype(float)
    # 擴展回日頻率 (Strict T+1)
    daily_sma_sig = monthly_sma_sig.reindex(data.index).ffill().shift(1)
    daily_sma_sig.columns = MAPPING.keys()

    h_risk_weights = pd.DataFrame(index=data.index, columns=MAPPING.keys())
    
    for ticker_3x in MAPPING.keys():
        col_1x = f"RAW_{ticker_3x}"
        if col_1x not in data.columns: continue
        
        s = data[col_1x]
        r = s.pct_change() * 100
        
        # GARCH Fast Fit (Full Window for speed in Streamlit)
        # Note: Strictly it should be rolling, but for "Fast Dashboard" we keep full fit
        # consistent with the user's "Fast Backtest" request, but apply new thresholds.
        win = r.dropna()
        if len(win) < 100: continue
        try:
            am = arch_model(win, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            vol = res.conditional_volatility * np.sqrt(252)
            
            df = pd.DataFrame({'Vol': vol})
            df = df.reindex(data.index)
            cfg = RISK_CONFIG[ticker_3x]
            
            df['Exit'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
            df['Entry'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
            
            g = pd.Series(np.nan, index=df.index)
            valid = df['Exit'].notna()
            g.loc[valid & (df['Vol'] > df['Exit'])] = 0.0
            g.loc[valid & (df['Vol'] < df['Entry'])] = 1.0
            g = g.ffill().fillna(0.0)
            
            # 結合月均線訊號
            s_sig = daily_sma_sig[ticker_3x]
            h_risk_weights[ticker_3x] = 0.5*g + 0.5*s_sig
        except: continue
        
    h_risk_weights = h_risk_weights.dropna()
    
    # B. Selection (Original Monthly)
    monthly_prices = get_monthly_data(data[list(MAPPING.keys())])
    daily_vol = data[list(MAPPING.keys())].pct_change().rolling(126).std() * np.sqrt(252)
    monthly_vol = get_monthly_data(daily_vol)
    
    scores = pd.DataFrame(0.0, index=monthly_prices.index, columns=monthly_prices.columns)
    for m in MOM_PERIODS:
        ret = monthly_prices.pct_change(m)
        risk_adj = ret / (monthly_vol + 1e-6)
        z = risk_adj.sub(risk_adj.mean(axis=1), axis=0).div(risk_adj.std(axis=1)+1e-6, axis=0)
        scores += z
    hist_winners = scores.idxmax(axis=1)
    
    # C. Safe (GLD vs TLT 12M Rotation)
    # [修改 5] 確保避險資產輪動
    safe_monthly = get_monthly_data(data[SAFE_POOL])
    hist_safe = safe_monthly.pct_change(12).idxmax(axis=1).fillna('TLT')
    
    return h_risk_weights, hist_winners, hist_safe

def run_fast_backtest(data, risk_weights, winners_series, safe_signals):
    dates = data.index
    # 這裡只需要 504 (GARCH生成) + 252 (Quantile) 
    start_idx = 756 
    
    # Align with VT if needed
    vt_start = data['VT'].first_valid_index()
    if vt_start:
        vt_idx = data.index.get_loc(vt_start)
        start_idx = max(start_idx, vt_idx)
        
    if start_idx >= len(dates): return None, None, None, None
    
    strategy_ret = []
    valid_dates = []
    hold_counts = defaultdict(float)
    prev_pos = {} 
    
    progress = st.progress(0)
    total = len(dates) - start_idx
    
    for idx, i in enumerate(range(start_idx, len(dates))):
        if idx % 100 == 0: progress.progress(idx / total)
        today = dates[i]
        yesterday = dates[i-1]
        
        # Monthly Lock Selection
        past_wins = winners_series[winners_series.index <= yesterday]
        if past_wins.empty: continue
        target_risky = past_wins.iloc[-1]
        
        # [修改 6] 動態避險資產
        past_safe = safe_signals[safe_signals.index <= yesterday]
        if past_safe.empty: target_safe = 'TLT'
        else: target_safe = past_safe.iloc[-1]
        
        # Daily Risk Weight
        if target_risky in risk_weights.columns and yesterday in risk_weights.index:
            w_risk = risk_weights.loc[yesterday, target_risky]
            if pd.isna(w_risk): w_risk = 0.0
        else: w_risk = 0.0
        w_safe = 1.0 - w_risk
        
        # Exec
        curr_pos = {}
        if w_risk > 0: curr_pos[target_risky] = w_risk
        if w_safe > 0: curr_pos[target_safe] = w_safe
        
        cost = 0.0
        all_assets = set(list(prev_pos.keys()) + list(curr_pos.keys()))
        for asset in all_assets:
            w_prev = prev_pos.get(asset, 0.0)
            w_curr = curr_pos.get(asset, 0.0)
            if w_prev != w_curr: cost += abs(w_curr - w_prev) * TRANSACTION_COST
            
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
        
    progress.empty()
    eq = pd.Series(strategy_ret, index=valid_dates)
    cum_eq = (1 + eq).cumprod()
    
    b_sub = data[list(MAPPING.keys())].loc[valid_dates].copy()
    b_eq = pd.Series(1.0, index=b_sub.index)
    curr = 1.0
    q_ends = b_sub.groupby(pd.Grouper(freq='QE')).apply(lambda x: x.index[-1] if len(x)>0 else None).dropna()
    cps = sorted(list(set([b_sub.index[0]] + list(q_ends) + [b_sub.index[-1]])))
    for i in range(len(cps)-1):
        t_s, t_e = cps[i], cps[i+1]
        if t_s >= t_e: continue
        seg = b_sub.loc[t_s:t_e]
        if len(seg)<2: continue
        rel = seg.div(seg.iloc[0])
        val = rel.mean(axis=1) * curr
        b_eq.loc[t_s:t_e] = val
        curr = val.iloc[-1]
    
    vt_eq = (1 + data['VT'].loc[valid_dates].pct_change().fillna(0)).cumprod()
    
    return cum_eq, b_eq, vt_eq, hold_counts

# ==========================================
# 4. Dashboard 介面
# ==========================================
st.title("🛡️ 雙重動能與動態風控 (Dashboard v3.1)")
st.caption(f"架構: Monthly Lock (SMA {SMA_MONTHS}M) / Daily GARCH (Q{RISK_CONFIG['UPRO']['exit_q']}) / Safe Rotation (TLT/GLD)")

live_data = get_live_data()
risk_live = calculate_live_risk(live_data)
sel_df, sel_date = calculate_live_selection(live_data)
safe_win, safe_df, safe_date = calculate_live_safe(live_data)

winner = sel_df.index[0] if not sel_df.empty else "N/A"
if winner in risk_live:
    latest_r = risk_live[winner].iloc[-1]
    final_w = latest_r['Weight']
    g_state = latest_r['GARCH_State']
else:
    final_w = 0.0
    g_state = 0.0

if sel_date:
    st.info(f"🔒 **訊號鎖定日**: {sel_date.strftime('%Y-%m-%d')} (上個月最後交易日)")

with st.expander("📖 策略白皮書", expanded=False):
    st.markdown(f"""
    ### 策略邏輯摘要
    * **選股**: 月初鎖定上個月底贏家 (3/6/9/12M Z-Score)。
    * **趨勢風控**: {SMA_MONTHS} 個月均線 (月底鎖定)。
    * **波動風控**: 每日 GARCH (Exit Q{RISK_CONFIG['UPRO']['exit_q']} / Entry Q{RISK_CONFIG['UPRO']['entry_q']})。
    * **避險**: 月初鎖定上個月底 GLD/TLT 贏家 (12M Momentum)。
    """)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🏆 本月進攻贏家", winner)
with c2: 
    color = "green" if final_w==1 else "orange" if final_w==0.5 else "red"
    st.markdown(f"### 🎯 權重: :{color}[{final_w*100:.0f}%]")
with c3: 
    st.metric("GARCH 風控", "安全" if g_state==1 else "危險", delta="✅" if g_state==1 else "🔻")
with c4: 
    s_val = safe_df.loc[safe_win, '12M Return'] if not safe_df.empty else 0
    st.metric("🛡️ 避險資產", safe_win, f"12M: {s_val:.1%}")

st.divider()

t1, t2, t3, t4, t5, t6 = st.tabs(["1️⃣ 數據層", "2️⃣ 風控層", "3️⃣ 權重層", "4️⃣ 選股層", "5️⃣ 避險層", "6️⃣ 執行層"])
with t1: st.dataframe(live_data.tail(5).style.format("{:.2f}"), use_container_width=True)
with t2:
    if winner in risk_live:
        st.dataframe(risk_live[winner].tail(10)[['Price','Vol','Exit_Th','Entry_Th','GARCH_State']].style.format("{:.2f}"), use_container_width=True)
with t3:
    if winner in risk_live:
        st.dataframe(risk_live[winner].tail(10)[['GARCH_State','SMA_State','Weight']], use_container_width=True)
with t4: st.dataframe(sel_df.style.format("{:.2f}"), use_container_width=True)
with t5: st.dataframe(safe_df.style.format("{:.2%}"), use_container_width=True)
with t6: st.success(f"持有 **{final_w*100:.0f}% {winner}** + **{(1-final_w)*100:.0f}% {safe_win}**")

st.divider()
st.header("⏳ 歷史回測 (Synthetic - Fast Mode)")

syn_data = get_synthetic_backtest_data()
if not syn_data.empty:
    if st.button("🚀 執行回測"):
        h_risk, h_win, h_safe = calculate_backtest_signals_fast(syn_data)
        s_eq, b_eq, v_eq, holds = run_fast_backtest(syn_data, h_risk, h_win, h_safe)
        
        if s_eq is not None:
            def calc_stats(eq, dr):
                d = (eq.index[-1] - eq.index[0]).days
                cagr = (eq.iloc[-1]) ** (365.25/d) - 1
                mdd = (eq / eq.cummax() - 1).min()
                excess = dr - (RF_RATE/252)
                sharpe = (excess.mean()/excess.std())*np.sqrt(252)
                down = excess.copy(); down[down>0]=0
                down_std = np.sqrt((down**2).mean())*np.sqrt(252)
                sortino = (excess.mean()*252)/(down_std+1e-6)
                return cagr, sortino, sharpe, mdd
            
            s_s = calc_stats(s_eq, s_eq.pct_change().fillna(0))
            b_s = calc_stats(b_eq, b_eq.pct_change().fillna(0))
            v_s = calc_stats(v_eq, v_eq.pct_change().fillna(0))
            
            r5_s = s_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()
            r5_b = b_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()
            r5_v = v_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()

            st.write("### 績效指標")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            def m_box(label, v, b, vt, fmt="{:.2%}"):
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{label}</p>
                    <p class="metric-value">{fmt.format(v)}</p>
                    <p class="metric-sub">3x: {fmt.format(b)} | VT: {fmt.format(vt)}</p>
                </div>""", unsafe_allow_html=True)
                
            with m1: m_box("CAGR", s_s[0], b_s[0], v_s[0])
            with m2: m_box("Sortino", s_s[1], b_s[1], v_s[1], "{:.2f}")
            with m3: m_box("Sharpe", s_s[2], b_s[2], v_s[2], "{:.2f}")
            with m4: m_box("Avg 5Y", r5_s, r5_b, r5_v)
            with m5: m_box("MaxDD", s_s[3], b_s[3], v_s[3])
            
            t_3x = sum([v for k,v in holds.items() if 'Syn_' in k]) / len(s_eq)
            with m6: m_box("Time in 3x", t_3x, 1.0, 1.0)
            
            st.write("#### 詳細數據")
            res_df = pd.DataFrame([
                ["Strategy", *s_s, r5_s],
                ["Bench (3x)", *b_s, r5_b],
                ["Bench (VT)", *v_s, r5_v]
            ], columns=["Name", "CAGR", "Sortino", "Sharpe", "MaxDD", "Avg 5Y"])
            
            for c in ["CAGR", "MaxDD", "Avg 5Y"]: res_df[c] = res_df[c].apply(lambda x: f"{x:.2%}")
            for c in ["Sortino", "Sharpe"]: res_df[c] = res_df[c].apply(lambda x: f"{x:.2f}")
            st.table(res_df)

            st.write("#### 📊 權益曲線 (Log Scale)")
            df_chart = pd.DataFrame({'Date': s_eq.index, 'Strategy': s_eq, 'Bench (3x)': b_eq, 'VT': v_eq}).melt('Date', var_name='Asset', value_name='NAV')
            c1 = alt.Chart(df_chart).mark_line().encode(x='Date', y=alt.Y('NAV', scale=alt.Scale(type='log')), color='Asset', tooltip=['Date','Asset', alt.Tooltip('NAV', format='.2f')]).properties(width=800, height=350).interactive()
            st.altair_chart(c1, use_container_width=True)
            
            st.write("#### 📉 回撤幅度 (Drawdown)")
            dd_s = s_eq/s_eq.cummax()-1
            dd_b = b_eq/b_eq.cummax()-1
            dd_v = v_eq/v_eq.cummax()-1
            df_dd = pd.DataFrame({'Date': s_eq.index, 'Strategy': dd_s, 'Bench (3x)': dd_b, 'VT': dd_v}).melt('Date', var_name='Asset', value_name='DD')
            c2 = alt.Chart(df_dd).mark_line().encode(x='Date', y=alt.Y('DD', axis=alt.Axis(format='%')), color='Asset', tooltip=['Date','Asset', alt.Tooltip('DD', format='.2%')]).properties(width=800, height=250).interactive()
            st.altair_chart(c2, use_container_width=True)
            
            st.write("#### 🔄 滾動 5 年報酬率 (Rolling 5Y CAGR)")
            roll_s = s_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1)
            roll_b = b_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1)
            roll_v = v_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1)
            df_r5 = pd.DataFrame({'Date': s_eq.index, 'Strategy': roll_s, 'Bench (3x)': roll_b, 'VT': roll_v}).melt('Date', var_name='Asset', value_name='Roll5Y')
            c3 = alt.Chart(df_r5.dropna()).mark_line().encode(x='Date', y=alt.Y('Roll5Y', axis=alt.Axis(format='%')), color='Asset', tooltip=['Date','Asset', alt.Tooltip('Roll5Y', format='.2%')]).properties(width=800, height=250).interactive()
            st.altair_chart(c3, use_container_width=True)
