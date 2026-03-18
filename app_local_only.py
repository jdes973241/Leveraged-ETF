import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from arch import arch_model
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
import pytz 

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="Dynamic Momentum Strategy (Standard)", layout="wide")
warnings.simplefilter(action='ignore')
alt.data_transformers.disable_max_rows()

# CSS 優化
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label {font-size: 14px; color: #6c757d; margin-bottom: 0; font-weight: 600;}
    .metric-value {font-size: 24px; font-weight: bold; color: #212529 !important; margin: 5px 0;}
    .metric-sub {font-size: 12px; color: #adb5bd; margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心參數與時間顯性化
# ==========================================
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"} 
SAFE_POOL = ["GLD", "TLT"] 

RISK_CONFIG = {
    "UPRO": {"exit_q": 0.99, "entry_q": 0.90},
    "EURL": {"exit_q": 0.99, "entry_q": 0.90},
    "EDC":  {"exit_q": 0.99, "entry_q": 0.90}
}

SMA_MONTHS = 6               
LIVE_GARCH_WINDOW = 504      
BACKTEST_GARCH_WINDOW = 504  
REFIT_STEP = 5               
MOM_PERIODS = [3, 6, 9, 12]
TRANSACTION_COST = 0.001 
RF_RATE = 0.02 

TZ_TW = pytz.timezone('Asia/Taipei')

# ==========================================
# [新增] 維運與結算控制台
# ==========================================
with st.sidebar:
    st.header("🔧 系統診斷與維運")
    now_tw = datetime.now(TZ_TW)
    st.info(f"🇹🇼 台灣時間: {now_tw.strftime('%Y-%m-%d %H:%M')}")
    
    st.divider()
    st.markdown("### ⚙️ 結算控制器")
    FORCE_EOM = st.checkbox(
        "⚠️ 執行月底強制結算", 
        value=False,
        help="僅在『每月最後一個交易日』勾選，系統會立即以當下最新收盤價進行月結算，讓您提早取得次月部位訊號。平時請保持關閉以防訊號漂移。"
    )
    st.divider()
    
    if st.button("🗑️ 強制清除快取 (重抓數據)"):
        st.cache_data.clear()
        st.rerun()

def get_daily_leverage_cost(date):
    year = date.year
    if year <= 2007 or year >= 2022: return 0.05 / 252 
    else: return 0.02 / 252

# ==========================================
# [優化] 數據結算核心引擎
# ==========================================
def get_settled_monthly_data(df, force_eom=False):
    if df.empty: return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    period_idx = df.index.to_period('M')
    month_end_dates = df.index.to_series().groupby(period_idx).max()
    monthly = df.loc[month_end_dates]
    
    last_date = df.index[-1]
    now_tw = datetime.now(TZ_TW)
    last_data_period = last_date.to_period('M')
    current_tw_period = pd.Period(now_tw.strftime('%Y-%m'), freq='M')
    
    # 防禦機制：若未強制結算，且最新數據仍在當前自然月中，剔除未完結的月份
    if not force_eom and (last_data_period == current_tw_period):
        monthly = monthly.iloc[:-1]
        
    return monthly

# ==========================================
# 2. Live 面板數據與邏輯
# ==========================================
@st.cache_data(ttl=300) 
def get_live_data():
    tickers = list(MAPPING.keys()) + list(MAPPING.values()) + SAFE_POOL
    try:
        data = yf.download(
            tickers, 
            period="5y", 
            interval="1d", 
            auto_adjust=True, 
            progress=False, 
            threads=False
        )
        if data.empty: return pd.DataFrame()

        clean_df = pd.DataFrame(index=data.index)
        
        # 扁平化結構處理
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                data = data['Close']
            else:
                data = data.xs(data.columns.levels[0][0], level=0, axis=1)

        for t in tickers:
            if t in data.columns: clean_df[t] = data[t]
                
        if clean_df.index.tz is not None:
            clean_df.index = clean_df.index.tz_localize(None)

        return clean_df.ffill()
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

# 即時計算，嚴禁快取
def calculate_live_risk(data, force_eom=False):
    if data.empty: return {}
    avail_cols = [c for c in list(MAPPING.keys()) if c in data.columns]
    if not avail_cols: return {}
    
    sma_tickers = [c for c in list(MAPPING.values()) if c in data.columns]
    
    # 透過統一引擎取得已結算的月數據
    monthly_prices = get_settled_monthly_data(data[sma_tickers], force_eom)
    if monthly_prices.empty: return {}
    
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sig = (monthly_prices > monthly_sma).astype(float)
    
    # [消除滯後]: 取消 .shift(1)，因為回測與實單皆在 T+1 執行 T 日的訊號
    daily_sma_sig = monthly_sig.reindex(data.index).ffill().fillna(0)
    
    risk_details = {}
    for trade_t, signal_t in MAPPING.items():
        if signal_t not in data.columns: continue
        if trade_t not in data.columns or data[trade_t].isnull().all(): continue
            
        series = data[trade_t]
        ret = data[signal_t].pct_change() * 100
        
        window = ret.dropna().tail(LIVE_GARCH_WINDOW * 2) 
        if len(window) < 100: continue
        
        try:
            am = arch_model(window, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            
            in_sample_vol = res.conditional_volatility * np.sqrt(252)
            fc = res.forecast(horizon=1, reindex=False)
            next_vol = np.sqrt(fc.variance.iloc[-1].values[0]) * np.sqrt(252)
            vol_aligned = np.append(in_sample_vol.values[1:], next_vol)
            
            df = pd.DataFrame({'Price': series, 'Ret': ret})
            df['Vol'] = pd.Series(vol_aligned, index=window.index).reindex(df.index)
            
            if signal_t in daily_sma_sig.columns:
                df['SMA_State'] = daily_sma_sig[signal_t]
            else: df['SMA_State'] = 1.0 
            
            cfg = RISK_CONFIG[trade_t]
            df['Exit_Th'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
            df['Entry_Th'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
            
            df['GARCH_State'] = np.nan
            valid = df['Exit_Th'].notna() & df['Vol'].notna()
            mask_exit = valid & (df['Vol'] > df['Exit_Th'])
            mask_entry = valid & (df['Vol'] < df['Entry_Th'])
            
            df.loc[mask_exit, 'GARCH_State'] = 0.0 
            df.loc[mask_entry, 'GARCH_State'] = 1.0 
            df['GARCH_State'] = df['GARCH_State'].ffill().fillna(1.0)
            
            df['Weight'] = (0.5 * df['GARCH_State']) + (0.5 * df['SMA_State'])
            risk_details[trade_t] = df.dropna(subset=['Weight'])
        except: continue
    return risk_details

def calculate_live_selection(data, force_eom=False):
    if data.empty: return pd.DataFrame(), None
    avail_keys = [k for k in list(MAPPING.keys()) if k in data.columns and not data[k].isnull().all()]
    if not avail_keys: return pd.DataFrame(), None
    
    prices = data[avail_keys]
    monthly = get_settled_monthly_data(prices, force_eom)
    if monthly.empty: return pd.DataFrame(), None

    ref_date = monthly.index[-1]
    metrics = []
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        try:
            if ref_date not in monthly.index: continue
            p_now = monthly.loc[ref_date, ticker]
            
            for m in MOM_PERIODS:
                loc = monthly.index.get_loc(ref_date)
                if loc >= m:
                    p_prev = monthly.iloc[loc-m][ticker]
                    if pd.isna(p_prev) or p_prev == 0: row[f'Ret_{m}M'] = np.nan
                    else: row[f'Ret_{m}M'] = (p_now - p_prev) / p_prev
                else: row[f'Ret_{m}M'] = np.nan
            
            d_loc = data.index.get_indexer([ref_date], method='pad')[0]
            if d_loc >= 126:
                subset = prices[ticker].iloc[d_loc-126 : d_loc]
                row['Vol_Ann'] = subset.pct_change().std() * np.sqrt(252)
            else: row['Vol_Ann'] = np.nan
            metrics.append(row)
        except: continue
        
    if not metrics: return pd.DataFrame(), None
    df = pd.DataFrame(metrics).set_index('Ticker')
    z_sum = pd.Series(0.0, index=df.index)
    for m in MOM_PERIODS:
        col = f'Ret_{m}M'
        if col in df.columns:
            risk_adj = df[col] / (df['Vol_Ann'] + 1e-6)
            z = (risk_adj - risk_adj.mean()) / (risk_adj.std() + 1e-6)
            df[f'Z_{m}M'] = z
            z_sum += z.fillna(0)
    df['Total_Z'] = z_sum
    return df.sort_values('Total_Z', ascending=False), ref_date

def calculate_live_safe(data, force_eom=False):
    if data.empty: return "TLT", pd.DataFrame(), None
    avail_safe = [t for t in SAFE_POOL if t in data.columns]
    if not avail_safe: return "TLT", pd.DataFrame(), None

    monthly = get_settled_monthly_data(data[avail_safe], force_eom)
    if monthly.empty: return "TLT", pd.DataFrame(), None

    ref_date = monthly.index[-1]
    loc = monthly.index.get_loc(ref_date)
    if loc >= 12: ret_12m = (monthly.iloc[loc] / monthly.iloc[loc-12]) - 1
    else: ret_12m = pd.Series(0.0, index=avail_safe)
    
    winner = ret_12m.idxmax()
    details = pd.DataFrame({"Ticker": avail_safe, "12M Return": ret_12m.values}).set_index("Ticker")
    return winner, details, ref_date

# ==========================================
# 3. 回測邏輯 (Strict Rolling)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="準備回測數據 (合成三倍槓桿)...")
def get_synthetic_backtest_data():
    tickers = list(MAPPING.values()) + SAFE_POOL + ['VT']
    try:
        data_raw = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
        if isinstance(data_raw.columns, pd.MultiIndex):
            if 'Close' in data_raw.columns.levels[0]: data_raw = data_raw['Close']
            else: data_raw = data_raw.xs(data_raw.columns.levels[0][0], level=0, axis=1)
        
        if data_raw.index.tz is not None: data_raw.index = data_raw.index.tz_localize(None)
        data_raw = data_raw.ffill()
        
        synthetic_data = pd.DataFrame(index=data_raw.index)
        if 'VT' in data_raw.columns: synthetic_data['VT'] = data_raw['VT']
        for t in SAFE_POOL: 
            if t in data_raw.columns: synthetic_data[t] = data_raw[t]
            
        REVERSE_MAP = {v: k for k, v in MAPPING.items()} 
        for ticker_1x in MAPPING.values():
            if ticker_1x not in data_raw.columns: continue
            ticker_3x = REVERSE_MAP[ticker_1x]
            ret_1x = data_raw[ticker_1x].pct_change().fillna(0)
            costs = pd.Series([get_daily_leverage_cost(d) for d in ret_1x.index], index=ret_1x.index)
            synthetic_data[ticker_3x] = (1 + ((ret_1x * 3.0) - costs)).cumprod() * 100
            synthetic_data[f"RAW_{ticker_3x}"] = data_raw[ticker_1x] 
            
        return synthetic_data.dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="計算滾動回測訊號...")
def calculate_backtest_signals_rolling(data):
    raw_cols = [f"RAW_{k}" for k in MAPPING.keys() if f"RAW_{k}" in data.columns]
    if not raw_cols: return pd.DataFrame(), pd.Series(), pd.Series()

    # 回測直接使用完整序列轉換，保留最後一筆作為最新訊號
    monthly_prices = get_settled_monthly_data(data[raw_cols], force_eom=True) 
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sma_sig = (monthly_prices > monthly_sma).astype(float)
    
    # [消除滯後]: 移除 shift(1)，在 run_backtest_logic 中透過 loc[yesterday] 取用 T 日訊號執行 T+1 交易
    daily_sma_sig = monthly_sma_sig.reindex(data.index).ffill().fillna(0)
    daily_sma_sig.columns = [c.replace("RAW_", "") for c in raw_cols]

    target_tickers = [k for k in MAPPING.keys() if f"RAW_{k}" in data.columns]
    h_risk_weights = pd.DataFrame(index=data.index, columns=target_tickers)
    
    for i, ticker_3x in enumerate(target_tickers):
        col_1x = f"RAW_{ticker_3x}"
        s_ret = data[col_1x].pct_change() * 100
        forecasts = {}
        model_res = None
        loop_start = BACKTEST_GARCH_WINDOW
        dates = s_ret.index
        
        for t in range(loop_start, len(s_ret)):
            train = s_ret.iloc[t - BACKTEST_GARCH_WINDOW + 1 : t + 1]
            if len(train) < 50: continue
            
            if (t - loop_start) % REFIT_STEP == 0 or model_res is None:
                try: model_res = arch_model(train, vol='Garch', p=1, q=1, dist='t', rescale=False).fit(disp='off', show_warning=False)
                except: pass
            
            if model_res:
                try:
                    fc = model_res.forecast(horizon=1, reindex=False)
                    forecasts[dates[t]] = np.sqrt(fc.variance.iloc[-1].values[0]) * np.sqrt(252)
                except: pass
        
        vol_series = pd.Series(forecasts).reindex(data.index)
        cfg = RISK_CONFIG[ticker_3x]
        ex_th = vol_series.rolling(252).quantile(cfg['exit_q']).shift(1)
        en_th = vol_series.rolling(252).quantile(cfg['entry_q']).shift(1)
        
        g_sig = pd.Series(np.nan, index=data.index)
        valid = ex_th.notna()
        g_sig.loc[valid & (vol_series > ex_th)] = 0.0
        g_sig.loc[valid & (vol_series < en_th)] = 1.0
        g_sig = g_sig.ffill().fillna(0.0)
        
        if ticker_3x in daily_sma_sig.columns:
            h_risk_weights[ticker_3x] = 0.5*g_sig + 0.5*daily_sma_sig[ticker_3x]
        
    monthly_src = get_settled_monthly_data(data[raw_cols], force_eom=True)
    monthly_src.columns = [c.replace("RAW_", "") for c in raw_cols]
    daily_vol = data[raw_cols].pct_change().rolling(126).std() * np.sqrt(252)
    monthly_vol = get_settled_monthly_data(daily_vol, force_eom=True)
    monthly_vol.columns = monthly_src.columns
    
    scores = pd.DataFrame(0.0, index=monthly_src.index, columns=monthly_src.columns)
    for m in MOM_PERIODS:
        ret = monthly_src.pct_change(m)
        risk_adj = ret / (monthly_vol + 1e-6)
        z = risk_adj.sub(risk_adj.mean(axis=1), axis=0).div(risk_adj.std(axis=1)+1e-6, axis=0)
        scores += z.fillna(0)
    hist_winners = scores.idxmax(axis=1)
    
    avail_safe = [t for t in SAFE_POOL if t in data.columns]
    safe_monthly = get_settled_monthly_data(data[avail_safe], force_eom=True)
    hist_safe = safe_monthly.pct_change(12).idxmax(axis=1).fillna('TLT')
    
    return h_risk_weights.dropna(), hist_winners, hist_safe

def run_backtest_logic(data, risk_weights, winners_series, safe_signals):
    dates = data.index
    start_idx = BACKTEST_GARCH_WINDOW + 252
    vt_start = data['VT'].first_valid_index()
    if vt_start: start_idx = max(start_idx, data.index.get_loc(vt_start))
    if start_idx >= len(dates): return None, None, None, None
    
    strategy_ret, valid_dates = [], []
    hold_counts = defaultdict(float)
    prev_pos = {}
    
    for i in range(start_idx, len(dates)):
        today = dates[i]
        yesterday = dates[i-1] # T-1 的訊號，在 T 日承受損益
        
        past_wins = winners_series[winners_series.index <= yesterday]
        if past_wins.empty: continue
        target_risky = past_wins.iloc[-1]
        
        past_safe = safe_signals[safe_signals.index <= yesterday]
        target_safe = past_safe.iloc[-1] if not past_safe.empty else 'TLT'
        
        if target_risky in risk_weights.columns and yesterday in risk_weights.index:
            w_risk = risk_weights.loc[yesterday, target_risky]
            if pd.isna(w_risk): w_risk = 0.0
        else: w_risk = 0.0
        w_safe = 1.0 - w_risk
        
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
        if w_risk > 0 and target_risky in data.columns:
            r = data[target_risky].pct_change().iloc[i]
            day_ret += w_risk * (r if not np.isnan(r) else 0)
        if w_safe > 0 and target_safe in data.columns:
            r = data[target_safe].pct_change().iloc[i]
            day_ret += w_safe * (r if not np.isnan(r) else 0)
            
        strategy_ret.append(day_ret - cost)
        valid_dates.append(today)
        hold_counts[target_risky] += w_risk
        hold_counts[target_safe] += w_safe
        prev_pos = curr_pos
        
    eq = pd.Series(strategy_ret, index=valid_dates)
    cum_eq = (1 + eq).cumprod()
    
    b_cols = [c for c in list(MAPPING.keys()) if c in data.columns]
    b_sub = data[b_cols].loc[valid_dates].copy()
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
st.title("🛡️ 雙重動能與動態風控 (Live Ops Fix)")
st.caption(f"配置: SMA {SMA_MONTHS}M / GARCH (Q{RISK_CONFIG['UPRO']['exit_q']*100:.0f}) / Safe (GLD/TLT)")

with st.expander("🛠️ 數據除錯與狀態"):
    live_data = get_live_data()
    st.write("原始數據形狀:", live_data.shape)
    st.write(f"EDC 數據狀態: {'✅ 成功' if 'EDC' in live_data.columns else '❌ 缺失'}")
    st.write("最後更新日期:", live_data.index[-1].strftime('%Y-%m-%d') if not live_data.empty else "無")
    if live_data.empty: st.error("⚠️ 數據下載失敗。")

# --- Live Data Loading (導入 FORCE_EOM 控制) ---
risk_live = calculate_live_risk(live_data, force_eom=FORCE_EOM)
sel_df, sel_date = calculate_live_selection(live_data, force_eom=FORCE_EOM)
safe_win, safe_df, safe_date = calculate_live_safe(live_data, force_eom=FORCE_EOM)

winner = sel_df.index[0] if not sel_df.empty else "N/A"
if winner in risk_live and not risk_live[winner].empty:
    latest_r = risk_live[winner].iloc[-1]
    final_w = latest_r['Weight']
    g_state = latest_r['GARCH_State']
else:
    final_w, g_state = 0.0, 0.0

if sel_date:
    status_icon = "🔓 強制提早結算啟用中" if FORCE_EOM else "🔒 正規結算鎖定"
    st.info(f"{status_icon} | **訊號基準日**: {sel_date.strftime('%Y-%m-%d')}")

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🏆 本月進攻贏家", winner)
with c2: 
    color = "green" if final_w==1 else "orange" if final_w==0.5 else "red"
    st.markdown(f"### 🎯 權重: :{color}[{final_w*100:.0f}%]")
with c3: st.metric("GARCH 風控", "安全" if g_state==1 else "危險", delta="✅" if g_state==1 else "🔻")
with c4: 
    s_val = safe_df.loc[safe_win, '12M Return'] if not safe_df.empty else 0
    st.metric("🛡️ 避險資產", safe_win, f"12M: {s_val:.1%}")

st.divider()

t1, t2, t3, t4, t5, t6 = st.tabs(["數據", "風控細節", "權重狀態", "選股排名", "避險輪動", "部位總結"])
with t1: st.dataframe(live_data.tail(5).style.format("{:.2f}"), use_container_width=True)
with t2:
    if winner in risk_live and not risk_live[winner].empty:
        st.dataframe(risk_live[winner].tail(10)[['Price','Vol','Exit_Th','Entry_Th','GARCH_State']].style.format("{:.2f}"))
with t3:
    if winner in risk_live and not risk_live[winner].empty:
        st.dataframe(risk_live[winner].tail(10)[['GARCH_State','SMA_State','Weight']])
with t4: st.dataframe(sel_df.style.format("{:.2f}"), use_container_width=True)
with t5: st.dataframe(safe_df.style.format("{:.2%}"), use_container_width=True)
with t6: st.success(f"建議次日開盤持有: **{final_w*100:.0f}% {winner}** + **{(1-final_w)*100:.0f}% {safe_win}**")

st.divider()

# ==========================================
# 5. 回測區塊 (Strict Rolling)
# ==========================================
st.header("⏳ 嚴格滾動回測 (Synthetic 3x)")
syn_data = get_synthetic_backtest_data()

if not syn_data.empty:
    if st.button("🚀 開始滾動回測 (約需 30-60 秒)"):
        with st.spinner("正在進行 GARCH 滾動訓練與參數擬合..."):
            h_risk, h_win, h_safe = calculate_backtest_signals_rolling(syn_data)
        with st.spinner("正在執行交易回測..."):
            s_eq, b_eq, v_eq, holds = run_backtest_logic(syn_data, h_risk, h_win, h_safe)
        
        if s_eq is not None:
            def calc_stats(eq, dr):
                d = (eq.index[-1] - eq.index[0]).days
                cagr = (eq.iloc[-1]) ** (365.25/d) - 1
                mdd = (eq / eq.cummax() - 1).min()
                excess = dr - (RF_RATE/252)
                sharpe = (excess.mean()/excess.std())*np.sqrt(252)
                down = excess.copy(); down[down>0]=0
                down_std = np.sqrt((down**2).mean())*np.sqrt(252)
                return cagr, (excess.mean()*252)/(down_std+1e-6), sharpe, mdd
            
            s_s = calc_stats(s_eq, s_eq.pct_change().fillna(0))
            b_s = calc_stats(b_eq, b_eq.pct_change().fillna(0))
            v_s = calc_stats(v_eq, v_eq.pct_change().fillna(0))
            
            r5_s = s_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()
            r5_b = b_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()
            r5_v = v_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()

            st.write("### 📈 績效指標")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            def m_box(label, v, b, vt, fmt="{:.2%}"):
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{label}</p><p class="metric-value">{fmt.format(v)}</p>
                    <p class="metric-sub">3x: {fmt.format(b)} | VT: {fmt.format(vt)}</p>
                </div>""", unsafe_allow_html=True)
                
            with m1: m_box("CAGR", s_s[0], b_s[0], v_s[0])
            with m2: m_box("Sortino", s_s[1], b_s[1], v_s[1], "{:.2f}")
            with m3: m_box("Sharpe", s_s[2], b_s[2], v_s[2], "{:.2f}")
            with m4: m_box("Avg 5Y", r5_s, r5_b, r5_v)
            with m5: m_box("MaxDD", s_s[3], b_s[3], v_s[3])
            
            t_3x = sum([v for k,v in holds.items() if k in MAPPING.keys()]) / len(s_eq)
            with m6: m_box("Time in 3x", t_3x, 1.0, 1.0)
            
            st.divider()
            st.write("### 📊 權益曲線")
            df_chart = pd.DataFrame({'Date': s_eq.index, 'Strategy': s_eq, 'Bench (3x)': b_eq, 'VT': v_eq}).melt('Date', var_name='Asset', value_name='NAV')
            st.altair_chart(alt.Chart(df_chart).mark_line().encode(
                x='Date', y=alt.Y('NAV', scale=alt.Scale(type='log')), 
                color='Asset', tooltip=['Date','Asset', alt.Tooltip('NAV', format='.2f')]
            ).properties(width=800, height=350), use_container_width=True)
