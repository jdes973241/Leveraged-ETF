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
st.set_page_config(page_title="Dynamic Momentum Strategy (Institutional)", layout="wide")
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
# 快取管理與時間診斷工具
# ==========================================
with st.sidebar:
    st.header("🔧 系統診斷")
    
    try:
        tz_tw = pytz.timezone('Asia/Taipei')
        now_tw = datetime.now(tz_tw)
        st.info(f"🇹🇼 台灣時間: {now_tw.strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        st.error(f"時區錯誤: {e}")
    
    FORCE_EOM = st.checkbox("⚠️ 執行月底強制結算", value=False, help="僅在『月底當天』勾選，強制系統立刻以最新收盤價進行月結算。")
    
    if st.button("🗑️ 強制清除快取 (重抓數據)"):
        st.cache_data.clear()
        st.rerun()

# ==========================================
# 1. 核心參數
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
MOM_PERIODS = [12]           
TRANSACTION_COST = 0.001 
RF_RATE = 0.02 

def get_daily_leverage_cost(date):
    year = date.year
    if year <= 2007: base_rate = 0.050
    elif 2008 <= year <= 2015: base_rate = 0.0025
    elif 2016 <= year <= 2019: base_rate = 0.020
    elif 2020 <= year <= 2021: base_rate = 0.0025
    else: base_rate = 0.0525
        
    mgt_fee = 0.0095   
    swap_spread = 0.01 
    annual_cost = mgt_fee + 2 * (base_rate + swap_spread)
    return annual_cost / 252.0

def get_monthly_data(df):
    if df.empty: return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    period_idx = df.index.to_period('M')
    month_end_dates = df.index.to_series().groupby(period_idx).max()
    return df.loc[month_end_dates]

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
            threads=False, 
            group_by='ticker' 
        )
        if data.empty: return pd.DataFrame()

        clean_df = pd.DataFrame(index=data.index)
        for t in tickers:
            try:
                if t in data.columns:
                    ticker_df = data[t]
                    if 'Close' in ticker_df.columns: clean_df[t] = ticker_df['Close']
                    elif 'Price' in ticker_df.columns: clean_df[t] = ticker_df['Price']
                    else: clean_df[t] = ticker_df.iloc[:, 0]
            except Exception: pass
                
        if clean_df.index.tz is not None:
            clean_df.index = clean_df.index.tz_localize(None)

        clean_df = clean_df.ffill()
        return clean_df
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

def calculate_live_risk(data):
    if data.empty: return {}
    avail_cols = [c for c in list(MAPPING.keys()) if c in data.columns]
    if not avail_cols: return {}
    
    sma_tickers = [c for c in list(MAPPING.values()) if c in data.columns]
    monthly_prices = get_monthly_data(data[sma_tickers])
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sig = (monthly_prices > monthly_sma).astype(float)
    daily_sma_sig = monthly_sig.reindex(data.index).ffill()
    
    risk_details = {}
    for trade_t, signal_t in MAPPING.items():
        if signal_t not in data.columns: continue
        if trade_t not in data.columns or data[trade_t].isnull().all(): continue
            
        series = data[trade_t]
        ret = data[signal_t].pct_change() * 100
        window = ret.dropna().tail(LIVE_GARCH_WINDOW + 252 + 50) 
        if len(window) < LIVE_GARCH_WINDOW + 50: continue
        
        forecasts = {}
        model_res = None
        dates = window.index
        loop_start = LIVE_GARCH_WINDOW
        
        for t in range(loop_start, len(window)):
            train = window.iloc[t - LIVE_GARCH_WINDOW + 1 : t + 1]
            if len(train) < 50: continue
            if (t - loop_start) % REFIT_STEP == 0 or model_res is None:
                try:
                    am = arch_model(train, vol='Garch', p=1, q=1, dist='t', rescale=False)
                    model_res = am.fit(disp='off', show_warning=False)
                except: pass
            if model_res:
                try:
                    fc = model_res.forecast(horizon=1, reindex=False)
                    vol = np.sqrt(fc.variance.iloc[-1].values[0]) * np.sqrt(252)
                    forecasts[dates[t]] = vol
                except: pass
                
        df = pd.DataFrame({'Price': series, 'Ret': ret})
        df['Vol'] = pd.Series(forecasts).reindex(df.index)
        
        if signal_t in daily_sma_sig.columns: df['SMA_State'] = daily_sma_sig[signal_t]
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
        df = df.dropna(subset=['Weight'])
        risk_details[trade_t] = df

    return risk_details

def calculate_live_selection(data):
    if data.empty: return pd.DataFrame(), None
    avail_keys = [k for k in list(MAPPING.keys()) if k in data.columns and not data[k].isnull().all()]
    if not avail_keys: return pd.DataFrame(), None
    
    prices = data[avail_keys]
    monthly = get_monthly_data(prices)
    if monthly.empty: return pd.DataFrame(), None

    last_date = data.index[-1]
    try:
        tz_tw = pytz.timezone('Asia/Taipei')
        now_tw = datetime.now(tz_tw)
        last_data_period = last_date.to_period('M')
        current_tw_period = pd.Period(now_tw.strftime('%Y-%m'), freq='M')

        if FORCE_EOM or (last_data_period < current_tw_period):
            ref_date = monthly.index[-1]
        else:
            prev_months = monthly[monthly.index.to_period('M') < last_data_period]
            if prev_months.empty: return pd.DataFrame(), None
            ref_date = prev_months.index[-1]
    except Exception as e:
        st.error(f"日期計算錯誤: {e}")
        return pd.DataFrame(), None
    
    metrics = []
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        try:
            if ref_date not in monthly.index: continue
            p_now = monthly.loc[ref_date, ticker]
            
            m = MOM_PERIODS[0]
            loc = monthly.index.get_loc(ref_date)
            if loc >= m:
                p_prev = monthly.iloc[loc-m][ticker]
                if pd.isna(p_prev) or p_prev == 0: row[f'Ret_{m}M'] = np.nan
                else: row[f'Ret_{m}M'] = (p_now - p_prev) / p_prev
            else: row[f'Ret_{m}M'] = np.nan
            
            d_loc = data.index.get_indexer([ref_date], method='pad')[0]
            if d_loc >= 252:
                subset = prices[ticker].iloc[d_loc-252 : d_loc]
                row['Vol_Ann'] = subset.pct_change().std() * np.sqrt(252)
            else: row['Vol_Ann'] = np.nan
            metrics.append(row)
        except: continue
        
    if not metrics: return pd.DataFrame(), None
    
    df = pd.DataFrame(metrics).set_index('Ticker')
    m = MOM_PERIODS[0]
    col = f'Ret_{m}M'
    if col in df.columns:
        risk_adj = df[col] / (df['Vol_Ann'] + 1e-6)
        z = (risk_adj - risk_adj.mean()) / (risk_adj.std() + 1e-6)
        df[f'Z_{m}M'] = z
        df['Total_Z'] = z.fillna(0)
        
    return df.sort_values('Total_Z', ascending=False), ref_date

def calculate_live_safe(data):
    if data.empty: return "TLT", pd.DataFrame(), None
    avail_safe = [t for t in SAFE_POOL if t in data.columns]
    if not avail_safe: return "TLT", pd.DataFrame(), None

    monthly = get_monthly_data(data[avail_safe])
    if monthly.empty: return "TLT", pd.DataFrame(), None

    last_date = data.index[-1]
    try:
        tz_tw = pytz.timezone('Asia/Taipei')
        now_tw = datetime.now(tz_tw)
        last_data_period = last_date.to_period('M')
        current_tw_period = pd.Period(now_tw.strftime('%Y-%m'), freq='M')

        if FORCE_EOM or (last_data_period < current_tw_period):
            ref_date = monthly.index[-1]
        else:
            prev_months = monthly[monthly.index.to_period('M') < last_data_period]
            if prev_months.empty: return "TLT", pd.DataFrame(), None
            ref_date = prev_months.index[-1]
    except:
        return "TLT", pd.DataFrame(), None
    
    loc = monthly.index.get_loc(ref_date)
    if loc >= 12: ret_12m = (monthly.iloc[loc] / monthly.iloc[loc-12]) - 1
    else: ret_12m = pd.Series(0.0, index=avail_safe)
    
    winner = ret_12m.idxmax()
    details = pd.DataFrame({"Ticker": avail_safe, "12M Return": ret_12m.values}).set_index("Ticker")
    return winner, details, ref_date

# ==========================================
# 3. 回測邏輯 (嚴格導入 T+1 與 Open/Close 微結構)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="準備回測數據 (合成三倍槓桿與隔夜成本)...")
def get_synthetic_backtest_data():
    tickers = list(MAPPING.values()) + SAFE_POOL + ['IOO']
    try:
        data_raw = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
        
        if isinstance(data_raw.columns, pd.MultiIndex):
            new_cols = []
            for col in data_raw.columns:
                new_cols.append(f"{col[1]}_{col[0]}")
            data_raw.columns = new_cols

        if data_raw.index.tz is not None:
            data_raw.index = data_raw.index.tz_localize(None)

        req_cols = []
        for t in list(MAPPING.values()) + SAFE_POOL + ['IOO']:
            if f"{t}_Close" in data_raw.columns and f"{t}_Open" in data_raw.columns:
                req_cols.extend([f"{t}_Close", f"{t}_Open"])
            elif f"{t}_Close" in data_raw.columns:
                req_cols.append(f"{t}_Close")

        data_core = data_raw[req_cols].dropna()
        synthetic_data = pd.DataFrame(index=data_core.index)
        
        if 'IOO_Close' in data_core.columns: synthetic_data['IOO'] = data_core['IOO_Close']

        for t_1x in list(MAPPING.values()) + SAFE_POOL:
            if f"{t_1x}_Close" not in data_core.columns: continue
            c_today = data_core[f"{t_1x}_Close"]
            o_today = data_core[f"{t_1x}_Open"]
            c_prev = c_today.shift(1)

            synthetic_data[t_1x] = c_today 
            synthetic_data[f"{t_1x}_Ret_ON"] = (o_today / c_prev) - 1
            synthetic_data[f"{t_1x}_Ret_ID"] = (c_today / o_today) - 1

        REVERSE_MAP = {v: k for k, v in MAPPING.items()} 
        for ticker_1x in MAPPING.values():
            if f"{ticker_1x}_Close" not in data_core.columns: continue
            ticker_3x = REVERSE_MAP[ticker_1x]
            costs = pd.Series([get_daily_leverage_cost(d) for d in data_core.index], index=data_core.index)
            
            synthetic_data[f"{ticker_3x}_Ret_ON"] = (3.0 * synthetic_data[f"{ticker_1x}_Ret_ON"]) - costs
            synthetic_data[f"{ticker_3x}_Ret_ID"] = 3.0 * synthetic_data[f"{ticker_1x}_Ret_ID"]
            
            ret_3x_total = ((1 + synthetic_data[f"{ticker_3x}_Ret_ON"].fillna(0)) * (1 + synthetic_data[f"{ticker_3x}_Ret_ID"].fillna(0))) - 1
            synthetic_data[ticker_3x] = (1 + ret_3x_total).cumprod() * 100
            synthetic_data[f"RAW_{ticker_3x}"] = data_core[f"{ticker_1x}_Close"] 
            
        return synthetic_data.dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="計算滾動回測訊號 (這需要約 1 分鐘)...")
def calculate_backtest_signals_rolling(data):
    raw_cols = [f"RAW_{k}" for k in MAPPING.keys() if f"RAW_{k}" in data.columns]
    if not raw_cols: return pd.DataFrame(), pd.Series(), pd.Series()

    monthly_prices = get_monthly_data(data[raw_cols])
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sma_sig = (monthly_prices > monthly_sma).astype(float)
    daily_sma_sig = monthly_sma_sig.reindex(data.index).ffill()
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
                try:
                    am = arch_model(train, vol='Garch', p=1, q=1, dist='t', rescale=False)
                    model_res = am.fit(disp='off', show_warning=False)
                except: pass
            if model_res:
                try:
                    fc = model_res.forecast(horizon=1, reindex=False)
                    vol = np.sqrt(fc.variance.iloc[-1].values[0]) * np.sqrt(252)
                    forecasts[dates[t]] = vol
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
            s_sig = daily_sma_sig[ticker_3x]
            h_risk_weights[ticker_3x] = 0.5*g_sig + 0.5*s_sig
        
    h_risk_weights = h_risk_weights.dropna()
    
    monthly_src = get_monthly_data(data[raw_cols])
    monthly_src.columns = [c.replace("RAW_", "") for c in raw_cols]
    
    daily_vol = data[raw_cols].pct_change().rolling(252).std() * np.sqrt(252)
    monthly_vol = get_monthly_data(daily_vol)
    monthly_vol.columns = monthly_src.columns
    
    m = MOM_PERIODS[0]
    ret = monthly_src.pct_change(m)
    risk_adj = ret / (monthly_vol + 1e-6)
    z = risk_adj.sub(risk_adj.mean(axis=1), axis=0).div(risk_adj.std(axis=1)+1e-6, axis=0)
    hist_winners = z.fillna(0).idxmax(axis=1)
    
    # [BUG FIX]: 避開 pct_change(12) 產生的全 NaN 列導致的 idxmax 崩潰
    avail_safe = [t for t in SAFE_POOL if t in data.columns]
    safe_monthly = get_monthly_data(data[avail_safe])
    safe_ret_12m = safe_monthly.pct_change(12)
    hist_safe = safe_ret_12m.dropna(how='all').idxmax(axis=1)
    hist_safe = hist_safe.reindex(safe_monthly.index).fillna('TLT')
    
    return h_risk_weights, hist_winners, hist_safe

def run_backtest_logic(data, risk_weights, winners_series, safe_signals):
    dates = data.index
    start_idx = BACKTEST_GARCH_WINDOW + 252
    
    vt_start = data['IOO'].first_valid_index() if 'IOO' in data.columns else None
    if vt_start:
        vt_idx = data.index.get_loc(vt_start)
        start_idx = max(start_idx, vt_idx)
    
    if start_idx >= len(dates): return None, None, None, None
    
    strategy_ret = []
    valid_dates = []
    hold_counts = defaultdict(float)
    prev_pos = {}
    
    for i in range(start_idx, len(dates)):
        today = dates[i]
        yesterday = dates[i-1]
        
        past_wins = winners_series[winners_series.index <= yesterday]
        if past_wins.empty: continue
        target_risky = past_wins.iloc[-1]
        
        past_safe = safe_signals[safe_signals.index <= yesterday]
        if past_safe.empty: target_safe = 'TLT'
        else: target_safe = past_safe.iloc[-1]
        
        if target_risky in risk_weights.columns and yesterday in risk_weights.index:
            w_risk = risk_weights.loc[yesterday, target_risky]
            if pd.isna(w_risk): w_risk = 0.0
        else: w_risk = 0.0
        w_safe = 1.0 - w_risk
        
        curr_pos = {}
        if w_risk > 0: curr_pos[target_risky] = w_risk
        if w_safe > 0: curr_pos[target_safe] = w_safe
        
        drifted_pos = {}
        total_val = 0.0
        for asset, w_prev in prev_pos.items():
            r_on_col = f"{asset}_Ret_ON"
            r_on = data[r_on_col].iloc[i] if r_on_col in data.columns else 0
            if np.isnan(r_on): r_on = 0
            val = w_prev * (1 + r_on)
            drifted_pos[asset] = val
            total_val += val
            
        overnight_ret = total_val - 1.0 if prev_pos else 0.0
        
        if total_val > 0:
            for asset in drifted_pos: drifted_pos[asset] /= total_val
            
        cost = 0.0
        all_assets = set(list(drifted_pos.keys()) + list(curr_pos.keys()))
        for asset in all_assets:
            w_d = drifted_pos.get(asset, 0.0)
            w_t = curr_pos.get(asset, 0.0)
            if w_d != w_t: cost += abs(w_t - w_d) * TRANSACTION_COST
            
        intraday_ret = 0.0
        for asset, w_curr in curr_pos.items():
            r_id_col = f"{asset}_Ret_ID"
            r_id = data[r_id_col].iloc[i] if r_id_col in data.columns else 0
            if np.isnan(r_id): r_id = 0
            intraday_ret += w_curr * r_id
            
        day_ret = ((1 + overnight_ret) * (1 - cost) * (1 + intraday_ret)) - 1
        
        strategy_ret.append(day_ret)
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
        
    vt_eq = pd.Series(1.0, index=valid_dates)
    if 'IOO' in data.columns:
        vt_ret = data['IOO'].loc[valid_dates].pct_change().fillna(0)
        vt_eq = (1 + vt_ret).cumprod()
        
    return cum_eq, b_eq, vt_eq, hold_counts

# ==========================================
# 4. Dashboard 介面
# ==========================================
st.title("🛡️ 雙重動能與動態風控 (機構實盤版)")
st.caption(f"配置: 對稱夏普動能 [12M] / SMA {SMA_MONTHS}M / GARCH (Q{RISK_CONFIG['UPRO']['exit_q']*100:.0f}) / T+1 執行")

with st.expander("🛠️ 數據除錯與狀態 (若數據為 N/A 請點此)"):
    live_data = get_live_data()
    st.write("原始數據形狀:", live_data.shape)
    
    has_edc = 'EDC' in live_data.columns
    st.write(f"EDC 數據狀態: {'✅ 成功抓取' if has_edc else '❌ 缺失'}")
    
    st.write("最後更新日期:", live_data.index[-1] if not live_data.empty else "無")
    
    try:
        tz_tw = pytz.timezone('Asia/Taipei')
        st.write("系統時間 (Taiwan):", datetime.now(tz_tw))
    except:
        st.write("系統時間: 本地時區庫未加載")
    
    if live_data.empty:
        st.error("⚠️ 警告：所有數據下載失敗。")
    else:
        st.success("✅ 數據下載流程完成")

# --- Live Data Loading ---
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
    st.info(f"🔒 **訊號鎖定日**: {sel_date.strftime('%Y-%m-%d')} (根據 {sel_date.strftime('%Y-%m')} 月底收盤)")

with st.expander("📖 策略詳細規則 (黃金規格書)", expanded=False):
    st.markdown(r"""
    這份程式碼建構了一個符合頂尖量化機構標準的 **「雙重動能與動態雙層風控（Dual Momentum with Dynamic Dual-Layer Risk Control）」** 策略儀表板，並包含即時監控與嚴格微結構滾動回測兩大模組。
    
    ### 1. 投資全集與資產池 (Asset Universe)
    策略採用 **槓桿 ETF** 作為進攻資產，並透過 **原型 ETF (1x)** 的數據來生成訊號與合成回測歷史。
    
    | 角色 | 交易代號 (3x) | 訊號源代號 (1x) | 對應資產類別 |
    | :--- | :--- | :--- | :--- |
    | **進攻 (Risky)** | **UPRO** | SPY | 美股大型股 (S&P 500) |
    | **進攻 (Risky)** | **EURL** | VGK | 歐洲已開發市場 |
    | **進攻 (Risky)** | **EDC** | EEM | 新興市場 |
    | **避險 (Safe)** | **GLD** / **TLT** | (自身) | 黃金 / 20年期美債 |
    
    ### 2. 進攻資產選擇機制 (Alpha Selection)
    策略每月進行一次選股，挑選當下動能最強的 **1 檔** 進攻資產。
    * **頻率**：月頻（Monthly），於每個月最後一個交易日收盤後計算。
    * **動能指標**：對稱夏普動能 (Matched-Period Risk-Adjusted Momentum Z-Score)。
    * **計算步驟**：
        1. **絕對報酬**：計算過去 **12 個月**的累積報酬率。
        2. **對稱風險調整**：計算過去 **252 天（一年）**的年化波動率。將 12M 報酬除以 12M 波動率，得出對稱的 Sharpe-like Ratio。
        3. **橫向標準化**：計算橫向 Z-Score（減去平均除以標準差），最高分者勝出。
    
    ### 3. 雙層動態風控機制 (Beta Risk Control)
    透過兩層獨立風控決定曝險比例。每層貢獻 50% 權重（Ensemble 50/50），持倉水位為 **0%、50% 或 100%**。
    
    **第一層：趨勢濾網 (Trend Filter) - 權重 50%**
    * **指標**：6 個月簡單移動平均線 (SMA 6 Months)。
    * **邏輯**：月收盤價 > 6M均線 $\rightarrow$ **安全 (1)**；反之 $\rightarrow$ **危險 (0)**。
    
    **第二層：波動率濾網 (Volatility Filter) - 權重 50%**
    * **模型**：Standard GARCH(1,1) with Student's t-distribution。
    * **邏輯**：
        1. 預測 T 日的條件波動率。
        2. 計算該波動率在過去 252 天歷史中的 **百分位數**。
        3. **出場**：波動率 > 歷史 **99%** 分位數 $\rightarrow$ **危險 (0)**。
        4. **進場**：波動率 < 歷史 **90%** 分位數 $\rightarrow$ **安全 (1)**。
    
    ### 4. 嚴格交易微結構 (Microstructure & Friction)
    * **真實 3x 融資成本**：精確扣除 ETF 管理費 (0.95%) + **2 倍** 總報酬交換合約 (TRS) 利息（動態聯邦基金利率 + 1.0% 利差）。
    * **T+1 開盤執行 (MOO)**：T 日收盤結算訊號，T+1 日開盤市價執行，嚴格承擔隔夜跳空風險與權重漂移耗損。
    * **交易滑價**：單邊 **0.1% (10 bps)** 手續費與衝擊成本。
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

t1, t2, t3, t4, t5, t6 = st.tabs(["數據", "風控細節", "權重狀態", "選股排名", "避險輪動", "部位總結"])
with t1: st.dataframe(live_data.tail(5).style.format("{:.2f}"), use_container_width=True)
with t2:
    if winner in risk_live:
        st.dataframe(risk_live[winner].tail(10)[['Price','Vol','Exit_Th','Entry_Th','GARCH_State']].style.format("{:.2f}"), use_container_width=True)
with t3:
    if winner in risk_live:
        st.dataframe(risk_live[winner].tail(10)[['GARCH_State','SMA_State','Weight']], use_container_width=True)
with t4: st.dataframe(sel_df.style.format("{:.2f}"), use_container_width=True)
with t5: st.dataframe(safe_df.style.format("{:.2%}"), use_container_width=True)
with t6: st.success(f"建議持有: **{final_w*100:.0f}% {winner}** + **{(1-final_w)*100:.0f}% {safe_win}**")

st.divider()

# ==========================================
# 5. 回測區塊
# ==========================================
st.header("⏳ 嚴格微結構滾動回測 (T+1 MOO & TRS Costs)")
st.caption("回測數據使用 1x 原型 ETF 合成，並扣除真實 3x 槓桿融資利息與交易滑價。基準指數替換為 IOO (Global 100) 以對齊 2008 年起點。")

syn_data = get_synthetic_backtest_data()

if not syn_data.empty:
    if st.button("🚀 開始實盤級別滾動回測 (約需 1 分鐘)"):
        with st.spinner("正在進行 GARCH 滾動訓練與參數擬合..."):
            h_risk, h_win, h_safe = calculate_backtest_signals_rolling(syn_data)
            
        with st.spinner("正在執行 T+1 開盤交易結算..."):
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
                sortino = (excess.mean()*252)/(down_std+1e-6)
                return cagr, sortino, sharpe, mdd
            
            s_s = calc_stats(s_eq, s_eq.pct_change().fillna(0))
            b_s = calc_stats(b_eq, b_eq.pct_change().fillna(0))
            v_s = calc_stats(v_eq, v_eq.pct_change().fillna(0))
            
            r5_s = s_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()
            r5_b = b_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()
            r5_v = v_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1).mean()

            st.write(f"### 📈 績效指標 (回測起迄: {s_eq.index[0].strftime('%Y-%m-%d')} ~ {s_eq.index[-1].strftime('%Y-%m-%d')})")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            def m_box(label, v, b, vt, fmt="{:.2%}"):
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{label}</p>
                    <p class="metric-value">{fmt.format(v)}</p>
                    <p class="metric-sub">3x: {fmt.format(b)} | IOO: {fmt.format(vt)}</p>
                </div>""", unsafe_allow_html=True)
                
            with m1: m_box("CAGR", s_s[0], b_s[0], v_s[0])
            with m2: m_box("Sortino", s_s[1], b_s[1], v_s[1], "{:.2f}")
            with m3: m_box("Sharpe", s_s[2], b_s[2], v_s[2], "{:.2f}")
            with m4: m_box("Avg 5Y", r5_s, r5_b, r5_v)
            with m5: m_box("MaxDD", s_s[3], b_s[3], v_s[3])
            
            risky_hold_sum = sum([v for k,v in holds.items() if k in MAPPING.keys()])
            t_3x = risky_hold_sum / len(s_eq)
            with m6: m_box("Time in 3x", t_3x, 1.0, 1.0)
            
            st.divider()
            
            st.write("### 📊 權益曲線")
            df_chart = pd.DataFrame({'Date': s_eq.index, 'Strategy': s_eq, 'Bench (3x)': b_eq, 'IOO': v_eq}).melt('Date', var_name='Asset', value_name='NAV')
            c1 = alt.Chart(df_chart).mark_line().encode(
                x='Date', y=alt.Y('NAV', scale=alt.Scale(type='log')), 
                color='Asset', tooltip=['Date','Asset', alt.Tooltip('NAV', format='.2f')]
            ).properties(width=800, height=350)
            st.altair_chart(c1, use_container_width=True)

            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.write("### 📉 回撤幅度")
                dd_s = s_eq/s_eq.cummax()-1
                dd_b = b_eq/b_eq.cummax()-1
                dd_v = v_eq/v_eq.cummax()-1
                df_dd = pd.DataFrame({'Date': s_eq.index, 'Strategy': dd_s, 'Bench (3x)': dd_b, 'IOO': dd_v}).melt('Date', var_name='Asset', value_name='DD')
                c2 = alt.Chart(df_dd).mark_line().encode(
                    x='Date', y=alt.Y('DD', axis=alt.Axis(format='%')), 
                    color='Asset', tooltip=['Date','Asset', alt.Tooltip('DD', format='.2%')]
                ).properties(width=400, height=250)
                st.altair_chart(c2, use_container_width=True)
            
            with c_col2:
                st.write("### 🔄 滾動 5 年年化")
                roll_s = s_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1)
                roll_b = b_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1)
                roll_v = v_eq.rolling(1260).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(252/1260)-1)
                df_r5 = pd.DataFrame({'Date': s_eq.index, 'Strategy': roll_s, 'Bench (3x)': roll_b, 'IOO': roll_v}).melt('Date', var_name='Asset', value_name='Roll5Y')
                c3 = alt.Chart(df_r5.dropna()).mark_line().encode(
                    x='Date', y=alt.Y('Roll5Y', axis=alt.Axis(format='%')), 
                    color='Asset', tooltip=['Date','Asset', alt.Tooltip('Roll5Y', format='.2%')]
                ).properties(width=400, height=250)
                st.altair_chart(c3, use_container_width=True)
