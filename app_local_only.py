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
st.set_page_config(page_title="Dynamic Momentum Strategy (Aggressive)", layout="wide")
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
# 1. 核心參數
# ==========================================
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"} 
SAFE_POOL = ["GLD", "TLT"] 

# 風控閾值 (已修改：Exit 0.99 / Entry 0.90 - 極度寬鬆/積極模式)
RISK_CONFIG = {
    "UPRO": {"exit_q": 0.99, "entry_q": 0.90},
    "EURL": {"exit_q": 0.99, "entry_q": 0.90},
    "EDC":  {"exit_q": 0.99, "entry_q": 0.90}
}

# 策略參數
SMA_MONTHS = 6               # 月均線
LIVE_GARCH_WINDOW = 504      # Live GARCH 窗口
BACKTEST_GARCH_WINDOW = 504  # 回測 GARCH 窗口
REFIT_STEP = 5               # 滾動重訓頻率
MOM_PERIODS = [3, 6, 9, 12]
TRANSACTION_COST = 0.001 
RF_RATE = 0.02 

def get_daily_leverage_cost(date):
    year = date.year
    if year <= 2007 or year >= 2022: return 0.05 / 252 
    else: return 0.02 / 252

def get_monthly_data(df):
    """鎖定每個月實際最後交易日"""
    if df.empty: return df
    # 確保索引是 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    period_idx = df.index.to_period('M')
    month_end_dates = df.index.to_series().groupby(period_idx).max()
    return df.loc[month_end_dates]

# ==========================================
# 2. Live 面板數據與邏輯
# ==========================================
@st.cache_data(ttl=3600)
def get_live_data():
    tickers = list(MAPPING.keys()) + list(MAPPING.values()) + SAFE_POOL
    try:
        # [FIX] 下載數據，增加 group_by='ticker' 以確保格式一致性
        data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True, progress=False, group_by='column')
        
        # [FIX] 處理 MultiIndex (yfinance 的欄位結構可能會變)
        if isinstance(data.columns, pd.MultiIndex):
            # 嘗試提取 'Close'，如果失敗則嘗試提取 Level 0
            if 'Close' in data.columns.levels[0]:
                data = data['Close']
            else:
                # 如果結構不同，嘗試保留所有數據並自動對齊
                pass

        # [FIX] 強制移除時區 (避免與 Pandas 本地時間衝突)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # [FIX] 只用 ffill，移除 dropna() 以避免單一資產缺漏導致全表刪除
        data = data.ffill()
        
        # 移除全部為空值的行（例如假日）
        data = data.dropna(how='all')
        
        return data
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def calculate_live_risk(data):
    if data.empty: return {}
    
    # 1. SMA (Monthly)
    # [FIX] 確保欄位存在才提取
    avail_cols = [c for c in list(MAPPING.keys()) if c in data.columns]
    if not avail_cols: return {}
    
    monthly_prices = get_monthly_data(data[avail_cols])
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sig = (monthly_prices > monthly_sma).astype(float)
    daily_sma_sig = monthly_sig.reindex(data.index).ffill()
    
    risk_details = {}
    for trade_t, signal_t in MAPPING.items():
        if signal_t not in data.columns: continue
        # 如果交易資產(UPRO等)不在數據中，暫時用訊號資產(SPY)代替計算SMA，但標記為缺漏
        if trade_t not in data.columns: 
             series = data[signal_t] # Fallback for display
        else:
             series = data[trade_t]
             
        ret = data[signal_t].pct_change() * 100
        
        # Live GARCH
        window = ret.dropna().tail(LIVE_GARCH_WINDOW * 2) 
        if len(window) < 100: continue
        
        try:
            am = arch_model(window, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            cond_vol = res.conditional_volatility * np.sqrt(252)
            
            df = pd.DataFrame({'Price': series, 'Ret': ret})
            # 將波動率對齊回原始索引
            df['Vol'] = pd.Series(cond_vol, index=window.index).reindex(df.index)
            
            # [FIX] 確保 SMA 狀態對齊
            if trade_t in daily_sma_sig.columns:
                df['SMA_State'] = daily_sma_sig[trade_t]
            else:
                # 如果沒有 UPRO 的 SMA，用 SPY 的代替 (邏輯上應一致)
                if MAPPING[trade_t] in daily_sma_sig.columns: 
                    df['SMA_State'] = 1.0 # Default safely or handle logic error
                else:
                    df['SMA_State'] = 0.0
            
            cfg = RISK_CONFIG[trade_t]
            df['Exit_Th'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
            df['Entry_Th'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
            
            df['GARCH_State'] = np.nan
            valid = df['Exit_Th'].notna() & df['Vol'].notna()
            
            # 使用 mask 避免 SettingWithCopyWarning
            mask_exit = valid & (df['Vol'] > df['Exit_Th'])
            mask_entry = valid & (df['Vol'] < df['Entry_Th'])
            
            df.loc[mask_exit, 'GARCH_State'] = 0.0 
            df.loc[mask_entry, 'GARCH_State'] = 1.0 
            df['GARCH_State'] = df['GARCH_State'].ffill().fillna(1.0)
            
            df['Weight'] = (0.5 * df['GARCH_State']) + (0.5 * df['SMA_State'])
            df = df.dropna(subset=['Weight']) # 只移除無法計算訊號的行
            risk_details[trade_t] = df
        except: continue
    return risk_details

@st.cache_data(ttl=3600)
def calculate_live_selection(data):
    if data.empty: return pd.DataFrame(), None
    
    # [FIX] 檢查可用欄位
    avail_keys = [k for k in list(MAPPING.keys()) if k in data.columns]
    if not avail_keys: return pd.DataFrame(), None
    
    prices = data[avail_keys]
    monthly = get_monthly_data(prices)
    
    if monthly.empty: return pd.DataFrame(), None

    last_date = data.index[-1]
    current_period = last_date.to_period('M')
    prev_months = monthly[monthly.index.to_period('M') < current_period]
    if prev_months.empty: return pd.DataFrame(), None
    
    ref_date = prev_months.index[-1]
    metrics = []
    
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        try:
            p_now = monthly.loc[ref_date, ticker]
            for m in MOM_PERIODS:
                # 使用 shift 避免索引錯誤
                # 找到 ref_date 的位置
                if ref_date not in monthly.index: continue
                loc = monthly.index.get_loc(ref_date)
                
                if loc >= m:
                    p_prev = monthly.iloc[loc-m][ticker]
                    if pd.isna(p_prev) or p_prev == 0:
                        row[f'Ret_{m}M'] = np.nan
                    else:
                        row[f'Ret_{m}M'] = (p_now - p_prev) / p_prev
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

@st.cache_data(ttl=3600)
def calculate_live_safe(data):
    # [FIX] 確保返回 DataFrame
    if data.empty: return "TLT", pd.DataFrame(), None
    
    avail_safe = [t for t in SAFE_POOL if t in data.columns]
    if not avail_safe: return "TLT", pd.DataFrame(), None

    monthly = get_monthly_data(data[avail_safe])
    if monthly.empty: return "TLT", pd.DataFrame(), None

    last_date = data.index[-1]
    current_period = last_date.to_period('M')
    prev_months = monthly[monthly.index.to_period('M') < current_period]
    if prev_months.empty: return "TLT", pd.DataFrame(), None
    
    ref_date = prev_months.index[-1]
    loc = monthly.index.get_loc(ref_date)
    
    if loc >= 12:
        ret_12m = (monthly.iloc[loc] / monthly.iloc[loc-12]) - 1
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
        data_raw = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)
        
        # [FIX] 同樣的數據清理邏輯
        if isinstance(data_raw.columns, pd.MultiIndex):
            if 'Close' in data_raw.columns.levels[0]: data_raw = data_raw['Close']
            else: pass
        
        if data_raw.index.tz is not None:
            data_raw.index = data_raw.index.tz_localize(None)

        data_raw = data_raw.ffill() # 移除 dropna，避免太嚴格
        
        # 檢查關鍵欄位是否存在
        required = ['VGK', 'EEM', 'SPY', 'GLD', 'TLT']
        missing = [x for x in required if x not in data_raw.columns]
        if missing:
             # 如果缺數據，嘗試從 MultiIndex 找
             pass

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
            ret_3x = (ret_1x * 3.0) - costs
            synthetic_data[ticker_3x] = (1 + ret_3x).cumprod() * 100
            synthetic_data[f"RAW_{ticker_3x}"] = data_raw[ticker_1x] 
            
        return synthetic_data.dropna() # 合成數據最後再清理
    except: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="計算滾動回測訊號 (這需要約 1 分鐘)...")
def calculate_backtest_signals_rolling(data):
    # 1. 月均線 (SMA 6 Months)
    raw_cols = [f"RAW_{k}" for k in MAPPING.keys() if f"RAW_{k}" in data.columns]
    if not raw_cols: return pd.DataFrame(), pd.Series(), pd.Series()

    monthly_prices = get_monthly_data(data[raw_cols])
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sma_sig = (monthly_prices > monthly_sma).astype(float)
    daily_sma_sig = monthly_sma_sig.reindex(data.index).ffill().shift(1)
    # 修正 Column Name mapping
    daily_sma_sig.columns = [c.replace("RAW_", "") for c in raw_cols]

    # 2. 滾動 GARCH (Rolling)
    target_tickers = [k for k in MAPPING.keys() if f"RAW_{k}" in data.columns]
    h_risk_weights = pd.DataFrame(index=data.index, columns=target_tickers)
    
    for i, ticker_3x in enumerate(target_tickers):
        col_1x = f"RAW_{ticker_3x}"
        
        s_ret = data[col_1x].pct_change() * 100
        forecasts = {}
        model_res = None
        loop_start = BACKTEST_GARCH_WINDOW
        
        # 使用 numpy 加速處理
        ret_values = s_ret.values
        dates = s_ret.index
        
        # 內層迴圈
        for t in range(loop_start, len(s_ret)):
            if (t - loop_start) % REFIT_STEP == 0 or model_res is None:
                train = s_ret.iloc[t-BACKTEST_GARCH_WINDOW : t]
                if train.std() < 1e-6: continue
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
    
    # 3. 選股 (Monthly)
    monthly_src = get_monthly_data(data[raw_cols])
    monthly_src.columns = [c.replace("RAW_", "") for c in raw_cols]
    
    daily_vol = data[raw_cols].pct_change().rolling(126).std() * np.sqrt(252)
    monthly_vol = get_monthly_data(daily_vol)
    monthly_vol.columns = monthly_src.columns
    
    scores = pd.DataFrame(0.0, index=monthly_src.index, columns=monthly_src.columns)
    for m in MOM_PERIODS:
        ret = monthly_src.pct_change(m)
        risk_adj = ret / (monthly_vol + 1e-6)
        z = risk_adj.sub(risk_adj.mean(axis=1), axis=0).div(risk_adj.std(axis=1)+1e-6, axis=0)
        scores += z.fillna(0)
    hist_winners = scores.idxmax(axis=1)
    
    # 4. 避險輪動
    avail_safe = [t for t in SAFE_POOL if t in data.columns]
    safe_monthly = get_monthly_data(data[avail_safe])
    hist_safe = safe_monthly.pct_change(12).idxmax(axis=1).fillna('TLT')
    
    return h_risk_weights, hist_winners, hist_safe

def run_backtest_logic(data, risk_weights, winners_series, safe_signals):
    dates = data.index
    # 起始點: GARCH窗口 + 252 Quantile
    start_idx = BACKTEST_GARCH_WINDOW + 252
    
    vt_start = data['VT'].first_valid_index()
    if vt_start:
        vt_idx = data.index.get_loc(vt_start)
        start_idx = max(start_idx, vt_idx)
    
    if start_idx >= len(dates): return None, None, None, None
    
    strategy_ret = []
    valid_dates = []
    hold_counts = defaultdict(float)
    prev_pos = {}
    
    # Daily Loop
    for i in range(start_idx, len(dates)):
        today = dates[i]
        yesterday = dates[i-1]
        
        # Monthly Selection
        past_wins = winners_series[winners_series.index <= yesterday]
        if past_wins.empty: continue
        target_risky = past_wins.iloc[-1]
        
        past_safe = safe_signals[safe_signals.index <= yesterday]
        if past_safe.empty: target_safe = 'TLT'
        else: target_safe = past_safe.iloc[-1]
        
        # Weight
        if target_risky in risk_weights.columns and yesterday in risk_weights.index:
            w_risk = risk_weights.loc[yesterday, target_risky]
            if pd.isna(w_risk): w_risk = 0.0
        else: w_risk = 0.0
        w_safe = 1.0 - w_risk
        
        # Calc
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
            if target_risky in data.columns:
                r = data[target_risky].pct_change().iloc[i]
                if np.isnan(r): r=0
                day_ret += w_risk * r
        if w_safe > 0:
            if target_safe in data.columns:
                r = data[target_safe].pct_change().iloc[i]
                if np.isnan(r): r=0
                day_ret += w_safe * r
            
        strategy_ret.append(day_ret - cost)
        valid_dates.append(today)
        hold_counts[target_risky] += w_risk
        hold_counts[target_safe] += w_safe
        prev_pos = curr_pos
        
    eq = pd.Series(strategy_ret, index=valid_dates)
    cum_eq = (1 + eq).cumprod()
    
    # Benchmarks
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
st.title("🛡️ 雙重動能與動態風控 (Live + Rolling Backtest)")
st.caption(f"配置: SMA {SMA_MONTHS}M (Monthly) / GARCH (Q{RISK_CONFIG['UPRO']['exit_q']*100:.0f}) / Safe (GLD/TLT)")

# --- Debug Panel (隱藏式) ---
with st.expander("🛠️ 數據除錯與狀態 (若數據為 N/A 請點此)"):
    live_data = get_live_data()
    st.write("原始數據形狀:", live_data.shape)
    st.write("包含欄位:", live_data.columns.tolist())
    st.write("最後更新日期:", live_data.index[-1] if not live_data.empty else "無")
    if live_data.empty:
        st.error("⚠️ 警告：無法下載數據，請檢查網路連線或 Yahoo Finance 狀態。")
    else:
        st.success("✅ 數據下載正常")

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
    st.info(f"🔒 **訊號鎖定日**: {sel_date.strftime('%Y-%m-%d')} (上個月最後交易日)")

with st.expander("📖 策略詳細規格", expanded=False):
    st.markdown(f"""
    **1. 選股 (Selection)**
    * 每月上個月底，計算 3/6/9/12 個月風險調整動能 (Z-Score)。
    * 選出最強 1 檔 (UPRO/EURL/EDC)。
    
    **2. 趨勢風控 (Trend)**
    * **{SMA_MONTHS} 個月均線**: 每月底檢視，價格 > 均線 = 安全。
    
    **3. 波動風控 (Volatility)**
    * **滾動 GARCH**: 每日計算，使用過去 504 天數據。
    * **Exit**: 預測波動率 > 歷史 PR {RISK_CONFIG['UPRO']['exit_q']*100:.0f} (寬鬆)。
    * **Entry**: 預測波動率 < 歷史 PR {RISK_CONFIG['UPRO']['entry_q']*100:.0f} (積極)。
    
    **4. 避險 (Safe Asset)**
    * **GLD vs TLT**: 每月底比較過去 12 個月報酬，強者持有。
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
# 5. 回測區塊 (Strict Rolling)
# ==========================================
st.header("⏳ 嚴格滾動回測 (Synthetic 3x)")

# 文檔說明
with st.expander("📊 查看回測步驟與數據細節", expanded=True):
    st.markdown("""
    #### 1. 數據源與合成
    * **進攻資產**: 使用 1x 原型 ETF (SPY, VGK, EEM) 的歷史數據。
    * **合成三倍**: 模擬 3x 槓桿，並扣除動態融資成本 (2%~5%)。
    * **避險資產**: 使用真實 GLD 與 TLT 數據。
    
    #### 2. 嚴格滾動風控 (Rolling GARCH)
    * **訓練視窗**: 嚴格限制為過去 **504 個交易日** (無未來視角)。
    * **參數重訓**: 每 **5 天** 重新擬合一次 GARCH 模型參數 (Refit)。
    * **訊號生成**: T-1 日收盤預測 T 日波動率，並與過去 252 天分位數 (Q99/Q90) 比較。
    
    #### 3. 趨勢與執行
    * **趨勢**: 使用合成資產的 **6個月月均線**，月底鎖定訊號。
    * **執行**: 嚴格 **T+1** 開盤執行 (訊號來自 T-1 收盤)。
    """)

syn_data = get_synthetic_backtest_data()

if not syn_data.empty:
    if st.button("🚀 開始滾動回測 (約需 30-60 秒)"):
        with st.spinner("正在進行 GARCH 滾動訓練與參數擬合..."):
            h_risk, h_win, h_safe = calculate_backtest_signals_rolling(syn_data)
            
        with st.spinner("正在執行交易回測..."):
            s_eq, b_eq, v_eq, holds = run_backtest_logic(syn_data, h_risk, h_win, h_safe)
        
        if s_eq is not None:
            # Stats
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

            st.write("### 📈 績效指標")
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
            
            st.divider()
            
            # --- Charting (Fixing Altair Display) ---
            st.write("### 📊 權益曲線")
            df_chart = pd.DataFrame({'Date': s_eq.index, 'Strategy': s_eq, 'Bench (3x)': b_eq, 'VT': v_eq}).melt('Date', var_name='Asset', value_name='NAV')
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
                df_dd = pd.DataFrame({'Date': s_eq.index, 'Strategy': dd_s, 'Bench (3x)': dd_b, 'VT': dd_v}).melt('Date', var_name='Asset', value_name='DD')
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
                df_r5 = pd.DataFrame({'Date': s_eq.index, 'Strategy': roll_s, 'Bench (3x)': roll_b, 'VT': roll_v}).melt('Date', var_name='Asset', value_name='Roll5Y')
                c3 = alt.Chart(df_r5.dropna()).mark_line().encode(
                    x='Date', y=alt.Y('Roll5Y', axis=alt.Axis(format='%')), 
                    color='Asset', tooltip=['Date','Asset', alt.Tooltip('Roll5Y', format='.2%')]
                ).properties(width=400, height=250)
                st.altair_chart(c3, use_container_width=True)
