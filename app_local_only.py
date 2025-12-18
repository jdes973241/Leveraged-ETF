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

# 風控閾值 (Exit 0.99 / Entry 0.90)
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
        data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True, progress=False, group_by='column')
        
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                data = data['Close']
            else:
                pass

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = data.ffill()
        data = data.dropna(how='all')
        
        return data
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def calculate_live_risk(data):
    if data.empty: return {}
    
    avail_cols = [c for c in list(MAPPING.keys()) if c in data.columns]
    if not avail_cols: return {}
    
    monthly_prices = get_monthly_data(data[avail_cols])
    monthly_sma = monthly_prices.rolling(SMA_MONTHS).mean()
    monthly_sig = (monthly_prices > monthly_sma).astype(float)
    daily_sma_sig = monthly_sig.reindex(data.index).ffill()
    
    risk_details = {}
    for trade_t, signal_t in MAPPING.items():
        if signal_t not in data.columns: continue
        if trade_t not in data.columns: 
             series = data[signal_t] 
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
            df['Vol'] = pd.Series(cond_vol, index=window.index).reindex(df.index)
            
            if trade_t in daily_sma_sig.columns:
                df['SMA_State'] = daily_sma_sig[trade_t]
            else:
                if MAPPING[trade_t] in daily_sma_sig.columns: 
                    df['SMA_State'] = 1.0 
                else:
                    df['SMA_State'] = 0.0
            
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
        except: continue
    return risk_details

@st.cache_data(ttl=3600)
def calculate_live_selection(data):
    if data.empty: return pd.DataFrame(), None
    
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
        
        if isinstance(data_raw.columns, pd.MultiIndex):
            if 'Close' in data_raw.columns.levels[0]: data_raw = data_raw['Close']
            else: pass
        
        if data_raw.index.tz is not None:
            data_raw.index = data_raw.index.tz_localize(None)

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
            ret_3x = (ret_1x * 3.0) - costs
            synthetic_data[ticker_3x] = (1 + ret_3x).cumprod() * 100
            synthetic_data[f"RAW_{ticker_3x}"] = data_raw[ticker_1x] 
            
        return synthetic_data.dropna()
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
    daily_sma_sig.columns = [c.replace("RAW_", "") for c in raw_cols]

    # 2. 滾動 GARCH
    target_tickers = [k for k in MAPPING.keys() if f"RAW_{k}" in data.columns]
    h_risk_weights = pd.DataFrame(index=data.index, columns=target_tickers)
    
    for i, ticker_3x in enumerate(target_tickers):
        col_1x = f"RAW_{ticker_3x}"
        s_ret = data[col_1x].pct_change() * 100
        forecasts = {}
        model_res = None
        loop_start = BACKTEST_GARCH_WINDOW
        
        ret_values = s_ret.values
        dates = s_ret.index
        
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
    # 起始點: GARCH窗口(504) + Quantile窗口(252) = 756
    start_idx = BACKTEST_GARCH_WINDOW + 252
    
    # 檢查 VT 上市時間，避免回測早期 VT 數據為空
    vt_start = data['VT'].first_valid_index()
    if vt_start:
        vt_idx = data.index.get_loc(vt_start)
        # 取最大值，確保數據與模型皆已備妥
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
        
        # 記錄持倉以計算 Time in 3x
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

with st.expander("📖 策略詳細規則", expanded=False):
    st.markdown(r"""
    這份程式碼建構了一個 **「雙重動能與動態雙層風控（Dual Momentum with Dynamic Dual-Layer Risk Control）」** 的策略儀表板，並包含即時監控（Live）與嚴格的滾動回測（Strict Rolling Backtest）兩大模組。
    
    以下是依據程式碼邏輯拆解的完整策略規格與數據細節：
    
    ### 1. 投資全集與資產映射 (Asset Universe)
    策略採用 **槓桿 ETF** 作為進攻資產，並透過 **原型 ETF (1x)** 的數據來生成訊號與合成回測歷史，以解決槓桿 ETF 歷史數據過短的問題。
    
    | 角色 | 交易代號 (3x) | 訊號源代號 (1x) | 對應資產類別 |
    | :--- | :--- | :--- | :--- |
    | **進攻 (Risky)** | **UPRO** | SPY | 美股大型股 (S&P 500) |
    | **進攻 (Risky)** | **EURL** | VGK | 歐洲已開發市場 |
    | **進攻 (Risky)** | **EDC** | EEM | 新興市場 |
    | **避險 (Safe)** | **GLD** / **TLT** | (自身) | 黃金 / 20年期美債 |
    
    ### 2. 進攻資產選擇機制 (Selection Logic)
    策略每月進行一次選股，挑選當下動能最強的 **1 檔** 進攻資產。
    * **頻率**：月頻（Monthly），於每個月最後一個交易日計算。
    * **動能指標**：綜合風險調整動能 Z-Score (Composite Risk-Adjusted Momentum Z-Score)。
    * **計算步驟**：
        1. **多週期回報**：計算 3、6、9、12 個月的累積報酬率。
        2. **波動率調整**：將上述報酬率除以過去 126 天（約半年）的年化波動率，得到 Sharpe-like ratio。
        3. **標準化 (Z-Score)**：將三個標的在同一週期內的數值進行標準化（Z-Score）。
        4. **加總評分**：將四個週期 (3/6/9/12M) 的 Z-Score 相加，總分最高者勝出。
    
    ### 3. 雙層動態風控機制 (Risk Control Logic)
    選定標的後，透過兩層風控決定曝險比例（Weight）。每層風控貢獻 50% 權重，因此持倉水位可能為 **0% (全避險)、50% (半倉)、100% (全攻)**。
    
    **第一層：趨勢濾網 (Trend Filter) - 權重 50%**
    * **指標**：6 個月簡單移動平均線 (SMA 6 Months)。
    * **邏輯**：
        * 若 **月收盤價 > 6個月均線** $\rightarrow$ **SMA_State = 1 (安全)**。
        * 若 **月收盤價 < 6個月均線** $\rightarrow$ **SMA_State = 0 (危險)**。
    * **數據源**：使用原型 ETF (如 SPY) 判斷。
    
    **第二層：波動率濾網 (Volatility Filter) - 權重 50%**
    * **模型**：GARCH(1,1) with Student's t-distribution。
    * **訓練窗口**：
        * **Live**：最近 504 天。
        * **Backtest**：嚴格滾動視窗（Rolling Window），只看過去 504 天，絕不使用未來數據。
    * **重訓頻率 (Refit)**：每 5 天重新擬合一次 GARCH 參數。
    * **訊號邏輯 (Regime Switching)**：
        1. 預測 T 日的條件波動率 (Conditional Volatility)。
        2. 計算該波動率在過去 252 天歷史中的 **百分位數 (Quantile/Percentile)**。
        3. **出場 (Exit)**：若波動率 > 歷史 **99%** 分位數 $\rightarrow$ **GARCH_State = 0 (危險)**。
        4. **進場 (Entry)**：若波動率 < 歷史 **90%** 分位數 $\rightarrow$ **GARCH_State = 1 (安全)**。
        5. **滯後性**：具備 Hysteresis 特性，未觸發閾值前維持原狀態。
    
    ### 4. 避險資產輪動 (Safe Asset Rotation)
    當進攻資產權重未滿 100% 時，剩餘資金配置於避險資產。
    * **候選池**：GLD (黃金), TLT (美債)。
    * **選擇邏輯**：比較兩者 **過去 12 個月** 的累積報酬率，強者勝出。
    * **預設值**：若數據不足，預設持有 TLT。
    
    ### 5. 嚴格回測細節 (Strict Backtest Specifics)
    這部分程式碼非常強調「真實性」與「防偏誤」，具體實作如下：
    
    **合成數據 (Synthetic Data)**：
    * 不直接使用 3x ETF 歷史數據（因時間太短）。
    * **合成公式**：$Ret_{3x} = (Ret_{1x} \times 3.0) - Cost_{borrow}$。
    * **融資成本 ($Cost_{borrow}$)**：動態設定。
        * 2008-2021 (低利時期)：年化 2%。
        * 2022-至今 (升息時期) 或 2007以前：年化 5%。
    
    **無前視偏差 (Look-Ahead Bias Free)**：
    * GARCH 模型訓練嚴格限制在 **t-504** 到 **t-1** 的視窗內。
    * SMA 與動能訊號均使用 T-1 日或上個月底的數據。
    
    **交易執行細節**：
    * **T+1 執行**：T 日計算出的訊號，於 T+1 日開盤/收盤價執行（程式碼邏輯為日報酬結算，隱含 T+1 概念）。
    * **交易成本**：單邊 **0.1% (10 bps)**。
    * **無風險利率 (Risk-Free Rate)**：計算 Sharpe Ratio 時使用年化 2%。
    
    ### 6. 輸出指標 (Dashboard Metrics)
    儀表板最終計算並展示以下關鍵績效指標：
    * **CAGR**：年化複合成長率。
    * **Sharpe Ratio**：夏普比率 (超額報酬 / 標準差)。
    * **Sortino Ratio**：索提諾比率 (只考慮下行波動)。
    * **Max Drawdown**：最大回撤。
    * **Avg Roll 5Y**：滾動 5 年平均年化報酬率（評估長期持有穩定性）。
    * **Time in 3x**：持有 3x 槓桿資產的時間比例。
    
    **總結：策略核心公式**
    
    $$Weight_{Risky} = 0.5 \times I(Price > SMA_{6m}) + 0.5 \times I(Vol_{GARCH} < Threshold)$$
    
    $$Position = Weight_{Risky} \times \text{Best\_Momentum\_3x} + (1 - Weight_{Risky}) \times \text{Best\_Safe\_Asset}$$
    
    這是一個結合了 **「相對動能 (選股)」** 與 **「雙重絕對動能 (擇時)」** 的複合策略。
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
st.caption("回測數據使用 1x 原型 ETF 合成，並自動對齊 VT 上市日以確保基準一致。")

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
            
            # [FIX] Time in 3x 修正邏輯
            # holds 的 key 是 3x ticker 名稱 (e.g. UPRO, EURL)，檢查 key 是否在 MAPPING 中
            risky_hold_sum = sum([v for k,v in holds.items() if k in MAPPING.keys()])
            t_3x = risky_hold_sum / len(s_eq)
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
