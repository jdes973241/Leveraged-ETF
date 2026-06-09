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
    # [v8 變更] GARCH Exit Quantile 從 0.99 改為 0.975
    # 證據基礎: test10 (90 組合 GARCH 搜索) + deep_compare 顯示 97.5/90 在
    #   9/9 指標方向一致勝出、3/3 WFA 子期間穩健改善、配對 Bootstrap P~96%、
    #   MaxDD 完全相同、順風期完全相同 (純 Pareto 改進)、學術理論支持 (VaR-97.5)
    "UPRO": {"exit_q": 0.975, "entry_q": 0.90},
    "EURL": {"exit_q": 0.975, "entry_q": 0.90},
    "EDC":  {"exit_q": 0.975, "entry_q": 0.90}
}

SMA_MONTHS = 6               
LIVE_GARCH_WINDOW = 504      
BACKTEST_GARCH_WINDOW = 504  
REFIT_STEP = 5               
# [anchor 防護] GARCH refit 以 data.index 絕對位置錨定 (Live 與回測對齊)。
# 若 yfinance 修訂歷史、改變資料起點, 絕對位置會整體偏移, refit 點錯位。
# 下列基準用於偵測起點漂移並示警。起點由最晚上市標的 (VGK/EURL) 決定, 歷來穩定。
EXPECTED_DATA_START = pd.Timestamp("2005-03-11")
ANCHOR_TOLERANCE_DAYS = 7    # 容忍假日/小幅調整
# [更新] 進攻端動能改為非重疊三段: (0,3], (3,7], (7,12] 月
MOM_SEGMENTS = [(0, 3), (3, 7), (7, 12)]
TRANSACTION_COST = 0.001 
RF_RATE = 0.02 

# ==========================================
# FRED Fed Funds Rate 抓取 (24h 快取 + fallback)
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_fed_funds_rate():
    """
    從 FRED 抓取 Daily Federal Funds Rate (DFF)。
    成功: 返回 pd.Series (日頻，百分比數值如 5.25 代表 5.25%)。
    失敗: 返回 None，上層邏輯自動 fallback 到靜態分段表。
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
        df = pd.read_csv(url)
        date_col = df.columns[0]
        rate_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
        s = df[rate_col].dropna() / 100.0  # 轉為小數
        return s
    except Exception:
        return None

def _get_static_base_rate(year):
    """原始靜態分段表 (fallback 用)"""
    if year <= 2007: return 0.050
    elif 2008 <= year <= 2015: return 0.0025
    elif 2016 <= year <= 2019: return 0.020
    elif 2020 <= year <= 2021: return 0.0025
    else: return 0.0525

def get_daily_leverage_cost(date, fed_rate_series=None):
    """
    計算單日 3x 槓桿融資成本。
    若提供 fed_rate_series (FRED DFF)，優先使用；否則 fallback 到靜態表。
    """
    base_rate = None
    if fed_rate_series is not None and len(fed_rate_series) > 0:
        try:
            # 使用 asof: 找到 <= date 的最新值 (避免前視)
            base_rate = fed_rate_series.asof(date)
            if pd.isna(base_rate):
                base_rate = None
        except Exception:
            base_rate = None
    
    if base_rate is None:
        base_rate = _get_static_base_rate(date.year)
        
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

def check_data_anchor(data):
    """[anchor 防護] 偵測 data.index 起點漂移。
    GARCH refit 以 data.index 絕對位置錨定 (與回測端 (t-504)%5 對齊)。若 yfinance
    修訂歷史、新增更早交易日, 起點漂移會使歷史 refit 點整體偏移, 造成 Live 與回測
    訊號錯位、歷史 vol 整體跳變。返回 warning 字串或 None。純監控, 不改變計算邏輯。"""
    if data is None or data.empty:
        return None
    actual_start = data.index[0]
    drift = abs((actual_start - EXPECTED_DATA_START).days)
    if drift > ANCHOR_TOLERANCE_DAYS:
        return (f"資料起點 {actual_start.date()} 偏離預期基準 {EXPECTED_DATA_START.date()} "
                f"達 {drift} 天。GARCH refit 觸發點以 data.index 絕對位置錨定, 起點漂移會使 "
                f"Live 與回測的 refit 對齊失準、歷史波動率整體跳變。請檢查資料來源, "
                f"確認新起點正確後更新程式碼中的 EXPECTED_DATA_START。")
    return None

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
        
        # [索引漂移修正] 用 data.index 絕對位置判斷 refit, 與回測端 (t-504)%5 對齊。
        # 現狀 bug: window=tail(N) 每次刷新後同一日期的相對索引 t 縮小, 導致
        # (t-loop_start)%REFIT_STEP 漂移, 同一天的 vol 被回頭改寫。改用絕對位置鎖死。
        ref_pos = data.index.get_indexer(dates)
        
        for t in range(loop_start, len(window)):
            train = window.iloc[t - LIVE_GARCH_WINDOW + 1 : t + 1]
            if len(train) < 50: continue
            abs_pos = ref_pos[t]
            if abs_pos >= 0:
                refit_due = (abs_pos - LIVE_GARCH_WINDOW) % REFIT_STEP == 0
            else:
                refit_due = (t - loop_start) % REFIT_STEP == 0  # 保護: 日期不在 data.index
            if refit_due or model_res is None:
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
        else: df['SMA_State'] = 0.0  # [修正 B-4] 對齊 GARCH 保守原則：資料異常時預設危險
        
        cfg = RISK_CONFIG[trade_t]
        df['Exit_Th'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
        df['Entry_Th'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
        
        df['GARCH_State'] = np.nan
        valid = df['Exit_Th'].notna() & df['Vol'].notna()
        
        mask_exit = valid & (df['Vol'] > df['Exit_Th'])
        mask_entry = valid & (df['Vol'] < df['Entry_Th'])
        
        df.loc[mask_exit, 'GARCH_State'] = 0.0 
        df.loc[mask_entry, 'GARCH_State'] = 1.0 
        # [更新] 預設狀態統一為 0.0 (保守: 無資料預設危險)，與回測端一致
        df['GARCH_State'] = df['GARCH_State'].ffill().fillna(0.0)
        
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
            
            # 取得 252 日年化波動率
            d_loc = data.index.get_indexer([ref_date], method='pad')[0]
            if d_loc >= 252:
                subset = prices[ticker].iloc[d_loc-252 : d_loc]
                row['Vol_Ann'] = subset.pct_change().std() * np.sqrt(252)
            else: row['Vol_Ann'] = np.nan

            # [更新] 非重疊三段動能: (0,3], (3,7], (7,12]
            # 每段獨立計算當區間報酬率 (無重疊)
            loc = monthly.index.get_loc(ref_date)
            for (start_m, end_m) in MOM_SEGMENTS:
                seg_label = f'Ret_{start_m}to{end_m}M' if start_m > 0 else f'Ret_{end_m}M'
                if loc >= end_m:
                    p_end = monthly.iloc[loc - end_m][ticker]   # 較早的時點
                    if start_m == 0:
                        # (0, end_m] 區間: 現在 vs end_m 個月前
                        if pd.isna(p_end) or p_end == 0: row[seg_label] = np.nan
                        else: row[seg_label] = (p_now - p_end) / p_end
                    else:
                        # (start_m, end_m] 區間: start_m 個月前 vs end_m 個月前
                        p_start = monthly.iloc[loc - start_m][ticker]  # 較晚的時點
                        if pd.isna(p_start) or pd.isna(p_end) or p_end == 0:
                            row[seg_label] = np.nan
                        else:
                            row[seg_label] = (p_start - p_end) / p_end
                else:
                    row[seg_label] = np.nan
                
            metrics.append(row)
        except: continue
        
    if not metrics: return pd.DataFrame(), None
    
    df = pd.DataFrame(metrics).set_index('Ticker')
    
    # [更新] 非重疊三段 Raw Risk-Adjusted Return，分母統一 252d 年化波動率
    raw_scores = []
    for (start_m, end_m) in MOM_SEGMENTS:
        seg_label = f'Ret_{start_m}to{end_m}M' if start_m > 0 else f'Ret_{end_m}M'
        sr_label = f'Raw_SR_{start_m}to{end_m}M' if start_m > 0 else f'Raw_SR_{end_m}M'
        if seg_label in df.columns:
            risk_adj = df[seg_label] / (df['Vol_Ann'] + 1e-6)
            df[sr_label] = risk_adj
            raw_scores.append(risk_adj)
            
    if raw_scores:
        df['Total_Score'] = pd.concat(raw_scores, axis=1).mean(axis=1).fillna(0)
    else:
        df['Total_Score'] = 0.0
        
    return df.sort_values('Total_Score', ascending=False), ref_date

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
    
    # [更新] 避險池強制分離，死鎖 6+12M 原始超額報酬平均
    avg_ret = pd.Series(0.0, index=avail_safe)
    valid_m = 0
    for m in [6, 12]:
        if loc >= m:
            ret_m = (monthly.iloc[loc] / monthly.iloc[loc-m]) - 1
            avg_ret += ret_m
            valid_m += 1
            
    if valid_m > 0: avg_ret /= valid_m
    else: avg_ret = pd.Series(0.0, index=avail_safe)
    
    winner = avg_ret.idxmax()
    details = pd.DataFrame({"Ticker": avail_safe, "Comp_Ret": avg_ret.values}).set_index("Ticker")
    return winner, details, ref_date

# ==========================================
# 3. 回測邏輯 (嚴格導入 T+1 與 Open/Close 微結構)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="準備回測數據 (合成三倍槓桿與隔夜成本)...")
def get_synthetic_backtest_data():
    tickers = list(MAPPING.values()) + SAFE_POOL + ['IOO']
    try:
        # [更新] 取得 FRED DFF 利率序列 (24h 快取)
        fed_rate_series = get_fed_funds_rate()
        
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
            # [更新] 使用 FRED 動態利率 (失敗自動 fallback 至靜態表)
            costs = pd.Series([get_daily_leverage_cost(d, fed_rate_series) for d in data_core.index], index=data_core.index)
            
            synthetic_data[f"{ticker_3x}_Ret_ON"] = (3.0 * synthetic_data[f"{ticker_1x}_Ret_ON"]) - costs
            synthetic_data[f"{ticker_3x}_Ret_ID"] = 3.0 * synthetic_data[f"{ticker_1x}_Ret_ID"]
            
            ret_3x_total = ((1 + synthetic_data[f"{ticker_3x}_Ret_ON"].fillna(0)) * (1 + synthetic_data[f"{ticker_3x}_Ret_ID"].fillna(0))) - 1
            synthetic_data[ticker_3x] = (1 + ret_3x_total).cumprod() * 100
            synthetic_data[f"RAW_{ticker_3x}"] = data_core[f"{ticker_1x}_Close"] 
            
        return synthetic_data.dropna()
    except Exception as e:
        # [修正 A-1] 不再吞掉所有錯誤；顯示具體例外以利 debug
        st.error(f"合成回測資料失敗: {type(e).__name__}: {e}")
        return pd.DataFrame()

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
    
    # [更新] 回測端進攻資產：非重疊三段 (0,3], (3,7], (7,12] 月
    # 每段獨立計算當區間報酬，分母統一 252d 年化波動率
    raw_scores_list = []
    for (start_m, end_m) in MOM_SEGMENTS:
        if start_m == 0:
            # (0, end_m] 區間: 現在 vs end_m 個月前 (等同於 pct_change(end_m))
            seg_ret = monthly_src.pct_change(end_m)
        else:
            # (start_m, end_m] 區間: start_m 個月前 vs end_m 個月前
            p_start = monthly_src.shift(start_m)    # 較晚的時點
            p_end = monthly_src.shift(end_m)        # 較早的時點
            seg_ret = (p_start / p_end) - 1
        risk_adj = seg_ret / (monthly_vol + 1e-6)
        raw_scores_list.append(risk_adj.fillna(0))
        
    comp_raw = sum(raw_scores_list) / len(raw_scores_list)
    hist_winners = comp_raw.idxmax(axis=1)
    
    # [更新] 回測端避險資產：強制鎖死 6+12M 原始絕對報酬平均
    avail_safe = [t for t in SAFE_POOL if t in data.columns]
    safe_monthly = get_monthly_data(data[avail_safe])
    
    safe_scores_dict = {}
    for m in [6, 12]:
        safe_scores_dict[m] = safe_monthly.pct_change(m).fillna(0)
        
    comp_safe = sum([safe_scores_dict[m] for m in [6, 12]]) / 2.0
    hist_safe = comp_safe.dropna(how='all').idxmax(axis=1).reindex(safe_monthly.index).fillna('TLT')
    
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
    # [修正 A-2] 3x 等權重季度再平衡邏輯重寫
    # 原邏輯使用 loc[t_s:t_e] 閉區間，跨季邊界日會被相鄰兩段覆寫，造成微幅偏誤。
    # 新邏輯使用日報酬 + 漂移權重 + 季底再平衡，數學上嚴格無重複計算。
    b_returns = b_sub.pct_change().fillna(0)
    q_ends_set = set(b_sub.groupby(pd.Grouper(freq='QE')).apply(
        lambda x: x.index[-1] if len(x) > 0 else None).dropna().tolist())
    N = len(b_cols)
    if N > 0:
        weights = np.ones(N) / N  # 起始等權重
        b_eq = pd.Series(1.0, index=b_sub.index)
        curr_nav = 1.0
        for j, today in enumerate(b_sub.index):
            if j == 0:
                b_eq.iloc[j] = curr_nav
                continue
            today_rets = b_returns.iloc[j].values  # 各資產今日報酬
            port_ret = float(np.dot(weights, today_rets))  # 加權報酬
            curr_nav *= (1 + port_ret)
            b_eq.iloc[j] = curr_nav
            # 權重漂移
            weights = weights * (1 + today_rets)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            # 季底再平衡回 1/N
            if today in q_ends_set:
                weights = np.ones(N) / N
    else:
        b_eq = pd.Series(1.0, index=b_sub.index)
        
    vt_eq = pd.Series(1.0, index=valid_dates)
    if 'IOO' in data.columns:
        vt_ret = data['IOO'].loc[valid_dates].pct_change().fillna(0)
        vt_eq = (1 + vt_ret).cumprod()
        
    return cum_eq, b_eq, vt_eq, hold_counts

# ==========================================
# 4. Dashboard 介面
# ==========================================
st.title("🛡️ 雙重動能與動態風控 (機構實盤版)")
# [更新] 標題對齊非重疊三段動能引擎憲法
st.caption(f"配置: 進攻 [非重疊 3M/[3-7]M/[7-12]M 風險調整] / 避險 [6+12M 原始絕對報酬] / SMA {SMA_MONTHS}M / GARCH (Q{RISK_CONFIG['UPRO']['exit_q']*100:.0f})")

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
# [anchor 防護] 偵測資料起點漂移, 若 refit 對齊可能失準則示警
_anchor_warn = check_data_anchor(live_data)
if _anchor_warn:
    st.warning(f"⚠️ GARCH 對齊警示：{_anchor_warn}")

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
    
    ### 2. 動能選擇機制 (非對稱 Alpha 雙引擎)
    策略每月進行一次選股，挑選當下動能最強的進攻/避險資產。
    
    **⚔️ 進攻端引擎：非重疊三段風險調整動能 (結構性分離版)**
    * **邏輯**：將過去 12 個月切分為**三段不重疊區間**：**近期 3M**、**中期 (3, 7]M**、**遠期 (7, 12]M**。每段獨立計算該區間的累積報酬率，再除以**過去 252 天（一年）的年化波動率**（訊號與風險估計解耦），得到對稱的 Sharpe-like Ratio。最終取三段**等權重平均分數 (Total Score)**，最高分者勝出。
    * **為何非重疊**：原始重疊版 (3, 6, 9, 12M) 中 3M 報酬在四個分數裡被計算四次，近期資訊被隱性加權。非重疊設計讓每段資訊只貢獻一次，**近期、中期、遠期動能權重嚴格對稱**。實證上 Avg 5Y Rolling CAGR 提升約 +3.57%，且排除 2008 後差異仍為正（非源自單一事件）。
    * **為何採用斷點 [3, 7]**：經過 231 個非重疊配置的完整網格搜索驗證（11 二段 + 55 三段 + 165 四段）：
        1. **三段網格最佳**：在 55 個三段配置中，`[3, 7]` 的 CAGR (29.45%)、Sharpe (0.7479)、IR (0.6985) 三項綜合最佳；斷點「7」對應 Novy-Marx (2012) 實證中期動能 (t-12 到 t-7) 的預測力強區間，有獨立學術理論支撐，非純數據挖掘產物。
        2. **全網格第四名但為主動選擇**：若將四段配置納入比較，`[3, 7]` 的 Avg_5Y_Rolling 排名第 4（前三名為 `4seg_2_3_7`、`4seg_1_3_7`、`4seg_3_5_7`，均以 7 為最末斷點），但差距僅 +6 ~ +15 bps（≈ 0.2 ~ 0.3 個網格標準差），**落在統計噪音量級內**。且四段網格 (165 個) 的樣本空間是三段 (55 個) 的 3 倍，在更大空間中取最大值本來就會系統性偏高（selection bias），此 15 bps 優勢幾乎完全可由此效應解釋，不代表結構性真實優勢。
        3. **抗過擬合與最簡性**：新增第四段斷點 (如 0-2M vs 2-3M 的切分) 缺乏獨立學術支持，且 whipsaw 從 23 次增至 25 次（+8.7%）會部分抵銷表面收益。依「最簡穩健優先」原則拒絕複雜化。
        4. **穩定高原而非孤立尖峰**：231 個配置中有 107 個 (46%) 通過嚴格篩選 (WinRate + Worst5Y + IR 均 ≥ Baseline)；三段網格內 CAGR 標準差僅 0.86%，結果對具體斷點不敏感，確認 `[3, 7]` 位於穩定高原，非過擬合尖峰。
    * **已知取捨**：採用非重疊結構後，MaxDD 從 -54.37% 惡化至 -61.54%（全 231 配置統一，為結構性固有屬性，與斷點選擇無關）。此代價已被接受，換取滾動勝率、Worst 5Y、IR 等機構級穩定度指標的系統性改善。
    
    **🛡️ 避險端引擎：6+12M 原始複合絕對報酬**
    * **邏輯**：避開夏普動能的「凸性悖論」，避險端直接計算過去 **6 個月與 12 個月**的絕對超額報酬並平均。此均勻濾波器能過濾 3M 的政策雜訊，並鎖定大部位的總經與利率週期。
    
    ### 3. 雙層動態風控機制 (Beta Risk Control)
    透過兩層獨立風控決定進攻端的曝險比例。每層貢獻 50% 權重，持倉水位為 **0%、50% 或 100%**。
    
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
        5. **預設值**：無資料期預設為 **0 (危險)**，保守對齊回測端，避免 Live / Backtest 偏移。
    
    ### 4. 嚴格交易微結構 (Microstructure & Friction)
    * **真實 3x 融資成本**：精確扣除 ETF 管理費 (0.95%) + **2 倍** 總報酬交換合約 (TRS) 利息（**動態聯邦基金利率**由 FRED DFF 即時抓取 + 1.0% 利差；抓取失敗時自動 fallback 至靜態分段表）。
    * **T+1 開盤執行 (MOO)**：T 日收盤結算訊號，T+1 日開盤市價執行，嚴格承擔隔夜跳空風險與權重漂移耗損。
    * **交易滑價**：單邊 **0.1% (10 bps)** 手續費與衝擊成本。

    ### 5. 機構級穩健性驗證紀錄 (反過擬合證據鏈)
    
    > ⚠️ **v8 變更紀錄**：GARCH Exit Quantile 從 `0.99` 變更為 `0.975`。Entry Quantile 維持 `0.90`。REFIT_STEP 維持 5。動能/SMA/避險全部維持不變。此變更基於 test10 + deep_compare + test9_v3 完整證據鏈確認。
    
    本策略已通過完整反過擬合驗證鏈，採用因子 ETF 策略驗證的同套方法論：
    
    **元件層級驗證 (test8)**：對動能 (6 變體) / SMA (7 變體) / 避險 (15 變體) 三元件分別做 Anchored + Rolling 雙 WFA + Block Bootstrap (L=60, N=10,000) + Difference Bootstrap (配對抽樣) 鐵三角驗證。三元件 Baseline 全部通過 Bonferroni 校正，無變體顯著勝出。
    
    **聯合層級驗證 (test9 v1, GARCH 99/90)**：對 6 × 7 × 15 = 630 個動能/SMA/避險組合做聯合搜索，採用四層判決框架。Baseline 全期 CAGR 排名第 1 / 630 (29.58%)、OOS_CAGR 排名第 1 / 630 (27.54%)、Avg5Y 排名第 2 / 630 (27.95%)。**Level 1 粗篩通過 0 個對手**（僅 Baseline 自身）。
    
    **GARCH 參數層級驗證 (test10)**：對 10 refit × 9 exit/entry pairs = 90 個 GARCH 參數組合做聯合搜索，動能/SMA/避險固定為 v1 baseline。關鍵發現：
    * MaxDD = -64.37% **是策略結構性硬底**——90 配置中無一突破，validate 了「方向 E (MaxDD 歸因) 是唯一可能改善路徑」的結論
    * GARCH refit=5-8 是「不過度反應」甜蜜點
    * **Exit=99 略偏保守**：97.5/90 配對在所有指標上輕度勝過 Baseline
    * Level 1 通過 3 個候選，但 Bonferroni 校正 (α=5.56e-4) 全部不及格
    
    **GARCH Exit Quantile 深度對比 (deep_compare)**：對 refit5_q99_90 vs refit5_q97.5_90 做完整對比，採用：
    * 完整指標表 (20+ 指標含 Sortino, Calmar, CVaR, Skew/Kurt)
    * WFA 三子期間獨立對比
    * 七個市場制度時期對比 (含 OOS 範圍外的 2024 Q4+)
    * Block Bootstrap CI (各自獨立)
    * Difference Bootstrap 五指標配對 p-value (ΔCAGR, ΔSharpe, ΔSortino, ΔAvg5Y, ΔMaxDD)
    
    **deep_compare 結論支持採納 97.5/90**：
    * **9/9 指標方向一致勝出** (CAGR, Sharpe, Sortino, Calmar, Avg_1Y, Avg_5Y, Worst_1Y, Worst_5Y, Std_5Y)
    * **3/3 WFA 子期間穩健勝出** (ΔCAGR: +2.06%/+2.31%/+1.35%, 三視窗 Sharpe 均改善)
    * **MaxDD 完全相同 (-64.37%, 2008-09-17, 恢復 229 天)**——非用尾部風險換報酬
    * **順風期 (2020-2021 COVID 反彈、2023 AI 反彈、2024 Q1-Q3) 完全相同**——無 whipsaw 副作用
    * **逆風期 (2018 貿易戰、2022 雙殺、2024 Q4+) 顯著改善**——純 Pareto 改進
    * **2024 Q4 ~ 2026 Q2 (OOS 範圍外) 仍勝出**——排除 OOS 過擬合
    * 配對 Bootstrap **P(Cand 較佳) 達 94-96%**（雙尾 p=0.07-0.13 邊緣未過 α=0.05，但 5/5 指標方向一致）
    
    **誠實標註的限制**：
    * 雙尾 p-value 在 [0.07, 0.13]，**統計上未通過 Level 2 α=0.05 顯著性**
    * **Bonferroni 校正 (α=5.56e-4) 完全不及格**——這是 90 個搜索空間的代價
    * 切換決策基於「方向一致性 + 子期間穩健 + MaxDD 不退化 + 學術理論」**綜合判斷**，而非單一統計顯著性
    * 嚴格按「鐵則」應拒絕，但鐵則的設計假設（無方向先驗、典型過擬合特徵）在此情境不適用
    
    **聯合驗證 (test9 v3, GARCH 97.5/90 新 baseline)**：在新 baseline 下重跑 630 組合動能/SMA/避險聯合搜索，結果完全驗證 v8 切換決策：
    * Baseline 全期 CAGR / OOS_CAGR / Avg5Y / Worst5Y 四項排名與 v1 完全相同 (1/1/2/100)
    * Level 1 粗篩通過數 = 1 (僅 Baseline 自身)，與 v1 一致
    * MaxDD 結構不變 (-64.37%)，2seg_7 仍劣於 Baseline (raw p=0.0436)
    * 動能群並列現象 (4 個含「7」斷點配置 OOS 完全並列) 與避險排序 ({6,12} > {3,12} > ...) 完全保持
    * 結論：GARCH 參數微調不影響動能/SMA/避險的最佳組合選擇
    
    **方向 E：MaxDD 歸因分析 (Phase 1 + Phase 2)**
    
    為理解 MaxDD = -64.37% 為何是策略結構性硬底，執行完整歸因分析：
    
    **Phase 1 (事件層面拆解)**：用 diagnostic 回測引擎拆解 2008 MaxDD 事件：
    * Peak: 2008-05-19 (NAV=1.1117) → Trough: 2008-09-17 (NAV=0.3962, MaxDD=-64.37%)
    * 進攻贏家全期間都是 EDC (新興市場 3x)，同期 EDC 跌 -79.59%
    * **MaxDD 是 121 天慢性下跌，非閃電事件**：
      - Phase 1 (5/19 → 7/1, 43 天, 100% 進攻): -36.25% 損失 (佔總 NAV 損失 56%)
      - Phase 2 (7/1 → 9/17, 78 天, 50/50 部分避險): -44.11% 內部跌幅 (佔總 NAV 損失 44%)
    * **GARCH 在整個 MaxDD 期間從未觸發 Exit**——直到 9/23 (Trough 之後 6 天) 才觸發
    * **SMA 在 7/1 觸發 Exit (Peak+43 天)，比 GARCH 早 84 天**——SMA 反而是更敏感的訊號
    * GARCH 失效機制：2008 持續高 vol 被自己的歷史推高 rolling 252 日 97.5% quantile threshold，造成「自動 desensitization」
    
    **Phase 2 (三方向聯合搜索 100 組合)**：
    1. **方向 A 不對稱 risk_weight (5 變體)**：當「單一風控觸發 Exit」時偏向更保守
    2. **方向 B SMA 期數 (5 變體)**：3M/4M/5M/6M/8M
    3. **方向 C GARCH rolling quantile window (4 變體)**：126/189/252/378 日
    
    **Phase 2 關鍵發現**：
    * **方向 B 對 2008 MaxDD 完全沒影響** (所有 SMA 期數 2008 MaxDD 都是 -64.37%) — 月頻 SMA 仍需等月底才生效，B=6 是全局最佳
    * **方向 C 主導改善**：C=378 (1.5 年 quantile window) 將全期 MaxDD 從 -64.37% 改善至 -51.64% (2008 MaxDD 改善 22.95pp)
    * **方向 A 次要改善**：A1 (SMA 主導) 將 MaxDD 從 -64.37% → -54.62%
    * **最佳候選 A3_any_mid_B6_C378**：全期 MaxDD -50.98% (改善 13.39pp)、2008 MaxDD -28.15% (改善 36.22pp)、2022 MaxDD -37.14% (改善 11.67pp)、OOS_CAGR +0.39pp
    * 代價：CAGR -0.49pp、Worst5Y 從 3.70% 降至 2.07%
    
    **Phase 2 判決：未通過反過擬合鐵則，維持 v8 Baseline**
    * Level 1 嚴格通過 0 個 (所有候選 CAGR 都會略降)
    * Level 1 鬆通過 2 個 (A2/A3 + B6 + C378)
    * Level 2 (p_MaxDD<0.05) 通過 0 個
    * Level 3 (Bonferroni α=5e-4) 通過 0 個
    * ΔMaxDD Bootstrap p 值 0.44-0.79 (極不顯著)——MaxDD 為單一極值統計量，Bootstrap CI 寬導致統計檢定固有 limitation
    
    **MaxDD 結構性硬底結論的修正**：
    * **原結論**：「MaxDD = -64.37% 是 1820+ 配置確認的結構性硬底」
    * **修正後**：「MaxDD = -64.37% 在『對稱 risk_weight + 252 日 GARCH quantile window』設定下是硬底。改變這兩個維度之一可以打破，但需付出 CAGR ~0.5pp 代價且統計顯著性未通過嚴格反過擬合驗證。」
    * **維持 v8 Baseline 的理由**：(1) 統計上 Level 2/3 都未通過、(2) 2024 Q4+ 在邊緣候選下微退化、(3) Bootstrap p 值高、(4) Occam 原則：增加複雜度需更強證據
    
    **邊緣候選監控標註**：
    * 配置 `A3_any_mid_B6_C378` (任一風控警示時 0.25/0.25 weight + 6M SMA + 378 日 GARCH quantile window) 列為「未來實盤監控候選」
    * 若未來實盤期間此配置持續勝出，未來版本可重新評估切換
    * 此配置代表「Risk-Reward Pareto 改進方向」但需更多真實數據確認
    
    **方向 F：GARCH Live/回測一致性 (物理致盲對決 + 索引漂移修正)**
    
    起因: 實盤發現 6/5 大跌後, 同一天的 GARCH vol 在 6/8 顯示 23.03 (低), 6/9 卻變成 51.07 (超閾值)。
    
    **診斷出兩個獨立問題**:
    * **索引漂移 (歷史重繪)**: Live 端 `window = ret.dropna().tail(N)` 每次刷新往後移一天, 同一日期在 window 中的相對索引 `t` 縮小, 使 `(t - loop_start) % REFIT_STEP` 漂移, 同一天的 refit 狀態在不同刷新日跳動 → vol 被回頭改寫。**確定性 bug**。
    * **物理致盲 (Stale Forecast)**: arch 套件在非 refit 日呼叫舊 `model_res.forecast()`, conditional variance 不反映新資料 → 大跌當天 vol 不跳升, 須等下個 refit 日。**回測與 Live 端皆有此特性**。
    
    **物理致盲三方對比實證 (8 變體完整回測)**: 為判斷物理致盲是 bug 還是有益特徵, 測試 Baseline (物理致盲) vs Pure .fix() (每天更新 vol) vs Confirmation Lag (連續 N 天確認, N=2/3/5) vs Output Smoothing (MA11/MA22/EWMA λ=0.94, 依 Moreira-Muir 2017 與 RiskMetrics)。結果:
    * **Pure .fix() 重蹈 refit=1 覆轍**: MaxDD 從 -64.37% 惡化至 -69.96% (與 test9_v2 的 refit=1 結果數字完全一致), CAGR -1.84pp, Whipsaw 117→164。證實「消除物理致盲 = 過度敏感」。
    * **所有變體的 2008 MaxDD 全部維持 -64.37%**: 印證 Phase 1 結論——2008 病因是「GARCH quantile 閾值被推高 (desensitization)」而非雜訊, 故 confirmation/smoothing 等雜訊濾波工具完全無效。
    * **Output Smoothing 全面劣化**: EWMA λ=0.94 的 ΔCAGR=-5.22% (p=0.041, 唯一統計顯著, 且為顯著變差)。本地 3x 槓桿資料重現 Liu/Tang/Zhou (2021, JFE) 的反面結論——平滑使 turnover 暴增、扣成本後無 alpha。
    * **反過擬合框架判決**: Level 1/2/3 全部 0 個通過; 所有 7 變體 ΔMaxDD ≤ 0 (無一改善)。
    * **結論: 物理致盲確認為「有益特徵」而非 bug**。其「每 5 天降採樣」是零額外參數的免費濾波器, 優於 confirmation/smoothing。Gemini 主張的 `.fix()`「唯一正解」被實證推翻 (使 MaxDD 退化 5.59pp)。
    
    **採取的修正 (僅修確定性 bug, 物理致盲維持原狀)**:
    * **索引漂移修正**: `calculate_live_risk` 改用 `data.index` 絕對位置判斷 refit (`(abs_pos - 504) % REFIT_STEP == 0`), 與回測端 `(t - 504) % REFIT_STEP` 用同一把尺 (anchor 同為 504)。修正後同一天 refit 狀態鎖死、不再漂移, 且 Live 與回測完全對齊。回測端 `t` 本為絕對索引、無漂移, 不需修改。
    * **anchor 防護**: 新增 `check_data_anchor()` 偵測 `data.index` 起點漂移 (預期 EXPECTED_DATA_START=2005-03-11, 容忍 7 天)。若 yfinance 修訂歷史使起點偏移, 絕對位置會整體錯位, 此防護在儀表板示警。純監控, 不改變計算邏輯。
    * **物理致盲不動**: 經三方對比實證確認為有益特徵, 維持 refit=5 的 Stale Forecast 行為, 確保 Live 與回測在數學上一致。
    
    **遺留限制**：
    * 2024-04 後仍不在 WFA OOS 範圍內，需透過實盤監控
    * 方向 E Phase 2 後不再追蹤 MaxDD 進一步改善的可能性，除非實盤證據顯示顯著問題
    """)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🏆 本月進攻贏家", winner)
with c2: 
    color = "green" if final_w==1 else "orange" if final_w==0.5 else "red"
    st.markdown(f"### 🎯 權重: :{color}[{final_w*100:.0f}%]")
with c3: 
    st.metric("GARCH 風控", "安全" if g_state==1 else "危險", delta="✅" if g_state==1 else "🔻")
with c4: 
    s_val = safe_df.loc[safe_win, 'Comp_Ret'] if not safe_df.empty else 0
    st.metric("🛡️ 避險資產", safe_win, f"複合超額: {s_val:.1%}")

st.divider()

t1, t2, t3, t4, t5, t6 = st.tabs(["數據", "風控細節", "權重狀態", "選股排名", "避險輪動", "部位總結"])
with t1: st.dataframe(live_data.tail(5).style.format("{:.2f}"), use_container_width=True)
with t2:
    if winner in risk_live:
        st.dataframe(risk_live[winner].tail(10)[['Price','Vol','Exit_Th','Entry_Th','GARCH_State']].style.format("{:.2f}"), use_container_width=True)
with t3:
    if winner in risk_live:
        st.dataframe(risk_live[winner].tail(10)[['GARCH_State','SMA_State','Weight']], use_container_width=True)
with t4: st.dataframe(sel_df.style.format("{:.4f}"), use_container_width=True)
with t5: st.dataframe(safe_df.style.format("{:.2%}"), use_container_width=True)
with t6: st.success(f"建議持有: **{final_w*100:.0f}% {winner}** + **{(1-final_w)*100:.0f}% {safe_win}**")

st.divider()

# ==========================================
# 5. 回測區塊
# ==========================================
st.header("⏳ 嚴格微結構滾回測 (T+1 MOO & TRS Costs)")
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
                # [修正 B-3] CAGR 使用實際起始值，不假設 eq.iloc[0]=1.0
                # 原 (eq.iloc[-1]) ** (365.25/d) 假設起始為 1.0，但 cumprod() 起始 ≈ 1+第一日報酬
                start_val = max(eq.iloc[0], 1e-10)  # 防止除零
                cagr = (eq.iloc[-1] / start_val) ** (365.25/max(d, 1)) - 1
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
