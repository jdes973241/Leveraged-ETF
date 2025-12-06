import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    .buy {color: #28a745; font-weight: bold;}
    .sell {color: #dc3545; font-weight: bold;}
    .neutral {color: #6c757d; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# 策略參數
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"} # 3x -> 1x
RISK_CONFIG = {
    "UPRO": {"exit_q": 0.85, "entry_q": 0.70},
    "EURL": {"exit_q": 0.97, "entry_q": 0.82},
    "EDC":  {"exit_q": 0.70, "entry_q": 0.55}
}
ROLLING_WINDOW_SIZE = 1260
SMA_WINDOW = 200
MOM_PERIODS = [3, 6, 9, 12]

# ==========================================
# 1. 核心邏輯函數 (快取優化)
# ==========================================

@st.cache_data(ttl=3600, show_spinner="正在下載市場數據...")
def get_market_data():
    """下載所有相關標的數據"""
    tickers = list(MAPPING.keys()) + list(MAPPING.values())
    # 下載較長歷史以確保指標計算準確
    data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.levels[0]: data = data['Close']
        else: data = data['Close'] if 'Close' in data else data
    
    return data.ffill().dropna()

@st.cache_data(ttl=3600, show_spinner="正在計算 GARCH 風控模型...")
def calculate_risk_metrics(data):
    """計算風控層的所有數據：GARCH Vol, Thresholds, SMA"""
    risk_details = {}
    
    for trade_t, signal_t in MAPPING.items():
        # 取得訊號源 (1x) 數據
        series = data[signal_t]
        ret = series.pct_change() * 100
        sma = series.rolling(SMA_WINDOW).mean()
        
        # GARCH 計算 (為了 Dashboard 速度，我們只計算最近 2 年的滾動數據)
        # 注意：這裡為了即時顯示，我們用全區間擬合參數來預測最後一天的波動率 (快速近似)
        # 若要嚴謹的回測請用 backtest 專用代碼
        window = ret.dropna().tail(1260) 
        
        # 1. 計算 GARCH 波動率序列 (Conditional Volatility)
        # 這裡我們需要"序列"來計算歷史分位數，所以不能只算最後一天
        am = arch_model(window, vol='Garch', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        cond_vol = res.conditional_volatility * np.sqrt(252) # 年化
        
        # 2. 整合 DataFrame
        df = pd.DataFrame({
            'Price': series,
            'Ret': ret,
            'SMA': sma,
        }).join(cond_vol.rename('Vol'), how='inner')
        
        # 3. 計算動態閾值 (Rolling Quantile)
        cfg = RISK_CONFIG[trade_t]
        # Shift(1) 模擬實際交易：今天的閾值是由昨天以前的數據決定的
        df['Exit_Th'] = df['Vol'].rolling(252).quantile(cfg['exit_q']).shift(1)
        df['Entry_Th'] = df['Vol'].rolling(252).quantile(cfg['entry_q']).shift(1)
        
        # 4. 生成訊號狀態
        # GARCH 狀態機
        df['GARCH_State'] = np.nan
        valid = df['Exit_Th'].notna()
        df.loc[valid & (df['Vol'] > df['Exit_Th']), 'GARCH_State'] = 0.0 # 避險
        df.loc[valid & (df['Vol'] < df['Entry_Th']), 'GARCH_State'] = 1.0 # 持有
        df['GARCH_State'] = df['GARCH_State'].ffill().fillna(1.0) # 預設持有
        
        # SMA 狀態
        df['SMA_State'] = (df['Price'] > df['SMA']).astype(float)
        
        # 混合權重
        df['Weight'] = (0.5 * df['GARCH_State']) + (0.5 * df['SMA_State'])
        
        risk_details[trade_t] = df
        
    return risk_details

@st.cache_data(ttl=3600)
def calculate_selection_metrics(data):
    """計算動能選股層數據"""
    # 使用 3x 交易標的計算動能
    prices = data[list(MAPPING.keys())]
    
    # 取最新一天與各週期的回報
    latest_date = prices.index[-1]
    
    metrics = []
    for ticker in prices.columns:
        row = {'Ticker': ticker}
        p_now = prices[ticker].iloc[-1]
        
        # 計算各週期 Return
        for m in MOM_PERIODS:
            # 概抓交易日 (1個月約21天)
            lookback = m * 21
            if len(prices) > lookback:
                p_prev = prices[ticker].iloc[-1-lookback]
                ret = (p_now - p_prev) / p_prev
                row[f'Ret_{m}M'] = ret
            else:
                row[f'Ret_{m}M'] = np.nan
                
        # 計算波動率 (風險調整用) - 使用過去 6 個月日波動
        vol_window = 126
        daily_ret = prices[ticker].pct_change().tail(vol_window)
        vol = daily_ret.std() * np.sqrt(252)
        row['Vol_Ann'] = vol
        
        metrics.append(row)
        
    df = pd.DataFrame(metrics).set_index('Ticker')
    
    # 計算 Z-Score
    z_score_sum = pd.Series(0.0, index=df.index)
    
    for m in MOM_PERIODS:
        col = f'Ret_{m}M'
        # Risk Adjusted Return = Ret / Vol
        risk_adj = df[col] / (df['Vol_Ann'] + 1e-6)
        
        # Z-Score
        z = (risk_adj - risk_adj.mean()) / (risk_adj.std() + 1e-6)
        df[f'Z_{m}M'] = z
        z_score_sum += z
        
    df['Total_Z'] = z_score_sum
    df['Rank'] = df['Total_Z'].rank(ascending=False)
    
    return df.sort_values('Total_Z', ascending=False)

# ==========================================
# 2. 應用程式主邏輯
# ==========================================

# 載入數據
data = get_market_data()
risk_data = calculate_risk_metrics(data)
selection_df = calculate_selection_metrics(data)

# 取得最新日期狀態
latest_date = data.index[-1]
winner_ticker = selection_df.index[0] # 排名第一的標的

# 取得 Winner 的風控狀態
winner_risk_df = risk_data[winner_ticker]
latest_risk_row = winner_risk_df.iloc[-1]
final_weight = latest_risk_row['Weight']

# ==========================================
# 3. 前端顯示 (Streamlit Layout)
# ==========================================

st.title("🛡️ 雙重動能與動態風控策略儀表板")
st.markdown(f"**數據基準日**: {latest_date.strftime('%Y-%m-%d')}")

# --- 頂部摘要 (Top Summary) ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🏆 本月動能贏家", winner_ticker, f"Rank #1")
with c2:
    w_color = "normal"
    if final_weight == 1.0: w_color = "off" # green implied
    elif final_weight == 0.0: w_color = "inverse" # red implied
    st.metric("🎯 建議持倉權重", f"{final_weight*100:.0f}%", delta=None)
with c3:
    garch_st = "安全 (持有)" if latest_risk_row['GARCH_State'] == 1.0 else "危險 (避險)"
    st.metric("波動率風控 (GARCH)", garch_st)
with c4:
    sma_st = "多頭 (持有)" if latest_risk_row['SMA_State'] == 1.0 else "空頭 (避險)"
    st.metric("趨勢風控 (SMA)", sma_st)

st.divider()

# --- 詳細數據表格 (Tabs) ---
st.subheader("📊 策略透視 (Strategy Whitebox)")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 數據獲取層 (Data)", 
    "2️⃣ 風控計算層 (Risk Engine)", 
    "3️⃣ 權重計算層 (Weighting)",
    "4️⃣ 動能選股層 (Selection)",
    "5️⃣ 執行決策層 (Execution)"
])

# 1. 數據獲取層
with tab1:
    st.markdown("#### 原始市場數據 (最新 5 日)")
    st.caption("展示 1x 訊號源 (SPY, VGK, EEM) 與 3x 交易標的 (UPRO, EURL, EDC) 的收盤價與漲跌幅。")
    
    display_cols = list(MAPPING.keys()) + list(MAPPING.values())
    recent_data = data[display_cols].tail(5).sort_index(ascending=False)
    
    # 格式化顯示
    st.dataframe(recent_data.style.format("{:.2f}"), use_container_width=True)

# 2. 風控計算層
with tab2:
    st.markdown("#### 風控指標詳情 (GARCH & Thresholds)")
    st.caption("針對每個標的，展示當前的預測波動率 (Vol) 與動態歷史分位數閾值 (Exit/Entry)。")
    
    risk_summary = []
    for ticker, signal_t in MAPPING.items():
        df = risk_data[ticker]
        row = df.iloc[-1]
        
        # 準備表格數據
        cfg = RISK_CONFIG[ticker]
        vol_status = "🔴 避險" if row['GARCH_State'] == 0 else "🟢 安全"
        
        risk_summary.append({
            "交易標的 (3x)": ticker,
            "訊號源 (1x)": signal_t,
            "年化波動率 (Vol)": row['Vol'],
            "賣出閾值 (Exit)": row['Exit_Th'],
            "買進閾值 (Entry)": row['Entry_Th'],
            "設定參數": f"Q{int(cfg['exit_q']*100)} / Q{int(cfg['entry_q']*100)}",
            "GARCH 狀態": vol_status,
            "SMA (200)": row['SMA'],
            "目前價格": row['Price']
        })
        
    risk_df_show = pd.DataFrame(risk_summary)
    
    # Apply Styling
    def highlight_status(val):
        color = 'red' if '避險' in str(val) else 'green'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        risk_df_show.style
        .format({
            "年化波動率 (Vol)": "{:.2f}%", 
            "賣出閾值 (Exit)": "{:.2f}%", 
            "買進閾值 (Entry)": "{:.2f}%",
            "SMA (200)": "{:.2f}",
            "目前價格": "{:.2f}"
        })
        .map(highlight_status, subset=['GARCH 狀態']),
        use_container_width=True
    )

# 3. 混合權重層
with tab3:
    st.markdown("#### 權重混合邏輯")
    st.caption("公式：最終權重 = 0.5 * GARCH訊號 + 0.5 * SMA訊號")
    
    weight_summary = []
    for ticker in MAPPING.keys():
        df = risk_data[ticker]
        row = df.iloc[-1]
        
        weight_summary.append({
            "標的": ticker,
            "GARCH 訊號 (0/1)": int(row['GARCH_State']),
            "SMA 訊號 (0/1)": int(row['SMA_State']),
            "計算過程": f"0.5*{int(row['GARCH_State'])} + 0.5*{int(row['SMA_State'])}",
            "個別總權重": f"{row['Weight']:.1%}"
        })
        
    w_df = pd.DataFrame(weight_summary)
    st.table(w_df)

# 4. 動能選股層
with tab4:
    st.markdown("#### 動能選股排名 (基於 3x 標的)")
    st.caption("計算 3/6/9/12 個月的風險調整後回報，並進行 Z-Score 排序。")
    
    # 格式化選股表
    sel_style = selection_df.style.format({
        'Ret_3M': '{:.2%}', 'Ret_6M': '{:.2%}', 'Ret_9M': '{:.2%}', 'Ret_12M': '{:.2%}',
        'Vol_Ann': '{:.2%}',
        'Z_3M': '{:.2f}', 'Z_6M': '{:.2f}', 'Z_9M': '{:.2f}', 'Z_12M': '{:.2f}',
        'Total_Z': '{:.2f}',
        'Rank': '{:.0f}'
    }).background_gradient(subset=['Total_Z'], cmap='Greens')
    
    st.dataframe(sel_style, use_container_width=True)

# 5. 執行決策層
with tab5:
    st.markdown("#### 🚀 最終執行指令 (Action)")
    
    action_color = "green" if final_weight > 0 else "red"
    action_text = "BUY / HOLD" if final_weight > 0 else "SELL / CASH"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; border: 2px solid {action_color}; padding: 20px; border-radius: 10px;">
            <h2 style="color: {action_color}">{action_text}</h2>
            <h1>{winner_ticker}</h1>
            <h3>部位: {final_weight*100:.0f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.info(f"""
        **今日交易指令解析：**
        1. **選股**：本月動能最強的是 **{winner_ticker}** (Rank #1)。
        2. **風控**：檢查對應訊號源 **{MAPPING[winner_ticker]}** 的狀態。
           - GARCH 波動率模型顯示為 **{'安全' if latest_risk_row['GARCH_State']==1 else '危險'}**。
           - SMA 趨勢模型顯示為 **{'多頭' if latest_risk_row['SMA_State']==1 else '空頭'}**。
        3. **結論**：綜合得分為 **{final_weight}**。
           - 若您目前持有 {winner_ticker}，請調整倉位至 {final_weight*100:.0f}%。
           - 剩餘 {100 - final_weight*100:.0f}% 資金應持有現金或短期國債 (BIL/SHV)。
        """)

st.divider()
with st.expander("查看原始數據與參數說明"):
    st.json(RISK_CONFIG)
    st.write("GARCH Model: GARCH(1,1) with Student's T distribution.")
    st.write(f"Rolling Window: {ROLLING_WINDOW_SIZE} days.")
