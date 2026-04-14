import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Airline Delay Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================================
# LIGHT THEME CSS
# ========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp { background-color: #F5F7FA; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* TOP HEADER */
.top-header {
    background: #FFFFFF;
    border-bottom: 1.5px solid #E5E7EB;
    padding: 18px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand-row { display: flex; align-items: center; gap: 14px; }
.brand-icon {
    width: 42px; height: 42px;
    background: #1D4ED8;
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.brand-name { font-size: 17px; font-weight: 700; color: #111827; letter-spacing: -0.3px; }
.brand-sub  { font-size: 12px; color: #9CA3AF; font-weight: 400; margin-top: 2px; }
.header-pill {
    background: #EFF6FF; color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 20px; padding: 5px 14px;
    font-size: 12px; font-weight: 600;
}

/* PAGE BODY */
.page-body { padding: 32px 48px 48px; }

/* KPI STRIP */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 20px 22px;
    display: flex; align-items: center; gap: 16px;
}
.kpi-icon {
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.kpi-icon.blue  { background: #EFF6FF; }
.kpi-icon.red   { background: #FFF1F2; }
.kpi-icon.green { background: #F0FDF4; }
.kpi-icon.amber { background: #FFFBEB; }
.kpi-val   { font-size: 24px; font-weight: 700; color: #111827; line-height: 1; letter-spacing: -0.5px; }
.kpi-label { font-size: 12px; color: #6B7280; margin-top: 5px; font-weight: 400; }

/* SECTION HEADINGS */
.sec-title { font-size: 18px; font-weight: 700; color: #111827; letter-spacing: -0.3px; margin-bottom: 3px; }
.sec-sub   { font-size: 13px; color: #9CA3AF; margin-bottom: 20px; }

/* CHART CARD */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 22px 22px 18px;
    height: 100%;
}
.chart-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px; padding-bottom: 12px;
    border-bottom: 1px solid #F3F4F6;
}
.chart-title-inner { font-size: 13px; font-weight: 600; color: #374151; }
.chart-badge {
    font-size: 11px; font-weight: 500;
    padding: 3px 10px; border-radius: 20px;
    background: #F3F4F6; color: #6B7280;
}

/* DIVIDER */
.div-line { height: 1px; background: #E5E7EB; margin: 28px 0; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #FFFFFF;
    border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 4px; margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 38px; padding: 0 20px;
    background: transparent; border-radius: 9px;
    color: #6B7280; font-size: 13px; font-weight: 500;
    border: none !important;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: #1D4ED8 !important; color: white !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* EXPANDER */
details summary {
    font-size: 12px !important; font-family: 'DM Mono', monospace !important;
    color: #6B7280 !important;
}

/* FOOTER */
.dash-footer {
    text-align: center; padding: 20px;
    font-size: 12px; color: #9CA3AF;
    border-top: 1px solid #E5E7EB;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ========================================
# CHART HELPERS — uniform 4.2 × 6 inches
# ========================================
FH, FW = 4.2, 6.0

CLRS = {
    "blue":   "#1D4ED8",
    "sky":    "#0EA5E9",
    "green":  "#10B981",
    "amber":  "#F59E0B",
    "red":    "#EF4444",
    "purple": "#8B5CF6",
    "teal":   "#14B8A6",
    "pink":   "#EC4899",
    "indigo": "#6366F1",
}

def _style(ax, fig):
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F9FAFB")
    ax.tick_params(colors="#6B7280", labelsize=8.5)
    ax.xaxis.label.set_color("#374151"); ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_color("#374151"); ax.yaxis.label.set_fontsize(10)
    for s in ax.spines.values():
        s.set_color("#E5E7EB"); s.set_linewidth(0.8)
    ax.grid(axis='y', color="#E5E7EB", linewidth=0.6, linestyle='--', alpha=0.9)
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)

def make_fig():
    fig, ax = plt.subplots(figsize=(FW, FH))
    _style(ax, fig); fig.tight_layout(pad=1.6)
    return fig, ax

def bar_chart(df, x, y, color, xlabel="", ylabel=""):
    fig, ax = make_fig()
    bars = ax.bar(df[x].astype(str), df[y], color=color,
                  width=0.55, edgecolor="white", linewidth=1.2, zorder=3)
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if h is not None and not (isinstance(h, float) and np.isnan(h)):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f'{int(h):,}', ha='center', va='bottom', fontsize=7.5, color="#374151")
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    return fig

def line_chart(df, x, y, color, xlabel="", ylabel=""):
    fig, ax = make_fig()
    xr = range(len(df))
    ax.plot(df[x].astype(str), df[y], color=color, linewidth=2.5,
            marker='o', markersize=7, markerfacecolor='white',
            markeredgecolor=color, markeredgewidth=2, zorder=3)
    ax.fill_between(xr, df[y], alpha=0.07, color=color)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    return fig

def pie_chart(values, labels, colors):
    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.patch.set_facecolor("#FFFFFF")
    wp, texts, at = ax.pie(
        values, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.72,
        wedgeprops=dict(edgecolor='white', linewidth=2.5)
    )
    for t in texts:  t.set_fontsize(9.5); t.set_color("#374151")
    for a in at:     a.set_fontsize(8.5); a.set_color("#FFFFFF"); a.set_fontweight('600')
    fig.tight_layout(pad=1.0)
    return fig

def chart_card(title, badge):
    st.markdown(f"""
    <div class="chart-card">
        <div class="chart-header">
            <span class="chart-title-inner">{title}</span>
            <span class="chart-badge">{badge}</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ========================================
# LOAD DATA
# ========================================
@st.cache_data
def load_all():
    files = {
        1:"output.csv", 2:"q2_delay_percentage.csv", 3:"q3_routes.csv",
        4:"q4_daywise.csv", 5:"q5_avg_length.csv", 6:"q6_delay_vs_ontime.csv",
        7:"q7_peak_time.csv", 8:"q8_longest_flights.csv", 9:"q9_busiest_departure.csv",
        10:"q10_busiest_destination.csv", 11:"q11_ontime.csv", 12:"q12_peak_traffic.csv",
        13:"q13_least_delays.csv", 14:"q14_routes_max.csv", 15:"q15_delay_distribution.csv",
    }
    out = {}
    for k, f in files.items():
        try:
            d = pd.read_csv(f); d.columns = d.columns.str.strip(); out[k] = d
        except:
            out[k] = pd.DataFrame()
    return out

dfs = load_all()
df1=dfs[1]; df2=dfs[2]; df3=dfs[3]; df4=dfs[4]; df5=dfs[5]
df6=dfs[6]; df7=dfs[7]; df8=dfs[8]; df9=dfs[9]; df10=dfs[10]
df11=dfs[11]; df12=dfs[12]; df13=dfs[13]; df14=dfs[14]; df15=dfs[15]


# ========================================
# HEADER
# ========================================
st.markdown("""
<div class="top-header">
  <div class="brand-row">
    <div class="brand-icon">✈️</div>
    <div>
      <div class="brand-name">Flight Delay Analytics</div>
      <div class="brand-sub">Hive · Partitioned Tables · Streamlit</div>
    </div>
  </div>
  <div class="header-pill">2024 Dataset</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-body">', unsafe_allow_html=True)


# ========================================
# KPI STRIP
# ========================================
st.markdown("""
<div class="kpi-strip">
  <div class="kpi-card">
    <div class="kpi-icon blue">✈️</div>
    <div><div class="kpi-val">539K</div><div class="kpi-label">Total Flights</div></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon red">⚠️</div>
    <div><div class="kpi-val">45.2%</div><div class="kpi-label">Delay Rate</div></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon green">✅</div>
    <div><div class="kpi-val">54.8%</div><div class="kpi-label">On-Time Rate</div></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon amber">🛫</div>
    <div><div class="kpi-val">18</div><div class="kpi-label">Airlines Tracked</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ========================================
# TABS
# ========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Delay Overview",
    "🛫  Routes & Airports",
    "⏱️  Time Analysis",
    "✈️  Flight Operations",
])


# ─────────────────────────────────
# TAB 1 · Delay Overview
# ─────────────────────────────────
with tab1:
    st.markdown('<div class="sec-title">Delay Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Which airlines delay most — and by how much?</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">📊 Airlines with Most Delays</span><span class="chart-badge">Bar</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Airline, COUNT(*) AS total_delays\nFROM airline_partitioned\nWHERE Delay = 1\nGROUP BY Airline;", language="sql")
        if not df1.empty:
            st.pyplot(bar_chart(df1,'Airline','total_delays',CLRS["blue"],ylabel="Number of Delays"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">📈 Delay % per Airline</span><span class="chart-badge">Pie</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Airline,\n  COUNT(CASE WHEN Delay=1 THEN 1 END)*100.0/COUNT(*) AS delay_percentage\nFROM airline_partitioned\nGROUP BY Airline;", language="sql")
        if not df2.empty:
            cols = plt.cm.Blues(np.linspace(0.35, 0.85, len(df2)))
            st.pyplot(pie_chart(df2['delay_percentage'], df2['Airline'], cols), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">⏱️ Delay vs On-Time Split</span><span class="chart-badge">Donut</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Delay, COUNT(*) AS count\nFROM airline_partitioned\nGROUP BY Delay;", language="sql")
        if not df6.empty:
            st.pyplot(pie_chart(df6['count'],["On-Time","Delayed"],[CLRS["green"],CLRS["red"]]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">⭐ Airlines with Least Delays</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Airline, COUNT(*) AS delays\nFROM airline_partitioned\nWHERE Delay = 1\nGROUP BY Airline\nORDER BY delays ASC LIMIT 5;", language="sql")
        if not df13.empty:
            st.pyplot(bar_chart(df13,'Airline','delays',CLRS["green"],ylabel="Delay Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">✅ On-Time Flights per Airline</span><span class="chart-badge">Bar</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Airline, COUNT(*) AS on_time_flights\nFROM airline_partitioned\nWHERE Delay = 0\nGROUP BY Airline;", language="sql")
        if not df11.empty:
            st.pyplot(bar_chart(df11,'Airline','on_time_flights',CLRS["teal"],ylabel="On-Time Flights"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r3c2:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">📊 Delay Distribution by Flight Length</span><span class="chart-badge">Pie</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT\n  CASE WHEN Length < 100 THEN 'Short'\n       WHEN Length BETWEEN 100 AND 200 THEN 'Medium'\n       ELSE 'Long' END AS Flight_Type,\n  COUNT(*) AS total_delays\nFROM airline_partitioned WHERE Delay = 1\nGROUP BY Flight_Type;", language="sql")
        if not df15.empty:
            cmap = {'Short':CLRS["amber"],'Medium':CLRS["blue"],'Long':CLRS["indigo"]}
            cl = [cmap.get(ft,CLRS["blue"]) for ft in df15['Flight_Type']]
            st.pyplot(pie_chart(df15['total_delays'],df15['Flight_Type'],cl), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────
# TAB 2 · Routes & Airports
# ─────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">Routes & Airports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Most delayed routes and busiest hubs</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">🛫 Most Delayed Routes</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT AirportFrom, AirportTo, COUNT(*) AS delays\nFROM airline_partitioned\nWHERE Delay = 1\nGROUP BY AirportFrom, AirportTo\nORDER BY delays DESC LIMIT 5;", language="sql")
        if not df3.empty:
            df3['Route'] = df3['AirportFrom'] + "→" + df3['AirportTo']
            st.pyplot(bar_chart(df3,'Route','delays',CLRS["red"],ylabel="Delay Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">🚀 Routes with Most Flights</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT AirportFrom, AirportTo, COUNT(*) AS total_flights\nFROM airline_partitioned\nGROUP BY AirportFrom, AirportTo\nORDER BY total_flights DESC LIMIT 5;", language="sql")
        if not df14.empty:
            df14['Route'] = df14['AirportFrom'] + "→" + df14['AirportTo']
            st.pyplot(bar_chart(df14,'Route','total_flights',CLRS["purple"],ylabel="Flight Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">🛫 Busiest Departure Airports</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT AirportFrom, COUNT(*) AS total_flights\nFROM airline_partitioned\nGROUP BY AirportFrom\nORDER BY total_flights DESC LIMIT 5;", language="sql")
        if not df9.empty:
            st.pyplot(bar_chart(df9,'AirportFrom','total_flights',CLRS["sky"],ylabel="Flight Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">🛬 Busiest Destination Airports</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT AirportTo, COUNT(*) AS total_flights\nFROM airline_partitioned\nGROUP BY AirportTo\nORDER BY total_flights DESC LIMIT 5;", language="sql")
        if not df10.empty:
            st.pyplot(bar_chart(df10,'AirportTo','total_flights',CLRS["pink"],ylabel="Flight Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────
# TAB 3 · Time Analysis
# ─────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">Time Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">When do delays peak across the week and day?</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">📅 Day-wise Delay Pattern</span><span class="chart-badge">Line</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT DayOfWeek, COUNT(*) AS total_delays\nFROM airline_partitioned\nWHERE Delay = 1\nGROUP BY DayOfWeek;", language="sql")
        if not df4.empty:
            days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            df4['DayOfWeek'] = pd.Categorical(df4['DayOfWeek'], categories=days, ordered=True)
            df4s = df4.sort_values('DayOfWeek')
            st.pyplot(line_chart(df4s,'DayOfWeek','total_delays',CLRS["blue"],ylabel="Delay Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">🔴 Peak Delay Times</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Time, COUNT(*) AS delays\nFROM airline_partitioned\nWHERE Delay = 1\nGROUP BY Time ORDER BY delays DESC LIMIT 5;", language="sql")
        if not df7.empty:
            st.pyplot(bar_chart(df7,'Time','delays',CLRS["red"],xlabel="Time Slot",ylabel="Delay Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">📍 Peak Traffic Times</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Time, COUNT(*) AS total_flights\nFROM airline_partitioned\nGROUP BY Time ORDER BY total_flights DESC LIMIT 5;", language="sql")
        if not df12.empty:
            st.pyplot(bar_chart(df12,'Time','total_flights',CLRS["amber"],xlabel="Time Slot",ylabel="Flight Count"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        # Placeholder — put a note if nothing to show here
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">💡 Insight</span><span class="chart-badge">Summary</span></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:16px 0; color:#6B7280; font-size:13px; line-height:1.7;">
            <b style="color:#111827">Key findings from time analysis:</b><br><br>
            • Delays peak on <b style="color:#1D4ED8">weekdays</b>, particularly Thursday–Friday<br>
            • Evening time slots show the <b style="color:#EF4444">highest delay concentration</b><br>
            • Peak traffic and peak delays often overlap, suggesting <b style="color:#F59E0B">congestion</b> as a root cause<br>
            • Early morning slots have the <b style="color:#10B981">lowest delay rates</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────
# TAB 4 · Flight Operations
# ─────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">Flight Operations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Flight distances, durations and operational statistics</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">🛬 Longest Flights</span><span class="chart-badge">Top 5</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT Flight, MAX(Length) AS Length\nFROM airline_partitioned\nGROUP BY Flight\nORDER BY Length DESC LIMIT 5;", language="sql")
        if not df8.empty:
            st.pyplot(bar_chart(df8,'Flight','Length',CLRS["purple"],ylabel="Length (Miles)"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="chart-card"><div class="chart-header"><span class="chart-title-inner">✈️ Avg Flight Length — Delayed Flights</span><span class="chart-badge">Stat</span></div>', unsafe_allow_html=True)
        with st.expander("SQL Query"):
            st.code("SELECT AVG(Length) AS avg_length\nFROM airline_partitioned\nWHERE Delay = 1;", language="sql")
        if not df5.empty:
            val = df5['avg_length'].values[0] if 'avg_length' in df5.columns else 0
            st.markdown(f"""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:200px;">
                <div style="font-size:56px; font-weight:800; color:#1D4ED8; letter-spacing:-2px;">{val:.0f}</div>
                <div style="font-size:14px; color:#6B7280; margin-top:8px;">Average miles for delayed flights</div>
                <div style="margin-top:20px; background:#EFF6FF; border-radius:10px; padding:10px 24px; font-size:12px; color:#1D4ED8; font-weight:600;">
                    ✈️ Longer flights tend to accumulate more delay
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# FOOTER
# ========================================
st.markdown("""
<div class="dash-footer">
    ✈️ Airline Delay Analytics &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; Powered by Apache Hive &amp; HDFS
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close page-body