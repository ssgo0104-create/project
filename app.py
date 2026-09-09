import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import base64
import os

# -----------------------------------------------------------------------------
# 0. 파일 기준 동적 기본 경로(BASE_DIR) 및 폰트 설정
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "Pretendard-Regular.otf")

# 차트용 폰트 등록
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rc('font', family=font_prop.get_name())
else:
    plt.rc('font', family='sans-serif')

plt.rc('axes', unicode_minus=False)

st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# 아이콘 깨짐 방지 처리된 CSS 스타일링
if os.path.exists(FONT_PATH):
    with open(FONT_PATH, "rb") as f:
        font_data = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(f"""
    <style>
        @font-face {{
            font-family: 'LocalPretendard';
            src: url(data:font/otf;base64,{font_data}) format('opentype');
            font-weight: normal;
            font-style: normal;
        }}
        
        /* 텍스트 요소에만 Pretendard 적용 */
        html, body, p, span, div, h1, h2, h3, h4, h5, h6, label, input, button, select {{
            font-family: 'LocalPretendard', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        /* Streamlit 기본 머티리얼 아이콘 폰트는 원래대로 복원 */
        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        span[class*="material-symbols"] {{
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    trade_path = os.path.join(BASE_DIR, "baci_85_sample.csv")
    country_path = os.path.join(BASE_DIR, "country_codes_sample.csv")
    
    # 인코딩 대응
    try:
        trade_df = pd.read_csv(trade_path, encoding='utf-8')
    except UnicodeDecodeError:
        trade_df = pd.read_csv(trade_path, encoding='cp949')
        
    try:
        country_df = pd.read_csv(country_path, encoding='utf-8')
    except UnicodeDecodeError:
        country_df = pd.read_csv(country_path, encoding='cp949')
    
    trade_df.columns = trade_df.columns.str.strip().str.lower()
    country_df.columns = country_df.columns.str.strip().str.lower()
    
    # 1) 국가 코드 매핑 사전
    code_cols = [c for c in country_df.columns if any(k in c for k in ['code', 'num', 'iso', 'id', 'i'])]
    code_col = code_cols[0] if code_cols else country_df.columns[0]
    
    name_cols = [c for c in country_df.columns if any(k in c for k in ['name', 'country', 'desc', 'label', 'kor']) and c != code_col]
    name_col = name_cols[0] if name_cols else (country_df.columns[1] if len(country_df.columns) > 1 else code_col)
    
    country_map = {}
    for _, row in country_df.iterrows():
        raw_code = str(row[code_col]).strip()
        val_name = str(row[name_col]).strip()
        country_map[raw_code] = val_name
        try:
            country_map[str(int(float(raw_code)))] = val_name
        except (ValueError, TypeError):
            pass

    # 2) 기준 국가 매핑
    target_country_col = 'i'
    if 'i' in trade_df.columns:
        if trade_df['i'].nunique() == 1 and 'j' in trade_df.columns and trade_df['j'].nunique() > 1:
            target_country_col = 'j'

    def get_country_name(code):
        c_str = str(code).strip()
        if c_str in country_map:
            return country_map[c_str]
        try:
            norm_c = str(int(float(c_str)))
            return country_map.get(norm_c, c_str)
        except (ValueError, TypeError):
            return c_str

    trade_df['country_name'] = trade_df[target_country_col].apply(get_country_name)
    
    # 무역액 등급 (소, 중, 대)
    trade_df['무역액등급'] = pd.qcut(
        trade_df['v'], 
        q=[0, 0.33, 0.66, 1.0], 
        labels=['소', '중', '대']
    )
    
    return trade_df

trade_raw = load_data()

# -----------------------------------------------------------------------------
# 2. 사이드바 필터
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 필터 옵션")

all_countries = sorted(trade_raw['country_name'].unique().tolist())
# 수정 1: 🌐국가 선택 (미선택 시 전체)
selected_countries = st.sidebar.multiselect(
    "🌐국가 선택 (미선택 시 전체)",
    options=all_countries,
    default=[]
)

tier_options = ['소', '중', '대']
# 수정 2: 💲무역액 등급 선택
selected_tiers = st.sidebar.multiselect(
    "💲무역액 등급 선택",
    options=tier_options,
    default=tier_options
)

filtered_df = trade_raw.copy()

if selected_countries:
    filtered_df = filtered_df[filtered_df['country_name'].isin(selected_countries)]

if selected_tiers:
    filtered_df = filtered_df[filtered_df['무역액등급'].isin(selected_tiers)]

# -----------------------------------------------------------------------------
# 3. 메인 화면 구성
# -----------------------------------------------------------------------------

# 1. 타이틀
st.title("🚢 무역 분석 대시보드")
st.markdown("---")

# 2. baci_85_sample.csv 파일의 결측치
st.subheader("📋 baci_85_sample.csv 결측치 현황")
missing_df = pd.DataFrame({
    '컬럼명': trade_raw.columns,
    '결측치 개수': trade_raw.isnull().sum().values,
    '결측 비율(%)': (trade_raw.isnull().mean() * 100).round(2).values
})
st.dataframe(missing_df.set_index('컬럼명').T, use_container_width=True)

st.markdown("---")

# 3. 총 거래 건수 및 총 수출액(달러)
st.subheader("📊 주요 통계 지표")
col_m1, col_m2 = st.columns(2)

total_deals = len(filtered_df)
total_trade_val = filtered_df['v'].sum() * 1000

with col_m1:
    st.metric(label="총 거래 건수", value=f"{total_deals:,} 건")

with col_m2:
    st.metric(label="총 수출액 (달러)", value=f"${total_trade_val:,.0f}")

st.markdown("---")

# 4. 국가*연도 수출액 히트맵 & 무역액 등급 분포
st.subheader("📈 수출 동향 및 무역액 등급 분석")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("##### 🌐 국가 × 연도 수출액 히트맵 (상위 8개국)")
    if not filtered_df.empty:
        top8_countries = filtered_df.groupby('country_name')['v'].sum().nlargest(8).index
        heatmap_data = filtered_df[filtered_df['country_name'].isin(top8_countries)]
        
        if not heatmap_data.empty:
            pivot_heat = heatmap_data.pivot_table(
                index='country_name', 
                columns='t', 
                values='v', 
                aggfunc='sum', 
                fill_value=0
            )
            
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(pivot_heat, cmap='YlGnBu', annot=True, fmt='.0f', cbar=True, ax=ax)
            ax.set_xlabel("연도")
            ax.set_ylabel("국가")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("표시할 데이터가 없습니다.")
    else:
        st.info("데이터가 없습니다.")

with col_chart2:
    # 수정 3: 💲무역액 등급 분포
    st.markdown("##### 💲무역액 등급 분포")
    if not filtered_df.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        tier_counts = filtered_df['무역액등급'].value_counts().reindex(tier_options, fill_value=0)
        
        sns.barplot(x=tier_counts.index, y=tier_counts.values, palette='Blues_r', ax=ax2)
        ax2.set_xlabel("무역액 등급")
        ax2.set_ylabel("건수")
        
        for i, v in enumerate(tier_counts.values):
            ax2.text(i, v, f"{v:,}", ha='center', va='bottom', fontsize=9)
            
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("데이터가 없습니다.")

st.markdown("---")

# 5. 상위 5개국 * 무역액 등급 교차표
st.subheader("📑 상위 5개국 × 무역액 등급 교차 분석")

if not filtered_df.empty:
    top5_countries = filtered_df.groupby('country_name')['v'].sum().nlargest(5).index
    df_top5 = filtered_df[filtered_df['country_name'].isin(top5_countries)]
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("##### [원본 건수]")
        cross_count = pd.crosstab(
            df_top5['country_name'], 
            df_top5['무역액등급'], 
            margins=True, 
            margins_name="합계"
        )
        st.dataframe(cross_count, use_container_width=True)

    with col_t2:
        st.markdown("##### [정규화 비율 (%)]")
        cross_norm = pd.crosstab(
            df_top5['country_name'], 
            df_top5['무역액등급'], 
            normalize='index'
        ) * 100
        st.dataframe(cross_norm.style.format("{:.2f}%"), use_container_width=True)
else:
    st.info("선택된 조건에 해당하는 데이터가 없습니다.")