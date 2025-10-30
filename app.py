#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Web Dashboard for Trendyol Competitor Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Trendyol Rakip Analizi Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar'ı kapat
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def find_latest_competitor_analysis_file():
    """En güncel competitor_analysis dosyasını bul"""
    pattern = "competitor_analysis_*.json"
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def load_data():
    """Veriyi yükle"""
    latest_file = find_latest_competitor_analysis_file()
    
    if not latest_file:
        st.error("❌ competitor_analysis_*.json dosyası bulunamadı!")
        return None, None
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basit analiz
        analysis = {
            'total_products': len(data.get('products', [])),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_name': latest_file,
            'file_date': datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return data, analysis
    except Exception as e:
        st.error(f"❌ Veri yüklenirken hata: {e}")
        return None, None

def prepare_data_for_charts(data):
    """Grafikler için veri hazırla"""
    products = data.get('products', [])
    
    # Veri listeleri
    product_names = []
    our_prices = []
    avg_comp_prices = []
    min_comp_prices = []
    price_differences = []
    competitor_counts = []
    our_scores = []
    comp_scores = []
    
    # En yakın rakip metrikleri
    closest_comp_prices = []
    closest_comp_diff_pct = []
    closest_comp_diff_tl = []
    our_price_ranks = []
    cheaper_than_us_counts = []
    
    for product in products:
        if not product.get('our_merchant') or not product.get('competitors'):
            continue
        
        our_merchant = product['our_merchant']
        competitors = product['competitors']
        
        our_price = our_merchant.get('price', 0)
        our_score = our_merchant.get('seller_score', 0)
        comp_prices = [c.get('price', 0) for c in competitors if c.get('price', 0) > 0]
        comp_scores_list = [c.get('seller_score', 0) for c in competitors if c.get('seller_score', 0) > 0]
        
        if our_price > 0 and comp_prices:
            product_names.append(product['product_name'][:30])
            our_prices.append(our_price)
            competitor_counts.append(len(competitors))
            
            avg_comp_price = np.mean(comp_prices)
            min_comp_price = np.min(comp_prices)
            avg_comp_prices.append(avg_comp_price)
            min_comp_prices.append(min_comp_price)
            
            # Fiyat farkı yüzdesi
            price_diff = ((our_price - avg_comp_price) / avg_comp_price) * 100 if avg_comp_price > 0 else 0
            price_differences.append(price_diff)
            
            # EN YAKIN RAKİP ANALİZİ
            # En yakın rakip fiyatı bul (mutlak fark minimum olan)
            closest_price = min(comp_prices, key=lambda x: abs(x - our_price))
            closest_comp_prices.append(closest_price)
            
            # En yakın rakibe göre fark (TL ve %)
            closest_diff_tl = our_price - closest_price
            closest_diff_pct = (closest_diff_tl / closest_price) * 100 if closest_price > 0 else 0
            closest_comp_diff_tl.append(closest_diff_tl)
            closest_comp_diff_pct.append(closest_diff_pct)
            
            # Fiyat sıralaması
            all_prices_with_ours = [our_price] + comp_prices
            all_prices_sorted = sorted(all_prices_with_ours)
            our_rank = all_prices_sorted.index(our_price) + 1
            our_price_ranks.append(our_rank)
            
            # Bizden ucuz rakip sayısı
            cheaper_count = sum(1 for p in comp_prices if p < our_price)
            cheaper_than_us_counts.append(cheaper_count)
            
            if our_score > 0:
                our_scores.append(our_score)
            if comp_scores_list:
                comp_scores.extend(comp_scores_list)
    
    return {
        'product_names': product_names,
        'our_prices': our_prices,
        'avg_comp_prices': avg_comp_prices,
        'min_comp_prices': min_comp_prices,
        'price_differences': price_differences,
        'competitor_counts': competitor_counts,
        'our_scores': our_scores,
        'comp_scores': comp_scores,
        # En yakın rakip metrikleri
        'closest_comp_prices': closest_comp_prices,
        'closest_comp_diff_pct': closest_comp_diff_pct,
        'closest_comp_diff_tl': closest_comp_diff_tl,
        'our_price_ranks': our_price_ranks,
        'cheaper_than_us_counts': cheaper_than_us_counts
    }

def create_price_comparison_chart(chart_data):
    """Fiyat karşılaştırma grafiği"""
    fig = go.Figure()
    
    x = chart_data['product_names']
    
    fig.add_trace(go.Bar(
        name='Bizim Fiyat',
        x=x,
        y=chart_data['our_prices'],
        marker_color='#FF6B6B',
        text=chart_data['our_prices'],
        texttemplate='%{text:.0f} TL',
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Ortalama Rakip',
        x=x,
        y=chart_data['avg_comp_prices'],
        marker_color='#4ECDC4',
        text=chart_data['avg_comp_prices'],
        texttemplate='%{text:.0f} TL',
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='En Ucuz Rakip',
        x=x,
        y=chart_data['min_comp_prices'],
        marker_color='#45B7D1',
        text=chart_data['min_comp_prices'],
        texttemplate='%{text:.0f} TL',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Fiyat Karşılaştırması',
        xaxis_title='Ürünler',
        yaxis_title='Fiyat (TL)',
        hovermode='x unified',
        height=500,
        xaxis={'tickangle': -45}
    )
    
    return fig


def create_competitor_count_chart(chart_data):
    """Rakip sayısı grafiği"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=chart_data['product_names'],
        y=chart_data['competitor_counts'],
        marker_color='#4ECDC4',
        text=chart_data['competitor_counts'],
        texttemplate='%{text}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Ürün Başına Rakip Sayısı',
        xaxis_title='Ürünler',
        yaxis_title='Rakip Sayısı',
        hovermode='x unified',
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_price_distribution_chart(chart_data):
    """Fiyat dağılımı histogramı"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=chart_data['our_prices'],
        name='Bizim Fiyatlar',
        marker_color='#FF6B6B',
        opacity=0.7
    ))
    
    if chart_data['comp_scores']:  # Rakip fiyatları için
        fig.add_trace(go.Histogram(
            x=chart_data['avg_comp_prices'],
            name='Rakip Ortalama Fiyatlar',
            marker_color='#4ECDC4',
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Fiyat Dağılımı',
        xaxis_title='Fiyat (TL)',
        yaxis_title='Frekans',
        barmode='overlay',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_box_plot_comparison(chart_data):
    """Box plot karşılaştırması"""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=chart_data['our_prices'],
        name='Bizim Fiyatlar',
        marker_color='#FF6B6B'
    ))
    
    if chart_data['avg_comp_prices']:
        fig.add_trace(go.Box(
            y=chart_data['avg_comp_prices'],
            name='Rakip Ortalama Fiyatlar',
            marker_color='#4ECDC4'
        ))
    
    fig.update_layout(
        title='Fiyat Dağılımı Box Plot',
        yaxis_title='Fiyat (TL)',
        height=400
    )
    
    return fig

# EN YAKIN RAKİP ANALİZİ GRAFİKLERİ

def create_closest_competitor_comparison_chart(chart_data):
    """Ortalama vs En Yakın Rakip karşılaştırması"""
    fig = go.Figure()
    
    x = chart_data['product_names']
    
    fig.add_trace(go.Bar(
        name='Ortalama Rakibe Göre Fark',
        x=x,
        y=chart_data['price_differences'],
        marker_color='#4ECDC4',
        text=[f"{val:+.2f}%" for val in chart_data['price_differences']],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='En Yakın Rakibe Göre Fark',
        x=x,
        y=chart_data['closest_comp_diff_pct'],
        marker_color='#FF6B6B',
        text=[f"{val:+.2f}%" for val in chart_data['closest_comp_diff_pct']],
        textposition='outside'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    
    fig.update_layout(
        title='Ortalama vs En Yakın Rakip - Fiyat Farkı Karşılaştırması',
        xaxis_title='Ürünler',
        yaxis_title='Fiyat Farkı (%)',
        hovermode='x unified',
        height=500,
        barmode='group',
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_price_rank_chart(chart_data):
    """Fiyat sıralaması görselleştirmesi"""
    fig = go.Figure()
    
    x = chart_data['product_names']
    total_competitors = [c + 1 for c in chart_data['competitor_counts']]  # +1 biz
    cheaper = chart_data['cheaper_than_us_counts']
    more_expensive = [total_competitors[i] - cheaper[i] - 1 for i in range(len(total_competitors))]
    
    fig.add_trace(go.Bar(
        name='Bizden Ucuz',
        x=x,
        y=cheaper,
        marker_color='#E74C3C',
        text=cheaper,
        texttemplate='%{text}',
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Biz',
        x=x,
        y=[1] * len(x),
        marker_color='#FF6B6B',
        text=['👤'] * len(x),
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Bizden Pahalı',
        x=x,
        y=more_expensive,
        marker_color='#2ECC71',
        text=more_expensive,
        texttemplate='%{text}',
        textposition='inside'
    ))
    
    fig.update_layout(
        title='Fiyat Sıralaması - Bizden Ucuz / Biz / Bizden Pahalı',
        xaxis_title='Ürünler',
        yaxis_title='Satıcı Sayısı',
        hovermode='x unified',
        height=500,
        barmode='stack',
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_closest_distance_scatter(chart_data):
    """En yakın rakibe mesafe scatter plot"""
    colors = ['#FF4444' if diff > 0 else '#44FF44' for diff in chart_data['closest_comp_diff_tl']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=chart_data['competitor_counts'],
        y=chart_data['closest_comp_diff_tl'],
        mode='markers',
        marker=dict(
            color=chart_data['closest_comp_diff_tl'],
            size=12,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Fiyat Farkı (TL)")
        ),
        text=chart_data['product_names'],
        hovertemplate='<b>%{text}</b><br>Rakip: %{x}<br>Mesafe: %{y:.1f} TL<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.add_hline(y=5, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_hline(y=-5, line_dash="dot", line_color="orange", opacity=0.5)
    
    fig.update_layout(
        title='En Yakın Rakibe Mesafe Analizi',
        xaxis_title='Rakip Sayısı',
        yaxis_title='En Yakın Rakibe Fark (TL)',
        height=500,
        annotations=[
            dict(x=0.02, y=0.98, text='Yeşil: Ucuzuz | Kırmızı: Pahalıyız',
                 xref='paper', yref='paper', showarrow=False, bgcolor='white',
                 bordercolor='black', borderwidth=1)
        ]
    )
    
    return fig

def create_price_positioning_heatmap(chart_data, data):
    """Fiyat pozisyonu heatmap - Her ürün için tüm fiyatlar"""
    products = data.get('products', [])
    
    # Her ürün için fiyat listesi oluştur
    heatmap_data = []
    product_labels = []
    
    for i, product in enumerate(products):
        if not product.get('our_merchant') or not product.get('competitors'):
            continue
        
        our_price = product['our_merchant'].get('price', 0)
        comp_prices = [c.get('price', 0) for c in product['competitors'] if c.get('price', 0) > 0]
        
        if our_price > 0 and comp_prices:
            # Tüm fiyatları birlikte sırala
            all_prices = [our_price] + comp_prices
            all_prices_sorted = sorted(all_prices)
            
            # Maksimum 10 rakip göster
            max_display = min(11, len(all_prices_sorted))
            heatmap_data.append(all_prices_sorted[:max_display])
            
            # Eksik değerleri NaN ile doldur
            while len(heatmap_data[-1]) < 11:
                heatmap_data[-1].append(np.nan)
            
            product_labels.append(product['product_name'][:25])
    
    # DataFrame oluştur
    df_heatmap = pd.DataFrame(heatmap_data, 
                              index=product_labels,
                              columns=[f'{i+1}.' for i in range(11)])
    
    # Bizim pozisyonumuzu işaretle
    hover_text = []
    for i, row in df_heatmap.iterrows():
        hover_row = []
        for col in df_heatmap.columns:
            val = row[col]
            if pd.notna(val):
                hover_row.append(f'{val:.0f} TL')
            else:
                hover_row.append('')
        hover_text.append(hover_row)
    
    # Bizim fiyatlarımızı bul ve vurgula
    fig = go.Figure(data=go.Heatmap(
        z=df_heatmap.values,
        x=df_heatmap.columns,
        y=df_heatmap.index,
        colorscale='Viridis',
        text=hover_text,
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar=dict(title="Fiyat (TL)")
    ))
    
    # Bizim pozisyonumuza işaret ekle
    for i, (row_label, row_data) in enumerate(df_heatmap.iterrows()):
        our_price = chart_data['our_prices'][i] if i < len(chart_data['our_prices']) else 0
        if our_price > 0:
            col_idx = list(row_data.values).index(our_price) if our_price in row_data.values else -1
            if col_idx >= 0:
                fig.add_shape(type="rect",
                             xref='x', yref='y',
                             x0=col_idx-0.5, x1=col_idx+0.5,
                             y0=i-0.5, y1=i+0.5,
                             line=dict(color='red', width=3))
    
    fig.update_layout(
        title='Fiyat Pozisyonu Heatmap (Kırmızı Çerçeve: Bizim Fiyat)',
        xaxis_title='Fiyat Sırası',
        yaxis_title='Ürünler',
        height=max(600, len(product_labels) * 30)
    )
    
    return fig

def main():
    # Başlık
    st.markdown('<h1 class="main-header">🏪 Trendyol Rakip Analizi Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar kaldırıldı - ekstra alan için
    
    # Veriyi yükle
    @st.cache_data
    def load_cached_data():
        return load_data()
    
    data, analysis = load_cached_data()
    
    if data is None or analysis is None:
        st.warning("📂 Lütfen competitor_analysis_*.json dosyasının mevcut dizinde olduğundan emin olun.")
        return
    
    # Veri özeti
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Toplam Ürün", analysis['total_products'])
    
    with col2:
        st.metric("📁 Dosya", analysis['file_name'].split('/')[-1])
    
    with col3:
        st.metric("📅 Analiz Tarihi", analysis['analysis_date'].split(' ')[0])
    
    with col4:
        price_diffs = prepare_data_for_charts(data)['price_differences']
        if price_diffs:
            avg_diff = np.mean(price_diffs)
            st.metric("💰 Ort. Fiyat Farkı", f"{avg_diff:+.1f}%")
    
    st.divider()
    
    # Grafikler için veri hazırla
    chart_data = prepare_data_for_charts(data)
    
    if not chart_data['product_names']:
        st.warning("⚠️ Grafik oluşturmak için yeterli veri yok.")
        return
    
    # Sekme yapısı
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Fiyat Analizi", "🎯 Rekabet Analizi", "📈 Dağılım Analizi", "📊 Ek Grafikler", "🎯 En Yakın Rakip", "📋 Özet Rapor"])
    
    # Sekme 1: Fiyat Analizi
    with tab1:
        st.header("Fiyat Karşılaştırması")
        
        # — Filtreler: Ürün ve Fiyat —
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            product_query = st.text_input(
                "Ürün adı ara",
                value="",
                placeholder="Ürün adında ara...",
            ).strip().lower()
        with c2:
            # Bizim fiyatlarımız için min-max aralığı
            min_price = int(np.floor(min(chart_data['our_prices']))) if chart_data['our_prices'] else 0
            max_price = int(np.ceil(max(chart_data['our_prices']))) if chart_data['our_prices'] else 0
            selected_price_range = st.slider(
                "Bizim fiyat aralığı (TL)",
                min_value=min_price,
                max_value=max_price if max_price > min_price else min_price + 1,
                value=(min_price, max_price if max_price > min_price else min_price + 1),
                step=1,
            )
        with c3:
            # En yakın rakibe göre fark (%) için aralık
            min_closest = float(np.floor(min(chart_data['closest_comp_diff_pct'])) if chart_data['closest_comp_diff_pct'] else -100)
            max_closest = float(np.ceil(max(chart_data['closest_comp_diff_pct'])) if chart_data['closest_comp_diff_pct'] else 100)
            selected_closest_range = st.slider(
                "En yakın rakip farkı (%)",
                min_value=min_closest,
                max_value=max_closest if max_closest > min_closest else min_closest + 1,
                value=(min_closest, max_closest if max_closest > min_closest else min_closest + 1),
            )
        # İsteğe bağlı: doğrudan ürün seçimi
        selected_names = st.multiselect(
            "Ürün seç (opsiyonel, çoklu)",
            options=chart_data['product_names'],
        )

        # Filtreleri uygula (index bazlı)
        idx_all = list(range(len(chart_data['product_names'])))
        filtered_idx = []
        for i in idx_all:
            name = chart_data['product_names'][i] or ""
            our_p = chart_data['our_prices'][i]
            closest_pct = chart_data['closest_comp_diff_pct'][i] if i < len(chart_data['closest_comp_diff_pct']) else 0.0

            if not (selected_price_range[0] <= our_p <= selected_price_range[1]):
                continue
            if not (selected_closest_range[0] <= float(closest_pct) <= selected_closest_range[1]):
                continue
            if product_query and product_query not in name.lower():
                continue
            if selected_names and name not in set(selected_names):
                continue
            filtered_idx.append(i)

        def subset_price_tab(data_dict, idxs):
            return {
                'product_names': [data_dict['product_names'][i] for i in idxs],
                'our_prices': [data_dict['our_prices'][i] for i in idxs],
                'avg_comp_prices': [data_dict['avg_comp_prices'][i] for i in idxs],
                'min_comp_prices': [data_dict['min_comp_prices'][i] for i in idxs],
                'price_differences': [data_dict['price_differences'][i] for i in idxs],
                'competitor_counts': [data_dict['competitor_counts'][i] for i in idxs],
                'our_scores': chart_data['our_scores'],
                'comp_scores': chart_data['comp_scores'],
                'closest_comp_prices': [data_dict['closest_comp_prices'][i] for i in idxs],
                'closest_comp_diff_pct': [data_dict['closest_comp_diff_pct'][i] for i in idxs],
                'closest_comp_diff_tl': [data_dict['closest_comp_diff_tl'][i] for i in idxs],
                'our_price_ranks': [data_dict['our_price_ranks'][i] for i in idxs],
                'cheaper_than_us_counts': [data_dict['cheaper_than_us_counts'][i] for i in idxs],
            }

        chart_data_tab1 = subset_price_tab(chart_data, filtered_idx) if filtered_idx else subset_price_tab(chart_data, [])

        # İlk satır: Fiyat karşılaştırması (filtreli)
        fig_price = create_price_comparison_chart(chart_data_tab1)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # İstatistikler (filtreli)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**En Pahalı:** {max(chart_data_tab1['our_prices']) if chart_data_tab1['our_prices'] else 0:.0f} TL")
        with col2:
            st.info(f"**En Ucuz:** {min(chart_data_tab1['our_prices']) if chart_data_tab1['our_prices'] else 0:.0f} TL")
        with col3:
            mean_val = np.mean(chart_data_tab1['our_prices']) if chart_data_tab1['our_prices'] else 0
            st.info(f"**Ortalama:** {mean_val:.0f} TL")
    
    # Sekme 2: Rekabet Analizi
    with tab2:
        st.header("Rekabet Yoğunluğu")
        
        # — Filtreler: Rakip sayısı aralığı —
        min_comp, max_comp = int(np.min(chart_data['competitor_counts'])), int(np.max(chart_data['competitor_counts']))
        selected_min, selected_max = st.slider(
            "Rakip sayısı aralığı",
            min_value=min_comp,
            max_value=max_comp,
            value=(min_comp, max_comp),
            step=1,
            help="Grafikler ve metrikler seçili aralığa göre filtrelenir."
        )
        
        # — Ürün arama ve sıralama —
        col_search, col_sort, col_order = st.columns([2, 2, 1])
        with col_search:
            search_query = st.text_input(
                "Ürün adı ara",
                value="",
                placeholder="Ürün adında ara...",
            ).strip().lower()
        with col_sort:
            sort_label = st.selectbox(
                "Sırala",
                options=[
                    "Rakip Sayısı",
                    "Ort. Rakip Fiyatı",
                    "Fiyat Farkı (%)",
                    "Ürün Adı",
                ],
                index=0,
            )
        with col_order:
            sort_order = st.selectbox("Sıra", options=["Artan", "Azalan"], index=1)
        reverse_sort = (sort_order == "Azalan")
        
        # Filtreyi uygula (index bazlı senkron alt listeler)
        filtered_indices = [i for i, c in enumerate(chart_data['competitor_counts']) if selected_min <= c <= selected_max]
        if search_query:
            filtered_indices = [
                i for i in filtered_indices
                if search_query in (chart_data['product_names'][i] or "").lower()
            ]

        # Sıralama anahtarı
        def sort_key(i):
            if sort_label == "Rakip Sayısı":
                return chart_data['competitor_counts'][i]
            if sort_label == "Ort. Rakip Fiyatı":
                return chart_data['avg_comp_prices'][i]
            if sort_label == "Fiyat Farkı (%)":
                return chart_data['price_differences'][i]
            if sort_label == "Ürün Adı":
                return chart_data['product_names'][i]
            return chart_data['competitor_counts'][i]
        filtered_indices = sorted(filtered_indices, key=sort_key, reverse=reverse_sort)
        def subset(data_dict, idxs):
            return {
                'product_names': [data_dict['product_names'][i] for i in idxs],
                'our_prices': [data_dict['our_prices'][i] for i in idxs],
                'avg_comp_prices': [data_dict['avg_comp_prices'][i] for i in idxs],
                'min_comp_prices': [data_dict['min_comp_prices'][i] for i in idxs],
                'price_differences': [data_dict['price_differences'][i] for i in idxs],
                'competitor_counts': [data_dict['competitor_counts'][i] for i in idxs],
                'our_scores': chart_data['our_scores'],  # global list (grafikte doğrudan kullanılmıyor)
                'comp_scores': chart_data['comp_scores'], # global list (grafikte doğrudan kullanılmıyor)
                'closest_comp_prices': [data_dict['closest_comp_prices'][i] for i in idxs],
                'closest_comp_diff_pct': [data_dict['closest_comp_diff_pct'][i] for i in idxs],
                'closest_comp_diff_tl': [data_dict['closest_comp_diff_tl'][i] for i in idxs],
                'our_price_ranks': [data_dict['our_price_ranks'][i] for i in idxs],
                'cheaper_than_us_counts': [data_dict['cheaper_than_us_counts'][i] for i in idxs],
            }
        filtered_data = subset(chart_data, filtered_indices) if filtered_indices else subset(chart_data, [])
        
        # Rakip sayısı
        fig_comp = create_competitor_count_chart(filtered_data)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Rakip sayısı istatistikleri
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Ortalama Rakip", f"{np.mean(filtered_data['competitor_counts']) if filtered_data['competitor_counts'] else 0:.1f}")
        
        with col2:
            st.metric("Maksimum Rakip", f"{max(filtered_data['competitor_counts']) if filtered_data['competitor_counts'] else 0}")
        
        with col3:
            st.metric("Minimum Rakip", f"{min(filtered_data['competitor_counts']) if filtered_data['competitor_counts'] else 0}")
        
        # Rekabet seviyesi kategorileri
        st.subheader("Rekabet Seviyesi")
        
        low_comp = sum(1 for c in filtered_data['competitor_counts'] if c <= 5)
        med_comp = sum(1 for c in filtered_data['competitor_counts'] if 5 < c <= 10)
        high_comp = sum(1 for c in filtered_data['competitor_counts'] if c > 10)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"**Düşük Rekabet (≤5 rakip):** {low_comp} ürün")
        
        with col2:
            st.warning(f"**Orta Rekabet (6-10 rakip):** {med_comp} ürün")
        
        with col3:
            st.error(f"**Yüksek Rekabet (>10 rakip):** {high_comp} ürün")
    
    # Sekme 3: Dağılım Analizi
    with tab3:
        st.header("Fiyat Dağılımı")
        
        # Histogram
        fig_hist = create_price_distribution_chart(chart_data)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Box plot
        fig_box = create_box_plot_comparison(chart_data)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Seçili ürünler için Box plot
        st.subheader("Seçili Ürünler - Fiyat Dağılımı Box Plot")
        selected_products = st.multiselect(
            "Ürün seçin (çoklu seçim)",
            options=chart_data['product_names'],
        )

        if selected_products:
            selected_indices = [
                i for i, name in enumerate(chart_data['product_names'])
                if name in set(selected_products)
            ]
            selected_our_prices = [chart_data['our_prices'][i] for i in selected_indices]
            selected_avg_comp_prices = [chart_data['avg_comp_prices'][i] for i in selected_indices]

            fig_box_selected = go.Figure()
            fig_box_selected.add_trace(go.Box(
                y=selected_our_prices,
                name='Bizim Fiyatlar (Seçili)',
                marker_color='#FF6B6B'
            ))
            if selected_avg_comp_prices:
                fig_box_selected.add_trace(go.Box(
                    y=selected_avg_comp_prices,
                    name='Rakip Ortalama (Seçili)',
                    marker_color='#4ECDC4'
                ))
            fig_box_selected.update_layout(
                yaxis_title='Fiyat (TL)',
                height=400
            )
            st.plotly_chart(fig_box_selected, use_container_width=True)

        # İstatistiksel özet
        st.subheader("İstatistiksel Özet")
        
        df_summary = pd.DataFrame({
            'Bizim Fiyatlar': chart_data['our_prices'],
            'Rakip Ortalama': chart_data['avg_comp_prices']
        })
        
        st.dataframe(df_summary.describe(), use_container_width=True)
    
    # Sekme 4: Ek Grafikler
    with tab4:
        st.header("Ek Analiz Grafikleri")
        
        # 1. Scatter Plot - Fiyat vs Rakip Sayısı
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fiyat vs Rakip Sayısı")
            fig_scatter1 = go.Figure()
            fig_scatter1.add_trace(go.Scatter(
                x=chart_data['competitor_counts'],
                y=chart_data['our_prices'],
                mode='markers',
                marker=dict(
                    color=chart_data['price_differences'],
                    size=10,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Fiyat Farkı (%)")
                ),
                text=chart_data['product_names'],
                hovertemplate='<b>%{text}</b><br>Rakip: %{x}<br>Fiyat: %{y} TL<extra></extra>'
            ))
            fig_scatter1.update_layout(
                xaxis_title='Rakip Sayısı',
                yaxis_title='Bizim Fiyat (TL)',
                height=400
            )
            st.plotly_chart(fig_scatter1, use_container_width=True)
        
        with col2:
            st.subheader("Fiyat Farkı vs Rakip Sayısı")
            fig_scatter2 = go.Figure()
            fig_scatter2.add_trace(go.Scatter(
                x=chart_data['competitor_counts'],
                y=chart_data['price_differences'],
                mode='markers',
                marker=dict(
                    color=chart_data['competitor_counts'],
                    size=10,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Rakip Sayısı")
                ),
                text=chart_data['product_names'],
                hovertemplate='<b>%{text}</b><br>Rakip: %{x}<br>Fark: %{y:.1f}%<extra></extra>'
            ))
            fig_scatter2.add_hline(y=0, line_dash="dash", line_color="red")
            fig_scatter2.update_layout(
                xaxis_title='Rakip Sayısı',
                yaxis_title='Fiyat Farkı (%)',
                height=400
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)
        
        # 2. Sunburst Chart - Hiyerarşik analiz
        st.subheader("Fiyat Pozisyon Analizi (Güneş Patlaması Grafiği)")
        
        # Pozisyon kategorileri
        categories = []
        for diff in chart_data['price_differences']:
            if diff < -20:
                categories.append('Çok Ucuz (<-20%)')
            elif diff < -5:
                categories.append('Ucuz (-20% to -5%)')
            elif diff < 5:
                categories.append('Rekabetçi (-5% to 5%)')
            elif diff < 20:
                categories.append('Pahalı (5% to 20%)')
            else:
                categories.append('Çok Pahalı (>20%)')
        
        # Sunburst için veri hazırla
        df_sunburst = pd.DataFrame({
            'category': categories,
            'product': chart_data['product_names'],
            'price_diff': chart_data['price_differences']
        })
        
        fig_sunburst = px.sunburst(
            df_sunburst, 
            path=['category', 'product'], 
            values='price_diff',
            color='price_diff',
            color_continuous_scale='RdYlGn',
            title='Fiyat Pozisyon Hiyerarşisi'
        )
        fig_sunburst.update_layout(height=600)
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # 3. Violin Plot - Detaylı dağılım
        st.subheader("Fiyat Dağılımı Violin Plot")
        
        all_prices = []
        labels_for_violin = []
        
        for i, price in enumerate(chart_data['our_prices']):
            all_prices.append(price)
            labels_for_violin.append('Bizim')
        
        for i, price in enumerate(chart_data['avg_comp_prices']):
            all_prices.append(price)
            labels_for_violin.append('Rakip Ortalama')
        
        fig_violin = go.Figure()
        
        for label in ['Bizim', 'Rakip Ortalama']:
            prices = [all_prices[i] for i, l in enumerate(labels_for_violin) if l == label]
            fig_violin.add_trace(go.Violin(
                y=prices,
                name=label,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig_violin.update_layout(height=400)
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # 4. Heatmap - Korelasyon Matrisi
        st.subheader("Korelasyon Matrisi")
        
        df_corr = pd.DataFrame({
            'Bizim_Fiyat': chart_data['our_prices'],
            'Rakip_Ortalama': chart_data['avg_comp_prices'],
            'Rakip_Sayisi': chart_data['competitor_counts'],
            'Fiyat_Farki': chart_data['price_differences']
        })
        
        corr_matrix = df_corr.corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Sekme 5: En Yakın Rakip Analizi
    with tab5:
        st.header("En Yakın Rakip Analizi")
        st.info("Bu sekmede, ortalama fiyat yerine bizim fiyatımıza **en yakın rakip** ile karşılaştırma yapılmaktadır.")
        
        # Özet metrikler
        st.subheader("Özet Metrikler")
        
        cheapest_count = sum(1 for rank in chart_data['our_price_ranks'] if rank == 1)
        avg_closest_diff_tl = np.mean([abs(d) for d in chart_data['closest_comp_diff_tl']])
        avg_closest_diff_pct = np.mean([abs(d) for d in chart_data['closest_comp_diff_pct']])
        risky_count = sum(1 for diff in chart_data['closest_comp_diff_tl'] if abs(diff) <= 5)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("En Ucuz Ürün", f"{cheapest_count} adet")
        
        with col2:
            st.metric("Ort. En Yakın Mesafe", f"{avg_closest_diff_tl:.1f} TL")
        
        with col3:
            st.metric("Ort. % Fark", f"{avg_closest_diff_pct:.1f}%")
        
        with col4:
            st.metric("Riskli Ürünler (≤5TL)", f"{risky_count} adet")
        
        st.divider()
        
        # Karşılaştırmalı bar chart
        st.subheader("Ortalama vs En Yakın Rakip - Fiyat Farkı")
        fig_closest_comp = create_closest_competitor_comparison_chart(chart_data)
        st.plotly_chart(fig_closest_comp, use_container_width=True)
        
        # Fiyat sıralaması
        st.subheader("Fiyat Sıralaması")
        fig_rank = create_price_rank_chart(chart_data)
        st.plotly_chart(fig_rank, use_container_width=True)
        
        # Mesafe scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("En Yakın Rakibe Mesafe")
            fig_distance = create_closest_distance_scatter(chart_data)
            st.plotly_chart(fig_distance, use_container_width=True)
        
        with col2:
            st.subheader("En Riskli Ürünler")
            
            # En yakın rakibe çok yakın olan ürünler
            risky_products = []
            for i, diff_tl in enumerate(chart_data['closest_comp_diff_tl']):
                if abs(diff_tl) <= 5:
                    risky_products.append({
                        'Ürün': chart_data['product_names'][i],
                        'Mesafe (TL)': diff_tl,
                        'Mesafe (%)': chart_data['closest_comp_diff_pct'][i],
                        'Bizim Fiyat': chart_data['our_prices'][i],
                        'En Yakın Rakip': chart_data['closest_comp_prices'][i]
                    })
            
            if risky_products:
                df_risky = pd.DataFrame(risky_products)
                st.dataframe(df_risky, use_container_width=True)
            else:
                st.success("Riskli ürün bulunamadı!")
        
        # Fiyat pozisyon heatmap
        st.subheader("Fiyat Pozisyonu Detayı")
        fig_positioning = create_price_positioning_heatmap(chart_data, data)
        st.plotly_chart(fig_positioning, use_container_width=True)
    
    # Sekme 6: Özet Rapor
    with tab6:
        st.header("Özet Rapor")
        
        # Genel metrikler
        st.subheader("Genel Metrikler")
        

        cheapest_count_en_yakın_rakip = sum(1 for rank in chart_data['our_price_ranks'] if rank == 1)
        avg_closest_diff_tl_en_yakın_rakip = np.mean([abs(d) for d in chart_data['closest_comp_diff_tl']])
        avg_closest_diff_pct_en_yakın_rakip = np.mean([abs(d) for d in chart_data['closest_comp_diff_pct']])
        risky_count_en_yakın_rakip = sum(1 for diff in chart_data['closest_comp_diff_tl'] if abs(diff) <= 5)
        

        cheaper_count = cheapest_count_en_yakın_rakip
        expensive_count = 20 - cheapest_count_en_yakın_rakip # sum(1 for diff in chart_data['price_differences'] if diff > 0)
        competitive_count = sum(1 for diff in chart_data['price_differences'] if -5 <= diff <= 5)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"**En Ucuz Biziz:** {cheaper_count} üründe")
        
        with col2:
            st.error(f"**Bizden Ucuzlar Var:** {expensive_count} üründe")
        
        st.divider()
        
        # En yakın rakip metrikleri
        st.subheader("En Yakın Rakip Analizi Metrikleri")
        
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("En Ucuz Ürün", f"{cheapest_count} adet")
        
        with col2:
            st.metric("En Yakın Mesafe Ort.", f"{avg_closest_diff_tl:.1f} TL")
        
        with col3:
            st.metric("Ort. % Fark", f"{avg_closest_diff_pct:.1f}%")
        
        with col4:
            st.metric("Riskli Ürün (≤5TL)", f"{risky_count} adet")
        
        st.divider()
        
        # En çok rekabet edilen ürünler
        st.subheader("En Çok Rekabet Edilen Ürünler")
        
        # Rakip sayısına göre sırala
        sorted_indices = sorted(range(len(chart_data['competitor_counts'])), 
                               key=lambda i: chart_data['competitor_counts'][i], 
                               reverse=True)
        
        # Dinamik sütun: Biz en ucuzsak 2. en pahalı, değilsek en ucuz
        reference_prices = []
        reference_labels = []
        
        products = data.get('products', [])
        valid_products = [p for p in products if p.get('our_merchant') and p.get('competitors')]
        
        for i in sorted_indices[:10]:
            product_idx = i
            if product_idx < len(valid_products):
                product = valid_products[product_idx]
                our_price = product['our_merchant'].get('price', 0)
                comp_prices = [c.get('price', 0) for c in product['competitors'] if c.get('price', 0) > 0]
                
                if comp_prices:
                    all_prices = [our_price] + comp_prices
                    all_prices_sorted = sorted(all_prices)
                    
                    # Bizim sıramız
                    our_rank = all_prices_sorted.index(our_price) + 1
                    
                    if our_rank == 1:  # Biz en ucuzuz
                        # 2. en pahalıyı al (tersten 2.)
                        if len(all_prices_sorted) >= 2:
                            second_most_expensive = all_prices_sorted[-2]
                            reference_prices.append(second_most_expensive)
                            reference_labels.append('Bizden Sonra 2. En Pahalı')
                        else:
                            reference_prices.append(all_prices_sorted[-1])
                            reference_labels.append('En Pahalı')
                    else:  # Biz en ucuz değiliz
                        # En ucuzu al
                        cheapest = all_prices_sorted[0]
                        reference_prices.append(cheapest)
                        reference_labels.append('En Ucuz Rakibimiz')
                else:
                    reference_prices.append(0)
                    reference_labels.append('N/A')
            else:
                reference_prices.append(0)
                reference_labels.append('N/A')
        
        # Format fonksiyonu: Gereksiz sıfırları kaldır
        def format_price(price):
            """123 -> 123.00, 123.5 -> 123.50, 123.55 -> 123.55"""
            if price == 0:
                return 0.00
            return f"{price:.2f}"
        
        top_competitive = pd.DataFrame({
            'Ürün': [chart_data['product_names'][i] for i in sorted_indices[:10]],
            'Rakip Sayısı': [chart_data['competitor_counts'][i] for i in sorted_indices[:10]],
            'Bizim Fiyat (TL)': [format_price(chart_data['our_prices'][i]) for i in sorted_indices[:10]],
            'Referans Fiyat (TL)': [format_price(p) for p in reference_prices],
            'Referans Tipi': reference_labels,
            'Ort. Rakip Fiyat (TL)': [format_price(chart_data['avg_comp_prices'][i]) for i in sorted_indices[:10]],
            'Fiyat Farkı (%)': [format_price(chart_data['price_differences'][i]) for i in sorted_indices[:10]]
        })
        
        # Renkli dataframe oluştur
        def color_reference_type(row):
            """Referans tipine göre renklendirme"""
            if row['Referans Tipi'] == 'Bizden Sonra 2. En Pahalı':
                return ['background-color: #d4edda'] * len(row)  # Yeşil
            else:
                return ['background-color: #f8d7da'] * len(row)  # Kırmızı
        
        # Styled dataframe
        styled_df = top_competitive.style.apply(color_reference_type, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.divider()
        
        # Öneriler
        st.subheader("💡 Öneriler")
        
        # Fiyat optimizasyonu önerileri
        if expensive_count > cheaper_count:
            st.warning("⚠️ **Fiyat Optimizasyonu:** Çoğu ürününüzde rakiplerden pahalı durumdasınız. "
                      "Fiyat rekabetçiliği için düşünülmelidir.")
        else:
            st.success("✅ **Fiyat Pozisyonu:** Fiyat rekabetçiliğiniz iyi durumda.")
        
        # Yüksek rekabetli ürünler
        if high_comp > 0:
            st.info("ℹ️ **Rekabet Stratejisi:** Yüksek rekabetli ürünlerde farklılaşma stratejileri geliştirmelisiniz.")
        
        st.divider()
        
        # Alt bilgi
        st.caption(f"Son güncelleme: {analysis['analysis_date']}")

if __name__ == "__main__":
    main()

