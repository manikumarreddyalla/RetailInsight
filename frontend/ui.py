# Filename: frontend/ui.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import requests
import os

# ===============================
# BASIC CONFIG
# ===============================
st.set_page_config(page_title="RetailInsight", layout="wide")

# ===============================
# PATH TO BACKEND DATA
# ===============================
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../backend/data")
)

BACKEND_URL = "http://127.0.0.1:8000"

def load_csv(filename):
    filepath = os.path.join(DATA_PATH, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        st.error(f"❌ Missing file: {filepath}")
        return None


# ===============================
# LOAD DATASETS
# ===============================
sales_df = load_csv("sales_dataset.csv")
product_df = load_csv("products_master.csv")
calendar_df = load_csv("calendar_dataset.csv")


# ===============================
# SMALL HELPERS
# ===============================
def metric_card(label, value, subtitle="", color="#1f2937"):
    """Pretty metric card using HTML."""
    st.markdown(
        f"""
        <div style="
            padding: 12px 16px;
            border-radius: 10px;
            background: {color};
            color: white;
            margin-bottom: 8px;">
            <div style="font-size: 13px; opacity: 0.8;">{label}</div>
            <div style="font-size: 20px; font-weight: 700;">{value}</div>
            <div style="font-size: 12px; opacity: 0.9;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("🛒 RetailInsight")
st.sidebar.caption("Retail Sales & Profit Analytics")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Preview", "Product Analytics"],
)


# ===============================
# HOME PAGE
# ===============================
if page == "Home":
    st.markdown(
        """
        <h1 style="margin-bottom:0;">RetailInsight Dashboard</h1>
        <h4 style="margin-top:2px;color:#6b7280;">AI-Powered Retail Forecasting (Light MATPFN)</h4>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            """
            RetailInsight helps you understand **sales, profit, seasonality and future demand**
            for every product in your store using a lightweight MATPFN-style forecaster.

            **Key capabilities:**
            - 3-year sales trend visualization  
            - Profit & revenue breakdown  
            - Basic seasonality insights (day-of-week patterns)  
            - 30-day demand forecast (future daily quantities)  
            - Suggested stock level based on forecast  
            """
        )

    with col2:
        st.markdown("### Quick Glance")
        metric_card("Mode", "Single-Store Local Analytics", "", "#111827")
        metric_card("Model", "Light MATPFN", "Level + Trend + Seasonality", "#1f2937")
        metric_card("Frontend", "Streamlit", "Interactive Analytics UI", "#0f766e")


# ===============================
# DATA PREVIEW PAGE
# ===============================
elif page == "Data Preview":
    st.title("📊 Dataset Preview")

    if sales_df is not None:
        st.subheader("Sales Dataset")
        st.dataframe(sales_df, use_container_width=True)

    if product_df is not None:
        st.subheader("Products Dataset")
        st.dataframe(product_df, use_container_width=True)

    if calendar_df is not None:
        st.subheader("Calendar Dataset")
        st.dataframe(calendar_df, use_container_width=True)


# ===============================
# PRODUCT ANALYTICS PAGE
# ===============================
elif page == "Product Analytics":
    st.title("📈 Product Analytics & Forecast")

    # ---------- Basic validation ----------
    if sales_df is None or product_df is None:
        st.error("Sales or Product dataset missing.")
        st.stop()

    # Ensure correct dtypes globally
    sales_df = sales_df.copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    sales_df["quantity_sold"] = pd.to_numeric(sales_df["quantity_sold"], errors="coerce")
    sales_df = sales_df.dropna(subset=["date", "quantity_sold"])

    # ---------- Product search + selection ----------
    prod_info_min = product_df[["product_id", "product_name"]].copy()
    prod_info_min["label"] = (
        prod_info_min["product_id"].astype(str) + " - " + prod_info_min["product_name"].astype(str)
    )

    search_text = st.text_input("Search Product (by ID or Name)", "")

    if search_text:
        mask = prod_info_min["label"].str.contains(search_text, case=False, na=False)
        options = prod_info_min[mask]
    else:
        options = prod_info_min

    if options.empty:
        st.warning("No product matched your search.")
        st.stop()

    selected_label = st.selectbox(
        "Select Product",
        options["label"].tolist()
    )

    selected_product_id = selected_label.split(" - ")[0]

    # Filter data for selected product
    p_df = sales_df[sales_df["product_id"] == selected_product_id].copy()
    p_df = p_df.sort_values("date")

    if p_df.empty:
        st.warning("No sales history available for this product.")
        st.stop()

    # ---------- Product meta ----------
    prod_row = product_df[product_df["product_id"] == selected_product_id].iloc[0]

    product_name = prod_row.get("product_name", "Unknown")
    category = prod_row.get("category", "N/A")
    cost = float(prod_row.get("cost", 0))
    margin_percent = float(prod_row.get("margin_percent", 0))

    if margin_percent < 100:
        selling_price = cost / (1 - margin_percent / 100)
    else:
        selling_price = cost  # fallback

    st.markdown("### 🧾 Product Information")

    top_cols = st.columns(4)
    with top_cols[0]:
        metric_card("Product ID", selected_product_id, "", "#111827")
    with top_cols[1]:
        metric_card("Name", product_name, "", "#1f2937")
    with top_cols[2]:
        metric_card("Category", category, "", "#1f2937")
    with top_cols[3]:
        metric_card("Margin %", f"{margin_percent:.1f}%", f"Selling: ₹{selling_price:.2f}", "#1f2937")

    # ---------- Aggregate monthly ----------
    p_df["month"] = p_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        p_df.groupby("month")["quantity_sold"]
        .sum()
        .reset_index()
        .sort_values("month")
    )

    if monthly.empty:
        st.warning("Not enough data to analyze this product.")
        st.stop()

    # ---------- Profit & revenue ----------
    p_df["revenue"] = selling_price * p_df["quantity_sold"]
    p_df["profit"] = (selling_price - cost) * p_df["quantity_sold"]

    monthly_profit = p_df.groupby("month")["profit"].sum().reset_index()
    monthly_revenue = p_df.groupby("month")["revenue"].sum().reset_index()

    total_revenue = monthly_revenue["revenue"].sum()
    total_profit = monthly_profit["profit"].sum()
    avg_margin = round((total_profit / total_revenue) * 100, 2) if total_revenue else 0

    # ---------- KPIs ----------
    latest = monthly["quantity_sold"].iloc[-1]
    previous = monthly["quantity_sold"].iloc[-2] if len(monthly) > 1 else 0
    growth = latest - previous
    growth_pct = (growth / previous * 100) if previous > 0 else 0

    seasonality_score = round(
        monthly["quantity_sold"].std() / monthly["quantity_sold"].mean(), 2
    ) if monthly["quantity_sold"].mean() != 0 else 0

    peak_row = monthly.loc[monthly["quantity_sold"].idxmax()]
    low_row = monthly.loc[monthly["quantity_sold"].idxmin()]

    last_profit = monthly_profit["profit"].iloc[-1]
    prev_profit = monthly_profit["profit"].iloc[-2] if len(monthly_profit) > 1 else 0
    profit_growth = last_profit - prev_profit

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📉 Sales & Trend",
        "💰 Profit & Revenue",
        "🤖 Forecast & Stock",
        "📤 Export",
        "📆 3-Year Comparison"
    ]
)


    # ======== TAB 1: SALES & TREND ========
    with tab1:
        st.subheader("Sales Trend (Monthly)")

        sales_chart_df = monthly.rename(columns={"quantity_sold": "Sales"})
        line_chart = (
            alt.Chart(sales_chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("Sales:Q", title="Quantity Sold"),
                tooltip=["month:T", "Sales:Q"],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(line_chart, use_container_width=True)

        st.markdown("#### 📊 Sales KPIs")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Last Month Sales", int(latest), "units", "#0f766e")
        with col2:
            metric_card("MoM Growth", f"{growth:+}", f"{growth_pct:+.1f}% vs prev", "#0f766e")
        with col3:
            metric_card("Seasonality Score", seasonality_score, "0 = stable, >1 = high", "#0f766e")
        with col4:
            metric_card(
                "Peak Month",
                peak_row["month"].strftime("%b %Y"),
                f"Low: {low_row['month'].strftime('%b %Y')}",
                "#0f766e",
            )

        trend_word = "increasing" if growth > 0 else "decreasing" if growth < 0 else "stable"
        st.markdown("#### 🧠 Sales Insight")
        st.write(
            f"""
            - Sales for **{product_name}** are currently **{trend_word}** with a month-over-month change of **{growth:+} units ({growth_pct:+.1f}%)**.  
            - Seasonality score of **{seasonality_score}** suggests **{"strong seasonality" if seasonality_score > 1 else "low to moderate seasonality"}**.  
            - Best month so far: **{peak_row["month"].strftime("%b %Y")}**, weakest month: **{low_row["month"].strftime("%b %Y")}**.
            """
        )

    # ======== TAB 2: PROFIT & REVENUE ========
    with tab2:
        st.subheader("Profit & Revenue Overview")

        colP1, colP2, colP3, colP4 = st.columns(4)
        with colP1:
            metric_card("Total Revenue (All Time)", f"₹{int(total_revenue):,}", "", "#1d4ed8")
        with colP2:
            metric_card("Total Profit (All Time)", f"₹{int(total_profit):,}", "", "#1d4ed8")
        with colP3:
            metric_card("Avg Profit Margin", f"{avg_margin:.1f}%", "", "#1d4ed8")
        with colP4:
            metric_card("Profit Δ Last vs Prev", f"{int(profit_growth):+}", "", "#1d4ed8")

        st.markdown("#### 📈 Monthly Profit Trend")
        profit_chart_df = monthly_profit.rename(columns={"profit": "Profit"})
        profit_chart = (
            alt.Chart(profit_chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("Profit:Q", title="Profit (₹)"),
                tooltip=["month:T", "Profit:Q"],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(profit_chart, use_container_width=True)

        profit_word = "improving" if profit_growth > 0 else "shrinking" if profit_growth < 0 else "flat"
        st.markdown("#### 🧠 Profit Insight")
        st.write(
            f"""
            - Profit is **{profit_word}**, with last month changing by **₹{int(profit_growth):,}** compared to previous month.  
            - Average profit margin is **{avg_margin:.1f}%**, based on total revenue **₹{int(total_revenue):,}** and profit **₹{int(total_profit):,}**.
            """
        )

    # ======== TAB 3: FORECAST & STOCK ========
    with tab3:
        st.subheader("MATPFN Forecast (Next 30 Days)")

        forecast = []

        if st.button("🚀 Generate MATPFN Forecast"):
            with st.spinner("🔮 Calling Lightweight MATPFN Backend..."):
                try:
                    resp = requests.get(
                        f"{BACKEND_URL}/predict/{selected_product_id}?horizon=30",
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        forecast = data.get("forecast", [])
                    else:
                        st.error(f"Backend error: {resp.status_code}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if forecast:
            last_date = p_df["date"].max()
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(forecast), freq="D")

            hist_df = (
                p_df.groupby("date")["quantity_sold"]
                .sum()
                .reset_index()
                .rename(columns={"quantity_sold": "value"})
            )
            hist_df["type"] = "Actual"

            fc_df = pd.DataFrame({"date": future_dates, "value": forecast})
            fc_df["type"] = "Forecast"

            chart_df = pd.concat([hist_df.tail(120), fc_df], ignore_index=True)

            chart = (
                alt.Chart(chart_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("value:Q", title="Quantity"),
                    color=alt.Color("type:N", title="Series"),
                    tooltip=["date:T", "value:Q", "type:N"],
                )
                .properties(height=330)
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True)

            # Store average comparison (latest month across store)
            store_monthly = (
                sales_df.assign(month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
                .groupby("month")["quantity_sold"]
                .mean()
                .reset_index()
            )
            latest_store_avg = int(store_monthly["quantity_sold"].iloc[-1]) if not store_monthly.empty else 0

            st.markdown("#### 📦 Stock Recommendation")

            forecast_total = int(sum(forecast))
            recommended_stock = int(forecast_total * 1.15)  # 15% buffer

            colSR1, colSR2, colSR3 = st.columns(3)
            with colSR1:
                metric_card("Forecast Demand (30d)", forecast_total, "units", "#7c3aed")
            with colSR2:
                metric_card("Suggested Stock (30d + 15%)", recommended_stock, "units", "#7c3aed")
            with colSR3:
                metric_card("Store Avg Last Month", latest_store_avg, "units / product", "#7c3aed")

            st.write(
                f"""
                - Estimated demand for next **30 days**: **{forecast_total} units**  
                - Recommended stock (with ~15% safety buffer): **{recommended_stock} units**  
                - If current inventory is significantly below this level, consider **reordering**.  
                """
            )
        else:
            st.info("Click **Generate MATPFN Forecast** to see future demand.")

    # ======== TAB 4: EXPORT ========
    with tab4:
        st.subheader("Export Analytics Data")

        export_df = monthly.copy()
        export_df = export_df.merge(monthly_revenue, on="month", how="left")
        export_df = export_df.merge(monthly_profit, on="month", how="left", suffixes=("_qty", "_profit"))
        export_df.rename(
            columns={
                "quantity_sold": "monthly_qty",
                "revenue": "monthly_revenue",
                "profit": "monthly_profit",
            },
            inplace=True,
        )

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Monthly Analytics CSV",
            data=csv_bytes,
            file_name=f"{selected_product_id}_analytics.csv",
            mime="text/csv",
        )

       # ======== TAB 5: 3-YEAR COMPARISON ========
    with tab5:
        st.subheader("📆 Yearly Sales Comparison (Jan–Dec)")

        # Extract month and year
        p_df["year"] = p_df["date"].dt.year
        p_df["month"] = p_df["date"].dt.month

        # Aggregate monthly totals for each year
        monthly_year = (
            p_df.groupby(["year", "month"])["quantity_sold"]
            .sum()
            .reset_index()
            .sort_values(["year", "month"])
        )

        if monthly_year.empty:
            st.warning("No multi-year data available for this product.")
            st.stop()

        # Convert month numbers to names
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_year["month_name"] = monthly_year["month"].apply(lambda x: month_order[x-1])

        # Build chart
        chart = (
            alt.Chart(monthly_year)
            .mark_line(point=True)
            .encode(
                x=alt.X("month_name:N", title="Month", sort=month_order),
                y=alt.Y("quantity_sold:Q", title="Quantity Sold"),
                color=alt.Color("year:N", title="Year"),
                tooltip=["year:N", "month_name:N", "quantity_sold:Q"],
            )
            .properties(height=350)
            .interactive()
        )
 
        st.altair_chart(chart, use_container_width=True)

        # ===== Legend Under Chart =====
        st.markdown("### 🟦 Year Legend")

        years = sorted(monthly_year["year"].unique())

        color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c",
            "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]

        # Render legend rows
        for idx, yr in enumerate(years):
            color = color_palette[idx % len(color_palette)]
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <div style="width:14px;height:14px;background:{color};
                                border-radius:3px;margin-right:8px;"></div>
                    <span style="font-size:16px;">{yr}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
        # ===== Yearly Sales Summary (ONE TIME ONLY) =====
        st.markdown("### 📊 Yearly Total Sales")
    
        yearly_totals = (
                p_df.groupby("year")["quantity_sold"]
            .sum()
            .reset_index()
            .sort_values("year")
        )

        for _, row in yearly_totals.iterrows():
            st.write(f"- **{row['year']}** → **{row['quantity_sold']} units**")
