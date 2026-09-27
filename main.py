# ============================================================
# アドバンテスト AI売買システム
# main.py
#
# 現在の機能
# 1. アドバンテスト株価取得
# 2. 日経平均取得
# 3. NASDAQ取得
# 4. SOXX取得
# 5. ドル円取得
# 6. 株価チャート表示
# 7. 市場データ表示
#
# ※現在は分析のみ
# ※実際の売買注文は行いません
# ============================================================


import streamlit as st
import pandas as pd

from data.stock_data import get_stock_data

from data.market_data import (
    get_all_market_data,
    get_latest_market_values
)


# ============================================================
# Streamlitページ設定
# ============================================================

st.set_page_config(
    page_title="アドバンテスト AI売買システム",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# タイトル
# ============================================================

st.title(
    "📈 アドバンテスト AI売買システム"
)

st.write(
    "株価・市場環境データ取得システム"
)

st.divider()


# ============================================================
# システム状態
# ============================================================

st.subheader("システム状態")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.success(
        "✅ 株価データ"
    )


with col2:

    st.success(
        "✅ 市場データ"
    )


with col3:

    st.info(
        "🔵 AI分析：準備中"
    )


with col4:

    st.warning(
        "⚠️ 自動売買：OFF"
    )


st.divider()


# ============================================================
# データ取得ボタン
# ============================================================

st.subheader(
    "📥 データ取得"
)


if st.button(
    "株価・市場データを取得",
    type="primary"
):

    # ========================================================
    # アドバンテスト株価取得
    # ========================================================

    st.write(
        "### ① アドバンテスト株価"
    )

    try:

        with st.spinner(
            "アドバンテストの株価を取得しています..."
        ):

            stock_data = get_stock_data(
                stock_code="6857.T",
                period="5y",
                interval="1d"
            )


        if (
            stock_data is not None
            and not stock_data.empty
        ):

            st.success(
                "✅ アドバンテスト株価取得成功"
            )


            # ------------------------------------------------
            # 最新株価
            # ------------------------------------------------

            latest = stock_data.iloc[-1]


            close_price = float(
                latest["Close"]
            )


            open_price = float(
                latest["Open"]
            )


            high_price = float(
                latest["High"]
            )


            low_price = float(
                latest["Low"]
            )


            volume = int(
                latest["Volume"]
            )


            # ------------------------------------------------
            # 株価表示
            # ------------------------------------------------

            col1, col2, col3, col4, col5 = st.columns(5)


            with col1:

                st.metric(
                    "終値",
                    f"{close_price:,.0f} 円"
                )


            with col2:

                st.metric(
                    "始値",
                    f"{open_price:,.0f} 円"
                )


            with col3:

                st.metric(
                    "高値",
                    f"{high_price:,.0f} 円"
                )


            with col4:

                st.metric(
                    "安値",
                    f"{low_price:,.0f} 円"
                )


            with col5:

                st.metric(
                    "出来高",
                    f"{volume:,}"
                )


            # ------------------------------------------------
            # 株価チャート
            # ------------------------------------------------

            st.write(
                "#### アドバンテスト株価"
            )


            chart_data = stock_data[
                ["Close"]
            ].copy()


            st.line_chart(
                chart_data
            )


            # ------------------------------------------------
            # 最新データ
            # ------------------------------------------------

            with st.expander(
                "アドバンテスト最新20日分"
            ):

                st.dataframe(
                    stock_data.tail(20),
                    use_container_width=True
                )


        else:

            st.error(
                "❌ アドバンテスト株価を取得できませんでした。"
            )


    except Exception as e:

        st.error(
            "❌ アドバンテスト株価取得エラー"
        )

        st.exception(e)


    # ========================================================
    # 市場データ取得
    # ========================================================

    st.divider()

    st.write(
        "### ② 市場環境"
    )


    try:

        with st.spinner(
            "市場データを取得しています..."
        ):

            market_data = get_all_market_data(
                period="5y",
                interval="1d"
            )


        # ----------------------------------------------------
        # エラー確認
        # ----------------------------------------------------

        market_errors = market_data.get(
            "_errors",
            {}
        )


        if market_errors:

            st.warning(
                "一部の市場データを取得できませんでした。"
            )


        # ----------------------------------------------------
        # 最新値取得
        # ----------------------------------------------------

        latest_values = get_latest_market_values(
            period="5d"
        )


        # ----------------------------------------------------
        # 市場データ表示
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # 日経平均
        # ----------------------------------------------------

        with col1:

            nikkei = latest_values.get(
                "日経平均"
            )


            if (
                nikkei
                and nikkei.get("close") is not None
            ):

                st.metric(
                    "🇯🇵 日経平均",
                    f"{nikkei['close']:,.2f}"
                )

            else:

                st.error(
                    "日経平均を取得できませんでした"
                )


        # ----------------------------------------------------
        # NASDAQ
        # ----------------------------------------------------

        with col2:

            nasdaq = latest_values.get(
                "NASDAQ"
            )


            if (
                nasdaq
                and nasdaq.get("close") is not None
            ):

                st.metric(
                    "🇺🇸 NASDAQ",
                    f"{nasdaq['close']:,.2f}"
                )

            else:

                st.error(
                    "NASDAQを取得できませんでした"
                )


        # ----------------------------------------------------
        # SOXX
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            soxx = latest_values.get(
                "SOXX"
            )


            if (
                soxx
                and soxx.get("close") is not None
            ):

                st.metric(
                    "💻 SOXX",
                    f"{soxx['close']:,.2f}"
                )

            else:

                st.error(
                    "SOXXを取得できませんでした"
                )


        # ----------------------------------------------------
        # ドル円
        # ----------------------------------------------------

        with col2:

            usd_jpy = latest_values.get(
                "ドル円"
            )


            if (
                usd_jpy
                and usd_jpy.get("close") is not None
            ):

                st.metric(
                    "💴 ドル円",
                    f"{usd_jpy['close']:,.2f} 円"
                )

            else:

                st.error(
                    "ドル円を取得できませんでした"
                )


        # ====================================================
        # 市場データ一覧
        # ====================================================

        st.write(
            "#### 市場データ一覧"
        )


        market_rows = []


        for market_name, info in latest_values.items():

            if info is None:

                continue


            if info.get("close") is None:

                continue


            market_rows.append(
                {
                    "市場": market_name,
                    "ティッカー": info["ticker"],
                    "最新値": info["close"],
                    "データ日": info["date"]
                }
            )


        if market_rows:

            market_table = pd.DataFrame(
                market_rows
            )


            st.dataframe(
                market_table,
                use_container_width=True
            )


        # ====================================================
        # 市場チャート
        # ====================================================

        st.write(
            "#### 市場データチャート"
        )


        chart_market = pd.DataFrame()


        for market_name in [
            "日経平均",
            "NASDAQ",
            "SOXX",
            "ドル円"
        ]:

            if market_name not in market_data:

                continue


            data = market_data[
                market_name
            ]


            if data is None or data.empty:

                continue


            close_data = data[
                ["Close"]
            ].copy()


            close_data.rename(
                columns={
                    "Close": market_name
                },
                inplace=True
            )


            if chart_market.empty:

                chart_market = close_data

            else:

                chart_market = chart_market.join(
                    close_data,
                    how="outer"
                )


        if not chart_market.empty:

            chart_market = chart_market.ffill()


            st.line_chart(
                chart_market
            )


        # ====================================================
        # 成功表示
        # ====================================================

        st.success(
            "🎉 市場データの取得が完了しました。"
        )


    except Exception as e:

        st.error(
            "❌ 市場データ取得エラー"
        )

        st.exception(e)


# ============================================================
# 現在の開発状況
# ============================================================

st.divider()

st.subheader(
    "🚧 開発状況"
)


development_status = {
    "Streamlit画面": "完了",
    "アドバンテスト株価取得": "完了",
    "市場データ取得": "今回追加",
    "テクニカル分析": "次の段階",
    "AI予測": "未実装",
    "エントリー判断": "未実装",
    "売却判断": "未実装",
    "リスク管理": "未実装",
    "バックテスト": "未実装",
    "仮想売買": "未実装",
    "自動売買": "OFF"
}


status_df = pd.DataFrame(
    list(
        development_status.items()
    ),
    columns=[
        "機能",
        "状態"
    ]
)


st.dataframe(
    status_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 注意事項
# ============================================================

st.divider()

st.caption(
    "現在はデータ取得・分析システムの開発段階です。"
    "実際の売買注文は行いません。"
)
