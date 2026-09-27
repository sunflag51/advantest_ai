# ============================================================
# アドバンテスト AI売買システム
# main.py
#
# 現在の機能
# 1. アドバンテスト株価取得
# 2. 市場環境データ取得
# 3. テクニカル指標計算
# 4. 株価チャート表示
# 5. テクニカル指標表示
#
# ※まだAIによる売買判断は行いません
# ※実際の売買注文も行いません
# ============================================================


import streamlit as st
import pandas as pd


# ============================================================
# データ取得
# ============================================================

from data.stock_data import get_stock_data

from data.market_data import (
    get_all_market_data,
    get_latest_market_values
)


# ============================================================
# テクニカル分析
# ============================================================

from indicators.technical import (
    add_all_indicators,
    get_latest_indicators
)


# ============================================================
# 設定
# ============================================================

STOCK_CODE = "6857.T"

STOCK_NAME = "アドバンテスト"


# ============================================================
# Streamlit設定
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
    "株価・市場環境・テクニカル分析"
)

st.divider()


# ============================================================
# システム状態
# ============================================================

st.subheader(
    "システム状態"
)


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

    st.success(
        "✅ テクニカル分析"
    )


with col4:

    st.warning(
        "⚠️ AI売買判断：未実装"
    )


st.divider()


# ============================================================
# データ取得ボタン
# ============================================================

st.subheader(
    "📥 データ取得・分析"
)


run_analysis = st.button(
    "🚀 株価・市場・テクニカル分析を実行",
    type="primary"
)


# ============================================================
# 分析実行
# ============================================================

if run_analysis:

    # ========================================================
    # ① アドバンテスト株価
    # ========================================================

    st.header(
        "① アドバンテスト株価"
    )


    try:

        with st.spinner(
            "アドバンテストの株価を取得しています..."
        ):

            stock_data = get_stock_data(
                stock_code=STOCK_CODE,
                period="5y",
                interval="1d"
            )


        if stock_data is None or stock_data.empty:

            st.error(
                "アドバンテストの株価データを取得できませんでした。"
            )

            st.stop()


        st.success(
            "✅ 株価データ取得成功"
        )


        # ----------------------------------------------------
        # 最新データ
        # ----------------------------------------------------

        latest = stock_data.iloc[-1]


        latest_close = float(
            latest["Close"]
        )

        latest_open = float(
            latest["Open"]
        )

        latest_high = float(
            latest["High"]
        )

        latest_low = float(
            latest["Low"]
        )

        latest_volume = int(
            latest["Volume"]
        )


        # ----------------------------------------------------
        # 株価表示
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:

            st.metric(
                "終値",
                f"{latest_close:,.0f} 円"
            )


        with col2:

            st.metric(
                "始値",
                f"{latest_open:,.0f} 円"
            )


        with col3:

            st.metric(
                "高値",
                f"{latest_high:,.0f} 円"
            )


        with col4:

            st.metric(
                "安値",
                f"{latest_low:,.0f} 円"
            )


        with col5:

            st.metric(
                "出来高",
                f"{latest_volume:,}"
            )


        # ----------------------------------------------------
        # 株価チャート
        # ----------------------------------------------------

        st.subheader(
            "アドバンテスト終値"
        )


        st.line_chart(
            stock_data[
                ["Close"]
            ]
        )


        # ====================================================
        # ② 市場環境
        # ====================================================

        st.divider()

        st.header(
            "② 市場環境"
        )


        try:

            with st.spinner(
                "日経平均・NASDAQ・SOXX・ドル円を取得しています..."
            ):

                market_data = get_all_market_data(
                    period="5y",
                    interval="1d"
                )


                latest_market = get_latest_market_values(
                    period="5d"
                )


            st.success(
                "✅ 市場データ取得成功"
            )


            # ------------------------------------------------
            # 市場最新値
            # ------------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                nikkei = latest_market.get(
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

                    st.warning(
                        "日経平均データなし"
                    )


            with col2:

                nasdaq = latest_market.get(
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

                    st.warning(
                        "NASDAQデータなし"
                    )


            col1, col2 = st.columns(2)


            with col1:

                soxx = latest_market.get(
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

                    st.warning(
                        "SOXXデータなし"
                    )


            with col2:

                usd_jpy = latest_market.get(
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

                    st.warning(
                        "ドル円データなし"
                    )


            # ------------------------------------------------
            # 市場チャート
            # ------------------------------------------------

            st.subheader(
                "市場データ"
            )


            market_chart = pd.DataFrame()


            for market_name in [
                "日経平均",
                "NASDAQ",
                "SOXX",
                "ドル円"
            ]:

                if market_name not in market_data:

                    continue


                market_df = market_data[
                    market_name
                ]


                if (
                    market_df is None
                    or market_df.empty
                ):

                    continue


                close_df = market_df[
                    ["Close"]
                ].copy()


                close_df.rename(
                    columns={
                        "Close": market_name
                    },
                    inplace=True
                )


                if market_chart.empty:

                    market_chart = close_df

                else:

                    market_chart = market_chart.join(
                        close_df,
                        how="outer"
                    )


            if not market_chart.empty:

                market_chart = (
                    market_chart.ffill()
                )


                st.line_chart(
                    market_chart
                )


        except Exception as e:

            st.error(
                "市場データ取得中にエラーが発生しました。"
            )

            st.exception(e)


        # ====================================================
        # ③ テクニカル指標
        # ====================================================

        st.divider()

        st.header(
            "③ テクニカル分析"
        )


        try:

            with st.spinner(
                "テクニカル指標を計算しています..."
            ):

                technical_data = add_all_indicators(
                    stock_data
                )


            st.success(
                "✅ テクニカル指標計算成功"
            )


            # ------------------------------------------------
            # 最新テクニカルデータ
            # ------------------------------------------------

            latest_indicators = (
                get_latest_indicators(
                    technical_data
                )
            )


            # ------------------------------------------------
            # 移動平均
            # ------------------------------------------------

            st.subheader(
                "📊 移動平均"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                sma5 = latest_indicators.get(
                    "SMA_5"
                )


                if pd.notna(sma5):

                    st.metric(
                        "5日移動平均",
                        f"{sma5:,.0f} 円"
                    )


            with col2:

                sma25 = latest_indicators.get(
                    "SMA_25"
                )


                if pd.notna(sma25):

                    st.metric(
                        "25日移動平均",
                        f"{sma25:,.0f} 円"
                    )


            with col3:

                sma75 = latest_indicators.get(
                    "SMA_75"
                )


                if pd.notna(sma75):

                    st.metric(
                        "75日移動平均",
                        f"{sma75:,.0f} 円"
                    )


            # ------------------------------------------------
            # 株価＋移動平均チャート
            # ------------------------------------------------

            ma_chart = technical_data[
                [
                    "Close",
                    "SMA_5",
                    "SMA_25",
                    "SMA_75"
                ]
            ].copy()


            st.line_chart(
                ma_chart
            )


            # ------------------------------------------------
            # RSI
            # ------------------------------------------------

            st.subheader(
                "📈 RSI"
            )


            rsi = latest_indicators.get(
                "RSI_14"
            )


            if pd.notna(rsi):

                st.metric(
                    "RSI（14日）",
                    f"{rsi:.2f}"
                )


                rsi_chart = technical_data[
                    ["RSI_14"]
                ].copy()


                st.line_chart(
                    rsi_chart
                )


            # ------------------------------------------------
            # MACD
            # ------------------------------------------------

            st.subheader(
                "📉 MACD"
            )


            macd_col1, macd_col2, macd_col3 = (
                st.columns(3)
            )


            macd = latest_indicators.get(
                "MACD"
            )


            macd_signal = latest_indicators.get(
                "MACD_Signal"
            )


            macd_histogram = latest_indicators.get(
                "MACD_Histogram"
            )


            with macd_col1:

                if pd.notna(macd):

                    st.metric(
                        "MACD",
                        f"{macd:.2f}"
                    )


            with macd_col2:

                if pd.notna(macd_signal):

                    st.metric(
                        "シグナル",
                        f"{macd_signal:.2f}"
                    )


            with macd_col3:

                if pd.notna(macd_histogram):

                    st.metric(
                        "ヒストグラム",
                        f"{macd_histogram:.2f}"
                    )


            macd_chart = technical_data[
                [
                    "MACD",
                    "MACD_Signal"
                ]
            ].copy()


            st.line_chart(
                macd_chart
            )


            # ------------------------------------------------
            # ボリンジャーバンド
            # ------------------------------------------------

            st.subheader(
                "📐 ボリンジャーバンド"
            )


            bollinger_chart = technical_data[
                [
                    "Close",
                    "BB_Upper",
                    "BB_Middle",
                    "BB_Lower"
                ]
            ].copy()


            st.line_chart(
                bollinger_chart
            )


            # ------------------------------------------------
            # ATR・出来高・ボラティリティ
            # ------------------------------------------------

            st.subheader(
                "📊 その他の指標"
            )


            col1, col2, col3 = st.columns(3)


            atr = latest_indicators.get(
                "ATR_14"
            )


            volume_ratio = latest_indicators.get(
                "Volume_Ratio"
            )


            volatility = latest_indicators.get(
                "Volatility_20D"
            )


            with col1:

                if pd.notna(atr):

                    st.metric(
                        "ATR（14日）",
                        f"{atr:,.2f}"
                    )


            with col2:

                if pd.notna(volume_ratio):

                    st.metric(
                        "出来高比率",
                        f"{volume_ratio:.2f}"
                    )


            with col3:

                if pd.notna(volatility):

                    st.metric(
                        "20日ボラティリティ",
                        f"{volatility * 100:.2f}%"
                    )


            # ------------------------------------------------
            # リターン
            # ------------------------------------------------

            st.subheader(
                "📊 価格変化率"
            )


            col1, col2, col3, col4 = st.columns(4)


            return_1d = latest_indicators.get(
                "Return_1D"
            )


            return_5d = latest_indicators.get(
                "Return_5D"
            )


            return_20d = latest_indicators.get(
                "Return_20D"
            )


            return_60d = latest_indicators.get(
                "Return_60D"
            )


            with col1:

                if pd.notna(return_1d):

                    st.metric(
                        "1日",
                        f"{return_1d * 100:.2f}%"
                    )


            with col2:

                if pd.notna(return_5d):

                    st.metric(
                        "5日",
                        f"{return_5d * 100:.2f}%"
                    )


            with col3:

                if pd.notna(return_20d):

                    st.metric(
                        "20日",
                        f"{return_20d * 100:.2f}%"
                    )


            with col4:

                if pd.notna(return_60d):

                    st.metric(
                        "60日",
                        f"{return_60d * 100:.2f}%"
                    )


            # =================================================
            # 詳細データ
            # =================================================

            st.subheader(
                "📋 テクニカルデータ詳細"
            )


            with st.expander(
                "最新20営業日のテクニカルデータを見る"
            ):

                st.dataframe(
                    technical_data.tail(20),
                    use_container_width=True
                )


            # =================================================
            # 完了
            # =================================================

            st.success(
                "🎉 テクニカル分析まで正常に完了しました。"
            )


        except Exception as e:

            st.error(
                "❌ テクニカル分析中にエラーが発生しました。"
            )

            st.exception(e)


    except Exception as e:

        st.error(
            "❌ 株価データ取得中にエラーが発生しました。"
        )

        st.exception(e)


# ============================================================
# 開発状況
# ============================================================

st.divider()

st.subheader(
    "🚧 開発状況"
)


development_status = {

    "Streamlit画面":
        "✅ 完了",

    "アドバンテスト株価取得":
        "✅ 完了",

    "市場データ取得":
        "✅ 完了",

    "ニュース取得":
        "✅ 完了",

    "テクニカル分析":
        "✅ 今回追加",

    "AI予測モデル":
        "🔵 次の段階",

    "エントリー判断":
        "🔵 未実装",

    "売却判断":
        "🔵 未実装",

    "リスク管理":
        "🔵 未実装",

    "バックテスト":
        "🔵 未実装",

    "仮想売買":
        "🔵 未実装",

    "自動売買":
        "⚪ OFF"
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
    "現在はデータ取得・テクニカル分析の開発段階です。"
    "実際の売買注文は行いません。"
)
