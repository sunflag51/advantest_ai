# ============================================================
# アドバンテスト AI売買分析システム
# main.py 完全版
#
# ① 株価データ
# ② 市場データ
# ③ テクニカル分析
# ④ AI特徴量
# ⑤ AIモデル学習
# ⑥ 最新AI予測
# ⑦ 特徴量重要度
# ⑧ 固定80/20検証
# ⑨ ウォークフォワード検証
# ⑩ 旧方式売買バックテスト
# ⑪ AIエントリー判断
# ⑫ リスク・資金管理
# ⑬ 本格戦略バックテスト
# ============================================================


import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# データ
# ============================================================

from data.stock_data import get_stock_data

from data.market_data import (
    get_all_market_data,
    get_latest_market_values,
)


# ============================================================
# テクニカル
# ============================================================

from indicators.technical import (
    add_all_indicators,
    get_latest_indicators,
)


# ============================================================
# AI
# ============================================================

from ai.features import (
    build_ai_features,
    get_ai_feature_columns,
    get_feature_summary,
    get_available_market_features,
)

from ai.model import StockPredictionModel


# ============================================================
# バックテスト
# ============================================================

from backtest.engine import WalkForwardBacktest
from backtest.report import BacktestReport
from backtest.trading_engine import TradingBacktestEngine


# ============================================================
# 戦略
# ============================================================

from strategy.entry import EntryStrategy
from strategy.risk import RiskManager


# ============================================================
# 基本設定
# ============================================================

STOCK_CODE = "6857.T"
STOCK_NAME = "アドバンテスト"

INITIAL_CAPITAL = 1_000_000


# ============================================================
# Streamlit設定
# ============================================================

st.set_page_config(
    page_title="アドバンテスト AI売買分析",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# 共通関数
# ============================================================

def safe_float(value, default=None):
    """
    float / numpy数値 / Series / DataFrame / 配列などから
    安全に最新の数値1個を取得する。
    """

    try:

        if value is None:
            return default

        # --------------------------------------------
        # DataFrame
        # --------------------------------------------

        if isinstance(value, pd.DataFrame):

            if value.empty:
                return default

            numeric = (
                value
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .stack()
                .dropna()
            )

            if numeric.empty:
                return default

            return float(
                numeric.iloc[-1]
            )

        # --------------------------------------------
        # Series
        # --------------------------------------------

        if isinstance(value, pd.Series):

            if value.empty:
                return default

            numeric = (
                pd.to_numeric(
                    value,
                    errors="coerce",
                )
                .dropna()
            )

            if numeric.empty:
                return default

            return float(
                numeric.iloc[-1]
            )

        # --------------------------------------------
        # numpy配列 / list / tuple
        # --------------------------------------------

        if isinstance(
            value,
            (
                np.ndarray,
                list,
                tuple,
            ),
        ):

            array = np.asarray(
                value
            ).reshape(-1)

            if len(array) == 0:
                return default

            numeric = pd.to_numeric(
                pd.Series(array),
                errors="coerce",
            ).dropna()

            if numeric.empty:
                return default

            return float(
                numeric.iloc[-1]
            )

        # --------------------------------------------
        # 通常の数値
        # --------------------------------------------

        result = float(
            value
        )

        if not np.isfinite(
            result
        ):
            return default

        return result

    except Exception:
        return default


def safe_metric_number(
    value,
    decimals=2,
    suffix="",
    default_text="取得なし",
):
    """
    st.metric用の安全な文字列を作る。
    """

    number = safe_float(
        value
    )

    if number is None:
        return default_text

    return (
        f"{number:,.{decimals}f}"
        f"{suffix}"
    )


# ============================================================
# タイトル
# ============================================================

st.title(
    "📈 アドバンテスト AI売買分析システム"
)

st.caption(
    "株価・市場・テクニカル・AI・"
    "エントリー・リスク管理・"
    "本格戦略バックテストを統合"
)

st.warning(
    "このシステムは研究・検証用です。"
    "実際の売買注文は行いません。"
)


# ============================================================
# 実行ボタン
# ============================================================

run_button = st.button(
    "🚀 株価・市場・AI・戦略分析を実行",
    type="primary",
    use_container_width=True,
)


# ============================================================
# メイン処理
# ============================================================

if run_button:

    # ========================================================
    # ① 株価データ
    # ========================================================

    st.divider()

    st.header(
        "① 📊 アドバンテスト株価データ"
    )

    try:

        stock_data = get_stock_data(
            stock_code=STOCK_CODE,
            period="5y",
            interval="1d",
        )

        if (
            stock_data is None
            or stock_data.empty
        ):

            st.error(
                "株価データを取得できませんでした。"
            )

            st.stop()


        latest_stock = (
            stock_data.iloc[-1]
        )

        close_price = safe_float(
            latest_stock.get("Close")
        )

        open_price = safe_float(
            latest_stock.get("Open")
        )

        high_price = safe_float(
            latest_stock.get("High")
        )

        low_price = safe_float(
            latest_stock.get("Low")
        )

        volume = safe_float(
            latest_stock.get("Volume")
        )


        if close_price is None:

            st.error(
                "最新終値を取得できませんでした。"
            )

            st.stop()


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "終値",
                safe_metric_number(
                    close_price,
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col2:

            st.metric(
                "始値",
                safe_metric_number(
                    open_price,
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col3:

            st.metric(
                "出来高",
                safe_metric_number(
                    volume,
                    decimals=0,
                ),
            )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "高値",
                safe_metric_number(
                    high_price,
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col2:

            st.metric(
                "安値",
                safe_metric_number(
                    low_price,
                    decimals=0,
                    suffix=" 円",
                ),
            )


        st.subheader(
            "株価チャート"
        )

        st.line_chart(
            stock_data[
                ["Close"]
            ].tail(250)
        )

        st.success(
            "✅ 株価データ取得完了"
        )

    except Exception as e:

        st.error(
            "❌ 株価データ取得中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ② 市場データ
    # ========================================================

    st.divider()

    st.header(
        "② 🌍 市場データ"
    )

    try:

        market_data = (
            get_all_market_data(
                period="5y",
                interval="1d",
            )
        )

        latest_market = (
            get_latest_market_values(
                period="5d"
            )
        )


        market_names = [
            "日経平均",
            "NASDAQ",
            "SOXX",
            "ドル円",
        ]

        market_columns = st.columns(
            len(market_names)
        )


        # ====================================================
        # 修正版：
        # Series / DataFrame / 配列でも安全に表示
        # ====================================================

        for i, name in enumerate(
            market_names
        ):

            value = None

            if isinstance(
                latest_market,
                dict,
            ):

                value = (
                    latest_market.get(
                        name
                    )
                )

            elif isinstance(
                latest_market,
                pd.Series,
            ):

                if name in latest_market.index:

                    value = (
                        latest_market.loc[
                            name
                        ]
                    )

            elif isinstance(
                latest_market,
                pd.DataFrame,
            ):

                if name in latest_market.columns:

                    value = (
                        latest_market[
                            name
                        ]
                    )


            display_value = safe_float(
                value
            )


            with market_columns[i]:

                if display_value is None:

                    st.metric(
                        name,
                        "取得なし",
                    )

                else:

                    st.metric(
                        name,
                        f"{display_value:,.2f}",
                    )


        # ====================================================
        # 市場比較チャート
        # ====================================================

        normalized_market = {}


        if isinstance(
            market_data,
            dict,
        ):

            for name, df in (
                market_data.items()
            ):

                if (
                    df is None
                    or not isinstance(
                        df,
                        pd.DataFrame,
                    )
                    or df.empty
                    or "Close"
                    not in df.columns
                ):

                    continue


                series = (
                    pd.to_numeric(
                        df["Close"],
                        errors="coerce",
                    )
                    .dropna()
                )


                if series.empty:
                    continue


                first_value = safe_float(
                    series.iloc[0]
                )


                if (
                    first_value is None
                    or first_value == 0
                ):

                    continue


                normalized_market[
                    name
                ] = (
                    series
                    / first_value
                    * 100
                )


        if normalized_market:

            market_chart = (
                pd.DataFrame(
                    normalized_market
                )
            )

            st.subheader(
                "市場比較（開始日=100）"
            )

            st.line_chart(
                market_chart.tail(250)
            )


        st.success(
            "✅ 市場データ取得完了"
        )

    except Exception as e:

        st.error(
            "❌ 市場データ取得中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ③ テクニカル分析
    # ========================================================

    st.divider()

    st.header(
        "③ 📉 テクニカル分析"
    )

    try:

        technical_data = (
            add_all_indicators(
                stock_data
            )
        )

        latest_indicators = (
            get_latest_indicators(
                technical_data
            )
        )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "SMA 5",
                safe_metric_number(
                    latest_indicators.get(
                        "SMA_5"
                    ),
                    decimals=0,
                ),
            )

        with col2:

            st.metric(
                "SMA 25",
                safe_metric_number(
                    latest_indicators.get(
                        "SMA_25"
                    ),
                    decimals=0,
                ),
            )

        with col3:

            st.metric(
                "SMA 75",
                safe_metric_number(
                    latest_indicators.get(
                        "SMA_75"
                    ),
                    decimals=0,
                ),
            )


        chart_columns = [
            column
            for column in [
                "Close",
                "SMA_5",
                "SMA_25",
                "SMA_75",
            ]
            if column
            in technical_data.columns
        ]


        if chart_columns:

            st.line_chart(
                technical_data[
                    chart_columns
                ].tail(250)
            )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "RSI 14",
                safe_metric_number(
                    latest_indicators.get(
                        "RSI_14"
                    ),
                    decimals=2,
                ),
            )

        with col2:

            st.metric(
                "MACD",
                safe_metric_number(
                    latest_indicators.get(
                        "MACD"
                    ),
                    decimals=2,
                ),
            )

        with col3:

            st.metric(
                "MACD Signal",
                safe_metric_number(
                    latest_indicators.get(
                        "MACD_Signal"
                    ),
                    decimals=2,
                ),
            )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "ATR 14",
                safe_metric_number(
                    latest_indicators.get(
                        "ATR_14"
                    ),
                    decimals=2,
                ),
            )

        with col2:

            st.metric(
                "出来高比率",
                safe_metric_number(
                    latest_indicators.get(
                        "Volume_Ratio"
                    ),
                    decimals=2,
                ),
            )

        with col3:

            st.metric(
                "20日ボラティリティ",
                safe_metric_number(
                    latest_indicators.get(
                        "Volatility_20D"
                    ),
                    decimals=2,
                ),
            )


        st.success(
            "✅ テクニカル分析完了"
        )

    except Exception as e:

        st.error(
            "❌ テクニカル分析中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ④ AI特徴量
    # ========================================================

    st.divider()

    st.header(
        "④ 🧠 AI特徴量"
    )

    try:

        ai_data = (
            build_ai_features(
                technical_data=
                    technical_data,

                market_data=
                    market_data,
            )
        )


        ai_feature_columns = (
            get_ai_feature_columns(
                ai_data
            )
        )


        feature_summary = (
            get_feature_summary(
                ai_data
            )
        )


        available_market_features = (
            get_available_market_features(
                ai_data
            )
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "AI特徴量数",
                f"{len(ai_feature_columns):,}",
            )

        with col2:

            st.metric(
                "AIデータ行数",
                f"{len(ai_data):,}",
            )


        with st.expander(
            "AI特徴量一覧を見る"
        ):

            st.write(
                ai_feature_columns
            )


        with st.expander(
            "特徴量サマリーを見る"
        ):

            st.write(
                feature_summary
            )


        with st.expander(
            "市場特徴量の取得状況を見る"
        ):

            st.write(
                available_market_features
            )


        st.success(
            "✅ AI特徴量作成完了"
        )

    except Exception as e:

        st.error(
            "❌ AI特徴量作成中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑤ AIモデル学習
    # ========================================================

    st.divider()

    st.header(
        "⑤ 🤖 AIモデル学習"
    )

    try:

        ai_model = (
            StockPredictionModel()
        )


        metrics = ai_model.train(
            data=ai_data,
            feature_columns=
                ai_feature_columns,
            train_ratio=0.8,
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "Accuracy",
                safe_metric_number(
                    metrics.get(
                        "accuracy"
                    ),
                    decimals=3,
                ),
            )


        with col2:

            st.metric(
                "Precision",
                safe_metric_number(
                    metrics.get(
                        "precision"
                    ),
                    decimals=3,
                ),
            )


        with col3:

            st.metric(
                "Recall",
                safe_metric_number(
                    metrics.get(
                        "recall"
                    ),
                    decimals=3,
                ),
            )


        with col4:

            st.metric(
                "AUC",
                safe_metric_number(
                    metrics.get(
                        "auc"
                    ),
                    decimals=3,
                    default_text="N/A",
                ),
            )


        st.success(
            "✅ AIモデル学習完了"
        )

    except Exception as e:

        st.error(
            "❌ AIモデル学習中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑥ 最新AI予測
    # ========================================================

    st.divider()

    st.header(
        "⑥ 🔮 最新AI予測"
    )

    try:

        prediction = ai_model.predict(
            ai_data,
            threshold=0.50,
        )


        probability_up = safe_float(
            prediction.get(
                "probability_up"
            )
        )


        probability_down = safe_float(
            prediction.get(
                "probability_down"
            )
        )


        if probability_up is None:

            raise ValueError(
                "AI上昇確率を取得できません。"
            )


        if probability_down is None:

            probability_down = (
                1.0
                - probability_up
            )


        predicted_class = (
            prediction.get(
                "prediction"
            )
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "上昇確率",
                f"{probability_up * 100:.1f}%",
            )

        with col2:

            st.metric(
                "下落確率",
                f"{probability_down * 100:.1f}%",
            )


        st.progress(
            float(
                min(
                    max(
                        probability_up,
                        0.0,
                    ),
                    1.0,
                )
            )
        )


        if predicted_class == 1:

            st.success(
                "AI予測：上昇方向"
            )

        else:

            st.info(
                "AI予測：下落または"
                "上昇条件未達"
            )


        st.success(
            "✅ 最新AI予測完了"
        )

    except Exception as e:

        st.error(
            "❌ AI予測中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑦ 特徴量重要度
    # ========================================================

    st.divider()

    st.header(
        "⑦ 🔍 AI特徴量重要度"
    )

    try:

        feature_importance = (
            ai_model
            .get_feature_importance()
        )


        if (
            feature_importance is not None
            and not feature_importance.empty
        ):

            st.dataframe(
                feature_importance,
                use_container_width=True,
                hide_index=True,
            )


            if (
                "feature"
                in feature_importance.columns
                and "importance"
                in feature_importance.columns
            ):

                importance_chart = (
                    feature_importance
                    .head(15)
                    .set_index(
                        "feature"
                    )[
                        "importance"
                    ]
                )

                st.bar_chart(
                    importance_chart
                )

        else:

            st.info(
                "特徴量重要度を"
                "取得できませんでした。"
            )

    except Exception as e:

        st.warning(
            "特徴量重要度の表示を"
            "スキップしました。"
        )

        st.exception(e)


    # ========================================================
    # ⑧ 固定80/20検証
    # ========================================================

    st.divider()

    st.header(
        "⑧ 🧪 固定80/20検証"
    )

    try:

        test_results = (
            ai_model.test_results
        )


        if (
            test_results is not None
            and not test_results.empty
        ):

            st.dataframe(
                test_results.tail(100),
                use_container_width=True,
            )

            st.caption(
                "古い80%で学習し、"
                "新しい20%で検証しています。"
            )

        else:

            st.info(
                "固定検証結果がありません。"
            )

    except Exception as e:

        st.warning(
            "固定80/20検証の表示を"
            "スキップしました。"
        )

        st.exception(e)


    # ========================================================
    # ⑨ ウォークフォワード検証
    # ========================================================

    st.divider()

    st.header(
        "⑨ 🔄 ウォークフォワード検証"
    )

    try:

        walk_forward = (
            WalkForwardBacktest(
                initial_train_size=500,
                test_size=20,
                retrain_every=20,
                threshold=0.50,
            )
        )


        (
            walk_results,
            walk_metrics,
        ) = walk_forward.run(
            data=ai_data,
            feature_columns=
                ai_feature_columns,
        )


        walk_samples = len(
            walk_results
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Accuracy",
                safe_metric_number(
                    walk_metrics.get(
                        "accuracy"
                    ),
                    decimals=3,
                ),
            )


        with col2:

            st.metric(
                "AUC",
                safe_metric_number(
                    walk_metrics.get(
                        "auc"
                    ),
                    decimals=3,
                    default_text="N/A",
                ),
            )


        with col3:

            st.metric(
                "検証営業日",
                f"{walk_samples:,}",
            )


        high_confidence_count = (
            walk_metrics.get(
                "high_confidence_count",
                0,
            )
        )


        high_confidence_accuracy = (
            walk_metrics.get(
                "high_confidence_accuracy"
            )
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "高確信度件数",
                f"{high_confidence_count:,}",
            )

        with col2:

            st.metric(
                "高確信度Accuracy",
                safe_metric_number(
                    high_confidence_accuracy,
                    decimals=3,
                    default_text="N/A",
                ),
            )


        with st.expander(
            "ウォークフォワード結果を見る"
        ):

            st.dataframe(
                walk_results.tail(200),
                use_container_width=True,
            )


        st.success(
            "✅ ウォークフォワード検証完了"
        )

    except Exception as e:

        st.error(
            "❌ ウォークフォワード検証中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑩ 旧方式 売買バックテスト
    # ========================================================

    st.divider()

    st.header(
        "⑩ 📊 旧方式 売買バックテスト"
    )

    st.caption(
        "比較用として従来方式を残しています。"
    )


    old_backtest_metrics = {}
    old_trades = pd.DataFrame()
    old_equity_curve = pd.DataFrame()


    try:

        backtest_report = (
            BacktestReport(
                initial_capital=
                    INITIAL_CAPITAL,

                lot_size=100,

                entry_threshold=0.60,

                commission_rate=0.001,

                slippage_rate=0.001,
            )
        )


        (
            old_trades,
            old_equity_curve,
            old_backtest_metrics,
        ) = backtest_report.run(
            stock_data=stock_data,
            walk_results=walk_results,
        )


        old_final_capital = (
            safe_float(
                old_backtest_metrics.get(
                    "final_capital"
                ),
                INITIAL_CAPITAL,
            )
        )


        old_total_return = (
            safe_float(
                old_backtest_metrics.get(
                    "total_return"
                ),
                0.0,
            )
        )


        old_trade_count = (
            old_backtest_metrics.get(
                "trade_count",
                old_backtest_metrics.get(
                    "trades",
                    0,
                ),
            )
        )


        old_win_rate = safe_float(
            old_backtest_metrics.get(
                "win_rate"
            )
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "最終資産",
                f"{old_final_capital:,.0f} 円",
            )

        with col2:

            st.metric(
                "総合リターン",
                f"{old_total_return * 100:.2f}%",
            )

        with col3:

            st.metric(
                "取引回数",
                f"{int(old_trade_count):,}",
            )

        with col4:

            if old_win_rate is None:

                st.metric(
                    "勝率",
                    "N/A",
                )

            else:

                st.metric(
                    "勝率",
                    f"{old_win_rate * 100:.1f}%",
                )


        if (
            old_equity_curve is not None
            and not old_equity_curve.empty
        ):

            with st.expander(
                "旧方式の資産推移を見る"
            ):

                st.line_chart(
                    old_equity_curve
                )


        if (
            old_trades is not None
            and not old_trades.empty
        ):

            with st.expander(
                "旧方式の取引履歴を見る"
            ):

                st.dataframe(
                    old_trades,
                    use_container_width=True,
                )


        st.success(
            "✅ 旧方式バックテスト完了"
        )

    except Exception as e:

        st.warning(
            "⚠️ 旧方式バックテストで"
            "エラーが発生しました。"
        )

        st.exception(e)


    # ========================================================
    # ⑪ AIエントリー判断
    # ========================================================

    st.divider()

    st.header(
        "⑪ 🟢 AIエントリー判断"
    )

    try:

        entry_strategy = (
            EntryStrategy(
                minimum_score=6,
                minimum_probability=0.55,
                strong_probability=0.60,
                rsi_min=40.0,
                rsi_max=70.0,
                volume_ratio_min=1.0,
            )
        )


        entry_result = (
            entry_strategy.evaluate_latest(
                data=ai_data,
                probability_up=
                    probability_up,
            )
        )


        entry_action = (
            entry_result[
                "action"
            ]
        )

        entry_score = (
            entry_result[
                "score"
            ]
        )

        entry_max_score = (
            entry_result[
                "max_score"
            ]
        )

        entry_score_rate = safe_float(
            entry_result.get(
                "score_rate"
            ),
            0.0,
        )


        if entry_action == "BUY":

            st.success(
                "🟢 エントリー判断：BUY"
            )

        else:

            st.warning(
                "🟡 エントリー判断：WAIT"
            )


        st.write(
            entry_result.get(
                "summary",
                "",
            )
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "AI上昇確率",
                f"{probability_up * 100:.1f}%",
                delta="基準 55%以上",
            )

        with col2:

            st.metric(
                "エントリースコア",
                f"{entry_score} / {entry_max_score}",
            )

        with col3:

            st.metric(
                "スコア率",
                f"{entry_score_rate * 100:.1f}%",
            )


        st.progress(
            float(
                min(
                    max(
                        entry_score_rate,
                        0.0,
                    ),
                    1.0,
                )
            )
        )


        st.subheader(
            "エントリースコア内訳"
        )


        entry_details = (
            entry_result.get(
                "details",
                {},
            )
        )


        score_table = pd.DataFrame(
            [
                {
                    "項目": "AI",
                    "スコア":
                        entry_details.get(
                            "ai_score",
                            0,
                        ),
                    "最大": 3,
                },
                {
                    "項目": "トレンド",
                    "スコア":
                        entry_details.get(
                            "trend_score",
                            0,
                        ),
                    "最大": 3,
                },
                {
                    "項目": "RSI",
                    "スコア":
                        entry_details.get(
                            "rsi_score",
                            0,
                        ),
                    "最大": 1,
                },
                {
                    "項目": "MACD",
                    "スコア":
                        entry_details.get(
                            "macd_score",
                            0,
                        ),
                    "最大": 1,
                },
                {
                    "項目": "出来高",
                    "スコア":
                        entry_details.get(
                            "volume_score",
                            0,
                        ),
                    "最大": 1,
                },
                {
                    "項目": "市場環境",
                    "スコア":
                        entry_details.get(
                            "market_score",
                            0,
                        ),
                    "最大": 4,
                },
            ]
        )


        st.dataframe(
            score_table,
            use_container_width=True,
            hide_index=True,
        )


        with st.expander(
            "エントリー判断理由を見る"
        ):

            for reason in (
                entry_result.get(
                    "reasons",
                    [],
                )
            ):

                st.write(
                    f"・{reason}"
                )


        st.success(
            "🎉 AIエントリー判断まで"
            "正常に完了しました。"
        )

    except Exception as e:

        st.error(
            "❌ AIエントリー判断中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑫ リスク・資金管理
    # ========================================================

    st.divider()

    st.header(
        "⑫ 🛡️ リスク・資金管理"
    )

    try:

        risk_manager = (
            RiskManager(
                lot_size=100,
                risk_per_trade=0.01,
                max_position_rate=0.50,
                stop_loss_rate=0.05,
                take_profit_rate=0.10,
                commission_rate=0.001,
                slippage_rate=0.001,
            )
        )


        risk_result = (
            risk_manager.evaluate_trade(
                capital=
                    INITIAL_CAPITAL,

                market_price=
                    close_price,
            )
        )


        can_trade = bool(
            risk_result.get(
                "can_trade",
                False,
            )
        )


        st.subheader(
            "🚦 最終エントリー確認"
        )


        if (
            entry_action == "BUY"
            and can_trade
        ):

            st.success(
                "🟢 BUY条件成立 ＋ 資金管理OK"
            )

        elif (
            entry_action == "BUY"
            and not can_trade
        ):

            st.warning(
                "🟡 BUY条件成立 / "
                "資金管理で見送り"
            )

            st.write(
                risk_result.get(
                    "reason",
                    "",
                )
            )

        else:

            st.info(
                "⚪ 現在はWAIT"
            )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "運用資金",
                safe_metric_number(
                    risk_result.get(
                        "capital"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col2:

            st.metric(
                "1取引の許容損失",
                safe_metric_number(
                    risk_result.get(
                        "risk_budget"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col3:

            st.metric(
                "最大投資金額",
                safe_metric_number(
                    risk_result.get(
                        "max_position_value"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
            )


        st.subheader(
            "📦 ポジションサイズ"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "想定購入価格",
                safe_metric_number(
                    risk_result.get(
                        "entry_price"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col2:

            st.metric(
                "購入可能株数",
                f"{int(risk_result.get('shares', 0)):,} 株",
            )

        with col3:

            st.metric(
                "購入単位",
                f"{int(risk_result.get('lots', 0)):,} 単元",
            )


        st.subheader(
            "🎯 損切り・利益確定"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "損切り価格",
                safe_metric_number(
                    risk_result.get(
                        "stop_price"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
                delta="-5.0%",
            )

        with col2:

            st.metric(
                "利益確定価格",
                safe_metric_number(
                    risk_result.get(
                        "take_profit_price"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
                delta="+10.0%",
            )

        with col3:

            st.metric(
                "リスクリワード比",
                safe_metric_number(
                    risk_result.get(
                        "reward_risk_ratio"
                    ),
                    decimals=2,
                ),
            )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "必要資金",
                safe_metric_number(
                    risk_result.get(
                        "required_cash"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col2:

            st.metric(
                "最大想定価格損失",
                safe_metric_number(
                    risk_result.get(
                        "expected_loss"
                    ),
                    decimals=0,
                    suffix=" 円",
                ),
            )

        with col3:

            position_rate = safe_float(
                risk_result.get(
                    "position_rate"
                ),
                0.0,
            )

            st.metric(
                "資金投入率",
                f"{position_rate * 100:.2f}%",
            )


        with st.expander(
            "株数計算の詳細を見る"
        ):

            position_table = pd.DataFrame(
                [
                    {
                        "計算基準":
                            "1取引リスク上限",
                        "購入可能株数":
                            risk_result.get(
                                "shares_by_risk",
                                0,
                            ),
                    },
                    {
                        "計算基準":
                            "最大投資比率",
                        "購入可能株数":
                            risk_result.get(
                                "shares_by_position_limit",
                                0,
                            ),
                    },
                    {
                        "計算基準":
                            "現金残高",
                        "購入可能株数":
                            risk_result.get(
                                "shares_by_cash",
                                0,
                            ),
                    },
                ]
            )

            st.dataframe(
                position_table,
                use_container_width=True,
                hide_index=True,
            )


        if can_trade:

            st.success(
                "✅ 資金管理条件：購入可能"
            )

        else:

            st.warning(
                "⚠️ 資金管理条件：NO TRADE"
            )

            st.write(
                risk_result.get(
                    "reason",
                    "",
                )
            )


        st.success(
            "🎉 リスク・資金管理まで"
            "正常に完了しました。"
        )

    except Exception as e:

        st.error(
            "❌ リスク・資金管理中に"
            "エラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑬ 本格戦略バックテスト
    # ========================================================

    st.divider()

    st.header(
        "⑬ 🚀 本格戦略バックテスト"
    )

    st.write(
        "EntryStrategy・ExitStrategy・RiskManagerを"
        "統合して過去データを時系列で検証します。"
    )

    st.caption(
        "BUYシグナル後は翌営業日の始値を基準に購入し、"
        "SELL条件が成立するまでポジションを保有します。"
    )


    try:

        trading_engine = (
            TradingBacktestEngine(
                initial_capital=
                    INITIAL_CAPITAL,

                lot_size=100,

                # Entry
                entry_minimum_score=6,
                entry_minimum_probability=0.55,
                entry_strong_probability=0.60,

                # Exit
                stop_loss_rate=0.05,
                take_profit_rate=0.10,
                ai_exit_probability=0.45,
                max_holding_days=10,
                minimum_exit_score=3,

                # Risk
                risk_per_trade=0.01,
                max_position_rate=0.50,

                # Cost
                commission_rate=0.001,
                slippage_rate=0.001,
            )
        )


        (
            strategy_trades,
            strategy_equity,
            strategy_metrics,
        ) = trading_engine.run(
            stock_data=stock_data,
            ai_data=ai_data,
            walk_results=walk_results,
        )


        # ====================================================
        # 総合成績
        # ====================================================

        st.subheader(
            "📊 総合成績"
        )


        strategy_final_capital = (
            safe_float(
                strategy_metrics.get(
                    "final_capital"
                ),
                INITIAL_CAPITAL,
            )
        )


        strategy_total_profit = (
            safe_float(
                strategy_metrics.get(
                    "total_profit"
                ),
                0.0,
            )
        )


        strategy_total_return = (
            safe_float(
                strategy_metrics.get(
                    "total_return"
                ),
                0.0,
            )
        )


        strategy_trade_count = int(
            strategy_metrics.get(
                "trades",
                0,
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "初期資金",
                f"{INITIAL_CAPITAL:,.0f} 円",
            )

        with col2:

            st.metric(
                "最終資産",
                f"{strategy_final_capital:,.0f} 円",
                delta=
                    f"{strategy_total_profit:+,.0f} 円",
            )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "総利益",
                f"{strategy_total_profit:+,.0f} 円",
            )

        with col2:

            st.metric(
                "総合リターン",
                f"{strategy_total_return * 100:+.2f}%",
            )


        # ====================================================
        # 取引成績
        # ====================================================

        st.subheader(
            "🎯 取引成績"
        )


        wins = int(
            strategy_metrics.get(
                "wins",
                0,
            )
        )

        losses = int(
            strategy_metrics.get(
                "losses",
                0,
            )
        )

        win_rate = safe_float(
            strategy_metrics.get(
                "win_rate"
            )
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "取引回数",
                f"{strategy_trade_count:,}",
            )

        with col2:

            st.metric(
                "勝ち",
                f"{wins:,}",
            )

        with col3:

            st.metric(
                "負け",
                f"{losses:,}",
            )

        with col4:

            if win_rate is None:

                st.metric(
                    "勝率",
                    "N/A",
                )

            else:

                st.metric(
                    "勝率",
                    f"{win_rate * 100:.1f}%",
                )


        # ====================================================
        # リスク・効率
        # ====================================================

        st.subheader(
            "🛡️ リスク・効率"
        )


        profit_factor = (
            strategy_metrics.get(
                "profit_factor"
            )
        )

        max_drawdown = safe_float(
            strategy_metrics.get(
                "max_drawdown"
            ),
            0.0,
        )

        average_holding_days = (
            safe_float(
                strategy_metrics.get(
                    "average_holding_days"
                )
            )
        )

        skipped_entries_count = int(
            strategy_metrics.get(
                "skipped_entries",
                0,
            )
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            pf = safe_float(
                profit_factor
            )

            if pf is None:

                if profit_factor == float(
                    "inf"
                ):

                    st.metric(
                        "プロフィットファクター",
                        "∞",
                    )

                else:

                    st.metric(
                        "プロフィットファクター",
                        "N/A",
                    )

            elif np.isinf(pf):

                st.metric(
                    "プロフィットファクター",
                    "∞",
                )

            else:

                st.metric(
                    "プロフィットファクター",
                    f"{pf:.2f}",
                )


        with col2:

            st.metric(
                "最大ドローダウン",
                f"{max_drawdown * 100:.2f}%",
            )


        with col3:

            if average_holding_days is None:

                st.metric(
                    "平均保有日数",
                    "N/A",
                )

            else:

                st.metric(
                    "平均保有日数",
                    f"{average_holding_days:.1f} 日",
                )


        with col4:

            st.metric(
                "資金管理で見送り",
                f"{skipped_entries_count:,} 回",
            )


        # ====================================================
        # 1取引成績
        # ====================================================

        if strategy_trade_count > 0:

            st.subheader(
                "💰 1取引あたりの成績"
            )


            average_profit = safe_float(
                strategy_metrics.get(
                    "average_profit"
                )
            )

            average_profit_rate = safe_float(
                strategy_metrics.get(
                    "average_profit_rate"
                )
            )

            best_trade = safe_float(
                strategy_metrics.get(
                    "best_trade"
                )
            )

            worst_trade = safe_float(
                strategy_metrics.get(
                    "worst_trade"
                )
            )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "平均損益",
                    (
                        f"{average_profit:+,.0f} 円"
                        if average_profit
                        is not None
                        else "N/A"
                    ),
                )

            with col2:

                st.metric(
                    "平均損益率",
                    (
                        f"{average_profit_rate * 100:+.2f}%"
                        if average_profit_rate
                        is not None
                        else "N/A"
                    ),
                )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "最大利益取引",
                    (
                        f"{best_trade:+,.0f} 円"
                        if best_trade
                        is not None
                        else "N/A"
                    ),
                )

            with col2:

                st.metric(
                    "最大損失取引",
                    (
                        f"{worst_trade:+,.0f} 円"
                        if worst_trade
                        is not None
                        else "N/A"
                    ),
                )


        total_commission = safe_float(
            strategy_metrics.get(
                "total_commission"
            ),
            0.0,
        )


        st.metric(
            "累計売買手数料",
            f"{total_commission:,.0f} 円",
        )


        # ====================================================
        # 資産曲線
        # ====================================================

        st.subheader(
            "📈 資産推移"
        )


        if (
            strategy_equity is not None
            and not strategy_equity.empty
            and "Total_Equity"
            in strategy_equity.columns
        ):

            st.line_chart(
                strategy_equity[
                    ["Total_Equity"]
                ]
            )

        else:

            st.info(
                "資産曲線データがありません。"
            )


        # ====================================================
        # 取引履歴
        # ====================================================

        st.subheader(
            "📋 本格戦略 取引履歴"
        )


        if (
            strategy_trades is not None
            and not strategy_trades.empty
        ):

            display_trades = (
                strategy_trades.copy()
            )


            if (
                "Entry_Probability"
                in display_trades.columns
            ):

                display_trades[
                    "AI上昇確率%"
                ] = (
                    display_trades[
                        "Entry_Probability"
                    ]
                    * 100
                )


            if (
                "Profit_Rate"
                in display_trades.columns
            ):

                display_trades[
                    "損益率%"
                ] = (
                    display_trades[
                        "Profit_Rate"
                    ]
                    * 100
                )


            st.dataframe(
                display_trades,
                use_container_width=True,
                hide_index=True,
            )


            csv_data = (
                strategy_trades
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            )


            st.download_button(
                label=
                    "📥 本格戦略の取引履歴CSVを保存",

                data=
                    csv_data,

                file_name=
                    "advantest_strategy_trades.csv",

                mime=
                    "text/csv",

                use_container_width=True,
            )

        else:

            st.warning(
                "この条件では取引が"
                "発生しませんでした。"
            )

            st.write(
                "100株単位・最大投資比率50%などの"
                "資金管理条件によって、"
                "BUY候補が見送られた可能性があります。"
            )


        # ====================================================
        # 見送り履歴
        # ====================================================

        skipped_entries = (
            trading_engine
            .get_skipped_entries()
        )


        with st.expander(
            "⚠️ 資金管理で見送った"
            "エントリーを見る"
        ):

            if (
                skipped_entries is not None
                and not skipped_entries.empty
            ):

                st.dataframe(
                    skipped_entries,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.write(
                    "資金管理による"
                    "見送りはありません。"
                )


        # ====================================================
        # 売却理由
        # ====================================================

        with st.expander(
            "🔴 売却理由を見る"
        ):

            if (
                strategy_trades is not None
                and not strategy_trades.empty
                and "Exit_Reason"
                in strategy_trades.columns
            ):

                exit_summary = (
                    strategy_trades[
                        "Exit_Reason"
                    ]
                    .value_counts()
                    .rename_axis(
                        "売却理由"
                    )
                    .reset_index(
                        name="回数"
                    )
                )


                st.dataframe(
                    exit_summary,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.write(
                    "売却履歴はありません。"
                )


        # ====================================================
        # 設定表示
        # ====================================================

        with st.expander(
            "⚙️ 本格バックテストの"
            "設定を見る"
        ):

            st.write(
                "初期資金：1,000,000円"
            )

            st.write(
                "売買単位：100株"
            )

            st.write(
                "BUY最低AI確率：55%"
            )

            st.write(
                "BUY最低スコア：6点"
            )

            st.write(
                "1取引の最大リスク：資金の1%"
            )

            st.write(
                "1銘柄への最大投資比率：50%"
            )

            st.write(
                "損切り：-5%"
            )

            st.write(
                "利益確定：+10%"
            )

            st.write(
                "AI売却警戒確率：45%以下"
            )

            st.write(
                "最大保有期間：10営業日"
            )

            st.write(
                "売却最低スコア：3点"
            )

            st.write(
                "売買手数料：片道0.1%（検証用）"
            )

            st.write(
                "スリッページ：0.1%（検証用）"
            )


        # ====================================================
        # 旧方式との比較
        # ====================================================

        st.subheader(
            "🔄 旧方式との参考比較"
        )


        old_final = safe_float(
            old_backtest_metrics.get(
                "final_capital"
            ),
            INITIAL_CAPITAL,
        )


        old_return = safe_float(
            old_backtest_metrics.get(
                "total_return"
            ),
            0.0,
        )


        old_count = (
            old_backtest_metrics.get(
                "trade_count",
                old_backtest_metrics.get(
                    "trades",
                    0,
                ),
            )
        )


        comparison_table = (
            pd.DataFrame(
                [
                    {
                        "方式":
                            "⑩ 旧方式",

                        "最終資産":
                            old_final,

                        "総合リターン%":
                            old_return * 100,

                        "取引回数":
                            old_count,
                    },

                    {
                        "方式":
                            "⑬ 本格戦略",

                        "最終資産":
                            strategy_final_capital,

                        "総合リターン%":
                            strategy_total_return
                            * 100,

                        "取引回数":
                            strategy_trade_count,
                    },
                ]
            )
        )


        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True,
        )


        st.caption(
            "⑩と⑬は売買ルールが異なるため、"
            "戦略設計の違いを見るための参考比較です。"
        )


        st.success(
            "🎉 ⑬ 本格戦略バックテストまで"
            "正常に完了しました。"
        )

    except Exception as e:

        st.error(
            "❌ 本格戦略バックテスト中に"
            "エラーが発生しました。"
        )

        st.exception(e)


    # ========================================================
    # 開発状況
    # ========================================================

    st.divider()

    st.header(
        "🧩 開発状況"
    )


    development_status = (
        pd.DataFrame(
            [
                {
                    "機能":
                        "株価データ",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "市場データ",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "テクニカル分析",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "AI特徴量",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "AIモデル",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "ウォークフォワード",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "旧方式バックテスト",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "エントリー判断",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "売却判断",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "リスク管理",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "本格戦略バックテスト",
                    "状態":
                        "✅ 完了",
                },
                {
                    "機能":
                        "ペーパートレード",
                    "状態":
                        "🔵 次の段階",
                },
            ]
        )
    )


    st.dataframe(
        development_status,
        use_container_width=True,
        hide_index=True,
    )


    st.success(
        "🚀 全分析処理が終了しました。"
    )


# ============================================================
# 実行前
# ============================================================

else:

    st.info(
        "上の「株価・市場・AI・戦略分析を実行」"
        "ボタンを押してください。"
    )

    st.write(
        "現在のシステムでは、"
        "アドバンテストの株価・市場環境・"
        "テクニカル・AI予測・エントリー判断・"
        "資金管理・本格戦略バックテストまで"
        "まとめて確認できます。"
    )
