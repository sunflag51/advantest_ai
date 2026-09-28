# ============================================================
# アドバンテスト AI売買分析システム
# main.py
#
# 完全版
#
# ・①～⑫を st.session_state に保存
# ・⑬だけ条件変更して再計算
# ・バックテスト資金を選択
# ・最大投資比率を選択
# ・コピー用バックテスト結果
# ・成績CSVダウンロード
# ・売買履歴CSVダウンロード
# ・見送り履歴CSVダウンロード
# ・注文ログCSVダウンロード
# ============================================================


import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# データ
# ============================================================

from data.stock_data import (
    get_stock_data,
)

from data.market_data import (
    get_all_market_data,
    get_latest_market_values,
)


# ============================================================
# テクニカル
# ============================================================

from indicators.technical import (
    add_all_indicators,
)


# ============================================================
# AI
# ============================================================

from ai.features import (
    build_ai_features,
)

from ai.model import (
    StockPredictionModel,
)


# ============================================================
# バックテスト
# ============================================================

from backtest.engine import (
    WalkForwardBacktest,
)

from backtest.report import (
    BacktestReport,
)

from backtest.trading_engine import (
    TradingBacktestEngine,
)


# ============================================================
# 戦略
# ============================================================

from strategy.entry import (
    EntryStrategy,
)

from strategy.risk import (
    RiskManager,
)


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
# session_state 初期値
# ============================================================

SESSION_DEFAULTS = {

    "analysis_completed":
        False,

    "stock_data":
        None,

    "latest_close":
        None,

    "latest_date":
        None,

    "market_data":
        None,

    "latest_market":
        None,

    "technical_data":
        None,

    "ai_data":
        None,

    "ai_model":
        None,

    "ai_metrics":
        None,

    "latest_probability":
        None,

    "entry_result":
        None,

    "walk_results":
        None,

    "walk_metrics":
        None,

    "old_backtest_metrics":
        None,
}


for key, default_value in (
    SESSION_DEFAULTS.items()
):

    if key not in st.session_state:

        st.session_state[
            key
        ] = default_value


# ============================================================
# 安全なfloat変換
# ============================================================

def safe_float(
    value,
    default=None,
):

    try:

        if value is None:

            return default


        if isinstance(
            value,
            pd.DataFrame,
        ):

            if value.empty:

                return default

            value = value.iloc[
                -1,
                -1,
            ]


        if isinstance(
            value,
            pd.Series,
        ):

            numeric = (
                pd.to_numeric(
                    value,
                    errors="coerce",
                )
                .dropna()
            )


            if numeric.empty:

                return default


            value = numeric.iloc[-1]


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


            value = array[-1]


        number = float(
            value
        )


        if not np.isfinite(
            number
        ):

            return default


        return number


    except Exception:

        return default


# ============================================================
# 数値表示
# ============================================================

def safe_metric_number(
    value,
    digits=2,
    suffix="",
    default="N/A",
):

    number = safe_float(
        value
    )


    if number is None:

        return default


    return (
        f"{number:,.{digits}f}"
        f"{suffix}"
    )


# ============================================================
# パーセント表示
# ============================================================

def format_percent(
    value,
    digits=2,
):

    number = safe_float(
        value
    )


    if number is None:

        return "N/A"


    return (
        f"{number * 100:.{digits}f}%"
    )


# ============================================================
# CSV変換
# ============================================================

def dataframe_to_csv_bytes(
    dataframe,
    include_index=False,
):
    """
    ExcelやiPhoneでも日本語が
    文字化けしにくいUTF-8 BOM付きCSV。
    """

    if dataframe is None:

        dataframe = pd.DataFrame()


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):

        dataframe = pd.DataFrame(
            dataframe
        )


    csv_text = dataframe.to_csv(
        index=include_index
    )


    return csv_text.encode(
        "utf-8-sig"
    )


# ============================================================
# 分析リセット
# ============================================================

def reset_analysis():

    for key, default_value in (
        SESSION_DEFAULTS.items()
    ):

        st.session_state[
            key
        ] = default_value


# ============================================================
# ①～⑫ 実行
# ============================================================

def run_base_analysis():

    st.session_state[
        "analysis_completed"
    ] = False


    # ========================================================
    # ① 株価
    # ========================================================

    stock_data = get_stock_data(
        stock_code=STOCK_CODE,
        period="5y",
        interval="1d",
    )


    if (
        stock_data is None
        or stock_data.empty
    ):

        raise ValueError(
            "株価データを取得できませんでした。"
        )


    latest_close = safe_float(
        stock_data[
            "Close"
        ].iloc[-1]
    )


    latest_date = (
        stock_data.index[-1]
    )


    # ========================================================
    # ② 市場
    # ========================================================

    market_data = (
        get_all_market_data(
            period="5y",
            interval="1d",
        )
    )


    latest_market = (
        get_latest_market_values(
            period="5d",
            interval="1d",
        )
    )


    # ========================================================
    # ③ テクニカル
    # ========================================================

    technical_data = (
        add_all_indicators(
            stock_data.copy()
        )
    )


    # ========================================================
    # ④ AI特徴量
    # ========================================================

    ai_data = (
        build_ai_features(
            technical_data,
            market_data,
        )
    )


    if (
        ai_data is None
        or ai_data.empty
    ):

        raise ValueError(
            "AI特徴量を作成できませんでした。"
        )


    # ========================================================
    # ⑤ AIモデル
    # ========================================================

    ai_model = (
        StockPredictionModel()
    )


    ai_metrics = (
        ai_model.train(
            ai_data
        )
    )


    # ========================================================
    # ⑥ 最新予測
    # ========================================================

    latest_probability = (
        ai_model.predict_probability(
            ai_data
        )
    )


    latest_probability = (
        safe_float(
            latest_probability
        )
    )


    # ========================================================
    # ⑧ Entry
    # ========================================================

    entry_result = None


    if latest_probability is not None:

        entry_strategy = (
            EntryStrategy()
        )


        try:

            entry_result = (
                entry_strategy
                .evaluate_latest(
                    ai_data,
                    latest_probability,
                )
            )

        except Exception:

            entry_result = None


    # ========================================================
    # ⑨ Walk Forward
    # ========================================================

    walk_engine = (
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
    ) = walk_engine.run(
        ai_data
    )


    # ========================================================
    # ⑩ 従来型バックテスト
    # ========================================================

    old_backtest_metrics = {}


    try:

        old_report = (
            BacktestReport(
                initial_capital=
                    INITIAL_CAPITAL,

                lot_size=
                    100,

                entry_threshold=
                    0.60,

                commission_rate=
                    0.001,

                slippage_rate=
                    0.001,
            )
        )


        old_report.run(
            stock_data=
                stock_data,

            prediction_data=
                walk_results,
        )


        old_backtest_metrics = (
            old_report
            .calculate_metrics()
        )


    except Exception:

        old_backtest_metrics = {}


    # ========================================================
    # session_state 保存
    # ========================================================

    st.session_state[
        "stock_data"
    ] = stock_data


    st.session_state[
        "latest_close"
    ] = latest_close


    st.session_state[
        "latest_date"
    ] = latest_date


    st.session_state[
        "market_data"
    ] = market_data


    st.session_state[
        "latest_market"
    ] = latest_market


    st.session_state[
        "technical_data"
    ] = technical_data


    st.session_state[
        "ai_data"
    ] = ai_data


    st.session_state[
        "ai_model"
    ] = ai_model


    st.session_state[
        "ai_metrics"
    ] = ai_metrics


    st.session_state[
        "latest_probability"
    ] = latest_probability


    st.session_state[
        "entry_result"
    ] = entry_result


    st.session_state[
        "walk_results"
    ] = walk_results


    st.session_state[
        "walk_metrics"
    ] = walk_metrics


    st.session_state[
        "old_backtest_metrics"
    ] = old_backtest_metrics


    st.session_state[
        "analysis_completed"
    ] = True


# ============================================================
# タイトル
# ============================================================

st.title(
    "📈 アドバンテスト AI売買分析システム"
)


st.caption(
    "①～⑫の分析結果を保持し、"
    "⑬の資金条件だけを変更して"
    "再バックテストできます。"
)


# ============================================================
# 操作ボタン
# ============================================================

button_col1, button_col2 = (
    st.columns(
        [3, 1]
    )
)


run_button = (
    button_col1.button(
        "🚀 ①～⑫を最新データで分析",
        type="primary",
        use_container_width=True,
    )
)


reset_button = (
    button_col2.button(
        "🔄 分析結果をリセット",
        use_container_width=True,
    )
)


# ============================================================
# リセット
# ============================================================

if reset_button:

    reset_analysis()

    st.rerun()


# ============================================================
# ①～⑫ 新規分析
# ============================================================

if run_button:

    try:

        with st.spinner(
            "①～⑫を分析しています。"
            "AI学習とウォークフォワード検証を"
            "実行中です..."
        ):

            run_base_analysis()


        st.success(
            "✅ ①～⑫の分析が完了しました。"
            "結果を保持しました。"
        )


    except Exception as error:

        st.session_state[
            "analysis_completed"
        ] = False


        st.error(
            "①～⑫の分析中に"
            "エラーが発生しました。"
        )


        st.exception(
            error
        )


# ============================================================
# 分析前
# ============================================================

if not st.session_state[
    "analysis_completed"
]:

    st.info(
        "最初に「🚀 ①～⑫を最新データで分析」"
        "を押してください。"
    )

    st.stop()


# ============================================================
# 保存結果取得
# ============================================================

stock_data = (
    st.session_state[
        "stock_data"
    ]
)


latest_close = (
    st.session_state[
        "latest_close"
    ]
)


latest_date = (
    st.session_state[
        "latest_date"
    ]
)


market_data = (
    st.session_state[
        "market_data"
    ]
)


latest_market = (
    st.session_state[
        "latest_market"
    ]
)


technical_data = (
    st.session_state[
        "technical_data"
    ]
)


ai_data = (
    st.session_state[
        "ai_data"
    ]
)


ai_model = (
    st.session_state[
        "ai_model"
    ]
)


ai_metrics = (
    st.session_state[
        "ai_metrics"
    ]
)


latest_probability = (
    st.session_state[
        "latest_probability"
    ]
)


entry_result = (
    st.session_state[
        "entry_result"
    ]
)


walk_results = (
    st.session_state[
        "walk_results"
    ]
)


walk_metrics = (
    st.session_state[
        "walk_metrics"
    ]
)


old_backtest_metrics = (
    st.session_state[
        "old_backtest_metrics"
    ]
)


st.success(
    "💾 ①～⑫の分析結果を保持中です。"
    "⑬の設定変更では再学習しません。"
)


# ============================================================
# ① 株価
# ============================================================

st.header(
    "① 📊 アドバンテスト株価"
)


stock_cols = st.columns(
    3
)


stock_cols[
    0
].metric(
    "銘柄",
    STOCK_NAME,
)


stock_cols[
    1
].metric(
    "最新株価",
    (
        f"{latest_close:,.0f} 円"

        if latest_close is not None

        else "取得なし"
    ),
)


stock_cols[
    2
].metric(
    "最新日",
    (
        str(
            latest_date.date()
        )

        if latest_date is not None

        else "取得なし"
    ),
)


if (
    stock_data is not None
    and not stock_data.empty
):

    st.line_chart(
        stock_data[
            ["Close"]
        ]
    )


# ============================================================
# ② 市場
# ============================================================

st.header(
    "② 🌏 市場データ"
)


market_names = [
    "日経平均",
    "NASDAQ",
    "SOXX",
    "ドル円",
]


market_cols = st.columns(
    4
)


for index, name in enumerate(
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


    value = safe_float(
        value
    )


    market_cols[
        index
    ].metric(
        name,
        (
            f"{value:,.2f}"

            if value is not None

            else "取得なし"
        ),
    )


# ============================================================
# 市場比較
# ============================================================

normalized_market = (
    pd.DataFrame()
)


if isinstance(
    market_data,
    dict,
):

    for (
        market_name,
        df,
    ) in market_data.items():

        if (
            isinstance(
                df,
                pd.DataFrame,
            )
            and not df.empty
            and "Close"
            in df.columns
        ):

            close = (
                pd.to_numeric(
                    df[
                        "Close"
                    ],
                    errors="coerce",
                )
                .dropna()
            )


            if not close.empty:

                first_value = (
                    safe_float(
                        close.iloc[0]
                    )
                )


                if (
                    first_value is not None
                    and first_value != 0
                ):

                    normalized_market[
                        market_name
                    ] = (
                        close
                        / first_value
                        * 100
                    )


if not normalized_market.empty:

    st.subheader(
        "市場比較（開始日=100）"
    )


    st.line_chart(
        normalized_market
    )


# ============================================================
# ③ テクニカル
# ============================================================

st.header(
    "③ 📐 テクニカル分析"
)


if (
    technical_data is not None
    and not technical_data.empty
):

    latest_technical = (
        technical_data.iloc[-1]
    )


    technical_cols = (
        st.columns(
            4
        )
    )


    technical_cols[
        0
    ].metric(
        "SMA 5",
        safe_metric_number(
            latest_technical.get(
                "SMA_5"
            ),
            0,
            " 円",
        ),
    )


    technical_cols[
        1
    ].metric(
        "SMA 25",
        safe_metric_number(
            latest_technical.get(
                "SMA_25"
            ),
            0,
            " 円",
        ),
    )


    technical_cols[
        2
    ].metric(
        "SMA 75",
        safe_metric_number(
            latest_technical.get(
                "SMA_75"
            ),
            0,
            " 円",
        ),
    )


    technical_cols[
        3
    ].metric(
        "RSI 14",
        safe_metric_number(
            latest_technical.get(
                "RSI_14"
            ),
            2,
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
            ]
        )


# ============================================================
# ④ AI特徴量
# ============================================================

st.header(
    "④ 🧩 AI特徴量"
)


st.write(
    f"AI特徴量データ数: "
    f"{len(ai_data):,} 行"
)


market_feature_columns = [
    column

    for column
    in ai_data.columns

    if (
        "日経平均_" in column
        or "NASDAQ_" in column
        or "SOXX_" in column
        or "ドル円_" in column
    )
]


st.write(
    "市場AI特徴量数:",
    len(
        market_feature_columns
    ),
)


if market_feature_columns:

    with st.expander(
        "市場AI特徴量を確認"
    ):

        st.dataframe(
            ai_data[
                market_feature_columns
            ].tail(10),
            use_container_width=True,
        )


# ============================================================
# ⑤ AIモデル
# ============================================================

st.header(
    "⑤ 🤖 AIモデル学習結果"
)


ai_cols = st.columns(
    4
)


ai_cols[
    0
].metric(
    "Accuracy",
    safe_metric_number(
        ai_metrics.get(
            "accuracy"
        ),
        3,
    ),
)


ai_cols[
    1
].metric(
    "Precision",
    safe_metric_number(
        ai_metrics.get(
            "precision"
        ),
        3,
    ),
)


ai_cols[
    2
].metric(
    "Recall",
    safe_metric_number(
        ai_metrics.get(
            "recall"
        ),
        3,
    ),
)


ai_cols[
    3
].metric(
    "F1",
    safe_metric_number(
        ai_metrics.get(
            "f1"
        ),
        3,
    ),
)


# ============================================================
# ⑥ 最新AI予測
# ============================================================

st.header(
    "⑥ 🔮 最新AI予測"
)


if latest_probability is not None:

    prediction_cols = (
        st.columns(
            2
        )
    )


    prediction_cols[
        0
    ].metric(
        "翌営業日 上昇確率",
        (
            f"{latest_probability * 100:.1f}%"
        ),
    )


    prediction_cols[
        1
    ].metric(
        "AI判断",
        (
            "上昇予測"

            if latest_probability >= 0.50

            else "下落予測"
        ),
    )


else:

    st.warning(
        "最新AI予測を取得できません。"
    )


# ============================================================
# ⑦ 特徴量重要度
# ============================================================

st.header(
    "⑦ 🧠 AI特徴量重要度"
)


try:

    importance = (
        ai_model
        .get_feature_importance()
    )


    if (
        importance is not None
        and not importance.empty
    ):

        st.dataframe(
            importance.head(20),
            use_container_width=True,
        )


except Exception as error:

    st.info(
        f"特徴量重要度を表示できません: "
        f"{error}"
    )


# ============================================================
# ⑧ Entry
# ============================================================

st.header(
    "⑧ 🎯 AIエントリー参考判定"
)


if entry_result is not None:

    entry_cols = (
        st.columns(
            4
        )
    )


    entry_cols[
        0
    ].metric(
        "判断",
        entry_result.get(
            "action",
            "N/A",
        ),
    )


    entry_cols[
        1
    ].metric(
        "スコア",
        (
            f"{entry_result.get('score', 0)}"
            f" / "
            f"{entry_result.get('max_score', 13)}"
        ),
    )


    entry_cols[
        2
    ].metric(
        "AI確率",
        (
            f"{latest_probability * 100:.1f}%"

            if latest_probability is not None

            else "N/A"
        ),
    )


    entry_cols[
        3
    ].metric(
        "最低必要スコア",
        entry_result.get(
            "minimum_score",
            6,
        ),
    )


    reasons = (
        entry_result.get(
            "reasons",
            [],
        )
    )


    if reasons:

        with st.expander(
            "エントリー判定詳細"
        ):

            for reason in reasons:

                st.write(
                    "・",
                    reason,
                )


else:

    st.info(
        "最新エントリー判定はありません。"
    )


# ============================================================
# ⑨ Walk Forward
# ============================================================

st.header(
    "⑨ 🔁 ウォークフォワード検証"
)


walk_samples = (
    len(
        walk_results
    )

    if walk_results is not None

    else 0
)


walk_cols = st.columns(
    5
)


walk_cols[
    0
].metric(
    "検証件数",
    f"{walk_samples:,}",
)


walk_cols[
    1
].metric(
    "Accuracy",
    safe_metric_number(
        walk_metrics.get(
            "accuracy"
        ),
        3,
    ),
)


walk_cols[
    2
].metric(
    "Precision",
    safe_metric_number(
        walk_metrics.get(
            "precision"
        ),
        3,
    ),
)


walk_cols[
    3
].metric(
    "Recall",
    safe_metric_number(
        walk_metrics.get(
            "recall"
        ),
        3,
    ),
)


walk_cols[
    4
].metric(
    "AUC",
    safe_metric_number(
        walk_metrics.get(
            "auc"
        ),
        3,
    ),
)


if (
    walk_results is not None
    and not walk_results.empty
):

    with st.expander(
        "ウォークフォワード予測結果"
    ):

        st.dataframe(
            walk_results.tail(100),
            use_container_width=True,
        )


# ============================================================
# ⑩ 従来型バックテスト
# ============================================================

st.header(
    "⑩ 📋 従来型売買バックテスト"
)


if isinstance(
    old_backtest_metrics,
    dict,
):

    old_cols = st.columns(
        4
    )


    old_final = safe_float(
        old_backtest_metrics.get(
            "final_capital"
        ),
        INITIAL_CAPITAL,
    )


    old_trade_count = int(
        safe_float(
            old_backtest_metrics.get(
                "trade_count"
            ),
            0,
        )
    )


    old_cols[
        0
    ].metric(
        "最終資産",
        f"{old_final:,.0f} 円",
    )


    old_cols[
        1
    ].metric(
        "取引回数",
        old_trade_count,
    )


    old_cols[
        2
    ].metric(
        "勝率",
        format_percent(
            old_backtest_metrics.get(
                "win_rate"
            )
        ),
    )


    old_cols[
        3
    ].metric(
        "最大DD",
        format_percent(
            old_backtest_metrics.get(
                "max_drawdown"
            )
        ),
    )


# ============================================================
# ⑪ 最新Entry判断
# ============================================================

st.header(
    "⑪ 🚦 最新エントリー判断"
)


if entry_result is not None:

    st.write(
        entry_result.get(
            "summary",
            "",
        )
    )


else:

    st.info(
        "最新エントリー判断はありません。"
    )


# ============================================================
# ⑫ リスク管理
# ============================================================

st.header(
    "⑫ 🛡️ リスク・資金管理"
)


if latest_close is not None:

    base_risk_manager = (
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


    base_risk = (
        base_risk_manager
        .evaluate_trade(
            capital=
                INITIAL_CAPITAL,

            market_price=
                latest_close,
        )
    )


    risk_cols = st.columns(
        4
    )


    risk_cols[
        0
    ].metric(
        "購入可能",
        (
            "YES"

            if base_risk.get(
                "can_trade",
                False,
            )

            else "NO"
        ),
    )


    risk_cols[
        1
    ].metric(
        "推奨株数",
        (
            f"{int(base_risk.get('shares', 0)):,} 株"
        ),
    )


    risk_cols[
        2
    ].metric(
        "損切り価格",
        safe_metric_number(
            base_risk.get(
                "stop_price"
            ),
            0,
            " 円",
        ),
    )


    risk_cols[
        3
    ].metric(
        "利益確定価格",
        safe_metric_number(
            base_risk.get(
                "take_profit_price"
            ),
            0,
            " 円",
        ),
    )


    st.caption(
        "⑫は基準条件として"
        "100万円・最大投資比率50%です。"
    )


# ============================================================
# ⑬ 本格戦略バックテスト
# ============================================================

st.header(
    "⑬ 🚀 本格戦略バックテスト"
)


st.write(
    "①～⑫の保存済みデータを使用します。"
)


st.write(
    "設定変更時は⑬だけ再計算します。"
)


st.write(
    "当日の終値でBUY / SELLを判定し、"
    "翌営業日の始値で約定します。"
)


# ============================================================
# ⑬ 設定
# ============================================================

st.subheader(
    "⚙️ バックテスト設定"
)


setting_col1, setting_col2 = (
    st.columns(
        2
    )
)


capital_options = {

    "100万円":
        1_000_000,

    "300万円":
        3_000_000,

    "500万円":
        5_000_000,

    "1,000万円":
        10_000_000,
}


selected_capital_label = (
    setting_col1.selectbox(
        "バックテスト資金",
        options=list(
            capital_options.keys()
        ),
        index=0,
        key=
            "strategy_capital_selector",
    )
)


backtest_capital = (
    capital_options[
        selected_capital_label
    ]
)


position_rate_options = {

    "25%":
        0.25,

    "50%":
        0.50,

    "75%":
        0.75,

    "100%":
        1.00,
}


selected_position_label = (
    setting_col2.selectbox(
        "最大投資比率",
        options=list(
            position_rate_options.keys()
        ),
        index=1,
        key=
            "strategy_position_selector",
    )
)


max_position_rate = (
    position_rate_options[
        selected_position_label
    ]
)


maximum_position_value = (
    backtest_capital
    * max_position_rate
)


setting_cols = st.columns(
    3
)


setting_cols[
    0
].metric(
    "設定資金",
    (
        f"{backtest_capital:,.0f} 円"
    ),
)


setting_cols[
    1
].metric(
    "最大投資比率",
    selected_position_label,
)


setting_cols[
    2
].metric(
    "1回の最大投資額",
    (
        f"{maximum_position_value:,.0f} 円"
    ),
)


st.caption(
    "売買単位100株・"
    "許容リスク1%・"
    "損切り5%・利益確定10%で検証します。"
)


# ============================================================
# ⑬ 実行
# ============================================================

try:

    with st.spinner(
        "⑬だけ再計算しています..."
    ):

        strategy_engine = (
            TradingBacktestEngine(
                initial_capital=
                    backtest_capital,

                lot_size=
                    100,

                entry_minimum_score=
                    6,

                entry_minimum_probability=
                    0.55,

                entry_strong_probability=
                    0.60,

                stop_loss_rate=
                    0.05,

                take_profit_rate=
                    0.10,

                ai_exit_probability=
                    0.45,

                max_holding_days=
                    10,

                minimum_exit_score=
                    3,

                risk_per_trade=
                    0.01,

                max_position_rate=
                    max_position_rate,

                commission_rate=
                    0.001,

                slippage_rate=
                    0.001,
            )
        )


        (
            strategy_trades,
            strategy_equity,
            strategy_metrics,
        ) = (
            strategy_engine.run(
                stock_data=
                    stock_data,

                ai_data=
                    ai_data,

                walk_results=
                    walk_results,
            )
        )


    # ========================================================
    # 数値取得
    # ========================================================

    initial_value = safe_float(
        strategy_metrics.get(
            "initial_capital"
        ),
        backtest_capital,
    )


    final_value = safe_float(
        strategy_metrics.get(
            "final_capital"
        ),
        backtest_capital,
    )


    total_profit = safe_float(
        strategy_metrics.get(
            "total_profit"
        ),
        0.0,
    )


    total_return = safe_float(
        strategy_metrics.get(
            "total_return"
        ),
        0.0,
    )


    trade_count = int(
        safe_float(
            strategy_metrics.get(
                "trade_count",
                strategy_metrics.get(
                    "trades",
                    0,
                ),
            ),
            0,
        )
    )


    wins = int(
        safe_float(
            strategy_metrics.get(
                "wins"
            ),
            0,
        )
    )


    losses = int(
        safe_float(
            strategy_metrics.get(
                "losses"
            ),
            0,
        )
    )


    flat = int(
        safe_float(
            strategy_metrics.get(
                "flat"
            ),
            0,
        )
    )


    win_rate = safe_float(
        strategy_metrics.get(
            "win_rate"
        )
    )


    profit_factor = safe_float(
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


    average_holding = safe_float(
        strategy_metrics.get(
            "average_holding_days"
        )
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


    gross_profit = safe_float(
        strategy_metrics.get(
            "gross_profit"
        ),
        0.0,
    )


    gross_loss = safe_float(
        strategy_metrics.get(
            "gross_loss"
        ),
        0.0,
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


    skipped_count = int(
        safe_float(
            strategy_metrics.get(
                "skipped_entries"
            ),
            0,
        )
    )


    total_commission = safe_float(
        strategy_metrics.get(
            "total_commission"
        ),
        0.0,
    )


    # ========================================================
    # 総合成績
    # ========================================================

    st.subheader(
        "📊 総合成績"
    )


    result_cols = st.columns(
        4
    )


    result_cols[
        0
    ].metric(
        "初期資金",
        (
            f"{initial_value:,.0f} 円"
        ),
    )


    result_cols[
        1
    ].metric(
        "最終資産",
        (
            f"{final_value:,.0f} 円"
        ),
        delta=(
            f"{total_profit:+,.0f} 円"
        ),
    )


    result_cols[
        2
    ].metric(
        "総利益",
        (
            f"{total_profit:+,.0f} 円"
        ),
    )


    result_cols[
        3
    ].metric(
        "総合リターン",
        (
            f"{total_return * 100:+.2f}%"
        ),
    )


    # ========================================================
    # 取引成績
    # ========================================================

    st.subheader(
        "🎯 取引成績"
    )


    trade_cols = st.columns(
        4
    )


    trade_cols[
        0
    ].metric(
        "取引回数",
        trade_count,
    )


    trade_cols[
        1
    ].metric(
        "勝ち",
        wins,
    )


    trade_cols[
        2
    ].metric(
        "負け",
        losses,
    )


    trade_cols[
        3
    ].metric(
        "勝率",
        (
            f"{win_rate * 100:.1f}%"

            if win_rate is not None

            else "N/A"
        ),
    )


    # ========================================================
    # リスク・効率
    # ========================================================

    st.subheader(
        "🛡️ リスク・効率"
    )


    risk_cols = st.columns(
        5
    )


    risk_cols[
        0
    ].metric(
        "プロフィットファクター",
        (
            f"{profit_factor:.2f}"

            if (
                profit_factor is not None
                and np.isfinite(
                    profit_factor
                )
            )

            else (
                "∞"

                if (
                    profit_factor is not None
                    and np.isinf(
                        profit_factor
                    )
                )

                else "N/A"
            )
        ),
    )


    risk_cols[
        1
    ].metric(
        "最大ドローダウン",
        (
            f"{max_drawdown * 100:.2f}%"
        ),
    )


    risk_cols[
        2
    ].metric(
        "平均保有日数",
        (
            f"{average_holding:.1f} 日"

            if average_holding is not None

            else "N/A"
        ),
    )


    risk_cols[
        3
    ].metric(
        "資金管理で見送り",
        (
            f"{skipped_count:,} 回"
        ),
    )


    risk_cols[
        4
    ].metric(
        "累計売買手数料",
        (
            f"{total_commission:,.0f} 円"
        ),
    )


    # ========================================================
    # 資産曲線
    # ========================================================

    if (
        strategy_equity is not None
        and not strategy_equity.empty
        and "Total_Equity"
        in strategy_equity.columns
    ):

        st.subheader(
            "📈 資産推移"
        )


        st.line_chart(
            strategy_equity[
                ["Total_Equity"]
            ]
        )


    # ========================================================
    # 売買履歴
    # ========================================================

    if (
        strategy_trades is not None
        and not strategy_trades.empty
    ):

        st.subheader(
            "🧾 売買履歴"
        )


        st.dataframe(
            strategy_trades,
            use_container_width=True,
        )


    else:

        st.info(
            "この設定では実際に成立した"
            "売買はありません。"
        )


    # ========================================================
    # 見送り履歴
    # ========================================================

    skipped_entries = (
        strategy_engine
        .get_skipped_entries()
    )


    if (
        skipped_entries is not None
        and not skipped_entries.empty
    ):

        with st.expander(
            "🔍 資金管理で見送ったBUY候補"
        ):

            st.dataframe(
                skipped_entries,
                use_container_width=True,
            )


    # ========================================================
    # 注文ログ
    # ========================================================

    order_log = (
        strategy_engine
        .get_order_log()
    )


    if (
        order_log is not None
        and not order_log.empty
    ):

        with st.expander(
            "📋 BUY / SELL シグナル・注文ログ"
        ):

            st.dataframe(
                order_log,
                use_container_width=True,
            )


    # ========================================================
    # ⑬ コピー・保存
    # ========================================================

    st.divider()


    st.subheader(
        "📋 バックテスト結果をコピー・保存"
    )


    st.write(
        "下のコピー用テキストは、"
        "そのままChatGPTに貼り付けて"
        "分析に使用できます。"
    )


    # --------------------------------------------------------
    # 表示文字列
    # --------------------------------------------------------

    if win_rate is None:

        win_rate_text = "N/A"

    else:

        win_rate_text = (
            f"{win_rate * 100:.1f}%"
        )


    if profit_factor is None:

        profit_factor_text = "N/A"

    elif np.isinf(
        profit_factor
    ):

        profit_factor_text = "∞"

    else:

        profit_factor_text = (
            f"{profit_factor:.2f}"
        )


    if average_holding is None:

        average_holding_text = "N/A"

    else:

        average_holding_text = (
            f"{average_holding:.1f} 日"
        )


    if average_profit is None:

        average_profit_text = "N/A"

    else:

        average_profit_text = (
            f"{average_profit:+,.0f} 円"
        )


    if average_profit_rate is None:

        average_profit_rate_text = "N/A"

    else:

        average_profit_rate_text = (
            f"{average_profit_rate * 100:+.2f}%"
        )


    if best_trade is None:

        best_trade_text = "N/A"

    else:

        best_trade_text = (
            f"{best_trade:+,.0f} 円"
        )


    if worst_trade is None:

        worst_trade_text = "N/A"

    else:

        worst_trade_text = (
            f"{worst_trade:+,.0f} 円"
        )


    # --------------------------------------------------------
    # コピー用テキスト
    # --------------------------------------------------------

    copy_text = f"""【⑬ 本格戦略バックテスト結果】

銘柄: {STOCK_NAME} ({STOCK_CODE})

■ バックテスト設定
バックテスト資金: {backtest_capital:,.0f} 円
最大投資比率: {selected_position_label}
1回の最大投資額: {maximum_position_value:,.0f} 円
売買単位: 100 株
1回の許容リスク: 1.00%
損切り: 5.00%
利益確定: 10.00%
BUY最低AI確率: 55.0%
BUY最低スコア: 6
AI SELL基準: 45.0%
最大保有日数: 10 日
売買手数料率: 0.10%
スリッページ率: 0.10%

■ 約定ルール
当日終値でBUY / SELL判定
翌営業日の始値で約定

■ 総合成績
初期資金: {initial_value:,.0f} 円
最終資産: {final_value:,.0f} 円
総利益: {total_profit:+,.0f} 円
総合リターン: {total_return * 100:+.2f}%

■ 取引成績
取引回数: {trade_count} 回
勝ち: {wins} 回
負け: {losses} 回
引き分け: {flat} 回
勝率: {win_rate_text}

■ リスク・効率
プロフィットファクター: {profit_factor_text}
最大ドローダウン: {max_drawdown * 100:.2f}%
平均保有日数: {average_holding_text}
平均損益: {average_profit_text}
平均損益率: {average_profit_rate_text}
総利益（勝ち取引）: {gross_profit:+,.0f} 円
総損失（負け取引）: -{gross_loss:,.0f} 円
ベストトレード: {best_trade_text}
ワーストトレード: {worst_trade_text}
資金管理で見送り: {skipped_count} 回
累計売買手数料: {total_commission:,.0f} 円
"""


    # --------------------------------------------------------
    # コピーしやすい表示
    # --------------------------------------------------------

    st.code(
        copy_text,
        language=None,
    )


    st.caption(
        "iPhoneでは上のテキスト欄の"
        "コピーアイコン、または長押しで"
        "コピーできます。"
    )


    # ========================================================
    # 成績CSV
    # ========================================================

    summary_data = {

        "銘柄":
            STOCK_NAME,

        "コード":
            STOCK_CODE,

        "バックテスト資金":
            backtest_capital,

        "最大投資比率":
            max_position_rate,

        "最大投資額":
            maximum_position_value,

        "売買単位":
            100,

        "許容リスク率":
            0.01,

        "損切り率":
            0.05,

        "利益確定率":
            0.10,

        "初期資金":
            initial_value,

        "最終資産":
            final_value,

        "総利益":
            total_profit,

        "総合リターン":
            total_return,

        "取引回数":
            trade_count,

        "勝ち":
            wins,

        "負け":
            losses,

        "引き分け":
            flat,

        "勝率":
            win_rate,

        "プロフィットファクター":
            profit_factor,

        "最大ドローダウン":
            max_drawdown,

        "平均保有日数":
            average_holding,

        "平均損益":
            average_profit,

        "平均損益率":
            average_profit_rate,

        "総利益_勝ち取引":
            gross_profit,

        "総損失_負け取引":
            gross_loss,

        "ベストトレード":
            best_trade,

        "ワーストトレード":
            worst_trade,

        "資金管理見送り":
            skipped_count,

        "累計売買手数料":
            total_commission,
    }


    summary_df = pd.DataFrame(
        [
            summary_data
        ]
    )


    # ========================================================
    # ダウンロードボタン
    # ========================================================

    st.subheader(
        "📥 CSVダウンロード"
    )


    download_col1, download_col2 = (
        st.columns(
            2
        )
    )


    download_col1.download_button(
        label=
            "📥 バックテスト成績CSV",

        data=
            dataframe_to_csv_bytes(
                summary_df
            ),

        file_name=
            "advantest_backtest_summary.csv",

        mime=
            "text/csv",

        use_container_width=
            True,
    )


    download_col2.download_button(
        label=
            "📥 売買履歴CSV",

        data=
            dataframe_to_csv_bytes(
                strategy_trades
            ),

        file_name=
            "advantest_trades.csv",

        mime=
            "text/csv",

        use_container_width=
            True,

        disabled=(
            strategy_trades is None
            or strategy_trades.empty
        ),
    )


    download_col3, download_col4 = (
        st.columns(
            2
        )
    )


    download_col3.download_button(
        label=
            "📥 見送り履歴CSV",

        data=
            dataframe_to_csv_bytes(
                skipped_entries
            ),

        file_name=
            "advantest_skipped_entries.csv",

        mime=
            "text/csv",

        use_container_width=
            True,

        disabled=(
            skipped_entries is None
            or skipped_entries.empty
        ),
    )


    download_col4.download_button(
        label=
            "📥 注文ログCSV",

        data=
            dataframe_to_csv_bytes(
                order_log
            ),

        file_name=
            "advantest_order_log.csv",

        mime=
            "text/csv",

        use_container_width=
            True,

        disabled=(
            order_log is None
            or order_log.empty
        ),
    )


    # ========================================================
    # 資産推移CSV
    # ========================================================

    if (
        strategy_equity is not None
        and not strategy_equity.empty
    ):

        equity_download = (
            strategy_equity
            .reset_index()
        )


        st.download_button(
            label=
                "📥 資産推移CSV",

            data=
                dataframe_to_csv_bytes(
                    equity_download
                ),

            file_name=
                "advantest_equity_curve.csv",

            mime=
                "text/csv",

            use_container_width=
                True,
        )


    st.success(
        "✅ ⑬のバックテストが完了しました。"
    )


except Exception as error:

    st.error(
        "⑬のバックテスト中に"
        "エラーが発生しました。"
    )


    st.exception(
        error
    )


# ============================================================
# 注意
# ============================================================

st.info(
    "バックテスト結果は過去データによる"
    "シミュレーションであり、"
    "将来の運用成果を保証するものではありません。"
)
