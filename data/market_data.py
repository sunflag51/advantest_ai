# ============================================================
# アドバンテスト AI売買システム
# data/market_data.py
#
# 市場データ取得 完全版
#
# 対象
# ・日経平均
# ・NASDAQ
# ・SOXX
# ・ドル円
#
# yfinanceを使用
# ============================================================


import pandas as pd
import yfinance as yf


# ============================================================
# 市場ティッカー
# ============================================================

MARKET_TICKERS = {
    "日経平均": "^N225",
    "NASDAQ": "^IXIC",
    "SOXX": "SOXX",
    "ドル円": "JPY=X",
}


# ============================================================
# 数値変換
# ============================================================

def _safe_float(value):
    """
    Seriesやnumpy数値などを
    安全にfloatへ変換する。
    """

    try:

        if value is None:
            return None

        # --------------------------------------------
        # pandas Series
        # --------------------------------------------

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
                return None

            return float(
                numeric.iloc[-1]
            )

        # --------------------------------------------
        # 通常の値
        # --------------------------------------------

        number = pd.to_numeric(
            value,
            errors="coerce",
        )

        if pd.isna(
            number
        ):

            return None

        return float(
            number
        )

    except Exception:

        return None


# ============================================================
# DataFrame整理
# ============================================================

def _normalize_market_dataframe(
    data,
    ticker=None,
):
    """
    yfinanceから取得したデータを
    通常のDataFrameへ整理する。

    yfinanceのバージョンによって
    MultiIndexになる場合にも対応。
    """

    if data is None:

        return pd.DataFrame()


    if not isinstance(
        data,
        pd.DataFrame,
    ):

        return pd.DataFrame()


    if data.empty:

        return pd.DataFrame()


    df = data.copy()


    # ========================================================
    # MultiIndex対策
    # ========================================================

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        # ----------------------------------------------------
        # tickerを指定して取り出せる場合
        # ----------------------------------------------------

        extracted = False


        if ticker is not None:

            # tickerが第2階層にあるケース
            try:

                if (
                    ticker
                    in df.columns.get_level_values(
                        1
                    )
                ):

                    df = df.xs(
                        ticker,
                        axis=1,
                        level=1,
                    )

                    extracted = True

            except Exception:

                pass


            # tickerが第1階層にあるケース
            if not extracted:

                try:

                    if (
                        ticker
                        in df.columns.get_level_values(
                            0
                        )
                    ):

                        df = df.xs(
                            ticker,
                            axis=1,
                            level=0,
                        )

                        extracted = True

                except Exception:

                    pass


        # ----------------------------------------------------
        # まだMultiIndexならOHLCV階層を探す
        # ----------------------------------------------------

        if isinstance(
            df.columns,
            pd.MultiIndex,
        ):

            standard_columns = {
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            }


            level0 = set(
                df.columns.get_level_values(
                    0
                )
            )


            level1 = set(
                df.columns.get_level_values(
                    1
                )
            )


            if (
                len(
                    standard_columns
                    & level0
                )
                > 0
            ):

                df.columns = (
                    df.columns.get_level_values(
                        0
                    )
                )

            elif (
                len(
                    standard_columns
                    & level1
                )
                > 0
            ):

                df.columns = (
                    df.columns.get_level_values(
                        1
                    )
                )

            else:

                df.columns = [
                    "_".join(
                        str(x)
                        for x in column
                    )

                    for column
                    in df.columns
                ]


    # ========================================================
    # 重複列を除去
    # ========================================================

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]


    # ========================================================
    # 日付インデックス
    # ========================================================

    try:

        df.index = pd.to_datetime(
            df.index
        )

    except Exception:

        pass


    # timezone除去
    try:

        if df.index.tz is not None:

            df.index = (
                df.index.tz_localize(
                    None
                )
            )

    except Exception:

        pass


    df = (
        df[
            ~df.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )


    # ========================================================
    # 数値列変換
    # ========================================================

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]


    for column in numeric_columns:

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
            )


    return df


# ============================================================
# 1市場データ取得
# ============================================================

def get_market_data(
    ticker,
    period="5y",
    interval="1d",
):
    """
    指定したティッカーの市場データを取得する。

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker

    period : str
        例: "5d", "1mo", "1y", "5y"

    interval : str
        例: "1d"

    Returns
    -------
    pandas.DataFrame
    """

    try:

        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )


        data = (
            _normalize_market_dataframe(
                data=data,
                ticker=ticker,
            )
        )


        if data.empty:

            return pd.DataFrame()


        # Closeが存在しないデータは使用しない
        if "Close" not in data.columns:

            return pd.DataFrame()


        # CloseがNaNの行を除外
        data = data.dropna(
            subset=[
                "Close"
            ]
        )


        return data


    except TypeError:

        # ====================================================
        # 古いyfinanceで
        # multi_level_indexが使えない場合
        # ====================================================

        try:

            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )


            data = (
                _normalize_market_dataframe(
                    data=data,
                    ticker=ticker,
                )
            )


            if data.empty:

                return pd.DataFrame()


            if "Close" not in data.columns:

                return pd.DataFrame()


            data = data.dropna(
                subset=[
                    "Close"
                ]
            )


            return data


        except Exception:

            return pd.DataFrame()


    except Exception:

        return pd.DataFrame()


# ============================================================
# 全市場データ取得
# ============================================================

def get_all_market_data(
    period="5y",
    interval="1d",
):
    """
    日経平均・NASDAQ・SOXX・ドル円を
    まとめて取得する。

    Returns
    -------
    dict

    {
        "日経平均": DataFrame,
        "NASDAQ": DataFrame,
        "SOXX": DataFrame,
        "ドル円": DataFrame
    }
    """

    market_data = {}


    for (
        market_name,
        ticker,
    ) in MARKET_TICKERS.items():

        data = get_market_data(
            ticker=ticker,
            period=period,
            interval=interval,
        )


        market_data[
            market_name
        ] = data


    return market_data


# ============================================================
# 最新値取得
# ============================================================

def get_latest_market_values(
    period="5d",
    interval="1d",
):
    """
    各市場の最新終値を取得する。

    重要：
    この関数は必ず

    {
        "日経平均": float または None,
        "NASDAQ": float または None,
        "SOXX": float または None,
        "ドル円": float または None
    }

    の形式で返す。
    """

    results = {}


    for (
        market_name,
        ticker,
    ) in MARKET_TICKERS.items():

        data = get_market_data(
            ticker=ticker,
            period=period,
            interval=interval,
        )


        # --------------------------------------------
        # 取得失敗
        # --------------------------------------------

        if (
            data is None
            or data.empty
        ):

            results[
                market_name
            ] = None

            continue


        # --------------------------------------------
        # Close列確認
        # --------------------------------------------

        if "Close" not in data.columns:

            results[
                market_name
            ] = None

            continue


        # --------------------------------------------
        # 最新の有効終値
        # --------------------------------------------

        close_series = (
            pd.to_numeric(
                data["Close"],
                errors="coerce",
            )
            .dropna()
        )


        if close_series.empty:

            results[
                market_name
            ] = None

            continue


        latest_value = (
            _safe_float(
                close_series.iloc[-1]
            )
        )


        results[
            market_name
        ] = latest_value


    return results


# ============================================================
# 市場データ結合
# ============================================================

def combine_market_data(
    market_data,
):
    """
    各市場のCloseを1つのDataFrameにまとめる。

    出力例：

                日経平均    NASDAQ    SOXX    ドル円
    Date
    ...
    """

    combined = {}


    if (
        market_data is None
        or not isinstance(
            market_data,
            dict,
        )
    ):

        return pd.DataFrame()


    for (
        market_name,
        data,
    ) in market_data.items():

        if (
            data is None
            or not isinstance(
                data,
                pd.DataFrame,
            )
            or data.empty
        ):

            continue


        if "Close" not in data.columns:

            continue


        close_series = (
            pd.to_numeric(
                data["Close"],
                errors="coerce",
            )
        )


        close_series.name = (
            market_name
        )


        combined[
            market_name
        ] = close_series


    if not combined:

        return pd.DataFrame()


    result = pd.concat(
        combined.values(),
        axis=1,
    )


    result.columns = list(
        combined.keys()
    )


    result = (
        result
        .sort_index()
        .ffill()
    )


    return result


# ============================================================
# 市場リターン計算
# ============================================================

def calculate_market_returns(
    market_data,
):
    """
    各市場のリターンを計算する。

    Returns
    -------
    DataFrame

    例:
    日経平均_Return_1D
    NASDAQ_Return_1D
    SOXX_Return_1D
    ドル円_Return_1D
    """

    combined = (
        combine_market_data(
            market_data
        )
    )


    if combined.empty:

        return pd.DataFrame()


    returns = pd.DataFrame(
        index=combined.index
    )


    for column in combined.columns:

        series = (
            combined[
                column
            ]
        )


        returns[
            f"{column}_Return_1D"
        ] = (
            series.pct_change(
                periods=1,
                fill_method=None,
            )
        )


        returns[
            f"{column}_Return_5D"
        ] = (
            series.pct_change(
                periods=5,
                fill_method=None,
            )
        )


        returns[
            f"{column}_Return_20D"
        ] = (
            series.pct_change(
                periods=20,
                fill_method=None,
            )
        )


    return returns


# ============================================================
# 市場データ取得状況
# ============================================================

def get_market_data_status(
    market_data,
):
    """
    デバッグ・確認用。

    各市場について
    取得できているか、
    行数、
    最新日、
    最新終値を返す。
    """

    status = {}


    for (
        market_name,
        ticker,
    ) in MARKET_TICKERS.items():

        data = None


        if isinstance(
            market_data,
            dict,
        ):

            data = market_data.get(
                market_name
            )


        if (
            data is None
            or not isinstance(
                data,
                pd.DataFrame,
            )
            or data.empty
        ):

            status[
                market_name
            ] = {
                "ticker":
                    ticker,

                "available":
                    False,

                "rows":
                    0,

                "latest_date":
                    None,

                "latest_close":
                    None,
            }

            continue


        latest_close = None


        if "Close" in data.columns:

            close_series = (
                pd.to_numeric(
                    data["Close"],
                    errors="coerce",
                )
                .dropna()
            )


            if not close_series.empty:

                latest_close = (
                    _safe_float(
                        close_series.iloc[-1]
                    )
                )


        status[
            market_name
        ] = {
            "ticker":
                ticker,

            "available":
                True,

            "rows":
                len(data),

            "latest_date":
                data.index[-1],

            "latest_close":
                latest_close,
        }


    return status


# ============================================================
# 単独テスト
# ============================================================

if __name__ == "__main__":

    print(
        "市場データ取得テスト"
    )

    print(
        "=" * 50
    )


    # --------------------------------------------------------
    # 5年データ
    # --------------------------------------------------------

    market_data = (
        get_all_market_data(
            period="5y",
            interval="1d",
        )
    )


    status = (
        get_market_data_status(
            market_data
        )
    )


    for (
        market_name,
        info,
    ) in status.items():

        print()

        print(
            market_name
        )

        print(
            info
        )


    # --------------------------------------------------------
    # 最新値
    # --------------------------------------------------------

    print()

    print(
        "=" * 50
    )

    print(
        "最新市場値"
    )

    print(
        "=" * 50
    )


    latest_values = (
        get_latest_market_values(
            period="5d",
            interval="1d",
        )
    )


    for (
        market_name,
        value,
    ) in latest_values.items():

        print(
            market_name,
            value,
        )
