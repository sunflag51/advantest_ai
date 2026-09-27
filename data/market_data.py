# ============================================================
# アドバンテスト AI売買システム
# data/market_data.py
#
# 市場全体のデータを取得するプログラム
#
# 取得対象
# ・日経平均
# ・NASDAQ
# ・半導体関連ETF（SOXX）
# ・ドル円
#
# ※このファイルでは売買判断を行いません
# ============================================================


import yfinance as yf
import pandas as pd


# ============================================================
# 市場データの設定
# ============================================================

MARKET_TICKERS = {

    # 日本株市場
    "日経平均": "^N225",

    # 米国ハイテク市場
    "NASDAQ": "^IXIC",

    # 半導体関連
    "SOXX": "SOXX",

    # ドル円
    "ドル円": "JPY=X",
}


# ============================================================
# 1つの市場データを取得
# ============================================================

def get_market_data(
    ticker,
    period="5y",
    interval="1d"
):
    """
    指定した市場データを取得する

    Parameters
    ----------
    ticker : str
        Yahoo! Financeのティッカー

    period : str
        取得期間

    interval : str
        データ間隔

    Returns
    -------
    pandas.DataFrame
        市場データ
    """

    try:

        # ----------------------------------------------------
        # データ取得
        # ----------------------------------------------------

        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            multi_level_index=False
        )


        # ----------------------------------------------------
        # データが空の場合
        # ----------------------------------------------------

        if data is None or data.empty:

            raise ValueError(
                f"{ticker} のデータを取得できませんでした。"
            )


        # ----------------------------------------------------
        # 必要な列
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]


        # ----------------------------------------------------
        # 存在する列だけ確認
        # ----------------------------------------------------

        available_columns = [
            column
            for column in required_columns
            if column in data.columns
        ]


        if "Close" not in available_columns:

            raise ValueError(
                f"{ticker} に終値データがありません。"
            )


        # ----------------------------------------------------
        # 必要な列だけ使用
        # ----------------------------------------------------

        data = data[
            available_columns
        ].copy()


        # ----------------------------------------------------
        # 数値化
        # ----------------------------------------------------

        for column in available_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )


        # ----------------------------------------------------
        # 欠損値削除
        # ----------------------------------------------------

        data.dropna(
            subset=["Close"],
            inplace=True
        )


        # ----------------------------------------------------
        # 日付順に並べる
        # ----------------------------------------------------

        data.sort_index(
            inplace=True
        )


        # ----------------------------------------------------
        # データ確認
        # ----------------------------------------------------

        if data.empty:

            raise ValueError(
                f"{ticker} の有効なデータがありません。"
            )


        return data


    except Exception as e:

        raise RuntimeError(
            f"市場データ取得エラー [{ticker}]：{e}"
        ) from e


# ============================================================
# すべての市場データを取得
# ============================================================

def get_all_market_data(
    period="5y",
    interval="1d"
):
    """
    日経平均・NASDAQ・SOXX・ドル円をまとめて取得する

    Returns
    -------
    dict
        各市場データを辞書として返す
    """

    market_data = {}

    errors = {}


    # --------------------------------------------------------
    # 各市場データを取得
    # --------------------------------------------------------

    for market_name, ticker in MARKET_TICKERS.items():

        try:

            data = get_market_data(
                ticker=ticker,
                period=period,
                interval=interval
            )

            market_data[market_name] = data


        except Exception as e:

            errors[market_name] = str(e)


    # --------------------------------------------------------
    # 取得できなかったものがあれば記録
    # --------------------------------------------------------

    if errors:

        market_data["_errors"] = errors


    return market_data


# ============================================================
# 最新値取得
# ============================================================

def get_latest_market_values(
    period="5d"
):
    """
    各市場の最新終値を取得する

    Returns
    -------
    dict
        最新値
    """

    latest_values = {}

    for market_name, ticker in MARKET_TICKERS.items():

        try:

            data = get_market_data(
                ticker=ticker,
                period=period,
                interval="1d"
            )

            if data.empty:

                latest_values[market_name] = None

                continue


            latest_close = float(
                data["Close"].iloc[-1]
            )


            latest_values[market_name] = {
                "ticker": ticker,
                "date": data.index[-1],
                "close": latest_close
            }


        except Exception as e:

            latest_values[market_name] = {
                "ticker": ticker,
                "date": None,
                "close": None,
                "error": str(e)
            }


    return latest_values


# ============================================================
# 市場データを1つのDataFrameに統合
# ============================================================

def combine_market_data(
    period="5y"
):
    """
    各市場の終値を日付で統合する

    AIの特徴量作成時に利用することを想定
    """

    combined = pd.DataFrame()


    for market_name, ticker in MARKET_TICKERS.items():

        try:

            data = get_market_data(
                ticker=ticker,
                period=period,
                interval="1d"
            )


            close_data = data[
                ["Close"]
            ].copy()


            close_data.rename(
                columns={
                    "Close": market_name
                },
                inplace=True
            )


            if combined.empty:

                combined = close_data

            else:

                combined = combined.join(
                    close_data,
                    how="outer"
                )


        except Exception:

            # 取得できない市場はスキップ
            continue


    # --------------------------------------------------------
    # 日付順
    # --------------------------------------------------------

    combined.sort_index(
        inplace=True
    )


    # --------------------------------------------------------
    # 前日の値で欠損を補完
    #
    # 日本市場と米国市場では休場日が違うため、
    # 単純に削除せず前値を使用する
    # --------------------------------------------------------

    combined = combined.ffill()


    return combined


# ============================================================
# 市場騰落率を計算
# ============================================================

def calculate_market_returns(
    period="5y"
):
    """
    各市場の前日比を計算する
    """

    combined = combine_market_data(
        period=period
    )


    if combined.empty:

        return combined


    returns = combined.pct_change() * 100


    return returns
