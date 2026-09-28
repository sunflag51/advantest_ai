# research/holding_period_test.py
# ============================================================
# アドバンテスト（6857.T）
# 保有営業日別 基礎調査プログラム
#
# 目的：
# AIを一切使わず、
# 「何営業日保有した場合に利益になりやすかったか」
# を直近2年間の実際の株価から調べる。
#
# 買い：
# 各営業日の始値（Open）
#
# 売り：
# 1～10営業日後の終値（Close）
#
# 例：
# 月曜日の始値で買う
# 1営業日後 → 火曜日の終値で売る
# 2営業日後 → 水曜日の終値で売る
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# 設定
# ============================================================

TICKER = "6857.T"
PERIOD = "2y"

# 何営業日後まで調査するか
MAX_HOLDING_DAYS = 10


# ============================================================
# 株価データ取得
# ============================================================

def get_stock_data():

    print("=" * 60)
    print("アドバンテスト 保有営業日別 基礎調査")
    print("=" * 60)

    print()
    print("株価データを取得しています...")

    data = yf.download(
        TICKER,
        period=PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError("株価データを取得できませんでした。")

    # yfinanceのバージョンによっては
    # 列がMultiIndexになるため通常の列名へ直す
    if isinstance(data.columns, pd.MultiIndex):

        if TICKER in data.columns.get_level_values(-1):
            data = data.xs(
                TICKER,
                axis=1,
                level=-1,
                drop_level=True,
            )

        else:
            data.columns = data.columns.get_level_values(0)

    # 必要なデータだけ残す
    required_columns = ["Open", "Close"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"{column} が株価データにありません。"
            )

    data = data.dropna(
        subset=["Open", "Close"]
    ).copy()

    print("株価データ取得完了")
    print()

    print(
        f"データ期間："
        f"{data.index.min().date()} ～ "
        f"{data.index.max().date()}"
    )

    print(
        f"営業日数：{len(data):,} 日"
    )

    print()

    return data


# ============================================================
# 保有日数別調査
# ============================================================

def calculate_holding_periods(data):

    results = []

    print("=" * 60)
    print("1～10営業日の調査を開始します")
    print("=" * 60)

    print()

    for holding_days in range(
        1,
        MAX_HOLDING_DAYS + 1
    ):

        df = data.copy()

        # ----------------------------------------------------
        # 買値
        # ----------------------------------------------------
        #
        # その日の始値で買う
        #
        df["Buy_Price"] = df["Open"]

        # ----------------------------------------------------
        # 売値
        # ----------------------------------------------------
        #
        # holding_days 営業日後の終値で売る
        #
        # 1営業日なら
        # 今日の始値 → 次営業日の終値
        #
        df["Sell_Price"] = (
            df["Close"].shift(-holding_days)
        )

        # ----------------------------------------------------
        # 利益率
        # ----------------------------------------------------

        df["Return"] = (
            df["Sell_Price"]
            / df["Buy_Price"]
            - 1
        )

        # 売値が存在しない最後の数日を除外
        trades = df.dropna(
            subset=["Buy_Price", "Sell_Price"]
        ).copy()

        if trades.empty:
            continue

        # ----------------------------------------------------
        # 基本成績
        # ----------------------------------------------------

        trade_count = len(trades)

        wins = trades[
            trades["Return"] > 0
        ]

        losses = trades[
            trades["Return"] < 0
        ]

        win_count = len(wins)
        loss_count = len(losses)

        win_rate = (
            win_count / trade_count
        )

        average_return = (
            trades["Return"].mean()
        )

        median_return = (
            trades["Return"].median()
        )

        max_return = (
            trades["Return"].max()
        )

        min_return = (
            trades["Return"].min()
        )

        # ----------------------------------------------------
        # 上昇時・下落時の平均
        # ----------------------------------------------------

        if not wins.empty:
            average_win = wins["Return"].mean()
        else:
            average_win = np.nan

        if not losses.empty:
            average_loss = losses["Return"].mean()
        else:
            average_loss = np.nan

        # ----------------------------------------------------
        # 結果保存
        # ----------------------------------------------------

        results.append(
            {
                "保有営業日": holding_days,
                "取引回数": trade_count,
                "利益回数": win_count,
                "損失回数": loss_count,
                "勝率": win_rate,
                "平均利益率": average_return,
                "中央値利益率": median_return,
                "最大利益率": max_return,
                "最大損失率": min_return,
                "勝ち平均": average_win,
                "負け平均": average_loss,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# 結果表示
# ============================================================

def display_results(results):

    print()
    print("=" * 60)
    print("調査結果")
    print("=" * 60)

    print()

    display_df = results.copy()

    percentage_columns = [
        "勝率",
        "平均利益率",
        "中央値利益率",
        "最大利益率",
        "最大損失率",
        "勝ち平均",
        "負け平均",
    ]

    for column in percentage_columns:

        display_df[column] = (
            display_df[column]
            .apply(
                lambda x:
                f"{x * 100:+.2f}%"
                if pd.notna(x)
                else "-"
            )
        )

    # 勝率だけ + を消す
    display_df["勝率"] = (
        results["勝率"]
        .apply(
            lambda x:
            f"{x * 100:.1f}%"
        )
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    print()


# ============================================================
# 簡単なまとめ
# ============================================================

def display_summary(results):

    print("=" * 60)
    print("簡単なまとめ")
    print("=" * 60)

    print()

    # --------------------------------------------------------
    # 勝率が最も高い
    # --------------------------------------------------------

    best_win_rate = results.loc[
        results["勝率"].idxmax()
    ]

    print(
        "勝率が最も高かった保有期間："
        f"{int(best_win_rate['保有営業日'])}営業日"
    )

    print(
        f"勝率："
        f"{best_win_rate['勝率'] * 100:.1f}%"
    )

    print()

    # --------------------------------------------------------
    # 平均利益率が最も高い
    # --------------------------------------------------------

    best_average = results.loc[
        results["平均利益率"].idxmax()
    ]

    print(
        "平均利益率が最も高かった保有期間："
        f"{int(best_average['保有営業日'])}営業日"
    )

    print(
        f"平均利益率："
        f"{best_average['平均利益率'] * 100:+.2f}%"
    )

    print()

    # --------------------------------------------------------
    # 中央値が最も高い
    # --------------------------------------------------------

    best_median = results.loc[
        results["中央値利益率"].idxmax()
    ]

    print(
        "中央値利益率が最も高かった保有期間："
        f"{int(best_median['保有営業日'])}営業日"
    )

    print(
        f"中央値利益率："
        f"{best_median['中央値利益率'] * 100:+.2f}%"
    )

    print()

    print("-" * 60)

    print()

    print(
        "※これは直近2年間の過去データを"
        "調査した結果です。"
    )

    print(
        "※AIによる予測は一切使用していません。"
    )

    print(
        "※手数料・スリッページ・税金は"
        "含めていません。"
    )

    print(
        "※過去に良かった保有日数が"
        "将来も良いとは限りません。"
    )

    print()


# ============================================================
# CSV保存
# ============================================================

def save_results(results):

    filename = (
        "advantest_holding_period_results.csv"
    )

    results.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"結果を保存しました：{filename}"
    )

    print()


# ============================================================
# メイン処理
# ============================================================

def main():

    try:

        # 株価取得
        data = get_stock_data()

        # 1～10営業日を調査
        results = calculate_holding_periods(
            data
        )

        # 結果表示
        display_results(
            results
        )

        # 簡単なまとめ
        display_summary(
            results
        )

        # CSV保存
        save_results(
            results
        )

        print("=" * 60)
        print("調査完了")
        print("=" * 60)

    except Exception as e:

        print()
        print("=" * 60)
        print("エラーが発生しました")
        print("=" * 60)

        print()
        print(str(e))
        print()


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()
