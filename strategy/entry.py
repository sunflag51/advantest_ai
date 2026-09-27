# ============================================================
# アドバンテスト AI売買システム
# strategy/entry.py
#
# エントリー（買い）判断
#
# 使用する情報
# ・AI上昇確率
# ・移動平均トレンド
# ・RSI
# ・MACD
# ・出来高
# ・日経平均
# ・NASDAQ
# ・SOXX
# ・ドル円
#
# 出力
# ・BUY / WAIT
# ・エントリースコア
# ・判定理由
#
# ※実際の注文は行わない
# ============================================================


import numpy as np
import pandas as pd


# ============================================================
# エントリー戦略
# ============================================================

class EntryStrategy:

    """
    AIとテクニカル指標、市場環境を組み合わせて
    エントリー判断を行うクラス。
    """

    # ========================================================
    # 初期化
    # ========================================================

    def __init__(
        self,
        minimum_score=6,
        minimum_probability=0.55,
        strong_probability=0.60,
        rsi_min=40.0,
        rsi_max=70.0,
        volume_ratio_min=1.0
    ):

        """
        minimum_score
            BUYに必要な最低スコア

        minimum_probability
            エントリー検討に必要な最低AI上昇確率

        strong_probability
            AIを強い上昇予測とみなす確率

        rsi_min / rsi_max
            RSIの許容範囲

        volume_ratio_min
            出来高確認に使用する基準
        """

        self.minimum_score = int(
            minimum_score
        )

        self.minimum_probability = float(
            minimum_probability
        )

        self.strong_probability = float(
            strong_probability
        )

        self.rsi_min = float(
            rsi_min
        )

        self.rsi_max = float(
            rsi_max
        )

        self.volume_ratio_min = float(
            volume_ratio_min
        )


        if self.minimum_score < 1:

            raise ValueError(
                "minimum_scoreは1以上にしてください。"
            )


        if not 0.0 < self.minimum_probability < 1.0:

            raise ValueError(
                "minimum_probabilityは0〜1の間にしてください。"
            )


        if not 0.0 < self.strong_probability < 1.0:

            raise ValueError(
                "strong_probabilityは0〜1の間にしてください。"
            )


        if (
            self.strong_probability
            < self.minimum_probability
        ):

            raise ValueError(
                "strong_probabilityは"
                "minimum_probability以上にしてください。"
            )


        if self.rsi_min >= self.rsi_max:

            raise ValueError(
                "rsi_minはrsi_maxより"
                "小さくしてください。"
            )


    # ========================================================
    # 数値取得
    # ========================================================

    @staticmethod
    def _get_value(
        row,
        column
    ):

        """
        行データから安全に数値を取得する。
        """

        if row is None:

            return None


        try:

            value = row.get(
                column
            )

        except AttributeError:

            return None


        if value is None:

            return None


        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return None


        if not np.isfinite(
            value
        ):

            return None


        return value


    # ========================================================
    # AI評価
    # ========================================================

    def evaluate_ai(
        self,
        probability_up
    ):

        """
        AI上昇確率を評価する。

        最大3点
        """

        score = 0

        reasons = []


        if probability_up is None:

            reasons.append(
                "AI上昇確率なし"
            )

            return score, reasons


        probability_up = float(
            probability_up
        )


        if (
            probability_up
            >= 0.70
        ):

            score += 3

            reasons.append(
                "AI上昇確率70%以上"
            )


        elif (
            probability_up
            >= self.strong_probability
        ):

            score += 2

            reasons.append(
                "AI上昇確率が強い"
            )


        elif (
            probability_up
            >= self.minimum_probability
        ):

            score += 1

            reasons.append(
                "AI上昇確率が最低基準以上"
            )


        else:

            reasons.append(
                "AI上昇確率が最低基準未満"
            )


        return score, reasons


    # ========================================================
    # トレンド評価
    # ========================================================

    def evaluate_trend(
        self,
        row
    ):

        """
        移動平均と株価位置から
        トレンドを評価する。

        最大3点
        """

        score = 0

        reasons = []


        close = self._get_value(
            row,
            "Close"
        )

        sma5 = self._get_value(
            row,
            "SMA_5"
        )

        sma25 = self._get_value(
            row,
            "SMA_25"
        )

        sma75 = self._get_value(
            row,
            "SMA_75"
        )


        # ----------------------------------------------------
        # 株価 > 25日線
        # ----------------------------------------------------

        if (
            close is not None
            and sma25 is not None
        ):

            if close > sma25:

                score += 1

                reasons.append(
                    "株価が25日移動平均より上"
                )

            else:

                reasons.append(
                    "株価が25日移動平均以下"
                )


        # ----------------------------------------------------
        # 5日線 > 25日線
        # ----------------------------------------------------

        if (
            sma5 is not None
            and sma25 is not None
        ):

            if sma5 > sma25:

                score += 1

                reasons.append(
                    "5日線が25日線より上"
                )

            else:

                reasons.append(
                    "5日線が25日線以下"
                )


        # ----------------------------------------------------
        # 25日線 > 75日線
        # ----------------------------------------------------

        if (
            sma25 is not None
            and sma75 is not None
        ):

            if sma25 > sma75:

                score += 1

                reasons.append(
                    "25日線が75日線より上"
                )

            else:

                reasons.append(
                    "25日線が75日線以下"
                )


        return score, reasons


    # ========================================================
    # RSI評価
    # ========================================================

    def evaluate_rsi(
        self,
        row
    ):

        """
        RSI評価。

        最大1点
        """

        score = 0

        reasons = []


        rsi = self._get_value(
            row,
            "RSI_14"
        )


        if rsi is None:

            reasons.append(
                "RSIデータなし"
            )

            return score, reasons


        if (
            self.rsi_min
            <= rsi
            <= self.rsi_max
        ):

            score += 1

            reasons.append(
                f"RSIが適正範囲 ({rsi:.1f})"
            )


        elif rsi > self.rsi_max:

            reasons.append(
                f"RSIが過熱気味 ({rsi:.1f})"
            )


        else:

            reasons.append(
                f"RSIが弱い ({rsi:.1f})"
            )


        return score, reasons


    # ========================================================
    # MACD評価
    # ========================================================

    def evaluate_macd(
        self,
        row
    ):

        """
        MACD評価。

        最大1点
        """

        score = 0

        reasons = []


        macd = self._get_value(
            row,
            "MACD"
        )

        signal = self._get_value(
            row,
            "MACD_Signal"
        )


        if (
            macd is None
            or signal is None
        ):

            reasons.append(
                "MACDデータなし"
            )

            return score, reasons


        if macd > signal:

            score += 1

            reasons.append(
                "MACDがシグナルより上"
            )


        else:

            reasons.append(
                "MACDがシグナル以下"
            )


        return score, reasons


    # ========================================================
    # 出来高評価
    # ========================================================

    def evaluate_volume(
        self,
        row
    ):

        """
        出来高評価。

        最大1点
        """

        score = 0

        reasons = []


        volume_ratio = self._get_value(
            row,
            "Volume_Ratio"
        )


        if volume_ratio is None:

            reasons.append(
                "出来高比率データなし"
            )

            return score, reasons


        if (
            volume_ratio
            >= self.volume_ratio_min
        ):

            score += 1

            reasons.append(
                f"出来高が平均以上 ({volume_ratio:.2f}倍)"
            )


        else:

            reasons.append(
                f"出来高が平均未満 ({volume_ratio:.2f}倍)"
            )


        return score, reasons


    # ========================================================
    # 市場評価
    # ========================================================

    def evaluate_market(
        self,
        row
    ):

        """
        市場環境を評価する。

        日経平均
        NASDAQ
        SOXX
        ドル円

        最大4点
        """

        score = 0

        reasons = []


        # ----------------------------------------------------
        # 日経平均
        # ----------------------------------------------------

        nikkei = self._get_value(
            row,
            "日経平均_Return_1D"
        )


        if nikkei is not None:

            if nikkei > 0:

                score += 1

                reasons.append(
                    "日経平均が上昇"
                )

            else:

                reasons.append(
                    "日経平均が下落"
                )


        else:

            reasons.append(
                "日経平均データなし"
            )


        # ----------------------------------------------------
        # NASDAQ
        # ----------------------------------------------------

        nasdaq = self._get_value(
            row,
            "NASDAQ_Return_1D"
        )


        if nasdaq is not None:

            if nasdaq > 0:

                score += 1

                reasons.append(
                    "NASDAQが上昇"
                )

            else:

                reasons.append(
                    "NASDAQが下落"
                )


        else:

            reasons.append(
                "NASDAQデータなし"
            )


        # ----------------------------------------------------
        # SOXX
        # ----------------------------------------------------

        soxx = self._get_value(
            row,
            "SOXX_Return_1D"
        )


        if soxx is not None:

            if soxx > 0:

                score += 1

                reasons.append(
                    "SOXXが上昇"
                )

            else:

                reasons.append(
                    "SOXXが下落"
                )


        else:

            reasons.append(
                "SOXXデータなし"
            )


        # ----------------------------------------------------
        # ドル円
        #
        # 上昇 = 円安方向
        #
        # ここでは単純に1点加算するが、
        # 将来バックテストで効果を検証する。
        # ----------------------------------------------------

        usd_jpy = self._get_value(
            row,
            "ドル円_Return_1D"
        )


        if usd_jpy is not None:

            if usd_jpy > 0:

                score += 1

                reasons.append(
                    "ドル円が円安方向"
                )

            else:

                reasons.append(
                    "ドル円が円高方向"
                )


        else:

            reasons.append(
                "ドル円データなし"
            )


        return score, reasons


    # ========================================================
    # 総合エントリー判断
    # ========================================================

    def evaluate(
        self,
        row,
        probability_up
    ):

        """
        総合エントリー判断。

        戻り値例

        {
            "action": "BUY",
            "score": 9,
            "max_score": 13,
            "score_rate": 0.69,
            "probability_up": 0.64,
            "reasons": [...]
        }
        """

        if row is None:

            raise ValueError(
                "エントリー判断用データがありません。"
            )


        try:

            probability_up = float(
                probability_up
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                "AI上昇確率が正しくありません。"
            )


        if not np.isfinite(
            probability_up
        ):

            raise ValueError(
                "AI上昇確率が正しくありません。"
            )


        if not 0.0 <= probability_up <= 1.0:

            raise ValueError(
                "AI上昇確率は0〜1で指定してください。"
            )


        total_score = 0

        all_reasons = []


        # ----------------------------------------------------
        # AI
        # 最大3点
        # ----------------------------------------------------

        ai_score, ai_reasons = (
            self.evaluate_ai(
                probability_up
            )
        )


        total_score += ai_score

        all_reasons.extend(
            ai_reasons
        )


        # ----------------------------------------------------
        # トレンド
        # 最大3点
        # ----------------------------------------------------

        trend_score, trend_reasons = (
            self.evaluate_trend(
                row
            )
        )


        total_score += trend_score

        all_reasons.extend(
            trend_reasons
        )


        # ----------------------------------------------------
        # RSI
        # 最大1点
        # ----------------------------------------------------

        rsi_score, rsi_reasons = (
            self.evaluate_rsi(
                row
            )
        )


        total_score += rsi_score

        all_reasons.extend(
            rsi_reasons
        )


        # ----------------------------------------------------
        # MACD
        # 最大1点
        # ----------------------------------------------------

        macd_score, macd_reasons = (
            self.evaluate_macd(
                row
            )
        )


        total_score += macd_score

        all_reasons.extend(
            macd_reasons
        )


        # ----------------------------------------------------
        # 出来高
        # 最大1点
        # ----------------------------------------------------

        volume_score, volume_reasons = (
            self.evaluate_volume(
                row
            )
        )


        total_score += volume_score

        all_reasons.extend(
            volume_reasons
        )


        # ----------------------------------------------------
        # 市場
        # 最大4点
        # ----------------------------------------------------

        market_score, market_reasons = (
            self.evaluate_market(
                row
            )
        )


        total_score += market_score

        all_reasons.extend(
            market_reasons
        )


        # ----------------------------------------------------
        # 最大スコア
        # ----------------------------------------------------

        max_score = 13


        score_rate = (
            total_score
            / max_score
        )


        # ====================================================
        # BUY条件
        #
        # 1. AI確率が最低基準以上
        # 2. 総合スコアが最低基準以上
        #
        # AI確率が低い場合、
        # 他の条件だけでBUYにならないようにする。
        # ====================================================

        probability_ok = (
            probability_up
            >= self.minimum_probability
        )


        score_ok = (
            total_score
            >= self.minimum_score
        )


        if (
            probability_ok
            and score_ok
        ):

            action = "BUY"

        else:

            action = "WAIT"


        # ----------------------------------------------------
        # 判定説明
        # ----------------------------------------------------

        if action == "BUY":

            summary = (
                "AI確率と総合スコアが"
                "エントリー基準を満たしています。"
            )

        elif not probability_ok:

            summary = (
                "AI上昇確率が"
                "エントリー最低基準を満たしていません。"
            )

        else:

            summary = (
                "AI上昇確率は基準以上ですが、"
                "総合スコアが不足しています。"
            )


        return {

            "action":
                action,

            "score":
                int(
                    total_score
                ),

            "max_score":
                int(
                    max_score
                ),

            "score_rate":
                float(
                    score_rate
                ),

            "minimum_score":
                int(
                    self.minimum_score
                ),

            "probability_up":
                float(
                    probability_up
                ),

            "minimum_probability":
                float(
                    self.minimum_probability
                ),

            "probability_ok":
                bool(
                    probability_ok
                ),

            "score_ok":
                bool(
                    score_ok
                ),

            "summary":
                summary,

            "reasons":
                all_reasons,

            "details": {

                "ai_score":
                    int(
                        ai_score
                    ),

                "trend_score":
                    int(
                        trend_score
                    ),

                "rsi_score":
                    int(
                        rsi_score
                    ),

                "macd_score":
                    int(
                        macd_score
                    ),

                "volume_score":
                    int(
                        volume_score
                    ),

                "market_score":
                    int(
                        market_score
                    )
            }
        }


    # ========================================================
    # DataFrameの特定日を評価
    # ========================================================

    def evaluate_date(
        self,
        data,
        date,
        probability_up
    ):

        """
        DataFrameの指定日の情報を使って
        エントリー判断を行う。
        """

        if data is None or data.empty:

            raise ValueError(
                "エントリー判断用データがありません。"
            )


        target_date = pd.Timestamp(
            date
        )


        if target_date not in data.index:

            raise ValueError(
                f"{target_date.date()} のデータがありません。"
            )


        row = data.loc[
            target_date
        ]


        if isinstance(
            row,
            pd.DataFrame
        ):

            row = row.iloc[-1]


        return self.evaluate(
            row=row,
            probability_up=probability_up
        )


    # ========================================================
    # 最新日の評価
    # ========================================================

    def evaluate_latest(
        self,
        data,
        probability_up
    ):

        """
        最新データを使って
        エントリー判断を行う。
        """

        if data is None or data.empty:

            raise ValueError(
                "エントリー判断用データがありません。"
            )


        row = data.iloc[-1]


        result = self.evaluate(
            row=row,
            probability_up=probability_up
        )


        result[
            "date"
        ] = data.index[-1]


        return result


# ============================================================
# 簡単実行関数
# ============================================================

def evaluate_entry(
    data,
    probability_up,
    minimum_score=6,
    minimum_probability=0.55,
    strong_probability=0.60,
    rsi_min=40.0,
    rsi_max=70.0,
    volume_ratio_min=1.0
):

    """
    最新データのエントリー判断を
    一度に実行する便利関数。
    """

    strategy = EntryStrategy(

        minimum_score=
            minimum_score,

        minimum_probability=
            minimum_probability,

        strong_probability=
            strong_probability,

        rsi_min=
            rsi_min,

        rsi_max=
            rsi_max,

        volume_ratio_min=
            volume_ratio_min
    )


    return strategy.evaluate_latest(

        data=data,

        probability_up=
            probability_up
    )
