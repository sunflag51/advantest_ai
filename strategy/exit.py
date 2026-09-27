# ============================================================
# アドバンテスト AI売買システム
# strategy/exit.py
#
# 売却（EXIT）判断
#
# 使用する情報
# ・現在価格
# ・購入価格
# ・含み損益
# ・AI上昇確率
# ・移動平均
# ・MACD
# ・RSI
# ・保有日数
#
# 優先順位
# 1. 損切り
# 2. 利益確定
# 3. 最大保有日数
# 4. AI悪化
# 5. テクニカル悪化
#
# 出力
# ・SELL / HOLD
# ・売却理由
# ・損益率
# ・売却スコア
#
# ※実際の注文は行わない
# ============================================================


import numpy as np
import pandas as pd


# ============================================================
# 売却戦略
# ============================================================

class ExitStrategy:

    """
    保有中の株式について、
    売却するか保有継続するかを判断するクラス。
    """

    # ========================================================
    # 初期化
    # ========================================================

    def __init__(
        self,
        stop_loss_rate=0.05,
        take_profit_rate=0.10,
        ai_exit_probability=0.45,
        max_holding_days=10,
        minimum_exit_score=3,
        rsi_overbought=75.0
    ):

        """
        stop_loss_rate
            損切り率
            0.05 = -5%

        take_profit_rate
            利益確定率
            0.10 = +10%

        ai_exit_probability
            AI上昇確率がこの値以下なら
            売却方向として評価

        max_holding_days
            最大保有営業日数

        minimum_exit_score
            テクニカル・AIによる
            通常売却に必要な最低スコア

        rsi_overbought
            RSI過熱判定
        """

        self.stop_loss_rate = float(
            stop_loss_rate
        )

        self.take_profit_rate = float(
            take_profit_rate
        )

        self.ai_exit_probability = float(
            ai_exit_probability
        )

        self.max_holding_days = int(
            max_holding_days
        )

        self.minimum_exit_score = int(
            minimum_exit_score
        )

        self.rsi_overbought = float(
            rsi_overbought
        )


        if not 0.0 < self.stop_loss_rate < 1.0:

            raise ValueError(
                "stop_loss_rateは0〜1の間にしてください。"
            )


        if not 0.0 < self.take_profit_rate < 1.0:

            raise ValueError(
                "take_profit_rateは0〜1の間にしてください。"
            )


        if not 0.0 <= self.ai_exit_probability <= 1.0:

            raise ValueError(
                "ai_exit_probabilityは0〜1の間にしてください。"
            )


        if self.max_holding_days < 1:

            raise ValueError(
                "max_holding_daysは1以上にしてください。"
            )


        if self.minimum_exit_score < 1:

            raise ValueError(
                "minimum_exit_scoreは1以上にしてください。"
            )


    # ========================================================
    # 安全に数値を取得
    # ========================================================

    @staticmethod
    def _get_value(
        row,
        column
    ):

        """
        Series等から安全に数値を取得する。
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
    # 損益計算
    # ========================================================

    @staticmethod
    def calculate_return(
        entry_price,
        current_price
    ):

        """
        購入価格から現在価格までの
        損益率を計算する。
        """

        entry_price = float(
            entry_price
        )

        current_price = float(
            current_price
        )


        if entry_price <= 0:

            raise ValueError(
                "購入価格は0より大きくしてください。"
            )


        if current_price <= 0:

            raise ValueError(
                "現在価格は0より大きくしてください。"
            )


        return (
            current_price
            / entry_price
            - 1.0
        )


    # ========================================================
    # 強制売却条件
    # ========================================================

    def evaluate_hard_exit(
        self,
        entry_price,
        current_price,
        holding_days
    ):

        """
        損切り・利益確定・最大保有日数を判定。

        これらは通常のスコア判定より優先する。
        """

        return_rate = (
            self.calculate_return(
                entry_price=entry_price,
                current_price=current_price
            )
        )


        # ----------------------------------------------------
        # 損切り
        # ----------------------------------------------------

        if (
            return_rate
            <= -self.stop_loss_rate
        ):

            return {

                "triggered":
                    True,

                "reason_code":
                    "STOP_LOSS",

                "reason":
                    (
                        "損切り基準に到達 "
                        f"({return_rate * 100:+.2f}%)"
                    )
            }


        # ----------------------------------------------------
        # 利益確定
        # ----------------------------------------------------

        if (
            return_rate
            >= self.take_profit_rate
        ):

            return {

                "triggered":
                    True,

                "reason_code":
                    "TAKE_PROFIT",

                "reason":
                    (
                        "利益確定基準に到達 "
                        f"({return_rate * 100:+.2f}%)"
                    )
            }


        # ----------------------------------------------------
        # 最大保有日数
        # ----------------------------------------------------

        if (
            holding_days
            >= self.max_holding_days
        ):

            return {

                "triggered":
                    True,

                "reason_code":
                    "MAX_HOLDING_DAYS",

                "reason":
                    (
                        "最大保有日数に到達 "
                        f"({holding_days}営業日)"
                    )
            }


        return {

            "triggered":
                False,

            "reason_code":
                None,

            "reason":
                None
        }


    # ========================================================
    # AI評価
    # ========================================================

    def evaluate_ai(
        self,
        probability_up
    ):

        """
        AI予測悪化を評価。

        最大2点。
        """

        score = 0

        reasons = []


        if probability_up is None:

            reasons.append(
                "AI上昇確率なし"
            )

            return score, reasons


        try:

            probability_up = float(
                probability_up
            )

        except (
            TypeError,
            ValueError
        ):

            reasons.append(
                "AI上昇確率が不正"
            )

            return score, reasons


        if not np.isfinite(
            probability_up
        ):

            reasons.append(
                "AI上昇確率が不正"
            )

            return score, reasons


        # ----------------------------------------------------
        # かなり弱い
        # ----------------------------------------------------

        if probability_up <= 0.35:

            score += 2

            reasons.append(
                "AI上昇確率35%以下"
            )


        # ----------------------------------------------------
        # 売却基準以下
        # ----------------------------------------------------

        elif (
            probability_up
            <= self.ai_exit_probability
        ):

            score += 1

            reasons.append(
                (
                    "AI上昇確率が"
                    "売却警戒基準以下"
                )
            )


        else:

            reasons.append(
                "AI上昇確率は売却基準より上"
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
        トレンド崩れを評価。

        最大3点。
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
        # 株価 < 25日線
        # ----------------------------------------------------

        if (
            close is not None
            and sma25 is not None
        ):

            if close < sma25:

                score += 1

                reasons.append(
                    "株価が25日移動平均を下回っている"
                )

            else:

                reasons.append(
                    "株価は25日移動平均以上"
                )


        # ----------------------------------------------------
        # 5日線 < 25日線
        # ----------------------------------------------------

        if (
            sma5 is not None
            and sma25 is not None
        ):

            if sma5 < sma25:

                score += 1

                reasons.append(
                    "5日線が25日線を下回っている"
                )

            else:

                reasons.append(
                    "5日線は25日線以上"
                )


        # ----------------------------------------------------
        # 25日線 < 75日線
        # ----------------------------------------------------

        if (
            sma25 is not None
            and sma75 is not None
        ):

            if sma25 < sma75:

                score += 1

                reasons.append(
                    "25日線が75日線を下回っている"
                )

            else:

                reasons.append(
                    "25日線は75日線以上"
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
        MACD悪化を評価。

        最大1点。
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


        if macd < signal:

            score += 1

            reasons.append(
                "MACDがシグナルを下回っている"
            )


        else:

            reasons.append(
                "MACDはシグナル以上"
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
        RSI過熱を評価。

        最大1点。

        RSIが高いだけで即売却にはせず、
        他の条件と組み合わせる。
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
            rsi >= self.rsi_overbought
        ):

            score += 1

            reasons.append(
                f"RSIが過熱水準 ({rsi:.1f})"
            )


        else:

            reasons.append(
                f"RSIは過熱水準未満 ({rsi:.1f})"
            )


        return score, reasons


    # ========================================================
    # 総合売却判断
    # ========================================================

    def evaluate(
        self,
        row,
        entry_price,
        current_price=None,
        probability_up=None,
        holding_days=0
    ):

        """
        SELL / HOLDを総合判定する。

        強制売却:
        ・損切り
        ・利益確定
        ・最大保有日数

        通常売却:
        ・AI悪化
        ・トレンド悪化
        ・MACD悪化
        ・RSI過熱
        """


        if row is None:

            raise ValueError(
                "売却判断用データがありません。"
            )


        entry_price = float(
            entry_price
        )


        if entry_price <= 0:

            raise ValueError(
                "購入価格は0より大きくしてください。"
            )


        # ----------------------------------------------------
        # 現在価格
        # ----------------------------------------------------

        if current_price is None:

            current_price = (
                self._get_value(
                    row,
                    "Close"
                )
            )


        if current_price is None:

            raise ValueError(
                "現在価格を取得できません。"
            )


        current_price = float(
            current_price
        )


        # ----------------------------------------------------
        # 保有日数
        # ----------------------------------------------------

        holding_days = int(
            holding_days
        )


        if holding_days < 0:

            raise ValueError(
                "holding_daysは0以上にしてください。"
            )


        # ----------------------------------------------------
        # 現在損益
        # ----------------------------------------------------

        return_rate = (
            self.calculate_return(
                entry_price=entry_price,
                current_price=current_price
            )
        )


        unrealized_profit = (
            current_price
            - entry_price
        )


        # ====================================================
        # 強制売却判定
        # ====================================================

        hard_exit = (
            self.evaluate_hard_exit(
                entry_price=entry_price,
                current_price=current_price,
                holding_days=holding_days
            )
        )


        if hard_exit[
            "triggered"
        ]:

            return {

                "action":
                    "SELL",

                "exit_type":
                    "HARD_EXIT",

                "reason_code":
                    hard_exit[
                        "reason_code"
                    ],

                "summary":
                    hard_exit[
                        "reason"
                    ],

                "exit_score":
                    0,

                "minimum_exit_score":
                    int(
                        self.minimum_exit_score
                    ),

                "probability_up":
                    (
                        float(
                            probability_up
                        )
                        if probability_up is not None
                        else None
                    ),

                "entry_price":
                    float(
                        entry_price
                    ),

                "current_price":
                    float(
                        current_price
                    ),

                "return_rate":
                    float(
                        return_rate
                    ),

                "unrealized_profit_per_share":
                    float(
                        unrealized_profit
                    ),

                "holding_days":
                    int(
                        holding_days
                    ),

                "reasons":
                    [
                        hard_exit[
                            "reason"
                        ]
                    ],

                "details": {

                    "ai_score":
                        0,

                    "trend_score":
                        0,

                    "macd_score":
                        0,

                    "rsi_score":
                        0
                }
            }


        # ====================================================
        # AI評価
        # ====================================================

        ai_score, ai_reasons = (
            self.evaluate_ai(
                probability_up
            )
        )


        # ====================================================
        # トレンド評価
        # ====================================================

        trend_score, trend_reasons = (
            self.evaluate_trend(
                row
            )
        )


        # ====================================================
        # MACD評価
        # ====================================================

        macd_score, macd_reasons = (
            self.evaluate_macd(
                row
            )
        )


        # ====================================================
        # RSI評価
        # ====================================================

        rsi_score, rsi_reasons = (
            self.evaluate_rsi(
                row
            )
        )


        # ====================================================
        # 総合スコア
        #
        # AI       最大2
        # Trend    最大3
        # MACD     最大1
        # RSI      最大1
        #
        # 合計     最大7
        # ====================================================

        exit_score = (
            ai_score
            + trend_score
            + macd_score
            + rsi_score
        )


        max_score = 7


        all_reasons = []

        all_reasons.extend(
            ai_reasons
        )

        all_reasons.extend(
            trend_reasons
        )

        all_reasons.extend(
            macd_reasons
        )

        all_reasons.extend(
            rsi_reasons
        )


        # ====================================================
        # SELL / HOLD
        # ====================================================

        if (
            exit_score
            >= self.minimum_exit_score
        ):

            action = "SELL"

            exit_type = (
                "SIGNAL_EXIT"
            )

            reason_code = (
                "EXIT_SCORE"
            )

            summary = (
                "AI・テクニカルの売却スコアが"
                "基準に到達しました。"
            )


        else:

            action = "HOLD"

            exit_type = None

            reason_code = None

            summary = (
                "現在は売却条件を満たしていません。"
            )


        # ====================================================
        # 結果
        # ====================================================

        return {

            "action":
                action,

            "exit_type":
                exit_type,

            "reason_code":
                reason_code,

            "summary":
                summary,

            "exit_score":
                int(
                    exit_score
                ),

            "max_score":
                int(
                    max_score
                ),

            "minimum_exit_score":
                int(
                    self.minimum_exit_score
                ),

            "score_rate":
                float(
                    exit_score
                    / max_score
                ),

            "probability_up":
                (
                    float(
                        probability_up
                    )
                    if probability_up is not None
                    else None
                ),

            "entry_price":
                float(
                    entry_price
                ),

            "current_price":
                float(
                    current_price
                ),

            "return_rate":
                float(
                    return_rate
                ),

            "unrealized_profit_per_share":
                float(
                    unrealized_profit
                ),

            "holding_days":
                int(
                    holding_days
                ),

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

                "macd_score":
                    int(
                        macd_score
                    ),

                "rsi_score":
                    int(
                        rsi_score
                    )
            }
        }


    # ========================================================
    # 指定日の売却判断
    # ========================================================

    def evaluate_date(
        self,
        data,
        date,
        entry_price,
        probability_up=None,
        holding_days=0
    ):

        """
        指定日のデータを使って
        売却判断する。
        """

        if (
            data is None
            or data.empty
        ):

            raise ValueError(
                "売却判断用データがありません。"
            )


        target_date = pd.Timestamp(
            date
        )


        if (
            target_date
            not in data.index
        ):

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
            entry_price=entry_price,
            probability_up=probability_up,
            holding_days=holding_days
        )


    # ========================================================
    # 最新日の売却判断
    # ========================================================

    def evaluate_latest(
        self,
        data,
        entry_price,
        probability_up=None,
        holding_days=0
    ):

        """
        最新データを使って
        売却判断する。
        """

        if (
            data is None
            or data.empty
        ):

            raise ValueError(
                "売却判断用データがありません。"
            )


        row = data.iloc[-1]


        result = self.evaluate(
            row=row,
            entry_price=entry_price,
            probability_up=probability_up,
            holding_days=holding_days
        )


        result[
            "date"
        ] = data.index[-1]


        return result


# ============================================================
# 簡単実行関数
# ============================================================

def evaluate_exit(
    data,
    entry_price,
    probability_up=None,
    holding_days=0,
    stop_loss_rate=0.05,
    take_profit_rate=0.10,
    ai_exit_probability=0.45,
    max_holding_days=10,
    minimum_exit_score=3,
    rsi_overbought=75.0
):

    """
    最新データの売却判断を
    一度に実行する便利関数。
    """

    strategy = ExitStrategy(

        stop_loss_rate=
            stop_loss_rate,

        take_profit_rate=
            take_profit_rate,

        ai_exit_probability=
            ai_exit_probability,

        max_holding_days=
            max_holding_days,

        minimum_exit_score=
            minimum_exit_score,

        rsi_overbought=
            rsi_overbought
    )


    return strategy.evaluate_latest(

        data=data,

        entry_price=
            entry_price,

        probability_up=
            probability_up,

        holding_days=
            holding_days
    )
