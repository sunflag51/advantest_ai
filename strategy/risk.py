# ============================================================
# アドバンテスト AI売買システム
# strategy/risk.py
#
# リスク・資金管理
#
# 主な役割
# ・1回の取引で許容する最大損失額
# ・購入可能株数
# ・100株単位への調整
# ・最大投資比率
# ・損切り価格
# ・利益確定価格
# ・必要資金
# ・資金不足判定
#
# ※実際の注文は行わない
# ============================================================


import math
import numpy as np


# ============================================================
# リスク管理クラス
# ============================================================

class RiskManager:

    """
    売買時の資金管理・ポジションサイズを
    計算するクラス。
    """

    # ========================================================
    # 初期化
    # ========================================================

    def __init__(
        self,
        lot_size=100,
        risk_per_trade=0.01,
        max_position_rate=0.50,
        stop_loss_rate=0.05,
        take_profit_rate=0.10,
        commission_rate=0.001,
        slippage_rate=0.001
    ):

        """
        lot_size
            売買単位
            日本株の標準として100株

        risk_per_trade
            1回の取引で許容する
            総資産に対する最大リスク

            0.01 = 1%

        max_position_rate
            1銘柄に投入できる
            最大資金比率

            0.50 = 総資産の50%

        stop_loss_rate
            損切り率

            0.05 = -5%

        take_profit_rate
            利益確定率

            0.10 = +10%

        commission_rate
            売買手数料率

        slippage_rate
            スリッページ想定率
        """

        self.lot_size = int(
            lot_size
        )

        self.risk_per_trade = float(
            risk_per_trade
        )

        self.max_position_rate = float(
            max_position_rate
        )

        self.stop_loss_rate = float(
            stop_loss_rate
        )

        self.take_profit_rate = float(
            take_profit_rate
        )

        self.commission_rate = float(
            commission_rate
        )

        self.slippage_rate = float(
            slippage_rate
        )


        # ====================================================
        # 入力チェック
        # ====================================================

        if self.lot_size < 1:

            raise ValueError(
                "lot_sizeは1以上にしてください。"
            )


        if not 0.0 < self.risk_per_trade <= 1.0:

            raise ValueError(
                "risk_per_tradeは0〜1の間にしてください。"
            )


        if not 0.0 < self.max_position_rate <= 1.0:

            raise ValueError(
                "max_position_rateは0〜1の間にしてください。"
            )


        if not 0.0 < self.stop_loss_rate < 1.0:

            raise ValueError(
                "stop_loss_rateは0〜1の間にしてください。"
            )


        if not 0.0 < self.take_profit_rate < 1.0:

            raise ValueError(
                "take_profit_rateは0〜1の間にしてください。"
            )


        if self.commission_rate < 0:

            raise ValueError(
                "commission_rateは0以上にしてください。"
            )


        if self.slippage_rate < 0:

            raise ValueError(
                "slippage_rateは0以上にしてください。"
            )


    # ========================================================
    # 数値チェック
    # ========================================================

    @staticmethod
    def _validate_positive_number(
        value,
        name
    ):

        """
        正の有限数か確認する。
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"{name}が正しくありません。"
            )


        if (
            not np.isfinite(value)
            or value <= 0
        ):

            raise ValueError(
                f"{name}は0より大きい数値にしてください。"
            )


        return value


    # ========================================================
    # 許容損失額
    # ========================================================

    def calculate_risk_budget(
        self,
        capital
    ):

        """
        1回の取引で許容する
        最大損失額を計算する。

        例:
        capital = 1,000,000円
        risk_per_trade = 1%

        → 10,000円
        """

        capital = (
            self._validate_positive_number(
                capital,
                "資金"
            )
        )


        return (
            capital
            * self.risk_per_trade
        )


    # ========================================================
    # 最大投資金額
    # ========================================================

    def calculate_max_position_value(
        self,
        capital
    ):

        """
        1銘柄に投入できる
        最大金額を計算する。
        """

        capital = (
            self._validate_positive_number(
                capital,
                "資金"
            )
        )


        return (
            capital
            * self.max_position_rate
        )


    # ========================================================
    # 想定購入価格
    # ========================================================

    def calculate_buy_price(
        self,
        market_price
    ):

        """
        スリッページを考慮した
        想定購入価格。
        """

        market_price = (
            self._validate_positive_number(
                market_price,
                "市場価格"
            )
        )


        return (
            market_price
            * (
                1.0
                + self.slippage_rate
            )
        )


    # ========================================================
    # 損切り価格
    # ========================================================

    def calculate_stop_price(
        self,
        entry_price
    ):

        """
        損切り価格を計算。
        """

        entry_price = (
            self._validate_positive_number(
                entry_price,
                "購入価格"
            )
        )


        return (
            entry_price
            * (
                1.0
                - self.stop_loss_rate
            )
        )


    # ========================================================
    # 利益確定価格
    # ========================================================

    def calculate_take_profit_price(
        self,
        entry_price
    ):

        """
        利益確定価格を計算。
        """

        entry_price = (
            self._validate_positive_number(
                entry_price,
                "購入価格"
            )
        )


        return (
            entry_price
            * (
                1.0
                + self.take_profit_rate
            )
        )


    # ========================================================
    # 1株あたりリスク
    # ========================================================

    def calculate_risk_per_share(
        self,
        entry_price,
        stop_price=None
    ):

        """
        1株あたりの
        想定最大損失額を計算。
        """

        entry_price = (
            self._validate_positive_number(
                entry_price,
                "購入価格"
            )
        )


        if stop_price is None:

            stop_price = (
                self.calculate_stop_price(
                    entry_price
                )
            )

        else:

            stop_price = (
                self._validate_positive_number(
                    stop_price,
                    "損切り価格"
                )
            )


        risk_per_share = (
            entry_price
            - stop_price
        )


        if risk_per_share <= 0:

            raise ValueError(
                "損切り価格は購入価格より"
                "低く設定してください。"
            )


        return risk_per_share


    # ========================================================
    # リスク基準株数
    # ========================================================

    def calculate_shares_by_risk(
        self,
        capital,
        entry_price,
        stop_price=None
    ):

        """
        許容損失額から
        最大株数を計算する。
        """

        risk_budget = (
            self.calculate_risk_budget(
                capital
            )
        )


        risk_per_share = (
            self.calculate_risk_per_share(
                entry_price=entry_price,
                stop_price=stop_price
            )
        )


        raw_shares = (
            risk_budget
            / risk_per_share
        )


        shares = (
            math.floor(
                raw_shares
                / self.lot_size
            )
            * self.lot_size
        )


        return max(
            shares,
            0
        )


    # ========================================================
    # 投資金額基準株数
    # ========================================================

    def calculate_shares_by_position_limit(
        self,
        capital,
        entry_price
    ):

        """
        最大投資比率から
        最大株数を計算する。
        """

        capital = (
            self._validate_positive_number(
                capital,
                "資金"
            )
        )


        entry_price = (
            self._validate_positive_number(
                entry_price,
                "購入価格"
            )
        )


        max_position_value = (
            self.calculate_max_position_value(
                capital
            )
        )


        raw_shares = (
            max_position_value
            / entry_price
        )


        shares = (
            math.floor(
                raw_shares
                / self.lot_size
            )
            * self.lot_size
        )


        return max(
            shares,
            0
        )


    # ========================================================
    # 資金基準株数
    # ========================================================

    def calculate_shares_by_cash(
        self,
        capital,
        entry_price
    ):

        """
        手数料を含めて、
        現金で実際に購入可能な
        最大株数を計算する。
        """

        capital = (
            self._validate_positive_number(
                capital,
                "資金"
            )
        )


        entry_price = (
            self._validate_positive_number(
                entry_price,
                "購入価格"
            )
        )


        # 1株購入時の必要資金
        cost_per_share = (
            entry_price
            * (
                1.0
                + self.commission_rate
            )
        )


        raw_shares = (
            capital
            / cost_per_share
        )


        shares = (
            math.floor(
                raw_shares
                / self.lot_size
            )
            * self.lot_size
        )


        return max(
            shares,
            0
        )


    # ========================================================
    # 最終購入株数
    # ========================================================

    def calculate_position_size(
        self,
        capital,
        entry_price,
        stop_price=None
    ):

        """
        以下3条件のうち、
        最も小さい株数を採用する。

        1. リスク許容額
        2. 最大投資比率
        3. 実際の現金残高

        最後に100株単位へ調整。
        """

        shares_by_risk = (
            self.calculate_shares_by_risk(
                capital=capital,
                entry_price=entry_price,
                stop_price=stop_price
            )
        )


        shares_by_position = (
            self.calculate_shares_by_position_limit(
                capital=capital,
                entry_price=entry_price
            )
        )


        shares_by_cash = (
            self.calculate_shares_by_cash(
                capital=capital,
                entry_price=entry_price
            )
        )


        shares = min(
            shares_by_risk,
            shares_by_position,
            shares_by_cash
        )


        shares = (
            math.floor(
                shares
                / self.lot_size
            )
            * self.lot_size
        )


        return max(
            shares,
            0
        )


    # ========================================================
    # 必要購入資金
    # ========================================================

    def calculate_required_cash(
        self,
        entry_price,
        shares
    ):

        """
        購入代金 + 購入手数料を計算。
        """

        entry_price = (
            self._validate_positive_number(
                entry_price,
                "購入価格"
            )
        )


        shares = int(
            shares
        )


        if shares < 0:

            raise ValueError(
                "株数は0以上にしてください。"
            )


        purchase_value = (
            entry_price
            * shares
        )


        commission = (
            purchase_value
            * self.commission_rate
        )


        required_cash = (
            purchase_value
            + commission
        )


        return {

            "purchase_value":
                float(
                    purchase_value
                ),

            "commission":
                float(
                    commission
                ),

            "required_cash":
                float(
                    required_cash
                )
        }


    # ========================================================
    # 想定最大損失
    # ========================================================

    def calculate_expected_loss(
        self,
        entry_price,
        stop_price,
        shares
    ):

        """
        損切り価格まで下落した場合の
        想定価格損失を計算。

        ※ここでは価格差による損失。
        手数料・追加スリッページは
        別途表示する。
        """

        risk_per_share = (
            self.calculate_risk_per_share(
                entry_price=entry_price,
                stop_price=stop_price
            )
        )


        return (
            risk_per_share
            * int(shares)
        )


    # ========================================================
    # リスクリワード比
    # ========================================================

    def calculate_reward_risk_ratio(
        self,
        entry_price,
        stop_price,
        take_profit_price
    ):

        """
        利益目標 / 損失幅を計算。

        例:
        損切り -5%
        利確 +10%

        → 約2.0
        """

        entry_price = (
            self._validate_positive_number(
                entry_price,
                "購入価格"
            )
        )


        stop_price = (
            self._validate_positive_number(
                stop_price,
                "損切り価格"
            )
        )


        take_profit_price = (
            self._validate_positive_number(
                take_profit_price,
                "利益確定価格"
            )
        )


        risk = (
            entry_price
            - stop_price
        )


        reward = (
            take_profit_price
            - entry_price
        )


        if risk <= 0:

            raise ValueError(
                "損切り価格は購入価格より"
                "低くしてください。"
            )


        if reward <= 0:

            raise ValueError(
                "利益確定価格は購入価格より"
                "高くしてください。"
            )


        return (
            reward
            / risk
        )


    # ========================================================
    # 総合リスク計算
    # ========================================================

    def evaluate_trade(
        self,
        capital,
        market_price
    ):

        """
        1回の取引について
        必要なリスク情報をまとめて返す。

        market_price
            次回購入予定価格などを指定。

        戻り値:
        ・購入可能か
        ・株数
        ・購入価格
        ・損切り価格
        ・利益確定価格
        ・必要資金
        ・最大想定損失
        ・リスクリワード比
        """


        capital = (
            self._validate_positive_number(
                capital,
                "資金"
            )
        )


        market_price = (
            self._validate_positive_number(
                market_price,
                "市場価格"
            )
        )


        # ----------------------------------------------------
        # スリッページ込み購入価格
        # ----------------------------------------------------

        entry_price = (
            self.calculate_buy_price(
                market_price
            )
        )


        # ----------------------------------------------------
        # 損切り価格
        # ----------------------------------------------------

        stop_price = (
            self.calculate_stop_price(
                entry_price
            )
        )


        # ----------------------------------------------------
        # 利益確定価格
        # ----------------------------------------------------

        take_profit_price = (
            self.calculate_take_profit_price(
                entry_price
            )
        )


        # ----------------------------------------------------
        # 許容損失額
        # ----------------------------------------------------

        risk_budget = (
            self.calculate_risk_budget(
                capital
            )
        )


        # ----------------------------------------------------
        # 最大投資金額
        # ----------------------------------------------------

        max_position_value = (
            self.calculate_max_position_value(
                capital
            )
        )


        # ----------------------------------------------------
        # 各制限による株数
        # ----------------------------------------------------

        shares_by_risk = (
            self.calculate_shares_by_risk(
                capital=capital,
                entry_price=entry_price,
                stop_price=stop_price
            )
        )


        shares_by_position = (
            self.calculate_shares_by_position_limit(
                capital=capital,
                entry_price=entry_price
            )
        )


        shares_by_cash = (
            self.calculate_shares_by_cash(
                capital=capital,
                entry_price=entry_price
            )
        )


        # ----------------------------------------------------
        # 最終株数
        # ----------------------------------------------------

        shares = min(
            shares_by_risk,
            shares_by_position,
            shares_by_cash
        )


        shares = (
            math.floor(
                shares
                / self.lot_size
            )
            * self.lot_size
        )


        shares = max(
            shares,
            0
        )


        lots = (
            shares
            // self.lot_size
        )


        # ----------------------------------------------------
        # 購入資金
        # ----------------------------------------------------

        cash_info = (
            self.calculate_required_cash(
                entry_price=entry_price,
                shares=shares
            )
        )


        # ----------------------------------------------------
        # 想定損失
        # ----------------------------------------------------

        if shares > 0:

            expected_loss = (
                self.calculate_expected_loss(
                    entry_price=entry_price,
                    stop_price=stop_price,
                    shares=shares
                )
            )

        else:

            expected_loss = 0.0


        # ----------------------------------------------------
        # リスクリワード
        # ----------------------------------------------------

        reward_risk_ratio = (
            self.calculate_reward_risk_ratio(
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price
            )
        )


        # ----------------------------------------------------
        # 投資比率
        # ----------------------------------------------------

        if capital > 0:

            position_rate = (
                cash_info[
                    "purchase_value"
                ]
                / capital
            )

        else:

            position_rate = 0.0


        # ----------------------------------------------------
        # 実際のリスク率
        # ----------------------------------------------------

        actual_risk_rate = (
            expected_loss
            / capital
        )


        # ====================================================
        # 購入可能判定
        # ====================================================

        can_trade = (
            shares
            >= self.lot_size
            and cash_info[
                "required_cash"
            ] <= capital
        )


        if can_trade:

            status = "OK"

            reason = (
                "リスク管理条件の範囲内で"
                "購入可能です。"
            )


        else:

            status = "NO_TRADE"

            # ------------------------------------------------
            # どの制限が原因か確認
            # ------------------------------------------------

            if shares_by_risk < self.lot_size:

                reason = (
                    "許容損失額の範囲では"
                    "最低売買単位を購入できません。"
                )


            elif (
                shares_by_position
                < self.lot_size
            ):

                reason = (
                    "最大投資比率の範囲では"
                    "最低売買単位を購入できません。"
                )


            elif (
                shares_by_cash
                < self.lot_size
            ):

                reason = (
                    "現在の資金では"
                    "最低売買単位を購入できません。"
                )


            else:

                reason = (
                    "リスク管理条件により"
                    "取引を見送ります。"
                )


        # ====================================================
        # 結果
        # ====================================================

        return {

            "can_trade":
                bool(
                    can_trade
                ),

            "status":
                status,

            "reason":
                reason,

            "capital":
                float(
                    capital
                ),

            "market_price":
                float(
                    market_price
                ),

            "entry_price":
                float(
                    entry_price
                ),

            "stop_price":
                float(
                    stop_price
                ),

            "take_profit_price":
                float(
                    take_profit_price
                ),

            "shares":
                int(
                    shares
                ),

            "lots":
                int(
                    lots
                ),

            "lot_size":
                int(
                    self.lot_size
                ),

            "risk_budget":
                float(
                    risk_budget
                ),

            "expected_loss":
                float(
                    expected_loss
                ),

            "actual_risk_rate":
                float(
                    actual_risk_rate
                ),

            "max_position_value":
                float(
                    max_position_value
                ),

            "position_rate":
                float(
                    position_rate
                ),

            "purchase_value":
                float(
                    cash_info[
                        "purchase_value"
                    ]
                ),

            "buy_commission":
                float(
                    cash_info[
                        "commission"
                    ]
                ),

            "required_cash":
                float(
                    cash_info[
                        "required_cash"
                    ]
                ),

            "reward_risk_ratio":
                float(
                    reward_risk_ratio
                ),

            "shares_by_risk":
                int(
                    shares_by_risk
                ),

            "shares_by_position_limit":
                int(
                    shares_by_position
                ),

            "shares_by_cash":
                int(
                    shares_by_cash
                ),

            "settings": {

                "risk_per_trade":
                    float(
                        self.risk_per_trade
                    ),

                "max_position_rate":
                    float(
                        self.max_position_rate
                    ),

                "stop_loss_rate":
                    float(
                        self.stop_loss_rate
                    ),

                "take_profit_rate":
                    float(
                        self.take_profit_rate
                    ),

                "commission_rate":
                    float(
                        self.commission_rate
                    ),

                "slippage_rate":
                    float(
                        self.slippage_rate
                    )
            }
        }


# ============================================================
# 簡単実行関数
# ============================================================

def calculate_trade_risk(
    capital,
    market_price,
    lot_size=100,
    risk_per_trade=0.01,
    max_position_rate=0.50,
    stop_loss_rate=0.05,
    take_profit_rate=0.10,
    commission_rate=0.001,
    slippage_rate=0.001
):

    """
    リスク計算を一度に行う
    便利関数。
    """

    manager = RiskManager(

        lot_size=
            lot_size,

        risk_per_trade=
            risk_per_trade,

        max_position_rate=
            max_position_rate,

        stop_loss_rate=
            stop_loss_rate,

        take_profit_rate=
            take_profit_rate,

        commission_rate=
            commission_rate,

        slippage_rate=
            slippage_rate
    )


    return manager.evaluate_trade(

        capital=capital,

        market_price=market_price
    )
