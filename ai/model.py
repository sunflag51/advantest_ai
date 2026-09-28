# ============================================================
# ai/model.py
#
# StockPredictionModel v2
#
# 1日・3日・5日ターゲット対応
#
# ------------------------------------------------------------
# シグナル日 = t
#
# 実際の売買ルールに合わせて
#
# BUY価格:
#     Open(t + 1)
#
# 評価価格:
#     Close(t + horizon)
#
# horizon:
#     1 / 3 / 5 営業日
#
# 例:
#
# horizon = 3
#
# t日の終値でAI判断
# ↓
# t+1日の始値でBUY
# ↓
# t+3日の終値で評価
#
# Target = 1
#   future_return > target_return_threshold
#
# Target = 0
#   それ以外
#
# ------------------------------------------------------------
# 重要:
# 将来価格はTarget作成だけに使用。
# AI特徴量には入れない。
# ============================================================


import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ai.features import (
    get_ai_feature_columns,
)


# ============================================================
# 対応ターゲット
# ============================================================

SUPPORTED_HORIZONS = (
    1,
    3,
    5,
)


# ============================================================
# StockPredictionModel
# ============================================================

class StockPredictionModel:

    def __init__(
        self,
        target_horizon=3,
        target_return_threshold=0.0,

        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",

        random_state=42,
    ):

        # ====================================================
        # ターゲット設定
        # ====================================================

        target_horizon = int(
            target_horizon
        )

        if (
            target_horizon
            not in SUPPORTED_HORIZONS
        ):

            raise ValueError(
                "target_horizon は "
                "1, 3, 5 のいずれかを"
                "指定してください。"
            )

        self.target_horizon = (
            target_horizon
        )

        self.target_return_threshold = (
            float(
                target_return_threshold
            )
        )


        # ====================================================
        # RandomForest設定
        # ====================================================

        self.n_estimators = int(
            n_estimators
        )

        self.max_depth = int(
            max_depth
        )

        self.min_samples_split = int(
            min_samples_split
        )

        self.min_samples_leaf = int(
            min_samples_leaf
        )

        self.max_features = (
            max_features
        )

        self.random_state = int(
            random_state
        )


        # ====================================================
        # モデル
        # ====================================================

        self.model = (
            self._create_model()
        )


        # ====================================================
        # 学習状態
        # ====================================================

        self.feature_columns = []

        self.is_trained = False

        self.metrics = {}

        self.test_results = (
            pd.DataFrame()
        )

        self.training_data = (
            pd.DataFrame()
        )


    # ========================================================
    # モデル作成
    # ========================================================

    def _create_model(
        self,
    ):

        return RandomForestClassifier(

            n_estimators=
                self.n_estimators,

            max_depth=
                self.max_depth,

            min_samples_split=
                self.min_samples_split,

            min_samples_leaf=
                self.min_samples_leaf,

            max_features=
                self.max_features,

            random_state=
                self.random_state,

            class_weight=
                "balanced",

            n_jobs=-1,
        )


    # ========================================================
    # 安全な数値化
    # ========================================================

    @staticmethod
    def _numeric_series(
        series,
    ):

        return pd.to_numeric(
            series,
            errors="coerce",
        )


    # ========================================================
    # Target作成
    #
    # signal day = t
    #
    # Entry_Open
    #   = Open(t+1)
    #
    # Future_Close
    #   = Close(t+horizon)
    #
    # Future_Return
    #   = Future_Close / Entry_Open - 1
    # ========================================================

    def create_target(
        self,
        data,
    ):

        if data is None:
            raise ValueError(
                "data がありません。"
            )

        if data.empty:
            raise ValueError(
                "data が空です。"
            )

        if "Open" not in data.columns:
            raise ValueError(
                "Open 列がありません。"
            )

        if "Close" not in data.columns:
            raise ValueError(
                "Close 列がありません。"
            )


        result = data.copy()


        # ====================================================
        # 数値化
        # ====================================================

        open_price = (
            self._numeric_series(
                result["Open"]
            )
        )

        close_price = (
            self._numeric_series(
                result["Close"]
            )
        )


        # ====================================================
        # 翌営業日始値
        # ====================================================

        result[
            "Target_Entry_Open"
        ] = open_price.shift(-1)


        # ====================================================
        # horizon営業日後の終値
        #
        # horizon=1:
        #   t+1 Close
        #
        # horizon=3:
        #   t+3 Close
        #
        # horizon=5:
        #   t+5 Close
        # ====================================================

        result[
            "Target_Future_Close"
        ] = close_price.shift(
            -self.target_horizon
        )


        # ====================================================
        # 将来リターン
        # ====================================================

        result[
            "Target_Future_Return"
        ] = (

            result[
                "Target_Future_Close"
            ]

            /

            result[
                "Target_Entry_Open"
            ]

            - 1.0
        )


        # ====================================================
        # Target
        #
        # Future_Return >
        # target_return_threshold
        #
        # なら1
        # ====================================================

        valid_target = (

            result[
                "Target_Entry_Open"
            ].notna()

            &

            result[
                "Target_Future_Close"
            ].notna()

            &

            result[
                "Target_Future_Return"
            ].notna()
        )


        result[
            "Target"
        ] = np.nan


        result.loc[
            valid_target,
            "Target",
        ] = (

            result.loc[
                valid_target,
                "Target_Future_Return",
            ]

            >

            self.target_return_threshold

        ).astype(int)


        return result


    # ========================================================
    # 学習データ準備
    # ========================================================

    def prepare_training_data(
        self,
        ai_data,
    ):

        if (
            ai_data is None
            or ai_data.empty
        ):

            raise ValueError(
                "AI学習データがありません。"
            )


        # ====================================================
        # Target追加
        # ====================================================

        data = self.create_target(
            ai_data
        )


        # ====================================================
        # 特徴量一覧
        # ====================================================

        requested_features = (
            get_ai_feature_columns()
        )


        available_features = [

            column

            for column
            in requested_features

            if column in data.columns
        ]


        if not available_features:

            raise ValueError(
                "AI特徴量が見つかりません。"
            )


        self.feature_columns = (
            available_features
        )


        # ====================================================
        # 数値化
        # ====================================================

        for column in self.feature_columns:

            data[column] = (
                pd.to_numeric(
                    data[column],
                    errors="coerce",
                )
            )


        # ====================================================
        # 学習に必要な列
        # ====================================================

        required_columns = (

            self.feature_columns

            + [
                "Target",
                "Target_Entry_Open",
                "Target_Future_Close",
                "Target_Future_Return",
            ]
        )


        training_data = (
            data[
                required_columns
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
            .copy()
        )


        training_data[
            "Target"
        ] = training_data[
            "Target"
        ].astype(int)


        # ====================================================
        # 最低データ数
        # ====================================================

        if len(
            training_data
        ) < 100:

            raise ValueError(
                "AI学習に使用できる"
                "データが100件未満です。"
            )


        self.training_data = (
            training_data.copy()
        )


        X = training_data[
            self.feature_columns
        ].copy()


        y = training_data[
            "Target"
        ].copy()


        return (
            X,
            y,
            training_data,
        )


    # ========================================================
    # 学習
    # ========================================================

    def train(
        self,
        ai_data,
    ):

        (
            X,
            y,
            training_data,
        ) = self.prepare_training_data(
            ai_data
        )


        # ====================================================
        # 時系列80 / 20
        # ====================================================

        split_index = int(
            len(X)
            * 0.80
        )


        # ====================================================
        # 安全チェック
        # ====================================================

        if split_index <= 0:

            raise ValueError(
                "学習データが不足しています。"
            )


        if split_index >= len(X):

            raise ValueError(
                "テストデータを確保できません。"
            )


        X_train = X.iloc[
            :split_index
        ].copy()

        X_test = X.iloc[
            split_index:
        ].copy()


        y_train = y.iloc[
            :split_index
        ].copy()

        y_test = y.iloc[
            split_index:
        ].copy()


        # ====================================================
        # 学習データに両クラスが必要
        # ====================================================

        if y_train.nunique() < 2:

            raise ValueError(
                "学習データのTargetが"
                "1種類しかありません。"
            )


        # ====================================================
        # モデルを毎回新規作成
        # ====================================================

        self.model = (
            self._create_model()
        )


        self.model.fit(
            X_train,
            y_train,
        )


        # ====================================================
        # Prediction
        # ====================================================

        prediction = (
            self.model.predict(
                X_test
            )
        )


        # ====================================================
        # Probability
        # ====================================================

        probability_up = (
            self._predict_positive_probability(
                X_test
            )
        )


        # ====================================================
        # 指標
        # ====================================================

        accuracy = (
            accuracy_score(
                y_test,
                prediction,
            )
        )


        precision = (
            precision_score(
                y_test,
                prediction,
                zero_division=0,
            )
        )


        recall = (
            recall_score(
                y_test,
                prediction,
                zero_division=0,
            )
        )


        f1 = (
            f1_score(
                y_test,
                prediction,
                zero_division=0,
            )
        )


        # ====================================================
        # AUC
        # ====================================================

        if y_test.nunique() >= 2:

            try:

                auc = (
                    roc_auc_score(
                        y_test,
                        probability_up,
                    )
                )

            except Exception:

                auc = None

        else:

            auc = None


        # ====================================================
        # Confusion Matrix
        # ====================================================

        matrix = confusion_matrix(
            y_test,
            prediction,
            labels=[
                0,
                1,
            ],
        )


        tn = int(
            matrix[0, 0]
        )

        fp = int(
            matrix[0, 1]
        )

        fn = int(
            matrix[1, 0]
        )

        tp = int(
            matrix[1, 1]
        )


        # ====================================================
        # Test Results
        # ====================================================

        test_source = (
            training_data.iloc[
                split_index:
            ]
        )


        self.test_results = pd.DataFrame(

            {
                "Actual":
                    y_test.values,

                "Prediction":
                    prediction,

                "Probability_Up":
                    probability_up,

                "Entry_Open":
                    test_source[
                        "Target_Entry_Open"
                    ].values,

                "Future_Close":
                    test_source[
                        "Target_Future_Close"
                    ].values,

                "Future_Return":
                    test_source[
                        "Target_Future_Return"
                    ].values,
            },

            index=X_test.index,
        )


        # ====================================================
        # Metrics
        # ====================================================

        self.metrics = {

            "accuracy":
                float(
                    accuracy
                ),

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "auc":
                (
                    float(auc)
                    if auc is not None
                    else None
                ),

            "train_samples":
                int(
                    len(X_train)
                ),

            "test_samples":
                int(
                    len(X_test)
                ),

            "total_samples":
                int(
                    len(X)
                ),

            "feature_count":
                int(
                    len(
                        self.feature_columns
                    )
                ),

            "train_up_rate":
                float(
                    y_train.mean()
                ),

            "test_up_rate":
                float(
                    y_test.mean()
                ),

            "predicted_up_rate":
                float(
                    np.mean(
                        prediction
                    )
                ),

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,

            # ================================================
            # v2 Target情報
            # ================================================

            "target_horizon":
                int(
                    self.target_horizon
                ),

            "target_return_threshold":
                float(
                    self.target_return_threshold
                ),

            "target_definition":
                (
                    "Close(t+"
                    + str(
                        self.target_horizon
                    )
                    + ") / Open(t+1) - 1"
                ),

            "average_future_return":
                float(
                    test_source[
                        "Target_Future_Return"
                    ].mean()
                ),

            "median_future_return":
                float(
                    test_source[
                        "Target_Future_Return"
                    ].median()
                ),
        }


        self.is_trained = True


        return dict(
            self.metrics
        )


    # ========================================================
    # Positive class probability
    #
    # predict_proba[:, 1] を固定で使わず
    # classes_から「1」の位置を取得する
    # ========================================================

    def _predict_positive_probability(
        self,
        X,
    ):

        probabilities = (
            self.model.predict_proba(
                X
            )
        )


        classes = list(
            self.model.classes_
        )


        if 1 not in classes:

            return np.zeros(
                len(X),
                dtype=float,
            )


        positive_index = (
            classes.index(1)
        )


        return probabilities[
            :,
            positive_index
        ]


    # ========================================================
    # 最新特徴量行
    # ========================================================

    def _get_latest_feature_row(
        self,
        ai_data,
    ):

        if not self.is_trained:

            raise RuntimeError(
                "AIモデルがまだ"
                "学習されていません。"
            )


        if (
            ai_data is None
            or ai_data.empty
        ):

            raise ValueError(
                "AIデータがありません。"
            )


        missing = [

            column

            for column
            in self.feature_columns

            if column
            not in ai_data.columns
        ]


        if missing:

            raise ValueError(
                "最新予測に必要な特徴量が"
                "不足しています: "
                + ", ".join(
                    missing
                )
            )


        features = (
            ai_data[
                self.feature_columns
            ]
            .copy()
        )


        for column in self.feature_columns:

            features[column] = (
                pd.to_numeric(
                    features[column],
                    errors="coerce",
                )
            )


        features = (
            features
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )


        if features.empty:

            raise ValueError(
                "最新予測に使用できる"
                "特徴量データがありません。"
            )


        return features.iloc[
            [-1]
        ]


    # ========================================================
    # 最新上昇確率
    # ========================================================

    def predict_probability(
        self,
        ai_data,
    ):

        latest_X = (
            self._get_latest_feature_row(
                ai_data
            )
        )


        probability = (
            self._predict_positive_probability(
                latest_X
            )[0]
        )


        return float(
            probability
        )


    # ========================================================
    # 最新予測
    # ========================================================

    def predict(
        self,
        ai_data,
        threshold=0.50,
    ):

        probability_up = (
            self.predict_probability(
                ai_data
            )
        )


        prediction = int(
            probability_up
            >= float(
                threshold
            )
        )


        return {

            "prediction":
                prediction,

            "probability_up":
                probability_up,

            "probability_down":
                (
                    1.0
                    - probability_up
                ),

            "threshold":
                float(
                    threshold
                ),

            "target_horizon":
                int(
                    self.target_horizon
                ),

            "target_return_threshold":
                float(
                    self.target_return_threshold
                ),

            "target_definition":
                (
                    "翌営業日始値から"
                    + str(
                        self.target_horizon
                    )
                    + "営業日後終値まで"
                ),
        }


    # ========================================================
    # Feature Importance
    # ========================================================

    def get_feature_importance(
        self,
    ):

        if not self.is_trained:

            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance",
                ]
            )


        if not hasattr(
            self.model,
            "feature_importances_",
        ):

            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance",
                ]
            )


        importance = pd.DataFrame(

            {
                "Feature":
                    self.feature_columns,

                "Importance":
                    self.model
                    .feature_importances_,
            }
        )


        importance = (
            importance
            .sort_values(
                "Importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        return importance


    # ========================================================
    # Test Results
    # ========================================================

    def get_test_results(
        self,
    ):

        return self.test_results.copy()


    # ========================================================
    # Metrics
    # ========================================================

    def get_metrics(
        self,
    ):

        return dict(
            self.metrics
        )


    # ========================================================
    # Target情報
    # ========================================================

    def get_target_info(
        self,
    ):

        return {

            "target_horizon":
                self.target_horizon,

            "target_return_threshold":
                self.target_return_threshold,

            "supported_horizons":
                list(
                    SUPPORTED_HORIZONS
                ),

            "entry":
                "Open(t+1)",

            "exit":
                (
                    "Close(t+"
                    + str(
                        self.target_horizon
                    )
                    + ")"
                ),

            "formula":
                (
                    "Close(t+"
                    + str(
                        self.target_horizon
                    )
                    + ") / Open(t+1) - 1"
                ),
        }


    # ========================================================
    # Model Info
    # ========================================================

    def get_model_info(
        self,
    ):

        return {

            "model":
                "RandomForestClassifier",

            "is_trained":
                self.is_trained,

            "target_horizon":
                self.target_horizon,

            "target_return_threshold":
                self.target_return_threshold,

            "feature_count":
                len(
                    self.feature_columns
                ),

            "features":
                list(
                    self.feature_columns
                ),

            "n_estimators":
                self.n_estimators,

            "max_depth":
                self.max_depth,

            "min_samples_split":
                self.min_samples_split,

            "min_samples_leaf":
                self.min_samples_leaf,

            "max_features":
                self.max_features,

            "random_state":
                self.random_state,
        }


# ============================================================
# Helper
# ============================================================

def train_stock_model(
    ai_data,
    target_horizon=3,
    target_return_threshold=0.0,
):

    model = StockPredictionModel(

        target_horizon=
            target_horizon,

        target_return_threshold=
            target_return_threshold,
    )


    metrics = model.train(
        ai_data
    )


    return (
        model,
        metrics,
    )


# ============================================================
# Helper
# ============================================================

def predict_stock_probability(
    model,
    ai_data,
):

    if model is None:

        raise ValueError(
            "model がありません。"
        )


    return model.predict_probability(
        ai_data
    )
