# ============================================================
# backtest/engine.py
# WalkForwardBacktest v2.4
#
# 初心者向け:
# このファイルは「AIの過去テスト」をするファイルです。
# 実際の株注文は出しません。
#
# v2.4:
# ・未来の答えをカンニングしない日付方式
# ・Brier Score（AI確率の正確さ）
# ・確率帯別集計
# ・前半/後半の安定性確認
# ・main.py v4/v4.1互換
# ============================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    brier_score_loss,
)

from ai.features import get_ai_feature_columns

BACKTEST_ENGINE_VERSION = "v2.4"
SUPPORTED_HORIZONS = (1, 3, 5)


class WalkForwardBacktest:

    def __init__(
        self,
        initial_train_size=500,
        test_size=20,
        retrain_every=20,
        prediction_threshold=0.50,
        threshold=None,
        target_horizon=3,
        target_return_threshold=0.0,
        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ):
        self.initial_train_size = int(initial_train_size)
        self.test_size = int(test_size)
        self.retrain_every = int(retrain_every)

        self.prediction_threshold = float(
            threshold if threshold is not None
            else prediction_threshold
        )
        self.threshold = self.prediction_threshold

        self.target_horizon = int(target_horizon)
        self.target_return_threshold = float(target_return_threshold)

        if self.target_horizon not in SUPPORTED_HORIZONS:
            raise ValueError(
                "target_horizon は 1、3、5 のいずれかを指定してください。"
            )

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.class_weight = class_weight
        self.n_jobs = int(n_jobs)

        self.feature_columns = []
        self.results = pd.DataFrame()
        self.metrics = {}
        self.training_log = pd.DataFrame()
        self.calibration_table = pd.DataFrame()
        self.stability_table = pd.DataFrame()
        self.model_count = 0

    def _create_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
        )

    @staticmethod
    def _normalize_dataframe(data):
        if data is None:
            return pd.DataFrame()
        result = data.copy()
        if result.empty:
            return result
        result.index = pd.to_datetime(result.index)
        if getattr(result.index, "tz", None) is not None:
            result.index = result.index.tz_localize(None)
        result = result[~result.index.duplicated(keep="last")]
        return result.sort_index()

    def _get_feature_columns(self, data):
        try:
            requested = get_ai_feature_columns()
        except Exception:
            requested = []

        columns = [c for c in requested if c in data.columns]

        if not columns:
            excluded = {
                "Target", "Future_Return", "Entry_Open",
                "Future_Close", "Target_Known_Date",
            }
            columns = [
                c for c in data.columns
                if c not in excluded
                and pd.api.types.is_numeric_dtype(data[c])
            ]

        if not columns:
            raise ValueError("AI学習に使用できる特徴量がありません。")

        return columns

    def prepare_data(self, ai_data):
        data = self._normalize_dataframe(ai_data)

        if data.empty:
            raise ValueError("ai_data が空です。")

        for c in ("Open", "Close"):
            if c not in data.columns:
                raise ValueError(f"ai_data に必要な列がありません: {c}")
            data[c] = pd.to_numeric(data[c], errors="coerce")

        # 今日の終値で判断 → 翌日始値で買う想定
        data["Entry_Open"] = data["Open"].shift(-1)
        data["Future_Close"] = data["Close"].shift(-self.target_horizon)
        data["Future_Return"] = (
            data["Future_Close"] / data["Entry_Open"] - 1.0
        )

        # 正解が実際に分かる日。
        # v2.4ではこの日付を使って「未来カンニング」を防ぎます。
        known_dates = pd.Series(data.index, index=data.index)
        data["Target_Known_Date"] = pd.to_datetime(
            known_dates.shift(-self.target_horizon)
        )

        valid = (
            data["Future_Return"].notna()
            & data["Target_Known_Date"].notna()
        )

        data["Target"] = np.nan
        data.loc[valid, "Target"] = (
            data.loc[valid, "Future_Return"]
            > self.target_return_threshold
        ).astype(int)

        self.feature_columns = self._get_feature_columns(data)

        for c in self.feature_columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        required = self.feature_columns + [
            "Entry_Open", "Future_Close", "Future_Return",
            "Target", "Target_Known_Date",
        ]

        prepared = data.dropna(subset=required).copy()
        prepared["Target"] = prepared["Target"].astype(int)

        if prepared.empty:
            raise ValueError("ターゲット作成後のデータが空です。")

        return prepared

    def _get_purged_training_data(self, data, test_start_position):
        if test_start_position <= 0:
            return pd.DataFrame()

        test_start_date = data.index[test_start_position]
        candidates = data.iloc[:test_start_position].copy()

        # テスト開始日より前に「正解が判明済み」のデータだけ学習。
        return candidates[
            candidates["Target_Known_Date"] < test_start_date
        ].copy()

    @staticmethod
    def _probability_up(model, features):
        p = model.predict_proba(features)
        classes = list(model.classes_)
        if 1 in classes:
            return p[:, classes.index(1)]
        return np.zeros(len(features), dtype=float)

    def run(self, ai_data):
        data = self.prepare_data(ai_data)
        total_rows = len(data)

        minimum = (
            self.initial_train_size
            + self.target_horizon
            + 1
        )
        if total_rows < minimum:
            raise ValueError(
                f"データ数が不足しています。現在:{total_rows} / 必要:{minimum}以上"
            )

        result_records = []
        training_records = []
        model = None
        last_train_position = None
        model_number = 0

        test_start = (
            self.initial_train_size
            + self.target_horizon
        )

        while test_start < total_rows:
            test_end = min(test_start + self.test_size, total_rows)

            train_data = self._get_purged_training_data(
                data, test_start
            )

            if len(train_data) < self.initial_train_size:
                test_start = test_end
                continue

            should_retrain = (
                model is None
                or last_train_position is None
                or test_start - last_train_position >= self.retrain_every
            )

            if should_retrain:
                X_train = train_data[self.feature_columns]
                y_train = train_data["Target"].astype(int)

                test_start_date = data.index[test_start]
                test_end_date = data.index[test_end - 1]
                max_known = train_data["Target_Known_Date"].max()

                audit = {
                    "Train_Start": train_data.index[0],
                    "Train_End": train_data.index[-1],
                    "Train_Rows": int(len(train_data)),
                    "Test_Start": test_start_date,
                    "Test_End": test_end_date,
                    "Test_Rows": int(test_end - test_start),
                    "Purge_Method": "Target_Known_Date",
                    "Purge_Days": int(self.target_horizon),
                    "Purged_Rows": int(test_start - len(train_data)),
                    "Max_Target_Known_Date": max_known,
                    "Leakage_Check": bool(max_known < test_start_date),
                    "Target_Horizon": int(self.target_horizon),
                    "Target_Return_Threshold": float(
                        self.target_return_threshold
                    ),
                    "Prediction_Threshold": float(
                        self.prediction_threshold
                    ),
                    "Class_0_Count": int((y_train == 0).sum()),
                    "Class_1_Count": int((y_train == 1).sum()),
                    "Engine_Version": BACKTEST_ENGINE_VERSION,
                }

                if y_train.nunique() < 2:
                    audit["Model_Number"] = model_number + 1
                    audit["Status"] = "SKIPPED_ONE_CLASS"
                    training_records.append(audit)
                    test_start = test_end
                    continue

                model = self._create_model()
                model.fit(X_train, y_train)
                model_number += 1
                last_train_position = test_start

                audit["Model_Number"] = model_number
                audit["Status"] = "TRAINED"
                training_records.append(audit)

            if model is None:
                test_start = test_end
                continue

            test_data = data.iloc[test_start:test_end].copy()
            if test_data.empty:
                break

            probabilities = self._probability_up(
                model,
                test_data[self.feature_columns],
            )
            predictions = (
                probabilities >= self.prediction_threshold
            ).astype(int)

            for i, (date, row) in enumerate(test_data.iterrows()):
                p = float(probabilities[i])
                pred = int(predictions[i])
                actual = int(row["Target"])

                result_records.append({
                    "Date": date,
                    "Probability_Up": p,
                    "Probability_Down": float(1.0 - p),
                    "Prediction": pred,
                    "Actual": actual,
                    "Entry_Open": float(row["Entry_Open"]),
                    "Future_Close": float(row["Future_Close"]),
                    "Future_Return": float(row["Future_Return"]),
                    "Target_Known_Date": row["Target_Known_Date"],
                    "Correct": bool(pred == actual),
                    "Target_Horizon": int(self.target_horizon),
                    "Target_Return_Threshold": float(
                        self.target_return_threshold
                    ),
                    "Prediction_Threshold": float(
                        self.prediction_threshold
                    ),
                    "Model_Number": int(model_number),
                    "Purge_Days": int(self.target_horizon),
                    "Purge_Method": "Target_Known_Date",
                    "Engine_Version": BACKTEST_ENGINE_VERSION,
                })

            test_start = test_end

        self.training_log = pd.DataFrame(training_records)
        self.results = pd.DataFrame(result_records)

        if self.results.empty:
            raise ValueError(
                "ウォークフォワード検証結果が作成できませんでした。"
            )

        self.results["Date"] = pd.to_datetime(self.results["Date"])
        self.results["Target_Known_Date"] = pd.to_datetime(
            self.results["Target_Known_Date"]
        )

        self.results = (
            self.results
            .drop_duplicates(subset=["Date"], keep="last")
            .set_index("Date")
            .sort_index()
        )

        self.model_count = int(model_number)
        self.metrics = self._calculate_metrics(self.results)
        self.calibration_table = self._make_band_table(
            self.results, "全期間"
        )
        self.stability_table = self._make_stability_table(
            self.results
        )

        return self.results.copy(), dict(self.metrics)

    @staticmethod
    def _make_band_table(results, period_name):
        if results is None or results.empty:
            return pd.DataFrame()

        p = pd.to_numeric(results["Probability_Up"], errors="coerce")
        a = pd.to_numeric(results["Actual"], errors="coerce")
        r = pd.to_numeric(results["Future_Return"], errors="coerce")

        bands = [
            ("50%未満", None, 0.50),
            ("50～55%", 0.50, 0.55),
            ("55～60%", 0.55, 0.60),
            ("60～65%", 0.60, 0.65),
            ("65%以上", 0.65, None),
        ]

        rows = []
        for label, low, high in bands:
            if low is None:
                mask = p < high
            elif high is None:
                mask = p >= low
            else:
                mask = (p >= low) & (p < high)

            bp, ba, br = p[mask].dropna(), a[mask].dropna(), r[mask].dropna()

            rows.append({
                "Period": period_name,
                "Probability_Band": label,
                "Count": int(mask.sum()),
                "Average_Predicted_Probability":
                    float(bp.mean()) if not bp.empty else None,
                "Actual_Up_Rate":
                    float((ba == 1).mean()) if not ba.empty else None,
                "Average_Future_Return":
                    float(br.mean()) if not br.empty else None,
                "Median_Future_Return":
                    float(br.median()) if not br.empty else None,
            })

        return pd.DataFrame(rows)

    def _make_stability_table(self, results):
        ordered = results.sort_index()
        split = len(ordered) // 2

        first = self._make_band_table(
            ordered.iloc[:split], "前半"
        )
        second = self._make_band_table(
            ordered.iloc[split:], "後半"
        )

        return pd.concat(
            [first, second],
            ignore_index=True,
        )

    def _calculate_metrics(self, results):
        actual = pd.to_numeric(
            results["Actual"], errors="coerce"
        ).astype(int)

        prediction = pd.to_numeric(
            results["Prediction"], errors="coerce"
        ).astype(int)

        probability = pd.to_numeric(
            results["Probability_Up"], errors="coerce"
        )

        future_return = pd.to_numeric(
            results["Future_Return"], errors="coerce"
        )

        accuracy = float(accuracy_score(actual, prediction))
        precision = float(
            precision_score(actual, prediction, zero_division=0)
        )
        recall = float(
            recall_score(actual, prediction, zero_division=0)
        )
        f1 = float(
            f1_score(actual, prediction, zero_division=0)
        )

        try:
            auc = (
                float(roc_auc_score(actual, probability))
                if actual.nunique() >= 2
                else None
            )
        except Exception:
            auc = None

        try:
            brier = float(
                brier_score_loss(actual, probability)
            )
        except Exception:
            brier = None

        matrix = confusion_matrix(
            actual, prediction, labels=[0, 1]
        )
        tn, fp, fn, tp = [
            int(x) for x in matrix.ravel()
        ]

        up_returns = future_return[prediction == 1].dropna()
        down_returns = future_return[prediction == 0].dropna()

        high = probability >= 0.60
        trained_count = (
            int(
                (self.training_log["Status"] == "TRAINED").sum()
            )
            if (
                not self.training_log.empty
                and "Status" in self.training_log.columns
            )
            else 0
        )

        leakage_ok = (
            bool(self.training_log["Leakage_Check"].dropna().all())
            if (
                not self.training_log.empty
                and "Leakage_Check" in self.training_log.columns
            )
            else None
        )

        return {
            "engine_version": BACKTEST_ENGINE_VERSION,
            "target_horizon": int(self.target_horizon),
            "target_return_threshold": float(
                self.target_return_threshold
            ),
            "target_definition":
                "Close(t+horizon) / Open(t+1) - 1",
            "purged": True,
            "purge_method": "Target_Known_Date",
            "purge_days": int(self.target_horizon),
            "leakage_check_all_passed": leakage_ok,
            "model_count": int(self.model_count),
            "trained_model_count": trained_count,
            "prediction_count": int(len(results)),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "brier_score": brier,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "actual_up_count": int((actual == 1).sum()),
            "actual_down_count": int((actual == 0).sum()),
            "predicted_up_count": int((prediction == 1).sum()),
            "predicted_down_count": int((prediction == 0).sum()),
            "actual_up_rate": float((actual == 1).mean()),
            "predicted_up_rate": float((prediction == 1).mean()),
            "average_future_return": float(future_return.mean()),
            "average_return_predicted_up":
                float(up_returns.mean()) if not up_returns.empty else None,
            "median_return_predicted_up":
                float(up_returns.median()) if not up_returns.empty else None,
            "average_return_predicted_down":
                float(down_returns.mean()) if not down_returns.empty else None,
            "median_return_predicted_down":
                float(down_returns.median()) if not down_returns.empty else None,
            "high_confidence_count": int(high.sum()),
            "high_confidence_accuracy":
                float((actual[high] == prediction[high]).mean())
                if high.any() else None,
            "high_confidence_avg_return":
                float(future_return[high].mean())
                if high.any() else None,
        }

    # 旧main.py互換
    def backtest(self, ai_data):
        return self.run(ai_data)

    def run_backtest(self, ai_data):
        return self.run(ai_data)

    def get_results(self):
        return self.results.copy()

    def get_metrics(self):
        return dict(self.metrics)

    def get_training_log(self):
        return self.training_log.copy()

    # v2.4追加
    def get_calibration_table(self):
        return self.calibration_table.copy()

    # v2.4追加
    def get_stability_table(self):
        return self.stability_table.copy()

    def get_target_info(self):
        return {
            "target_horizon": int(self.target_horizon),
            "target_return_threshold": float(
                self.target_return_threshold
            ),
            "entry_price": "Open(t+1)",
            "evaluation_price":
                f"Close(t+{self.target_horizon})",
            "target_definition":
                "Future_Return > target_return_threshold",
            "future_return_definition":
                "Close(t+horizon) / Open(t+1) - 1",
            "purged": True,
            "purge_method": "Target_Known_Date",
            "purge_days": int(self.target_horizon),
            "engine_version": BACKTEST_ENGINE_VERSION,
        }

    def get_engine_info(self):
        return {
            "engine": "WalkForwardBacktest",
            "version": BACKTEST_ENGINE_VERSION,
            "method": "Expanding Walk-Forward",
            "purged": True,
            "purge_method": "Target_Known_Date",
            "purge_days": int(self.target_horizon),
            "initial_train_size": int(self.initial_train_size),
            "test_size": int(self.test_size),
            "retrain_every": int(self.retrain_every),
            "prediction_threshold": float(
                self.prediction_threshold
            ),
            "threshold": float(self.threshold),
            "target_horizon": int(self.target_horizon),
            "target_return_threshold": float(
                self.target_return_threshold
            ),
            "model_count": int(self.model_count),
        }
