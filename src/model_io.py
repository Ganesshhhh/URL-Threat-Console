"""
Model persistence helpers shared by every script that reads or writes
models/model.joblib.

WHY THIS EXISTS: joblib.dump() on a raw XGBClassifier just pickles the
Python object, which internally pickles the C++ Booster's memory buffer.
That buffer's exact layout is NOT guaranteed stable across xgboost
versions, platforms, or even Python versions -- loading a model pickled by
one xgboost build with a different one can raise
`XGBoostError: input stream corrupted` even though the file itself isn't
corrupted at all. This bit us in practice: a model trained in one
environment failed to load in another with a newer/older xgboost install.

XGBoost's own documented fix is to persist the model through its native
serialization (`Booster.save_model` / `.load_model`, using the UBJ format),
which IS versioned and guaranteed backward/forward compatible:
https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html

This project stores everything in a single `model.joblib` file by
convention (every script expects one path), so we keep that filename but
change WHAT joblib actually persists:
  - For XGBoost models: the raw bytes of the model in UBJ format (version-
    stable) plus feature_names.
  - For other sklearn models (LogisticRegression, RandomForest -- only
    relevant if compare_models.py ever picks one of those as the winner):
    a plain pickled object, which is the normal/expected way to persist
    those and doesn't have this problem.

Old bundles saved before this fix (with a raw "model" key holding a pickled
XGBClassifier) are still readable for backward compatibility, on whatever
xgboost version originally pickled them.
"""

import os
import tempfile

import joblib
import xgboost as xgb


def save_model_bundle(clf, feature_names, path):
    if isinstance(clf, xgb.XGBClassifier):
        tmp_path = tempfile.mktemp(suffix=".ubj")
        try:
            clf.save_model(tmp_path)
            with open(tmp_path, "rb") as f:
                raw = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        joblib.dump({"xgb_raw": raw, "feature_names": feature_names}, path)
    else:
        joblib.dump({"model": clf, "feature_names": feature_names}, path)


def load_model_bundle(path):
    bundle = joblib.load(path)

    if "xgb_raw" in bundle:
        tmp_path = tempfile.mktemp(suffix=".ubj")
        try:
            with open(tmp_path, "wb") as f:
                f.write(bundle["xgb_raw"])
            clf = xgb.XGBClassifier()
            clf.load_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return clf, bundle["feature_names"]

    # Backward compatibility with bundles saved before this fix.
    return bundle["model"], bundle["feature_names"]
