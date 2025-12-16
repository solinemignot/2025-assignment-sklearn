import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import check_classification_targets


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = validate_data(self, X, y=y, reset=True)
        y = np.asarray(y)
        y = np.ravel(y)

        check_classification_targets(y)

        if y.ndim != 1:
            raise ValueError("y should be a 1D array-like.")

        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a strictly positive int.")

        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors cannot be greater than n_samples "
                f"(n_samples = {X.shape[0]})."
            )

        # Store training data
        self.X_ = X

        # Encode classes to allow fast voting via bincount
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.y_encoded_ = y_encoded.astype(int)
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self, attributes=["X_", "y_encoded_", "classes_"])
        X = validate_data(self, X, reset=False)

        distances = pairwise_distances(X, self.X_, metric="euclidean")
        nn_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        n_test = X.shape[0]
        y_pred_encoded = np.empty(n_test, dtype=int)

        n_classes = self.classes_.shape[0]
        for i in range(n_test):
            neigh_labels = self.y_encoded_[nn_idx[i]]
            counts = np.bincount(neigh_labels, minlength=n_classes)
            y_pred_encoded[i] = int(np.argmax(counts))

        return self.classes_[y_pred_encoded]

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        check_is_fitted(self, attributes=["X_", "y_encoded_", "classes_"])
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def __repr__(self):
        """Representation for the cross-validator."""
        return f"MonthlySplit(time_col='{self.time_col}')"

    def _extract_times(self, X):
        """Validate input and return a DataFrame and datetime index."""
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, pd.DataFrame):
            X_df = X
        else:
            raise ValueError("X must be a pandas DataFrame or Series.")

        if self.time_col == "index":
            times = X_df.index
        else:
            if self.time_col not in X_df.columns:
                raise ValueError("time_col must be a column of X or 'index'.")
            times = X_df[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(times):
            raise ValueError("The time index/column must be datetime-like.")

        return X_df, pd.DatetimeIndex(times)

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        _, times = self._extract_times(X)
        months = times.to_period("M")
        n_months = months.unique().shape[0]
        return int(max(n_months - 1, 0))

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        X_df, times = self._extract_times(X)

        if y is not None and len(y) != len(X_df):
            raise ValueError("X and y must be the same length.")

        months = times.to_period("M")
        uniq_months = np.array(sorted(months.unique()))

        for current_month, next_month in zip(uniq_months[:-1],
                                             uniq_months[1:]):
            idx_train = np.where(months == current_month)[0]
            idx_test = np.where(months == next_month)[0]

            idx_train = idx_train[np.argsort(times[idx_train].values)]
            idx_test = idx_test[np.argsort(times[idx_test].values)]

            yield idx_train, idx_test
