import statsmodels.api as sm
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE, Isomap
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from typing import List

import warnings

warnings.filterwarnings("ignore")


class PQR:
    def __init__(self, n_components, quantile):
        self.n_components = n_components
        self.quantile = quantile

    def fit_transform(self, y, x):
        # only 2d data can be pass into standardscalar
        y = np.array(y).reshape(-1, 1)
        x = np.array(x)
        x = x.reshape(1, len(x)) if x.ndim == 1 else x

        row, col = x.shape
        quantile = self.quantile
        n_components = self.n_components
        component = np.full([row, n_components], np.nan)
        weight = np.full([col, n_components], np.nan)
        sc_x_list = []
        params_list = []

        assert col > self.n_components, f'n_components={self.n_components} must be between 0 and min(n_samples, n_features)={col} '

        for comp in range(n_components):
            # standardization
            sc_x, sc_y = StandardScaler(), StandardScaler()
            x = sc_x.fit_transform(x)
            # y = sc_y.fit_transform(y)
            sc_x_list.append(sc_x)

            # first pass    
            phi = np.full([col, 1], np.nan)

            for c in range(col):
                model = sm.QuantReg(y, sm.add_constant(x[:, c]), has_constant='add').fit(q=quantile)
                phi[c] = model.params[1]

            # second pass
            component_ = np.full([row, 1], np.nan)
            weight[:, comp] = phi.reshape(-1, )

            for r in range(row):
                # model_2 = sm.OLS(x[r,:], sm.add_constant(phi, has_constant='add')).fit()
                # component_[r] = model_2.params[1]
                model_2 = np.linalg.lstsq(sm.add_constant(phi, has_constant='add'), x[r, :])
                component_[r] = model_2[0][1]

            # to get second component, orthogonal is required
            # reassign resid to x
            model_3 = sm.OLS(x, sm.add_constant(component_, has_constant='add')).fit()
            x = model_3.resid
            params_list.append(model_3.params)
            # reassign resid to y
            y = (sm.OLS(y, sm.add_constant(component_, has_constant='add')).fit()).resid.reshape(-1, 1)
            # save component_

            component[:, comp] = component_.reshape(-1, )

        # assign attributes
        self.component = component
        self.weight = weight
        self.sc_x_list = sc_x_list
        self.params_list = params_list

        return component

    def transform(self, x):
        # ensure the dimension of x is 2d
        x = np.array(x)
        x = x.reshape(1, len(x)) if x.ndim == 1 else x

        row, col = x.shape
        n_components = self.n_components
        weight = self.weight
        sc_x_list = self.sc_x_list
        component = np.full([row, n_components], np.nan)
        params_list = self.params_list

        for comp in range(n_components):
            x = sc_x_list[comp].transform(x)
            component_ = np.full([row, 1], np.nan)

            for r in range(row):
                model_2 = sm.OLS(x[r, :], sm.add_constant(weight[:, comp], has_constant='add')).fit()
                component_[r] = model_2.params[1]

            # orthogonal
            _ = np.dot(sm.add_constant(component_, has_constant='add'), params_list[comp])
            _ = _.reshape(1, len(_)) if _.ndim == 1 else _
            x -= _
            component[:, comp] = component_.reshape(-1, )

        return component


class PLS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, y, x):
        # only 2d data can be pass into standardscalar
        y = np.array(y).reshape(-1, 1)
        x = np.array(x)
        x = x.reshape(1, len(x)) if x.ndim == 1 else x

        row, col = x.shape
        n_components = self.n_components
        component = np.full([row, n_components], np.nan)
        weight = np.full([col, n_components], np.nan)
        sc_x_list = []
        params_list = []

        assert col > self.n_components, f'n_components={self.n_components} must be between 0 and min(n_samples, n_features)={col} '

        for comp in range(n_components):
            # standardization
            sc_x, sc_y = StandardScaler(), StandardScaler()
            x = sc_x.fit_transform(x)
            # y = sc_y.fit_transform(y)
            sc_x_list.append(sc_x)

            # first pass    
            phi = np.full([col, 1], np.nan)

            for c in range(col):
                model = sm.OLS(y, sm.add_constant(x[:, c]), has_constant='add').fit()
                phi[c] = model.params[1]

            # second pass
            component_ = np.full([row, 1], np.nan)
            weight[:, comp] = phi.reshape(-1, )

            for r in range(row):
                model_2 = sm.OLS(x[r, :], sm.add_constant(phi, has_constant='add')).fit()
                component_[r] = model_2.params[1]

            # to get second component, orthogonal is required
            # reassign resid to x
            model_3 = sm.OLS(x, sm.add_constant(component_, has_constant='add')).fit()
            x = model_3.resid
            params_list.append(model_3.params)
            # reassign resid to y
            y = (sm.OLS(y, sm.add_constant(component_, has_constant='add')).fit()).resid.reshape(-1, 1)
            # save component_

            component[:, comp] = component_.reshape(-1, )

        # assign attributes
        self.component = component
        self.weight = weight
        self.sc_x_list = sc_x_list
        self.params_list = params_list

        return component

    def transform(self, x):
        # ensure the dimension of x is 2d
        x = np.array(x)
        x = x.reshape(1, len(x)) if x.ndim == 1 else x

        row, col = x.shape
        n_components = self.n_components
        weight = self.weight
        sc_x_list = self.sc_x_list
        component = np.full([row, n_components], np.nan)
        params_list = self.params_list

        for comp in range(n_components):
            x = sc_x_list[comp].transform(x)
            component_ = np.full([row, 1], np.nan)

            for r in range(row):
                model_2 = sm.OLS(x[r, :], sm.add_constant(weight[:, comp], has_constant='add')).fit()
                component_[r] = model_2.params[1]

            # orthogonal
            _ = np.dot(sm.add_constant(component_, has_constant='add'), params_list[comp])
            _ = _.reshape(1, len(_)) if _.ndim == 1 else _
            x -= _
            component[:, comp] = component_.reshape(-1, )

        return component


class ModuleBase:
    def __init__(self, data: pd.DataFrame, y_name_str: str, x_name_list: List[str] = None, oos_periods: int = None,
                 rescale_data: bool = True, base_model='ar'):
        self.df = data
        self.y_name = y_name_str
        self.x_name = x_name_list
        self.oos_periods = oos_periods
        self.rescale_data = rescale_data
        self.recursive = False
        self.base_model = base_model
        self._rescale() if rescale_data else self._train_test_split()  # train test split is included in _rescale method
        self._rescale_base() if rescale_data else self._train_test_split_base()
        self.x_train_origin = self.x_train
        self.y_train_origin = self.y_train
        self.y_train_origin = self.y_test
        self.y_test_origin = self.y_train

    def _train_test_split(self, oos_periods: int = None):
        df = self.df.copy()
        if oos_periods:
            oos_periods = -abs(oos_periods)
        else:
            oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods

        # ------------------------------------------
        # setup x, y
        # ------------------------------------------
        y = df[self.y_name]
        x = df[self.x_name] if self.x_name else df
        self.x_name = list(x.columns)

        # shift x variables forward one period
        x = x.shift(1).dropna()
        y = y.iloc[1:]
        self.index = x.index
        # ------------------------------------------
        # ins oos split, transform  x and y into np.array
        # ------------------------------------------
        if oos_periods:
            self.y_train, self.y_test = y.iloc[:oos_periods].values.reshape(-1, 1), y.iloc[oos_periods:].values.reshape(
                -1, 1)
            self.x_train, self.x_test = x.iloc[:oos_periods].values, x.iloc[oos_periods:].values

        # a condition when there is no oos data
        else:
            self.y_train, self.y_test = y.values.reshape(-1, 1), np.array([])
            self.x_train, self.x_test = x.values, np.array([])

    def _rescale(self, oos_periods: int = None):
        if oos_periods:
            oos_periods = -abs(oos_periods)
        else:
            oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods
        # ------------------------------------------
        # automatically split the data
        # ------------------------------------------
        sc_y, sc_x = StandardScaler(), StandardScaler()
        self._train_test_split(oos_periods=oos_periods)

        # ------------------------------------------
        # stanardize the data
        # ------------------------------------------
        # self.y_train = sc_y.fit_transform(self.y_train)
        self.x_train = sc_x.fit_transform(self.x_train)

        if oos_periods:
            # self.y_test = sc_y.transform(self.y_test)
            self.x_test = sc_x.transform(self.x_test)

    def statistics(self):
        assert self.recursive == False, 'use .statistics_recursive() instead'
        oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods
        base_model = self.base_model
        pred_ins = self.pred_ins.flatten()
        pred_oos = self.pred_oos.flatten()

        # combine pred & true value into dataframe 
        pred_base = self._fit_basemodel(base_model=base_model, recursive=False)
        pred = np.concatenate((pred_ins.flatten(), pred_oos.flatten()), axis=0)
        y = np.concatenate((self.y_train.flatten(), self.y_test.flatten()), axis=0)

        pred_df = pd.DataFrame(
            {
                f'pred {self.y_name}': pred,
                f'true {self.y_name}': y,
                f'base {self.y_name}': pred_base
            },
            index=self.index
        )

        # https://stackoverflow.com/questions/54614157/scikit-learn-statsmodels-which-r-squared-is-correct
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
        statistics = {
            'in sample':
                {
                    'mse': np.nanmean((y[:oos_periods] - pred[:oos_periods]) ** 2),
                    'pseudo_rsquare': self.prsquared(y[:oos_periods], pred[:oos_periods], pred_base[:oos_periods]),
                    'adj_rsquare': self.adjrsquared(y[:oos_periods], pred_ins),
                },

            'out of sample':
                {
                    'mse': np.nanmean((y[oos_periods:] - pred[oos_periods:]) ** 2),
                    'pseudo_rsquare': self.prsquared(y[oos_periods:], pred[oos_periods:], pred_base[oos_periods:]),
                    'adj_rsquare': self.adjrsquared(y[oos_periods:], pred_oos),
                },

            'full sample':
                {
                    'mse': np.nanmean((y - pred) ** 2),
                    'pseudo_rsquare': self.prsquared(y, pred, pred_base),
                    'adj_rsquare': self.adjrsquared(y, pred),
                },
        }

        return pred_df, pd.DataFrame(statistics)

    def statistics_recursive(self):
        assert self.recursive, 'use .statistics() instead'
        oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods
        base_model = self.base_model
        pred_ins = self.pred_ins.flatten()
        pred_base = self._fit_basemodel(base_model=base_model, recursive=True)
        pred = self.pred_recursive.flatten()
        y = self.y_recursive.flatten()

        # combine pred & true value into dataframe 
        pred_df = pd.DataFrame(
            {
                f'pred {self.y_name}': pred,
                f'true {self.y_name}': y,
                f'base {self.y_name}': pred_base
            },
            index=self.index
        )
        # calcualate stastics
        statistics = {
            'in sample':
                {
                    'mse': np.nanmean((y[:oos_periods] - pred[:oos_periods]) ** 2),
                    'pseudo_rsquare': self.prsquared(y[:oos_periods], pred[:oos_periods], pred_base[:oos_periods]),
                    'adj_rsquare': self.adjrsquared(y[:oos_periods], pred_ins)
                },

            'out of sample':
                {
                    'mse': np.nanmean((y[oos_periods:] - pred[oos_periods:]) ** 2),
                    'pseudo_rsquare': self.prsquared(y[oos_periods:], pred[oos_periods:], pred_base[oos_periods:]),
                    'adj_rsquare': None,
                },

            'full sample':
                {
                    'mse': np.nanmean((y - pred) ** 2),
                    'pseudo_rsquare': self.prsquared(y, pred, pred_base),
                    'adj_rsquare': None,
                },
        }

        return pred_df, pd.DataFrame(statistics)

    def _fit_basemodel(self, base_model, recursive: bool = False):
        if base_model == 'ar':
            pred_base = self._fit_base_ar(recursive)

        elif base_model == 'ols':
            pred_base = self._fit_base_ols(recursive)

        elif base_model == 'qr':
            pred_base = self._fit_base_qr(recursive)

        elif base_model == 'pca':
            pred_base = self._fit_base_pca(recursive)

        elif base_model == 'zero':
            pred_base = self._fit_base_zero(recursive)

        return pred_base

    def _fit_base_zero(self, recursive):
        oos_periods = self.oos_periods
        y_input_train = self.y_train
        y_input_test = self.y_test
        y = np.concatenate([y_input_train, y_input_test], axis=0) if oos_periods else y_input_train
        y = self.y_recursive if recursive else y
        self.pred_base = pred_base = np.zeros(len(y))

        return pred_base

    def _fit_base_ar(self, recursive):
        oos_periods = self.oos_periods
        y_input_train = self.y_train
        y_input_test = self.y_test
        y = np.concatenate([y_input_train, y_input_test], axis=0) if oos_periods else y_input_train
        y = self.y_recursive if recursive else y

        # choose ar model as baseline model
        # choose optimal lag for ar model base on y_train
        result = sm.tsa.arma_order_select_ic(y_input_train, 6, 0, ic="aic")  # ic=["aic", "bic"]
        p, _ = result.aic_min_order
        # fit ar model on full y
        ar = sm.tsa.ARIMA(y, order=(p, 0, 0)).fit()
        self.pred_base = pred_base = ar.fittedvalues

        return pred_base

    def _fit_base_ols(self, recursive):
        oos_periods = self.oos_periods
        x_input_train_ = self.x_train_base
        x_input_test_ = self.x_test_base
        y_input_train_ = self.y_train_base
        x_input_train = sm.add_constant(x_input_train_, has_constant='add')

        if recursive:
            x_input_test = sm.add_constant(x_input_test_, has_constant='add')
            model = sm.OLS(y_input_train_, x_input_train).fit()
            pred_ins = model.predict(x_input_train)
            pred_oos = model.predict(x_input_test)
            pred_base = np.concatenate([pred_ins, [pred_oos[0]]], axis=0)

            for t in range(abs(oos_periods) - 1, 0, -1):
                self._rescale_base(oos_periods=t) if self.rescale_data else self._train_test_split_base(oos_periods=t)
                y_input_train = self.y_train_base
                x_input_train = self.x_train_base
                x_input_test = self.x_test_base
                x_input_train = sm.add_constant(x_input_train, has_constant='add')
                x_input_test = sm.add_constant(x_input_test, has_constant='add')

                model = sm.OLS(y_input_train, x_input_train).fit()
                pred_oos = model.predict(x_input_test)
                pred_base = np.concatenate([pred_base, [pred_oos[0]]], axis=0)

            self.x_train_base = x_input_train_
            self.x_test_base = x_input_test_
            self.y_train_base = y_input_train_

        else:
            model = sm.OLS(y_input_train_, x_input_train).fit()
            pred_base = pred_ins = model.predict(x_input_train)

            if oos_periods:
                x_input_test = sm.add_constant(x_input_test_, has_constant='add')
                pred_oos = model.predict(x_input_test)
                pred_base = np.concatenate([pred_ins, pred_oos], axis=0)

        self.pred_base = pred_base

        return pred_base

    def _fit_base_qr(self, recursive):
        oos_periods = self.oos_periods
        x_input_train_ = self.x_train_base
        x_input_test_ = self.x_test_base
        y_input_train_ = self.y_train_base
        x_input_train = sm.add_constant(x_input_train_, has_constant='add')

        if recursive:
            x_input_test = sm.add_constant(x_input_test_, has_constant='add')
            model = sm.QuantReg(y_input_train_, x_input_train).fit(self.quantile)
            pred_ins = model.predict(x_input_train)
            pred_oos = model.predict(x_input_test)
            pred_base = np.concatenate([pred_ins, [pred_oos[0]]], axis=0)

            for t in range(abs(oos_periods) - 1, 0, -1):
                self._rescale_base(oos_periods=t) if self.rescale_data else self._train_test_split_base(oos_periods=t)
                y_input_train = self.y_train_base
                x_input_train = self.x_train_base
                x_input_test = self.x_test_base
                x_input_train = sm.add_constant(x_input_train, has_constant='add')
                x_input_test = sm.add_constant(x_input_test, has_constant='add')

                model = sm.QuantReg(y_input_train, x_input_train).fit(self.quantile)
                pred_oos = model.predict(x_input_test)
                pred_base = np.concatenate([pred_base, [pred_oos[0]]], axis=0)

            self.x_train_base = x_input_train_
            self.x_test_base = x_input_test_
            self.y_train_base = y_input_train_

        else:
            model = sm.QuantReg(y_input_train_, x_input_train).fit(self.quantile)
            pred_base = pred_ins = model.predict(x_input_train)

            if oos_periods:
                x_input_test = sm.add_constant(x_input_test_, has_constant='add')
                pred_oos = model.predict(x_input_test)
                pred_base = np.concatenate([pred_ins, pred_oos], axis=0)

        self.pred_base = pred_base

        return pred_base

    def _fit_base_pca(self, recursive):
        oos_periods = self.oos_periods
        x_input_train_ = self.x_train_base
        x_input_test_ = self.x_test_base
        y_input_train_ = self.y_train_base
        # x_input_train = sm.add_constant(x_input_train_, has_constant='add')

        if recursive:
            pca = PCA(n_components=7)
            factor_ins = pca.fit_transform(x_input_train_)
            factor_ins = sm.add_constant(factor_ins, has_constant='add')
            factor_oos = pca.transform(x_input_test_)
            factor_oos = sm.add_constant(factor_oos, has_constant='add')
            # x_input_test = sm.add_constant(x_input_test_, has_constant='add')
            model = sm.OLS(y_input_train_, factor_ins).fit()
            pred_ins = model.predict(factor_ins)
            pred_oos = model.predict(factor_oos)
            pred_base = np.concatenate([pred_ins, [pred_oos[0]]], axis=0)

            for t in range(abs(oos_periods) - 1, 0, -1):
                self._rescale_base(oos_periods=t) if self.rescale_data else self._train_test_split_base(oos_periods=t)
                y_input_train = self.y_train_base
                x_input_train = self.x_train_base
                x_input_test = self.x_test_base
                pca2 = PCA(n_components=7)
                factor_ins2 = pca2.fit_transform(x_input_train)
                factor_ins2 = sm.add_constant(factor_ins2, has_constant='add')
                factor_oos2 = pca2.transform(x_input_test)
                factor_oos2 = sm.add_constant(factor_oos2, has_constant='add')

                model = sm.OLS(y_input_train, factor_ins2).fit()
                pred_oos = model.predict(factor_oos2)
                pred_base = np.concatenate([pred_base, [pred_oos[0]]], axis=0)

            self.x_train_base = x_input_train_
            self.x_test_base = x_input_test_
            self.y_train_base = y_input_train_

        else:
            pca3 = PCA(n_components=7)
            factor_ = pca3.fit_transform(x_input_train_)
            factor_ = sm.add_constant(factor_, has_constant='add')
            model = sm.OLS(y_input_train_, factor_).fit()
            pred_base = pred_ins = model.predict(factor_)

            if oos_periods:
                pca4 = PCA(n_components=7)
                factor_oos4 = pca4.transform(x_input_test_)
                factor_oos4 = sm.add_constant(factor_oos4, has_constant='add')
                pred_oos = model.predict(factor_oos4)
                pred_base = np.concatenate([pred_ins, pred_oos], axis=0)

        self.pred_base = pred_base

        return pred_base

    def _train_test_split_base(self, oos_periods: int = None):
        df = self.df.copy()

        if oos_periods:
            oos_periods = -abs(oos_periods)
        else:
            oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods

        x_base = df.shift(1).dropna()
        # x_base = df.dropna()
        # x_base = df.drop(columns=['MthRet'])
        y_base = df[self.y_name].iloc[1:]

        if oos_periods:
            self.y_train_base, self.y_test_base = y_base.iloc[:oos_periods].values.reshape(-1, 1), y_base.iloc[
                                                                                                   oos_periods:].values.reshape(
                -1, 1)
            self.x_train_base, self.x_test_base = x_base.iloc[:oos_periods].values, x_base.iloc[oos_periods:].values
        else:
            self.y_train_base, self.y_test_basse = y_base.values.reshape(-1, 1), np.array([])
            self.x_train_base, self.x_test_base = x_base.values, np.array([])

    def _rescale_base(self, oos_periods: int = None):
        if oos_periods:
            oos_periods = -abs(oos_periods)
        else:
            oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods
        # ------------------------------------------
        # automatically split the data
        # ------------------------------------------
        sc_y, sc_x = StandardScaler(), StandardScaler()
        self._train_test_split_base(oos_periods=oos_periods)

        # ------------------------------------------
        # stanardize the data
        # ------------------------------------------
        # self.y_train_base = sc_y.fit_transform(self.y_train_base)
        self.x_train_base = sc_x.fit_transform(self.x_train_base)

        if oos_periods:
            # self.y_test_base = sc_y.transform(self.y_test_base)
            self.x_test_base = sc_x.transform(self.x_test_base)

    def _fit_recursive(self, *args, **kargs):
        oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods

        if self.dr:
            if self.sep:
                x_input_train_ = self.factor_ins
                x_input_test_ = self.factor_oos

            else:
                x_input_train_ = self.factor_ins
                x_input_test_ = self.factor_oos

        else:
            x_input_train_ = self.x_train
            x_input_test_ = self.x_test

        # initiate the first fit
        # fit the model
        self.fit(x_input_train_, x_input_test_, *args, **kargs)

        # store value before recursive fit
        model_ = self.model

        # combine y_train and the first y_test value
        y = np.concatenate([self.y_train, [self.y_test[0]]], axis=0)

        # combine x_input_train and the first x_input_test value
        x = np.concatenate([x_input_train_, [x_input_test_[0]]], axis=0)

        # combine pred_ins and the first pred_oos value
        pred = np.concatenate([self.pred_ins, [self.pred_oos[0]]], axis=0)
        pred_ins_ = self.pred_ins

        for t in range(abs(oos_periods) - 1, 0, -1):
            self._rescale(oos_periods=t) if self.rescale_data else self._train_test_split(oos_periods=t)

            if self.dr:
                if self.sep:
                    self.seperate(self.x_name_list_sep, *self.dr_args, dr_algo=self.dr_algo, oos_periods=t,
                                  **self.dr_kargs)
                    x_input_train = self.factor_ins
                    x_input_test = self.factor_oos

                else:
                    self.dr_setup(*self.dr_args, dr_algo=self.dr_algo, **self.dr_kargs)
                    x_input_train = self.factor_ins
                    x_input_test = self.factor_oos

            else:
                x_input_train = self.x_train
                x_input_test = self.x_test

            # fit the model
            self.fit(x_input_train, x_input_test, *args, **kargs)

            # combine the first y_test value
            y = np.concatenate([y, [self.y_test[0]]], axis=0)

            # combine the first x_input_test value
            x = np.concatenate([x, [x_input_test[0]]], axis=0)

            # combine the first pred_oos value
            pred = np.concatenate([pred, [self.pred_oos[0]]], axis=0)

            # self.pred_ins, self.pred_oos, self.factor_ins, self.factor_oos are meaningless in recursive fit
        # reassign pre-recursive value to attribute 
        self.model = model_
        self.x_input_train = x_input_train_
        self.x_input_test = x_input_test_
        self.pred_ins = pred_ins_

        # assign pro-recursive value to attribute
        self.y_recursive = y
        self.x_input_recursive = x
        self.pred_recursive = pred
        self.recursive = True

    def adjrsquared(self, y: np.array, pred: np.array):
        if pred.shape[0] == 0 or y.shape[0] == 0:
            adj_rsquare = None
        else:
            row, feature = pred.shape[0], self.x_input_train.shape[1]
            rsquare = r2_score(y, pred)
            adj_rsquare = 1 - (1 - rsquare) * (row - 1) / (row - feature - 1)

        return adj_rsquare

    @staticmethod
    def prsquared(y, pred: np.array, pred_base: np.array):
        # input : array, fitted value series, unconditional series
        # output: floatc
        numer = (y.reshape(-1, 1) - pred.reshape(-1, 1))
        numer = np.nansum(np.square(numer))

        denom = (y.reshape(-1, 1) - pred_base.reshape(-1, 1))
        denom = np.nansum(np.square(denom))

        pseudo_rsquare = 1 - (numer / denom)
        return pseudo_rsquare

    @staticmethod
    def interactive_plot(df: pd.DataFrame, title: str = '自定圖表標題', recession: bool = False,
                         oos_periods: int = None, height: int = 370, width: int = 1050):
        """ 使用 plotly 劃出 dataframe 裡的序列畫出，搭配衰退陰影

        Args:
            df (_type_): dataframe index 要是時間序列，column 的名字要先設定好
            title (str, optional): _description_. Defaults to '自訂圖表標題'.
        """
        # ===================================
        # setup
        # ===================================
        df = df.dropna()
        str_dt_index = df.index.strftime('%Y-%m-%d')

        recession_period = [
            ['1969-12', '1970-11'],
            ['1973-11', '1975-03'],
            ['1980-01', '1980-07'],
            ['1981-01', '1982-11'],
            ['1990-07', '1991-03'],
            ['2001-03', '2001-11'],
            ['2007-12', '2009-06'],
            ['2020-02', '2020-04']
        ]

        fig_layout = go.Layout(

            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, b=40, t=60, pad=0),
            legend=dict(
                orientation="h",
                yanchor="top",
                xanchor="left"),
            height=height,
            width=width
        )
        # ===================================
        # initiate
        # ===================================

        fig = go.Figure()
        fig.update_layout(fig_layout)
        fig.update_layout(
            template='seaborn')  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
        fig.update_layout(title_text=title, title_x=0.5)

        if recession:
            for recession in recession_period:
                if int(recession[0][:4]) >= int(str_dt_index[0][:4]) and int(recession[1][:4]) <= int(
                        str_dt_index[-1][:4]):
                    fig.add_vrect(
                        x0=recession[0],
                        x1=recession[1],
                        fillcolor='rgba(30,30,30,0.3)',
                        opacity=0.5,
                        line_width=0)
        if oos_periods:
            fig.add_vline(
                x=df.index[-abs(oos_periods)].strftime('%Y-%m-%d'),
                line_dash="dash",
                line_color="black",
                line_width=3)
        # ===================================
        # loop through data in df
        # ===================================

        for col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col],
                           mode='lines',
                           name=col,
                           showlegend=True,
                           )
            )

        fig.show()


class ModuleDR(ModuleBase):
    def __init__(self, data: pd.DataFrame, y_name_str: str, *args, **kargs):
        super().__init__(data, y_name_str, *args, **kargs)

        # unsupervised, linear

    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    def fit_pca(self, n_components: (float, int) = 1, **kargs):
        oos_periods = self.oos_periods
        pca = PCA(n_components=n_components, **kargs)

        factor_ins = pca.fit_transform(self.x_train)
        factor_oos = pca.transform(self.x_test) if oos_periods else np.array([])

        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return pca

    # unsupervised, nonlinear
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
    def fit_kpca(self, n_components: int = 1, kernel: str = 'rbf', **kargs):
        oos_periods = self.oos_periods
        kpca = KernelPCA(n_components=n_components, kernel=kernel, **kargs)

        factor_ins = kpca.fit_transform(self.x_train)
        factor_oos = kpca.transform(self.x_test) if oos_periods else np.array([])

        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return kpca

    # unsupervised, nonlinear
    # https://scikit-learn.org/stable/modules/random_projection.html#random-projection
    def fit_grp(self, n_components: int = 1, eps: float = 0.1, **kargs):
        oos_periods = self.oos_periods
        grp = random_projection.GaussianRandomProjection(n_components=n_components, eps=eps, **kargs)

        factor_ins = grp.fit_transform(self.x_train)
        factor_oos = grp.transform(self.x_test) if oos_periods else np.array([])

        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return grp

    # unsupervised, nonlinear
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    def fit_tsne(self, n_components: int = 1, **kargs):
        oos_periods = self.oos_periods
        tsne = TSNE(n_components=n_components, **kargs)

        factor_ins = tsne.fit_transform(self.x_train)
        factor_oos = tsne.fit_transform(self.x_test) if oos_periods else np.array([])

        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return tsne

    # unsupervised, nonlinear
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap
    def fit_isomap(self, n_components: int = 1, n_neighbors: int = 5, **kargs):
        oos_periods = self.oos_periods
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors, **kargs)

        factor_ins = isomap.fit_transform(self.x_train)
        factor_oos = isomap.transform(self.x_test) if oos_periods else np.array([])

        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return isomap

    def fit_pqr(self, n_components: int = 1, quantile: float = 0.5):
        oos_periods = self.oos_periods
        pqr = PQR(n_components=n_components, quantile=quantile)

        factor_ins = pqr.fit_transform(self.y_train, self.x_train)
        factor_oos = pqr.transform(self.x_test) if oos_periods else np.array([])

        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return pqr

    # supervised, linear
    # https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
    def fit_pls(self, n_components: int = 1, **kargs):
        oos_periods = self.oos_periods
        pls = PLSRegression(n_components=n_components, **kargs).fit(self.x_train, self.y_train)

        factor_ins = pls.transform(self.x_train)
        factor_oos = pls.transform(self.x_test) if oos_periods else np.array([])
        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return pls

    # supervised, linear
    # proposed by Giglio, Kelly and Pruitt (2016)
    # def fit_pls(self,n_components:int=1):
    #     oos_periods = self.oos_periods
    #     pls = PLS(n_components=n_components)

    #     factor_ins = pls.fit_transform(self.y_train, self.x_train)
    #     factor_oos = pls.transform(self.x_test) if oos_periods else np.array([])

    #     self.factor_ins, self.factor_oos = factor_ins, factor_oos

    #     return pls

    # supervised, linear (看起來應該是 unsupervised?)
    # 適用於非常態分配 
    # 結果都和 PCA 很像?
    # https://datascience.stackexchange.com/questions/40185/pca-and-fastica-in-scikit-learn-giving-near-identical-results
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    def fit_ica(self, n_components: int = 1, **kargs):
        oos_periods = self.oos_periods
        ica = FastICA(n_components=n_components, **kargs)

        factor_ins = ica.fit_transform(self.x_train)
        factor_oos = ica.transform(self.x_test) if oos_periods else np.array([])

        self.factor_ins, self.factor_oos = factor_ins, factor_oos

        return ica

    def seperate(self, x_name_list, *args, dr_algo: str = 'fit_pca', oos_periods: int = None, **kargs):
        assert all(isinstance(el, list) for el in x_name_list), 'x_name_list must be a list of list'

        factor_ins = np.empty((self.x_train.shape[0], 1))
        factor_oos = np.empty((self.x_test.shape[0], 1)) if oos_periods else np.array([])  # 加了條件，就能使用全樣本降維
        self.x_name_list_sep = x_name_list

        for block in x_name_list:
            self.x_name = block
            self._rescale(oos_periods=oos_periods) if self.rescale_data else self._train_test_split(
                oos_periods=oos_periods)
            if len(block) == 1:
                factor_ins = np.concatenate((factor_ins, self.x_train), axis=1)
                factor_oos = np.concatenate((factor_oos, self.x_test), axis=1) if oos_periods else np.array(
                    [])  # 加了條件，就能使用全樣本降維
            else:
                getattr(self, dr_algo)(*args, **kargs)
                factor_ins = np.concatenate((factor_ins, self.factor_ins), axis=1)
                factor_oos = np.concatenate((factor_oos, self.factor_oos), axis=1) if oos_periods else np.array(
                    [])  # 加了條件，就能使用全樣本降維

        self.factor_ins = factor_ins[:, 1:]
        self.factor_oos = factor_oos[:, 1:] if oos_periods else np.array([])  # 加了條件，就能使用全樣本降維
        self.dr_algo = dr_algo
        self.dr_args = args
        self.dr_kargs = kargs
        self.dr = True
        self.sep = True

    def dr_setup(self, *args, dr_algo: str = 'fit_pca', **kargs):
        getattr(self, dr_algo)(*args, **kargs)
        self.dr = True
        self.dr_algo = dr_algo
        self.dr_args = args
        self.dr_kargs = kargs

    def plot_factor(self):
        oos_periods = self.oos_periods
        df = pd.DataFrame(np.concatenate((self.factor_ins, self.factor_oos), axis=0), index=self.index)
        df[f'true {self.y_name}'] = np.concatenate((self.y_train.flatten(), self.y_test.flatten()), axis=0)
        self.interactive_plot(df, oos_periods=oos_periods)

    def plot_factor_recursive(self):
        oos_periods = self.oos_periods
        df = pd.DataFrame(self.x_input_recursive, index=self.index)
        df[f'true {self.y_name}'] = self.y_recursive.flatten()
        self.interactive_plot(df, oos_periods=oos_periods)


class ModuleOLS(ModuleDR):
    def __init__(self, data: pd.DataFrame, y_name_str: str, *args, **kargs):
        super().__init__(data, y_name_str, *args, **kargs)
        self.dr = False  # trigger in recursive_fit
        self.sep = False  # trigger in recursive_fit

    def fit(self, x_train: np.array, x_test: np.array):
        """_summary_

        Args:
            x_train (_type_): can be x or factor
            x_test (_type_): can be x or factor
        """
        oos_periods = self.oos_periods

        # x may be facotrs or explanatory variables
        # save input x for model evaluation
        x_input_train, x_input_test = x_train, x_test
        y_input_train = self.y_train

        self.x_input_train = x_input_train
        self.x_input_test = x_input_test

        x_input_train = sm.add_constant(x_input_train, has_constant='add')
        model = sm.OLS(y_input_train, x_input_train).fit()
        pred_ins = model.predict(x_input_train)

        if oos_periods:
            x_input_test = sm.add_constant(x_input_test, has_constant='add')
            pred_oos = model.predict(x_input_test)

        else:
            pred_oos, self.y_test = np.array([]), np.array([])

        self.pred_ins, self.pred_oos = pred_ins, pred_oos
        self.model = model

    def fit_recursive(self):
        oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods

        if self.dr:
            if self.sep:
                x_input_train_ = self.factor_ins
                x_input_test_ = self.factor_oos

            else:
                x_input_train_ = self.factor_ins
                x_input_test_ = self.factor_oos

        else:
            x_input_train_ = self.x_train
            x_input_test_ = self.x_test

        # initiate the first fit
        # fit the model
        self.fit(x_input_train_, x_input_test_)

        # store beta before recursive fit
        beta = self.model.params[1:].reshape(1, -1)

        # store value before recursive fit
        model_ = self.model

        # combine y_train and the first y_test value
        y = np.concatenate([self.y_train, [self.y_test[0]]], axis=0)

        # combine x_input_train and the first x_input_test value
        x = np.concatenate([x_input_train_, [x_input_test_[0]]], axis=0)

        # combine pred_ins and the first pred_oos value
        pred = np.concatenate([self.pred_ins, [self.pred_oos[0]]], axis=0)
        pred_ins_ = self.pred_ins

        for t in range(abs(oos_periods) - 1, 0, -1):
            self._rescale(oos_periods=t) if self.rescale_data else self._train_test_split(oos_periods=t)

            if self.dr:
                if self.sep:
                    self.seperate(self.x_name_list_sep, *self.dr_args, dr_algo=self.dr_algo, oos_periods=t,
                                  **self.dr_kargs)
                    x_input_train = self.factor_ins
                    x_input_test = self.factor_oos

                else:
                    self.dr_setup(*self.dr_args, dr_algo=self.dr_algo, **self.dr_kargs)
                    x_input_train = self.factor_ins
                    x_input_test = self.factor_oos

            else:
                x_input_train = self.x_train
                x_input_test = self.x_test

            # fit the model
            self.fit(x_input_train, x_input_test)

            # store beta for every recursive fit
            beta = np.concatenate((beta, self.model.params[1:].reshape(1, -1)), axis=0)

            # combine the first y_test value
            y = np.concatenate([y, [self.y_test[0]]], axis=0)

            # combine the first x_input_test value
            x = np.concatenate([x, [x_input_test[0]]], axis=0)

            # combine the first pred_oos value
            pred = np.concatenate([pred, [self.pred_oos[0]]], axis=0)

            # self.pred_ins, self.pred_oos, self.factor_ins, self.factor_oos are meaningless in recursive fit
        # reassign pre-recursive value to attribute 
        self.model = model_
        self.x_input_train = x_input_train_
        self.x_input_test = x_input_test_
        self.pred_ins = pred_ins_

        # assign pro-recursive value to attribute
        self.y_recursive = y
        self.x_input_recursive = x
        self.pred_recursive = pred
        self.beta = beta
        self.recursive = True


class ModuleQR(ModuleDR):
    def __init__(self, data: pd.DataFrame, y_name_str: str, *args, **kargs):
        super().__init__(data, y_name_str, *args, **kargs)
        self.dr = False  # trigger in recursive_fit
        self.sep = False  # trigger in recursive_fit

    def fit(self, x_train: np.array, x_test: np.array, quantile: float = 0.5):
        """_summary_

        Args:
            x_train (_type_): can be x or factor
            x_test (_type_): can be x or factor
        """
        oos_periods = self.oos_periods

        # x may be facotrs or explanatory variables
        # save input x for model evaluation
        x_input_train, x_input_test = x_train, x_test
        y_input_train = self.y_train

        self.x_input_train = x_input_train
        self.x_input_test = x_input_test

        x_input_train = sm.add_constant(x_input_train, has_constant='add')
        model = sm.QuantReg(y_input_train, x_input_train).fit(quantile)
        pred_ins = model.predict(x_input_train)

        if oos_periods:
            x_input_test = sm.add_constant(x_input_test, has_constant='add')
            pred_oos = model.predict(x_input_test)

        else:
            pred_oos, self.y_test = np.array([]), np.array([])

        self.pred_ins, self.pred_oos = pred_ins, pred_oos
        self.model = model
        self.quantile = quantile

    # def fit_recursive(self, quantile:float=0.5):
    #     super()._fit_recursive(quantile = quantile)

    def fit_recursive(self, quantile=0.5):
        oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods

        if self.dr:
            if self.sep:
                x_input_train_ = self.factor_ins
                x_input_test_ = self.factor_oos

            else:
                x_input_train_ = self.factor_ins
                x_input_test_ = self.factor_oos

        else:
            x_input_train_ = self.x_train
            x_input_test_ = self.x_test

        # initiate the first fit
        # fit the model
        self.fit(x_input_train_, x_input_test_, quantile=quantile)

        # store beta before recursive fit
        beta = self.model.params[1:].reshape(1, -1)

        # store value before recursive fit
        model_ = self.model

        # get unconditional quantile
        model_uncond = sm.QuantReg(self.y_train, np.ones((self.y_train.shape[0], 1))).fit(quantile)
        y_uncond = np.empty((self.y_train.shape[0] + 1, 1))
        y_uncond.fill(model_uncond.params[0])

        # combine y_train and the first y_test value
        y = np.concatenate([self.y_train, [self.y_test[0]]], axis=0)

        # combine x_input_train and the first x_input_test value
        x = np.concatenate([x_input_train_, [x_input_test_[0]]], axis=0)

        # combine pred_ins and the first pred_oos value
        pred = np.concatenate([self.pred_ins, [self.pred_oos[0]]], axis=0)
        pred_ins_ = self.pred_ins

        for t in range(abs(oos_periods) - 1, 0, -1):
            self._rescale(oos_periods=t) if self.rescale_data else self._train_test_split(oos_periods=t)

            if self.dr:
                if self.sep:
                    self.seperate(self.x_name_list_sep, *self.dr_args, dr_algo=self.dr_algo, oos_periods=t,
                                  **self.dr_kargs)
                    x_input_train = self.factor_ins
                    x_input_test = self.factor_oos

                else:
                    self.dr_setup(*self.dr_args, dr_algo=self.dr_algo, **self.dr_kargs)
                    x_input_train = self.factor_ins
                    x_input_test = self.factor_oos

            else:
                x_input_train = self.x_train
                x_input_test = self.x_test

            # fit the model
            self.fit(x_input_train, x_input_test, quantile=quantile)

            # store beta for every recursive fit
            beta = np.concatenate((beta, self.model.params[1:].reshape(1, -1)), axis=0)

            # get unconditional quantile
            model_uncond = sm.QuantReg(y, np.ones((y.shape[0], 1))).fit(quantile)
            y_uncond = np.concatenate([y_uncond, [[model_uncond.params[0]]]], axis=0)

            # combine the first y_test value
            y = np.concatenate([y, [self.y_test[0]]], axis=0)

            # combine the first x_input_test value
            x = np.concatenate([x, [x_input_test[0]]], axis=0)

            # combine the first pred_oos value
            pred = np.concatenate([pred, [self.pred_oos[0]]], axis=0)

            # self.pred_ins, self.pred_oos, self.factor_ins, self.factor_oos are meaningless in recursive fit
        # reassign pre-recursive value to attribute 
        self.model = model_
        self.x_input_train = x_input_train_
        self.x_input_test = x_input_test_
        self.pred_ins = pred_ins_

        # assign pro-recursive value to attribute
        self.y_recursive = y
        self.y_uncond_recursive = y_uncond
        self.x_input_recursive = x
        self.pred_recursive = pred
        self.beta = beta
        self.recursive = True

    def statistics_recursive(self):
        assert self.recursive, 'use .statistics() instead'
        oos_periods = -abs(self.oos_periods) if self.oos_periods else self.oos_periods
        pred_ins = self.pred_ins.flatten()
        pred_base = self.y_uncond_recursive.flatten()
        pred = self.pred_recursive.flatten()
        y = self.y_recursive.flatten()

        # combine pred & true value into dataframe 
        pred_df = pd.DataFrame(
            {
                f'pred {self.y_name}': pred,
                f'true {self.y_name}': y,
                f'base {self.y_name}': pred_base
            },
            index=self.index
        )
        # calcualate stastics
        statistics = {
            'in sample':
                {
                    'mse': np.nanmean((y[:oos_periods] - pred[:oos_periods]) ** 2),
                    'pseudo_rsquare': self.prsquared(y[:oos_periods], pred[:oos_periods], pred_base[:oos_periods]),
                    'adj_rsquare': self.adjrsquared(y[:oos_periods], pred_ins)
                },

            'out of sample':
                {
                    'mse': np.nanmean((y[oos_periods:] - pred[oos_periods:]) ** 2),
                    'pseudo_rsquare': self.prsquared(y[oos_periods:], pred[oos_periods:], pred_base[oos_periods:]),
                    'adj_rsquare': None,
                },

            'full sample':
                {
                    'mse': np.nanmean((y - pred) ** 2),
                    'pseudo_rsquare': self.prsquared(y, pred, pred_base),
                    'adj_rsquare': None,
                },
        }

        return pred_df, pd.DataFrame(statistics)

    def prsquared(self, y, pred: np.array, pred_base: np.array):
        # input : array, fitted value series, unconditional series
        # output: float
        q = self.quantile
        e = (y.reshape(-1, 1) - pred.reshape(-1, 1))
        e = np.where(e < 0, (1 - q) * e, q * e)
        e = np.abs(e)
        ered = y.reshape(-1, 1) - pred_base.reshape(-1, 1)
        ered = np.where(ered < 0, (1 - q) * ered, q * ered)
        ered = np.abs(ered)
        pseudo_rsquare = 1 - (np.sum(e) / np.sum(ered))
        return pseudo_rsquare
