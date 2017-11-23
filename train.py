import argparse
import datetime
import os
from shutil import copy2

import numpy as np
import pandas as pd
import xgboost as xgb

np.random.seed(1337)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import shuffle, resample

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras_util import ExponentialMovingAverage, batch_generator

from statsmodels.regression.quantile_regression import QuantReg

from pylightgbm.models import GBMRegressor

from scipy.stats import boxcox

from bayes_opt import BayesianOptimization

from util import Dataset, load_prediction, hstack

categoricals = Dataset.get_part_features('categorical')


class CategoricalAlphaEncoded(object):
    requirements = ['categorical']

    def __init__(self, combinations=[]):
        self.combinations = [map(categoricals.index, comb) for comb in combinations]

    def fit_transform(self, ds):
        return self.transform(ds)

    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros((test_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        for col in range(len(categoricals)):
            test_res[:, col] = self.transform_column(test_cat[:, col])

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            test_res[:, col] = self.transform_column(map(''.join, test_cat[:, comb]))

        return test_res

    def transform_column(self, arr):
        def encode(charcode):
            r = 0
            ln = len(charcode)
            for i in range(ln):
                r += (ord(charcode[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
            return r

        return np.array(map(encode, arr))

    def get_feature_names(self):
        return categoricals + ['_'.join(categoricals[c] for c in comb) for comb in self.combinations]


class CategoricalMeanEncoded(object):
    requirements = ['categorical']

    def __init__(self, C=100, loo=False, noisy=True, noise_std=None, random_state=11, combinations=[]):
        self.random_state = np.random.RandomState(random_state)
        self.C = C
        self.loo = loo
        self.noisy = noisy
        self.noise_std = noise_std
        self.combinations = [map(categoricals.index, comb) for comb in combinations]

    def fit_transform(self, ds):
        train_cat = ds['categorical']
        train_target = pd.Series(np.log(ds['prediction'] + 100))
        train_res = np.zeros((train_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        self.global_target_mean = train_target.mean()
        self.global_target_std = train_target.std() if self.noise_std is None else self.noise_std

        self.target_sums = {}
        self.target_cnts = {}

        for col in range(len(categoricals)):
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(train_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            train_res[:, col] = self.fit_transform_column(col, train_target,
                                                          pd.Series(map(''.join, train_cat[:, comb])))

        return train_res

    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros((test_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        for col in range(len(categoricals)):
            test_res[:, col] = self.transform_column(col, pd.Series(test_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            test_res[:, col] = self.transform_column(col, pd.Series(map(''.join, test_cat[:, comb])))

        return test_res

    def fit_transform_column(self, col, train_target, train_series):
        self.target_sums[col] = train_target.groupby(train_series).sum()
        self.target_cnts[col] = train_target.groupby(train_series).count()

        if self.noisy:
            train_res_reg = self.random_state.normal(
                loc=self.global_target_mean * self.C,
                scale=self.global_target_std * np.sqrt(self.C),
                size=len(train_series)
            )
        else:
            train_res_reg = self.global_target_mean * self.C

        train_res_num = train_series.map(self.target_sums[col]) + train_res_reg
        train_res_den = train_series.map(self.target_cnts[col]) + self.C

        if self.loo:  # Leave-one-out mode, exclude current observation
            train_res_num -= train_target
            train_res_den -= 1

        return np.exp(train_res_num / train_res_den).values

    def transform_column(self, col, test_series):
        test_res_num = test_series.map(self.target_sums[col]).fillna(0.0) + self.global_target_mean * self.C
        test_res_den = test_series.map(self.target_cnts[col]).fillna(0.0) + self.C

        return np.exp(test_res_num / test_res_den).values

    def get_feature_names(self):
        return categoricals + ['_'.join(categoricals[c] for c in comb) for comb in self.combinations]


class BaseAlgo(object):
    def fit_predict(self, train, val=None, test=None, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if val is None:
            return self.predict(test[0])
        else:
            return self.predict(val[0]), self.predict(test[0])


class Xgb(BaseAlgo):
    default_params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'nthread': -1,
    }

    def __init__(self, params, n_iter=400, huber=None, fair=None, fair_decay=0):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        self.huber = huber
        self.fair = fair
        self.fair_decay = fair_decay

        if self.huber is not None:
            self.objective = self.huber_approx_obj
        elif self.fair is not None:
            self.objective = self.fair_obj
        else:
            self.objective = None

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None,
            size_mult=None, name=None):
        feval = lambda y_pred, y_true: ('mae', eval_func(y_true.get_label(), y_pred))

        params = self.params.copy()
        params['seed'] = seed
        params['base_score'] = np.median(y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

        if X_eval is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_names)
            watchlist = [(deval, 'eval'), (dtrain, 'train')]

        if size_mult is None:
            n_iter = self.n_iter
        else:
            n_iter = int(self.n_iter * size_mult)

        self.iter = 0
        self.model = xgb.train(params, dtrain, n_iter, watchlist, self.objective, feval, verbose_eval=20)
        self.model.dump_model('xgb-%s.dump' % name, with_stats=True)
        self.feature_names = feature_names

        print("    Feature importances: %s" % ', '.join(
            '%s: %d' % t for t in sorted(self.model.get_fscore().items(), key=lambda t: -t[1])))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func=None, seed=42):
        feval = lambda y_pred, y_true: ('mae', eval_func(y_true.get_label(), y_pred))

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        def fun(**kw):
            params = self.params.copy()
            params['seed'] = seed
            params['base_score'] = np.median(y_train)

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print("Trying %s..." % str(params))

            self.iter = 0

            model = xgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], self.objective, feval,
                              verbose_eval=20, early_stopping_rounds=100)

            print("Score %.5f at iteration %d" % (model.best_score, model.best_iteration))

            return - model.best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))

    def huber_approx_obj(self, preds, dtrain):
        d = preds - dtrain.get_label()
        h = self.huber

        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)

        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt

        return grad, hess

    def fair_obj(self, preds, dtrain):
        x = preds - dtrain.get_label()
        c = self.fair

        den = np.abs(x) * np.exp(self.fair_decay * self.iter) + c

        grad = c * x / den
        hess = c * c / den ** 2

        self.iter += 1

        return grad, hess


class LightGBM(BaseAlgo):
    default_params = {
        'exec_path': '/home/mark/LightGBM/lightgbm',
        'num_threads': 4
    }

    def __init__(self, params):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        params = self.params.copy()
        params['bagging_seed'] = seed
        params['feature_fraction_seed'] = seed + 3

        self.model = GBMRegressor(**params)

        if X_eval is None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, test_data=[(X_eval, y_eval)])

    def predict(self, X):
        return self.model.predict(X)


class LibFM(BaseAlgo):
    default_params = {
    }

    def __init__(self, params={}):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.exec_path = 'libFM'
        self.tmp_dir = "libfm_models/{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M'))

    def __del__(self):
        # if os.path.exists(self.tmp_dir):
        #    rmtree(self.tmp_dir)
        pass

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        train_file = os.path.join(self.tmp_dir, 'train.svm')
        eval_file = os.path.join(self.tmp_dir, 'eval.svm')
        out_file = os.path.join(self.tmp_dir, 'out.txt')

        print("Exporting train...")
        with open(train_file, 'wb') as f:
            dump_svmlight_file(*shuffle(X_train, y_train, random_state=seed), f=f)

        if X_eval is None:
            eval_file = train_file
        else:
            print("Exporting eval...")
            with open(eval_file, 'wb') as f:
                dump_svmlight_file(X_eval, y_eval, f=f)

        params = self.params.copy()
        params['seed'] = seed
        params['task'] = 'r'
        params['train'] = train_file
        params['test'] = eval_file
        params['out'] = out_file
        params['save_model'] = os.path.join(self.tmp_dir, 'model.libfm')
        params = " ".join("-{} {}".format(k, params[k]) for k in params)

        command = "{} {}".format(self.exec_path, params)

        print(command)
        os.system(command)

    def predict(self, X):
        train_file = os.path.join(self.tmp_dir, 'train.svm')
        pred_file = os.path.join(self.tmp_dir, 'pred.svm')
        out_file = os.path.join(self.tmp_dir, 'out.txt')

        print("Exporting pred...")
        with open(pred_file, 'wb') as f:
            dump_svmlight_file(X, np.zeros(X.shape[0]), f=f)

        params = self.params.copy()
        params['iter'] = 0
        params['task'] = 'r'
        params['train'] = train_file
        params['test'] = pred_file
        params['out'] = out_file
        params['load_model'] = os.path.join(self.tmp_dir, 'model.libfm')
        params = " ".join("-{} {}".format(k, params[k]) for k in params)

        command = "{} {}".format(self.exec_path, params)

        print(command)
        os.system(command)

        return pd.read_csv(out_file, header=None).values.flatten()


class Sklearn(BaseAlgo):
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        self.model.fit(X_train, y_train)

        if X_eval is not None and hasattr(self.model, 'staged_predict'):
            for i, p_eval in enumerate(self.model.staged_predict(X_eval)):
                print("Iter %d score: %.5f" % (i, eval_func(y_eval, p_eval)))

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func, seed=42):
        def fun(**params):
            for k in params:
                if type(param_grid[k][0]) is int:
                    params[k] = int(params[k])

            print("Trying %s..." % str(params))

            self.model.set_params(**params)
            self.fit(X_train, y_train)

            if hasattr(self.model, 'staged_predict'):
                best_score = 1e9
                best_i = -1
                for i, p_eval in enumerate(self.model.staged_predict(X_eval)):
                    mae = eval_func(y_eval, p_eval)

                    if mae < best_score:
                        best_score = mae
                        best_i = i

                print("Best score after %d iters: %.5f" % (best_i, best_score))
            else:
                p_eval = self.predict(X_eval)
                best_score = eval_func(y_eval, p_eval)

                print("Score: %.5f" % best_score)

            return -best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))


class QuantileRegression(object):
    def fit_predict(self, train, val=None, test=None, **kwa):
        model = QuantReg(train[1], train[0]).fit(q=0.5, max_iter=10000)

        if val is None:
            return model.predict(test[0])
        else:
            return model.predict(val[0]), model.predict(test[0])


class Keras(BaseAlgo):
    def __init__(self, arch, params, scale=True, loss='mae', checkpoint=False):
        self.arch = arch
        self.params = params
        self.scale = scale
        self.loss = loss
        self.checkpoint = checkpoint

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        params = self.params

        if callable(params):
            params = params()

        np.random.seed(seed * 11 + 137)

        if self.scale:
            self.scaler = StandardScaler(with_mean=False)

            X_train = self.scaler.fit_transform(X_train)

            if X_eval is not None:
                X_eval = self.scaler.transform(X_eval)

        checkpoint_path = "/tmp/nn-weights-%d.h5" % seed

        self.model = self.arch((X_train.shape[1],), params)
        self.model.compile(optimizer=params.get('optimizer', 'adadelta'), loss=self.loss)

        callbacks = list(params.get('callbacks', []))

        if self.checkpoint:
            callbacks.append(ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0))

        self.model.fit_generator(
            generator=batch_generator(X_train, y_train, params['batch_size'], True), steps_per_epoch=X_train.shape[0],
            validation_data=batch_generator(X_eval, y_eval, 800) if X_eval is not None else None,
            validation_steps=X_eval.shape[0] if X_eval is not None else None,
            epochs=params['n_epoch'], verbose=1, callbacks=callbacks)

        if self.checkpoint and os.path.isfile(checkpoint_path):
            self.model.load_weights(checkpoint_path)

    def predict(self, X):
        if self.scale:
            X = self.scaler.transform(X)

            print(X.shape[0])
            print(X.shape[1])
            print(len(self.model.predict_generator(generator=batch_generator(X, batch_size=800), steps=X.shape[0])))

        return self.model.predict_generator(generator=batch_generator(X, batch_size=800), steps=X.shape[0]).reshape(
            (X.shape[0],))


def regularizer(params):
    if 'l1' in params and 'l2' in params:
        return regularizers.l1_l2(params['l1'], params['l2'])
    elif 'l1' in params:
        return regularizers.l1(params['l1'])
    elif 'l2' in params:
        return regularizers.l2(params['l2'])
    else:
        return None


def nn_lr(input_shape, params):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))

    return model


def nn_mlp(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(units=layer_size, kernel_initializer='he_normal', kernel_regularizer=reg,
                            input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

        model.add(PReLU())

    model.add(Dense(units=1, kernel_initializer='he_normal'))

    return model


def nn_mlp_2(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        model.add(PReLU())

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

    model.add(Dense(1, init='he_normal'))

    return model


def load_x(ds, preset):
    feature_parts = [Dataset.load_part(ds, part) for part in preset.get('features', [])]
    prediction_parts = [load_prediction(ds, p, mode=preset.get('predictions_mode', 'fulltrain')) for p in
                        preset.get('predictions', [])]
    prediction_parts = [p.clip(lower=0.1).values.reshape((p.shape[0], 1)) for p in prediction_parts]

    if 'prediction_transform' in preset:
        prediction_parts = map(preset['prediction_transform'], prediction_parts)

    return hstack(feature_parts + prediction_parts)


def extract_feature_names(preset):
    x = []

    for part in preset.get('features', []):
        x += Dataset.get_part_features(part)

    lp = 1
    for pred in preset.get('predictions', []):
        if type(pred) is list:
            x.append('pred_%d' % lp)
            lp += 1
        else:
            x.append(pred)

    return x


def add_powers(x, feature_names, powers):
    res_feature_names = list(feature_names)
    res = [x]

    for p in powers:
        res.append(x ** p)

        for f in feature_names:
            res_feature_names.append("%s^%s" % (f, str(p)))

    return hstack(res), res_feature_names


norm_y_lambda = 0.7


def norm_y(y):
    return boxcox(np.log1p(y), lmbda=norm_y_lambda)


def norm_y_inv(y_bc):
    return np.expm1((y_bc * norm_y_lambda + 1) ** (1 / norm_y_lambda))


y_norm = (norm_y, norm_y_inv)
y_log = (np.log, np.exp)


def y_log_ofs(ofs):
    def transform(y):
        return np.log(y + ofs)

    def inv_transform(yl):
        return np.clip(np.exp(yl) - ofs, 1.0, np.inf)

    return transform, inv_transform


def y_pow(p):
    def transform(y):
        return y ** p

    def inv_transform(y):
        return y ** (1 / p)

    return transform, inv_transform


def y_pow_ofs(p, ofs):
    def transform(y):
        return (y + ofs) ** p

    def inv_transform(y):
        return np.clip(y ** (1 / p) - ofs, 1.0, np.inf)

    return transform, inv_transform


## Main part


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')
parser.add_argument('--optimize', action='store_true', help='optimize model params')
parser.add_argument('--fold', type=int, help='specify fold')
parser.add_argument('--threads', type=int, default=4, help='specify thread count')

args = parser.parse_args()

Xgb.default_params['nthread'] = args.threads
LightGBM.default_params['num_threads'] = args.threads

n_folds = 8

l1_predictions = [
    '20171122-2025-lr-cd-1.08107',
    '20171122-2025-lr-ce-1.43686',
    '20171122-2027-lr-cd-2-1.08149',
    '20171122-2028-lr-cd-nr-1.07609',
    '20171122-2029-ab-ce-1.89150',
    '20171122-2030-et-ce-1.05009',
    '20171122-2032-et-ce-2-1.05009',
    '20171122-2035-et-ce-3-0.92469',
    '20171122-2035-rf-ce-0.87731',
    '20171122-2035-rf-ce-2-0.87493',
    '20171122-2037-gb-ce-0.86759',
    '20171122-2039-lgb-ce-0.94181',
    '20171122-2040-lgb-tst-0.94686',
    '20171122-2119-lgb-cd-2-0.76446',
    '20171122-2125-lgb-cd-tst-0.80870',
    '20171122-2126-xgb-tst-0.76615',
    '20171122-2128-xgb2-0.78846',
    '20171122-2129-xgb3-0.78809',
    '20171122-2130-xgb4-0.77659',
    '20171122-2132-xgb-ce-0.96801',
    '20171122-2134-xgb-ce-2-0.86578',
]

l2_predictions = [
    '20171122-2313-l2-xgbf-3-0.79776',
    '20171122-2300-l2-xgbf-0.74259',
    '20171122-2308-l2-xgbf-2-0.81875',
    '20171122-2317-l2-xgbf-4-0.81620',
]

presets = {
    'xgb-tst': {
        'features': ['numeric'],
        'model': Xgb({'max_depth': 5, 'eta': 0.5}, n_iter=1000),
        'param_grid': {'colsample_bytree': [0.2, 1.0], 'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}
    },

    'xgb2': {
        'features': ['numeric', 'categorical_counts'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.5,
            'subsample': 0.95,
        }, n_iter=1000),
        'param_grid': {'colsample_bytree': [0.2, 1.0], 'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgb3': {
        'features': ['numeric', 'categorical_counts'],
        'y_transform': y_log,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
        }, n_iter=1000),
        'param_grid': {'colsample_bytree': [0.2, 1.0], 'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'subsample': 0.95,
            'min_child_weight': 2,
            'gamma': 0.2,
        }, n_iter=1000),
        'param_grid': {'colsample_bytree': [0.2, 1.0], 'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},

    },

    'xgb-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
            'gamma': 0.2,
        }, n_iter=1000),
    },

    'xgb4': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
        }, n_iter=1000),
    },

    'xgb6': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=1000),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-clrbf-1': {
        'features': ['numeric', 'categorical_encoded', 'cluster_rbf_200'],
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
        }, n_iter=1000, fair=150),
    },

    'xgbf-ce-clrbf-2': {
        'features': ['numeric', 'categorical_encoded', 'cluster_rbf_50'],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
            'lambda': 2.1
        }, n_iter=1000, fair=150),
    },

    'lgb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'n_bags': 1,
        'model': LightGBM({
            'num_iterations': 1000,
            'learning_rate': 0.01,
            'num_leaves': 50,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.2,
            'bagging_fraction': 0.3,
            'bagging_freq': 20,
            'metric_freq': 10,
            'metric': 'l1',
        }),
    },

    'lgb-tst': {
        'features': ['numeric'],
        'y_transform': y_norm,
        'n_bags': 1,
        'model': LightGBM({
            'num_iterations': 1000,
            'learning_rate': 0.01,
            'num_leaves': 50,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.2,
            'bagging_fraction': 0.3,
            'bagging_freq': 20,
            'metric_freq': 10,
            'metric': 'l1',
        }),
    },

    'lgb-cd-1': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_norm,
        'model': LightGBM({
            'num_iterations': 1000,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10
        }),
    },

    'lgb-cd-2': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_log,
        'model': LightGBM({
            'num_iterations': 1000,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10
        }),
    },

    'lgb-cd-tst': {
        'features': ['numeric', 'categorical_dummy'],
        'model': LightGBM({
            'num_iterations': 1000,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10,
            'metric': 'l1',
        }),
    },

    'gb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=300, max_depth=7, max_features=0.2)),
        'param_grid': {'n_estimators': (500, 1000), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)},
    },

    'gb-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=300, max_depth=7, max_features=0.2)),
        'param_grid': {'n_estimators': (500, 1000), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)},
    },

    'gb-cd': {
        'features': ['numeric', 'categorical_dummy'],
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=300, max_depth=7, max_features=0.2)),
        'param_grid': {'n_estimators': (500, 1000), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)},
    },

    'gb-cd-2': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_log,
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=300, max_depth=7, max_features=0.2)),
        'param_grid': {'n_estimators': (500, 1000), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)},
    },

    'ab-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(AdaBoostRegressor(loss='linear', n_estimators=300)),
        'param_grid': {'n_estimators': (500, 1000), 'learning_rate': (0.1, 1.0)},
    },

    'ab-cd': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_log,
        'model': Sklearn(AdaBoostRegressor(loss='linear', n_estimators=300)),
        'param_grid': {'n_estimators': (500, 1000), 'learning_rate': (0.1, 1.0)},
    },

    'et-cd': {
        'features': ['numeric', 'categorical_dummy'],
        'model': Sklearn(ExtraTreesRegressor(1000, max_features=0.2, n_jobs=-1)),
    },

    'et-cd-2': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_log,
        'model': Sklearn(ExtraTreesRegressor(1000, max_features=0.2, n_jobs=-1)),
    },

    'et-cd-3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(ExtraTreesRegressor(1000, max_features=0.8, min_samples_split=26, max_depth=23, n_jobs=-1)),
    },

    'et-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(1000, max_features=0.2, n_jobs=-1)),
    },

    'et-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(ExtraTreesRegressor(1000, max_features=0.2, n_jobs=-1)),
    },

    'et-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(ExtraTreesRegressor(1000, max_features=0.8, min_samples_split=26, max_depth=23, n_jobs=-1)),
    },

    'rf-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(RandomForestRegressor(1000, min_samples_split=16, max_features=0.3, max_depth=26, n_jobs=-1)),
    },

    'rf-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'model': Sklearn(RandomForestRegressor(1000, min_samples_split=16, max_features=0.3, max_depth=26, n_jobs=-1)),
    },

    'rf-cd': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_log,
        'model': Sklearn(RandomForestRegressor(1000, min_samples_split=16, max_features=0.3, max_depth=26, n_jobs=-1)),
    },

    'rf-cd-2': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_norm,
        'model': Sklearn(RandomForestRegressor(1000, min_samples_split=16, max_features=0.3, max_depth=26, n_jobs=-1)),
    },

    'lr-tst': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'model': Sklearn(Ridge(1e-3)),
        'param_grid': {'C': (1e-3, 1e3)},
    },

    'lr-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'y_transform': y_log,
        'model': Sklearn(Ridge(1e-3)),
        'param_grid': {'C': (1e-3, 1e3)},
    },

    'lr-cd-2': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'y_transform': y_norm,
        'model': Sklearn(Ridge(1e-3)),
        'param_grid': {'C': (1e-3, 1e3)},
    },

    'lr-cd-nr': {
        'features': ['numeric_rank_norm', 'categorical_dummy'],
        'y_transform': y_log,
        'model': Sklearn(Ridge(1e-3)),
    },

    'lr-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(Pipeline([('sc', StandardScaler(with_mean=False)), ('lr', Ridge(1e-3))])),
    },

    'l2-xgbf': {
        'predictions': l1_predictions,
        'model': Xgb({
            'max_depth': 4,
            'eta': 0.5,
            'colsample_bytree': 0.4,
            'subsample': 0.75,
            'min_child_weight': 6,
        }, n_iter=1000, fair=1.0),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0),
                       'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-2': {
        'predictions': l1_predictions,
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.4,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 1.0,
        }, n_iter=1000, fair=1.0),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0),
                       'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-3': {
        'predictions': l1_predictions,
        'y_transform': y_log,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.4,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 0.5,
            'alpha': 0.4,
        }, n_iter=1000, fair=1.0),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0),
                       'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-4': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.5,
            'colsample_bytree': 0.3,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 1.5,
            'alpha': 1.3,
        }, n_iter=1000, fair=150),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0),
                       'subsample': (0.5, 1.0)},
    },

    'l3-xgbf': {
        'predictions': l2_predictions,
        'y_transform': y_log,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.005,
            'colsample_bytree': 0.3,
            'subsample': 0.55,
            'min_child_weight': 3,
        }, n_iter=1000, fair=50),
    },

}

print("Preset: %s" % args.preset)

preset = presets[args.preset]

feature_builders = preset.get('feature_builders', [])

n_bags = preset.get('n_bags', 1)
n_splits = preset.get('n_splits', 1)

y_aggregator = preset.get('agg', np.mean)
y_transform, y_inv_transform = preset.get('y_transform', (lambda y: y, lambda y: y))

print("Loading train data...")
train_x = load_x('train', preset)
train_y = Dataset.load_part('train', 'prediction')
train_p = np.zeros((train_x.shape[0], n_splits * n_bags))
train_r = Dataset.load('train', parts=np.unique(sum([b.requirements for b in feature_builders], ['prediction'])))

feature_names = extract_feature_names(preset)

if args.optimize:
    opt_train_idx, opt_eval_idx = train_test_split(range(len(train_y)), test_size=0.2)

    opt_train_x = train_x[opt_train_idx]
    opt_train_y = train_y[opt_train_idx]
    opt_train_r = train_r.slice(opt_train_idx)

    opt_eval_x = train_x[opt_eval_idx]
    opt_eval_y = train_y[opt_eval_idx]
    opt_eval_r = train_r.slice(opt_eval_idx)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print("    Building per-fold features...")

        opt_train_x = [opt_train_x]
        opt_eval_x = [opt_eval_x]

        for fb in feature_builders:
            opt_train_x.append(fb.fit_transform(opt_train_r))
            opt_eval_x.append(fb.transform(opt_eval_r))

        opt_train_x = hstack(opt_train_x)
        opt_eval_x = hstack(opt_eval_x)

    preset['model'].optimize(opt_train_x, y_transform(opt_train_y), opt_eval_x, y_transform(opt_eval_y),
                             preset['param_grid'],
                             eval_func=lambda yt, yp: mean_absolute_error(y_inv_transform(yt), y_inv_transform(yp)))

print("Loading test data...")
test_x = load_x('test', preset)
test_r = Dataset.load('test', parts=np.unique([b.requirements for b in feature_builders]))
test_foldavg_p = np.zeros((test_x.shape[0], n_splits * n_bags * n_folds))
test_fulltrain_p = np.zeros((test_x.shape[0], n_bags))

if 'powers' in preset:
    print("Adding power features...")

    train_x, feature_names = add_powers(train_x, feature_names, preset['powers'])
    test_x = add_powers(test_x, feature_names, preset['powers'])[0]

maes = []

for split in range(n_splits):
    print('\n')
    print("Training split %d..." % split)

    # for fold, (fold_train_idx, fold_eval_idx) in enumerate(
    #         KFold(len(train_y), n_folds, shuffle=True, random_state=2016 + 17 * split)):
    #     if args.fold is not None and fold != args.fold:
    #         continue

    for fold, (fold_train_idx, fold_eval_idx) in enumerate(
            KFold(n_splits=n_folds, shuffle=True, random_state=2016 + 17 * split).split(X=train_x, y=train_y)):
        if args.fold is not None and fold != args.fold:
            continue

        print('\n')
        print("  Fold %d..." % fold)

        fold_train_x = train_x[fold_train_idx]
        fold_train_y = train_y[fold_train_idx]
        fold_train_r = train_r.slice(fold_train_idx)

        fold_eval_x = train_x[fold_eval_idx]
        fold_eval_y = train_y[fold_eval_idx]
        fold_eval_r = train_r.slice(fold_eval_idx)

        fold_test_x = test_x
        fold_test_r = test_r

        fold_feature_names = list(feature_names)

        if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
            print("    Building per-fold features...")

            fold_train_x = [fold_train_x]
            fold_eval_x = [fold_eval_x]
            fold_test_x = [fold_test_x]

            for fb in feature_builders:
                fold_train_x.append(fb.fit_transform(fold_train_r))
                fold_eval_x.append(fb.transform(fold_eval_r))
                fold_test_x.append(fb.transform(fold_test_r))
                fold_feature_names += fb.get_feature_names()

            fold_train_x = hstack(fold_train_x)
            fold_eval_x = hstack(fold_eval_x)
            fold_test_x = hstack(fold_test_x)

        eval_p = np.zeros((fold_eval_x.shape[0], n_bags))

        for bag in range(n_bags):
            print("    Training model %d..." % bag)

            rs = np.random.RandomState(101 + 31 * split + 13 * fold + 29 * bag)

            bag_train_x = fold_train_x
            bag_train_y = fold_train_y

            bag_eval_x = fold_eval_x
            bag_eval_y = fold_eval_y

            bag_test_x = fold_test_x

            if 'sample' in preset:
                bag_train_x, bag_train_y = resample(fold_train_x, fold_train_y, replace=False,
                                                    n_samples=int(preset['sample'] * fold_train_x.shape[0]),
                                                    random_state=42 + 11 * split + 13 * fold + 17 * bag)

            if 'feature_sample' in preset:
                features = rs.choice(range(bag_train_x.shape[1]), int(bag_train_x.shape[1] * preset['feature_sample']),
                                     replace=False)

                bag_train_x = bag_train_x[:, features]
                bag_eval_x = bag_eval_x[:, features]
                bag_test_x = bag_test_x[:, features]

            if 'svd' in preset:
                svd = TruncatedSVD(preset['svd'])

                bag_train_x = svd.fit_transform(bag_train_x)
                bag_eval_x = svd.transform(bag_eval_x)
                bag_test_x = svd.transform(bag_test_x)

            pe, pt = preset['model'].fit_predict(train=(bag_train_x, y_transform(bag_train_y)),
                                                 val=(bag_eval_x, y_transform(bag_eval_y)),
                                                 test=(bag_test_x,),
                                                 seed=42 + 11 * split + 17 * fold + 13 * bag,
                                                 feature_names=fold_feature_names,
                                                 eval_func=lambda yt, yp: mean_absolute_error(y_inv_transform(yt),
                                                                                              y_inv_transform(yp)),
                                                 name='%s-fold-%d-%d' % (args.preset, fold, bag))

            eval_p[:, bag] += pe
            test_foldavg_p[:, split * n_folds * n_bags + fold * n_bags + bag] = pt

            train_p[fold_eval_idx, split * n_bags + bag] = pe

            print("    MAE of model: %.5f" % mean_absolute_error(fold_eval_y, y_inv_transform(pe)))

        print("  MAE of mean-transform: %.5f" % mean_absolute_error(fold_eval_y,
                                                                    y_inv_transform(np.mean(eval_p, axis=1))))
        print("  MAE of transform-mean: %.5f" % mean_absolute_error(fold_eval_y,
                                                                    np.mean(y_inv_transform(eval_p), axis=1)))
        print("  MAE of transform-median: %.5f" % mean_absolute_error(fold_eval_y,
                                                                      np.median(y_inv_transform(eval_p), axis=1)))

        # Calculate err
        maes.append(mean_absolute_error(fold_eval_y, y_aggregator(y_inv_transform(eval_p), axis=1)))

        print("  MAE: %.5f" % maes[-1])

        # Free mem
        del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y

if True:
    print('\n')
    print("  Full...")

    full_train_x = train_x
    full_train_y = train_y
    full_train_r = train_r

    full_test_x = test_x
    full_test_r = test_r

    full_feature_names = list(feature_names)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print("    Building per-fold features...")

        full_train_x = [full_train_x]
        full_test_x = [full_test_x]

        for fb in feature_builders:
            full_train_x.append(fb.fit_transform(full_train_r))
            full_test_x.append(fb.transform(full_test_r))
            full_feature_names += fb.get_feature_names()

        full_train_x = hstack(full_train_x)
        full_test_x = hstack(full_test_x)

    for bag in range(n_bags):
        print("    Training model %d..." % bag)

        rs = np.random.RandomState(101 + 31 * split + 13 * fold + 29 * bag)

        bag_train_x = full_train_x
        bag_train_y = full_train_y

        bag_test_x = full_test_x

        if 'sample' in preset:
            bag_train_x, bag_train_y = resample(bag_train_x, bag_train_y, replace=False,
                                                n_samples=int(preset['sample'] * bag_train_x.shape[0]),
                                                random_state=42 + 11 * split + 13 * fold + 17 * bag)

        if 'feature_sample' in preset:
            features = rs.choice(range(bag_train_x.shape[1]), int(bag_train_x.shape[1] * preset['feature_sample']),
                                 replace=False)

            bag_train_x = bag_train_x[:, features]
            bag_test_x = bag_test_x[:, features]

        if 'svd' in preset:
            svd = TruncatedSVD(preset['svd'])

            bag_train_x = svd.fit_transform(bag_train_x)
            bag_test_x = svd.transform(bag_test_x)

        pt = preset['model'].fit_predict(train=(bag_train_x, y_transform(bag_train_y)),
                                         test=(bag_test_x,),
                                         seed=42 + 11 * split + 17 * fold + 13 * bag,
                                         feature_names=fold_feature_names,
                                         eval_func=lambda yt, yp: mean_absolute_error(y_inv_transform(yt),
                                                                                      y_inv_transform(yp)),
                                         size_mult=n_folds / (n_folds - 1.0),
                                         name='%s-full-%d' % (args.preset, bag))

        test_fulltrain_p[:, bag] = pt

# Aggregate predictions
train_p = pd.Series(y_aggregator(y_inv_transform(train_p), axis=1), index=Dataset.load_part('train', 'record_id'))
test_foldavg_p = pd.Series(y_aggregator(y_inv_transform(test_foldavg_p), axis=1),
                           index=Dataset.load_part('test', 'record_id'))
test_fulltrain_p = pd.Series(y_aggregator(y_inv_transform(test_fulltrain_p), axis=1),
                             index=Dataset.load_part('test', 'record_id'))

# Analyze predictions
mae_mean = np.mean(maes)
mae_std = np.std(maes)
mae = mean_absolute_error(train_y, train_p)

print('\n')
print("CV MAE: %.5f +- %.5f" % (mae_mean, mae_std))
print("CV RES MAE: %.5f" % mae)

name = "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), args.preset, mae)

print('\n')
print("Saving predictions... (%s)" % name)

for part, pred in [('train', train_p), ('test-foldavg', test_foldavg_p), ('test-fulltrain', test_fulltrain_p)]:
    pred.rename('prediction', inplace=True)
    pred.index.rename('record_id', inplace=True)
    pred.to_csv('preds/%s-%s.csv' % (name, part), header=True)

copy2(os.path.realpath(__file__), os.path.join("preds", "%s-code.py" % name))

print("Done.")
