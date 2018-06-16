import numpy as np
import pymc as pm
from sklearn.preprocessing import OneHotEncoder

from convpy import functions


class ConversionModel:

    def __init__(self, feature_value_list, w_c, w_d):
        self.feature_value_list = feature_value_list
        self.w_c = w_c
        self.w_d = w_d
        self.data_ = self._generate_data()
        self.data = self._get_data()
        self.conv_prob = self._get_conv_prob()
        self.conv = self._get_conv()
        self.hazard = self._get_hazard()
        self.lag = self._get_lag()
        self.model = self._get_model()

    def _generate_data(self):
        l = self.feature_value_list

        @pm.stochastic(dtype=np.int)
        def data_(value=np.zeros(len(l)), l=l):

            def logp(value, l=l):
                return np.sum((pm.discrete_uniform_like(0, lower=0, upper=val - 1) for val in l))

            def random(l=l):
                return np.array([np.round((val - 1) * np.random.random()) for val in l])

        return data_

    def _get_data(self):
        l = self.feature_value_list
        enc = OneHotEncoder(n_values=l)
        data_ = self.data_

        @pm.deterministic
        def data(data_=data_):
            return enc.fit_transform([data_]).toarray()[0]

        return data

    def _get_conv_prob(self):
        w_c = self.w_c
        x = self.data

        @pm.deterministic
        def conv_prob(x=x, w_c=w_c):
            return functions.conv_prob(x, w_c)

        return conv_prob

    def _get_conv(self):
        conv = pm.Bernoulli('conv', p=self.conv_prob)
        return conv

    def _get_hazard(self):
        w_d = self.w_d
        x = self.data

        @pm.deterministic
        def hazard(x=x, w_d=w_d):
            return functions.hazard(x, w_d)

        return hazard

    def _get_lag(self):
        lag = pm.Exponential('lag', beta=self.hazard)
        return lag

    def _get_model(self):
        return pm.MCMC([self.w_c, self.w_d, self.data_, self.data, self.hazard, self.lag, self.conv_prob, self.conv])

    def sample(self, samples=100000, burn=1000, thin=10):
        self.model.sample(samples, burn=burn, thin=thin)

    def plot(self):
        return pm.Matplot.plot(self.model)
