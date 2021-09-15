# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


class FitNonlinearLeastSquares(object):

    def fit_func(self, data_x, data_y):

        """
        这里 p0 存放的是k、b的初始值，这个值会随着拟合的进行不断变化，使得误差
        error的值越来越小
        """
        p0 = np.random.rand(2)

        fit_res = optimize.least_squares(self.error, p0,
                                         args=(data_x, data_y),
                                         bounds=((0, 0), (1000, np.inf)))
        fit_res = dict(fit_res)
        print(fit_res)
        return fit_res

    @staticmethod
    def func(p, x):
        """
        Define the form of fitting function.
        """
        k, b = p
        return k * x + b

    def error(self, p, x, y):
        """
        Fitting residuals.
        """
        return y - self.func(p, x)

    def mapping(self, fit_res, data_x, data_y):
        y_fitted = self.func(fit_res['x'], data_x)
        plt.plot(data_x, data_y, 'o', label='Data')
        plt.plot(data_x, y_fitted, label='Fitted curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig('test_fit.jpg')
        # plt.show()
        plt.pause(0.5)
        plt.close()

    def main(self, data_x, data_y):
        fit_res = self.fit_func(data_x, data_y)
        self.mapping(fit_res, data_x, data_y)
        return fit_res['x']


if __name__ == '__main__':
    x = np.asarray(
        [0, 378.98261532, 712.54653485, 1041.63125103, 1535.36185542,
         3201.85772693])
    y = np.asarray([0.02806867, 0.10259962, 0.2211375, 0.34793035, 0.55847596,
                    1.19682617])
    fit_gamma2_dfdphi = FitNonlinearLeastSquares()
    k, b = fit_gamma2_dfdphi.main(x, y)
    print(k, b)
