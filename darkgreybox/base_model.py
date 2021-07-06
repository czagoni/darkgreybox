from abc import ABC
from copy import deepcopy
from typing import Callable, Dict, Optional, Union

from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
import numpy as np

from darkgreybox import logger


class DarkGreyModelResult:
    '''
    Container object that holds the results of the model
    '''

    def __init__(self, Z, **kwargs):
        '''
        Parameters
        ----------
        Z : np.array
            The measured variable's fit / predicted values
        kwargs: kwargs
            Any other parameters calculated can be passed here.

        ~~~~
        E.g. in case of a TiTeTh model, Ti is the measured variable and Ti, Te, Th
        variables can be passed as kwargs for easy access

        DarkGreyModelResult(Ti, Ti=Ti, Te=Te, Th=Th)
        ~~~~
        '''

        self.Z = Z
        for key, val in kwargs.items():
            setattr(self, key, val)


class DarkGreyModel(ABC):
    '''
    Abstract Base Class for DarkGrey Models
    '''

    def __init__(self, params: Union[Parameters, Dict], rec_duration: float):
        '''
        Initialises the model instance

        Parameters
        ----------
        params : dict
            A dictionary of parameters for the fitting. Key - value pairs should follow the
            `lmfit.Parameters` declaration:
            e.g. {'A' : {'value': 10, 'min': 0, 'max': 30}} - sets the initial value and the bounds
            for parameter `A`
        rec_duration : float
            The duration of each measurement record in hours
        '''

        self.result = MinimizerResult()

        # convert the params dict into lmfit parameters
        if isinstance(params, Parameters):
            self.params = deepcopy(params)
        else:
            self.params = Parameters()
            for k, v in params.items():
                self.params.add(k, **v)

        # set the number of records based on the measured variable's values
        self.rec_duration = rec_duration

    def fit(
        self,
        X: Dict,
        y: np.ndarray,
        method: str,
        ic_params: Optional[Dict] = None,
        obj_func: Optional[Callable] = None
    ):
        '''
        Fits the model by minimising the objective function value

        Parameters
        ----------
        X : dict
            A dictionary of input values for the fitting - these values are fixed during the fit.
        y : np.array
            The measured variable's values for the minimiser to fit to
        method : str
            Name of the fitting method to use. Valid values are described in:
            `lmfit.minimize`
        ic_params : Optional[Dict]
            The initial condition parameters - if passed in these will overwrite
            the initial conditions in self.params
        obj_func : Optional[Callable]
            The objective function that is passed to `lmfit.minimize`/
            It must have (params, *args, **kwargs) as its method signature.
            Default: `def_obj_func`

        Returns
        -------
        `lmfit.MinimizerResult`
            Object containing the optimized parameters and several
            goodness-of-fit statistics.
        '''

        # overwrite initial conditions
        if ic_params is not None:
            for k, v in ic_params.items():
                if k in self.params:
                    self.params[k].value = v
                else:
                    logger.warning(f'Key `{k}` not found in initial conditions params')

        # we are passing X, y to minimise as kwargs
        self.result = minimize(
            obj_func or self.def_obj_func,
            self.params,
            kws={'model': self.model, 'X': X, 'y': y},
            method=method
        )

        self.params = self.result.params

        return self

    def predict(self, X: Dict, ic_params: Optional[Dict] = None) -> DarkGreyModelResult:
        '''
        Generates a prediction based on the result parameters and X.

        Parameters
        ----------
        X : dict
            A dictionary of input values
        ic_params : dict
            The initial condition parameters - if passed in these will overwrite
            the initial conditions in self.params

        Returns
        -------
        The results of the model
        '''

        if ic_params is not None:
            for k, v in ic_params.items():
                self.params[k].value = v

        return self.model(self.params, X)

    def model(self, params: Parameters, X: Dict) -> DarkGreyModelResult:
        '''
        A system of differential equations describing the thermal model
        '''
        ...  # pragma: no cover

    def lock(self):
        '''
        Locks the parameters by setting `vary` to False
        '''

        for param in self.params.keys():
            self.params[param].vary = False

        return self

    @staticmethod
    def def_obj_func(params: Parameters, *args, **kwargs):
        '''
        Default objective function
        Computes the residual between measured data and fitted data
        The model, X and y are passed in as kwargs by `lmfit.minimize`
        '''
        return ((kwargs['model'](params=params, X=kwargs['X']).Z - kwargs['y'])).ravel()
