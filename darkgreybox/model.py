import numpy as np
from abc import ABC
from lmfit import minimize, Parameters


#TODO: allow different start conditions in predict


class DarkGreyModelError(ValueError): pass
class ModelNotFitError(DarkGreyModelError): pass


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
    
    def __init__(self, params, rec_duration):
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
        
        self.params = params
        self.result = None
        
        # convert the params dict into lmfit parameters
        if isinstance(params, Parameters):
            self.params = params
        else:
            self.params = Parameters()
            for k, v in params.items():
                self.params.add(k, **v)

        # set the number of records based on the measured variable's values
        self.rec_duration = rec_duration
            
    def fit(self, X, y, method, obj_func=None, refit=False):
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
        obj_func : function
            The objective function that is passed to `lmfit.minimize`/
            It must have (params, *args, **kwargs) as its method signature.
            Default: `def_obj_func`
        refit : boolean
            Whether to refit using the last results parameters.
            Use params passed in the constructor if False
            Use result.params if True (refit)

        Returns
        -------
        `lmfit.MinimizerResult`
            Object containing the optimized parameters and several
            goodness-of-fit statistics.
        '''    

        if self.result is None and refit:
            raise ModelNotFitError(
                "Model has no result to be used with refit. Call fit with refit=False"
                " at least once before a refit")
        
        # we are passing X, y to minimise as kwargs 
        self.result = minimize(obj_func or self.def_obj_func, 
                               self.result.params if refit else self.params, 
                               kws={'model': self.model, 'X': X, 'y': y}, 
                               method=method)

        return self

    def predict(self, X):
        '''
        Generates a prediction based on the result parameters and X.

        Parameters
        ----------
        X : dict
            A dictionary of input values

        Returns
        -------
        The results of the model
        '''

        if self.result is None:
            raise ModelNotFitError(
                "Model has no result to be used with predict. Call fit with refit=False"
                " at least once before predict")

        return self.model(self.result.params, X)
    
    def model(self, params, X):
        '''
        A system of differential equations describing the thermal model
        '''

        pass

    @staticmethod
    def def_obj_func(params, *args, **kwargs):
        '''
        Default objective function
        Computes the residual between measured data and fitted data
        The model, X and y are passed in as kwargs by `lmfit.minimize`
        '''
        return ((kwargs['model'](params=params, X=kwargs['X']).Z - kwargs['y'])).ravel()        
        

class TiTeThRia(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTeThRia RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings. 
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of TiTeThRia.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
        'Th': df['Heating Circuit Temperature [˚C]'].values
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Te0': {'value': y[0] - 2, 'vary': True, 'min': 10, 'max': 25},
        'Th0': {'value': y[0], 'vary': False, 'min': 10, 'max': 80},
        'Ci': {'value': 132},
        'Ce': {'value': 600},
        'Ch': {'value': 2.55, 'vary': False},
        'Rie': {'value': 0.1},
        'Rea': {'value': 1},
        'Ria': {'value': 2},
        'Rih': {'value': 0.65, 'vary': False}
    }

    # fit using the Nelder-Mead method
    result = TiTeTh(y, X, params, method='nelder').fit()
    ~~~~
    '''   

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Te0' : Thermal envelope temperature at t(0)
            - 'Th0' : Heating system temperature at t(0)  
            - 'Rih' : Thermal resistance between internal and heating system  
            - 'Rie' : Thermal resistance between internal and thermal envelope
            - 'Rea' : Thermal resistance between thermal envelope and ambient
            - 'Ria' : Thermal resistance between internal and ambient
            - 'Ci' : Thermal capacitance of internal
            - 'Ch' : Thermal capacitance of heating system
            - 'Ce' : Thermal capacitance of thermal envelope
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values  
        Te : np.array
            Fitted thermal envelope temperature values         
        Th : np.array
            Fitted heating system temperature values
        '''       

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Te = np.zeros(num_rec)
        Th = np.zeros(num_rec)
       
        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Te[0] = params['Te0']
        Th[0] = params['Th0']

        Rie = params['Rie'].value
        Rea = params['Rea'].value
        Rih = params['Rih'].value
        Ria = params['Ria'].value

        Ci = params['Ci'].value
        Ce = params['Ce'].value
        Ch = params['Ch'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci) +
                   (Ta[i-1] - Ti[i-1]) / (Ria * Ci)) * self.rec_duration 
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration 
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration 

            Ti[i] = Ti[i-1] + dTi 
            Te[i] = Te[i-1] + dTe 
            Th[i] = Th[i-1] + dTh

        return DarkGreyModelResult(Ti, Ti=Ti, Te=Te, Th=Th)

    
class TiTeTh(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTeTh RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings. 
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of TiTeTh.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
        'Th': df['Heating Circuit Temperature [˚C]'].values
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Te0': {'value': y[0] - 2, 'vary': True, 'min': 10, 'max': 25},
        'Th0': {'value': y[0], 'vary': False, 'min': 10, 'max': 80},
        'Ci': {'value': 132},
        'Ce': {'value': 600},
        'Ch': {'value': 2.55, 'vary': False},
        'Rie': {'value': 0.1},
        'Rea': {'value': 1},
        'Rih': {'value': 0.65, 'vary': False}
    }

    # fit using the Nelder-Mead method
    result = TiTeTh(y, X, params, method='nelder').fit()
    ~~~~
    '''

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Te0' : Thermal envelope temperature at t(0)
            - 'Th0' : Heating system temperature at t(0)  
            - 'Rih' : Thermal resistance between internal and heating system  
            - 'Rie' : Thermal resistance between internal and thermal envelope
            - 'Rea' : Thermal resistance between thermal envelope and ambient
            - 'Ci' : Thermal capacitance of internal
            - 'Ch' : Thermal capacitance of heating system
            - 'Ce' : Thermal capacitance of thermal envelope
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values  
        Te : np.array
            Fitted thermal envelope temperature values         
        Th : np.array
            Fitted heating system temperature values
        '''            

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Te = np.zeros(num_rec)
        Th = np.zeros(num_rec)
        
        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Te[0] = params['Te0']
        Th[0] = params['Th0']

        Rie = params['Rie'].value
        Rea = params['Rea'].value
        Rih = params['Rih'].value

        Ci = params['Ci'].value
        Ce = params['Ce'].value
        Ch = params['Ch'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci)) * self.rec_duration 
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration 
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration 

            Ti[i] = Ti[i-1] + dTi 
            Te[i] = Te[i-1] + dTe 
            Th[i] = Th[i-1] + dTh

        return DarkGreyModelResult(Ti, Ti=Ti, Te=Te, Th=Th)

    
class TiTh(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTh RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings. 
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of TiTh.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
        'Th': df['Heating Circuit Temperature [˚C]'].values
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Th0': {'value': y[0], 'vary': False, 'min': 10, 'max': 80},
        'Ci': {'value': 132},
        'Ch': {'value': 2.55, 'vary': False},
        'Ria': {'value': 1},
        'Rih': {'value': 0.65, 'vary': False}
    }

    # fit using the Nelder-Mead method
    result = TiTh(y, X, params, method='nelder').fit()
    ~~~~
    '''
    
    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Th0' : Heating system temperature at t(0)  
            - 'Rih' : Thermal resistance between internal and heating system  
            - 'Ria' : Thermal resistance between internal and ambient
            - 'Ci' : Thermal capacitance of internal
            - 'Ch' : Thermal capacitance of heating system
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values  
        Th : np.array
            Fitted heating system temperature values
        '''            

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Th = np.zeros(num_rec)
        
        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Th[0] = params['Th0']

        Rih = params['Rih'].value
        Ria = params['Ria'].value

        Ci = params['Ci'].value
        Ch = params['Ch'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Ta[i-1] - Ti[i-1]) / (Ria * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci)) * self.rec_duration 
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration 

            Ti[i] = Ti[i-1] + dTi 
            Th[i] = Th[i-1] + dTh

        return DarkGreyModelResult(Ti, Ti=Ti, Th=Th)
