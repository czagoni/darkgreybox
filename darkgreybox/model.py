import numpy as np
from abc import ABC
from lmfit import minimize, Parameters


class DarkGreyModel(ABC):
    '''
    Abstract Base Class for DarkGrey Models
    '''
    
    def __init__(self, y, rec_duration, inputs, params, method):
        '''
        Initialises the model instance

        Parameters
        ----------
        y : np.array
            The measured variable's values for the minimiser to fit to
        rec_duration : float
            The duration of each measurement record in hours 
        inputs : dict
            A dictionary of input values for the fitting - these values are fixed during the fit.
        params : dict
            A dictionary of parameters for the fitting. Key - value pairs should follow the 
            `lmfit.Parameters` declaration: 
            e.g. {'Y' : {'value': 10, 'min': 0, 'max': 30}} - sets the initial value and the bounds 
            for parameter `Y` 
        method : str
            Name of the fitting method to use. Valid values are described in:
            `lmfit.minimize`
        '''
        
        self.y = y
        self.inputs = inputs
        self.params = params
        self.method = method
        self.result = None
        
        # convert the params dict into lmfit parameters
        self.params = Parameters()
        for k, v in params.items():
            self.params.add(k, **v)

        # set the number of records based on the measured variable's values
        self.num_rec = len(self.y)
        self.rec_duration = rec_duration
            
    def fit(self):
        '''
        Fits the model by minimising the objective function value

        Returns
        -------
        `lmfit.MinimizerResult`
            Object containing the optimized parameters and several
            goodness-of-fit statistics.
        '''    

        # we are passing the inputs to minimise as kwargs 
        self.result = minimize(self.obj_func, self.params, kws=self.inputs, method=self.method)
        return self.result
 
    def obj_func(self, params, **inputs):
        '''
        Computes the residual between measured data and fitted data
        '''

        return ((self.model(params, **inputs)[0] - self.y)).ravel()
    
    def model(self, params, **inputs):
        '''
        A system of differential equations describing the thermal model
        '''

        pass
        
        
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
    inputs = {
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
        'Ria': {'value': 2}
        'Rih': {'value': 0.65, 'vary': False}
    }

    # fit using the Nelder-Mead method
    result = TiTeTh(y, inputs, params, method='nelder').fit()
    ~~~~
    '''   

    def model(self, params, **inputs):
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
        inputs : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values  
        Ti : np.array
            Fitted thermal envelope temperature values         
        Th : np.array
            Fitted heating system temperature values
        '''       

        Ti = np.zeros(self.num_rec)
        Te = np.zeros(self.num_rec)
        Th = np.zeros(self.num_rec)
       
        # alias these params/inputs so that the differential equations look pretty
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

        Ta = inputs['Ta']
        Ph = inputs['Ph']

        for i in range(1, self.num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci) +
                   (Ta[i-1] - Ti[i-1]) / (Ria * Ci)) * self.rec_duration 
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration 
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration 

            Ti[i] = Ti[i-1] + dTi 
            Te[i] = Te[i-1] + dTe 
            Th[i] = Th[i-1] + dTh

        return (Ti, Te, Th)

    
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
    inputs = {
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
    result = TiTeTh(y, inputs, params, method='nelder').fit()
    ~~~~
    '''

    def model(self, params, **inputs):
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
        inputs : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values  
        Ti : np.array
            Fitted thermal envelope temperature values         
        Th : np.array
            Fitted heating system temperature values
        '''            

        Ti = np.zeros(self.num_rec)
        Te = np.zeros(self.num_rec)
        Th = np.zeros(self.num_rec)
        
        # alias these params/inputs so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Te[0] = params['Te0']
        Th[0] = params['Th0']

        Rie = params['Rie'].value
        Rea = params['Rea'].value
        Rih = params['Rih'].value

        Ci = params['Ci'].value
        Ce = params['Ce'].value
        Ch = params['Ch'].value

        Ta = inputs['Ta']
        Ph = inputs['Ph']

        for i in range(1, self.num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci)) * self.rec_duration 
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration 
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration 

            Ti[i] = Ti[i-1] + dTi 
            Te[i] = Te[i-1] + dTe 
            Th[i] = Th[i-1] + dTh

        return (Ti, Te, Th)

    
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
    inputs = {
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
    result = TiTh(y, inputs, params, method='nelder').fit()
    ~~~~
    '''
    
    def model(self, params, **inputs):
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
        inputs : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values  
        Th : np.array
            Fitted heating system temperature values
        '''            

        Ti = np.zeros(self.num_rec)
        Th = np.zeros(self.num_rec)
        
        # alias these params/inputs so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Th[0] = params['Th0']

        Rih = params['Rih'].value
        Ria = params['Ria'].value

        Ci = params['Ci'].value
        Ch = params['Ch'].value

        Ta = inputs['Ta']
        Ph = inputs['Ph']

        for i in range(1, self.num_rec):

            # the model equations
            dTi = ((Ta[i-1] - Ti[i-1]) / (Ria * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci)) * self.rec_duration 
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration 

            Ti[i] = Ti[i-1] + dTi 
            Th[i] = Th[i-1] + dTh

        return (Ti, Th)
