import numpy as np

from darkgreybox.base_model import DarkGreyModel, DarkGreyModelResult


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


class TiTe(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTe RC-equivalent circuit

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
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Te0': {'value': y[0] - 2, 'vary': True, 'min': 10, 'max': 25},
        'Ci': {'value': 132},
        'Ce': {'value': 600},
        'Rie': {'value': 0.1},
        'Rea': {'value': 1},
    }

    # fit using the Nelder-Mead method
    result = TiTe(y, X, params, method='nelder').fit()
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
            - 'Rie' : Thermal resistance between internal and thermal envelope
            - 'Rea' : Thermal resistance between thermal envelope and ambient
            - 'Ci' : Thermal capacitance of internal
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
        '''

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Te = np.zeros(num_rec)

        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Te[0] = params['Te0']

        Rie = params['Rie'].value
        Rea = params['Rea'].value

        Ci = params['Ci'].value
        Ce = params['Ce'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Ph[i-1]) / (Ci)) * self.rec_duration
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi
            Te[i] = Te[i-1] + dTe

        return DarkGreyModelResult(Ti, Ti=Ti, Te=Te)


class Ti(DarkGreyModel):
    '''
    A DarkGrey Model representing a Ti RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings.
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of Ti.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Ci': {'value': 132},
        'Ria': {'value': 1},
    }

    # fit using the Nelder-Mead method
    result = Ti(y, X, params, method='nelder').fit()
    ~~~~
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ic_param_names = ['Ti0']
        self.rc_param_names = ['Ci', 'Ria']
        self.input_param_names = ['Ta', 'Ph']

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Ria' : Thermal resistance between internal and ambient
            - 'Ci' : Thermal capacitance of internal
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values
        '''

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)

        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0'].value

        Ria = params['Ria'].value
        Ci = params['Ci'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Ta[i-1] - Ti[i-1]) / (Ria * Ci) + (Ph[i-1]) / (Ci)) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi

        return DarkGreyModelResult(Ti, Ti=Ti)
