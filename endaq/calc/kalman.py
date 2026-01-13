import pandas as pd
from typing import List, Optional, Iterable, Union, Callable
import numpy as np


class Kalman:
    """
    An abstract mutable class that performs a Kalman filter, implementation based on 
    https://thekalmanfilter.com/kalman-filter-explained-simply/.
    A new `Kalman` class should be instantialized for filter operations on different inputs.
    The following functions 
    need to be implemented by the methods that wish to use this class:
    :py:func:`get_A()`
    :py:func:`get_H()`
    :py:func:`get_Q()`
    :py:func:`initialize_system()`
    naming conventions : capital letters are matrices, pr_xxx is priori, and ps_xxx is postori
    """

    def __init__(self, df: pd.DataFrame):
        """
        init for the Kalman class
        :param df: the dataset to perform the Kalma operation on
        """
        self.x, self.P = self.initialize_system()
        self.R = None #TODO
        self.K = self._compute_gain()
        self.A = self.get_A()
        self.H = self.get_H()
        self.Q = self.get_Q()
        #----- in between variables -----#
        self._pr_x = None
        self._pr_P = None

    #========== Methods to Implement ==========#
    def get_A(self):
        """
        Getter for the State Transition Matrix, whose dimensions are n by n.
        This method is abstract and needs to be defined in the implementing subclass.
        """
        raise NotImplementedError("Method get_A() needs to be implemented by the inheriting subclass")
    
    def get_H(self):
        """
        Getter for the State to Measurement matrix, whose dimensions are m by m.
        This method is abstract and needs to be defined in the implementing subclass.
        """
        raise NotImplementedError("Method get_H() needs to be implemented by the inheriting subclass")
    
    def get_Q(self):
        """
        Getter for the Noise Covariance Matrix, whose dimensions are n by n.
        This method is abstract and needs to be defined in the implementing subclass.
        """
        raise NotImplementedError("Method get_Q() needs to be implemented by the inheriting subclass")

    def initialize_system(self) -> tuple[np.array, np.array]:
        """
        initializes the Kalaman system by defining the initial state and certaintiy.
        This method is abstract and needs to be defined in the implementing subclass.
        :return: a tuple consisting of the state variable `x` 
            and state covariance `P` matrix, index respective.
        """
        raise NotImplementedError("Method initialize_system() " \
                                  "needs to be implemented by the ineriting subclass")
    
    #==========     Main Methods     ==========#
    def run(self) -> Union[pd.Series | pd.DataFrame]:
        """
        Runs a Kalman filter on all steps for `inputs` set in :py:func:`__init__`.
        :return: a pandas `Series` object, with original timesteps and the computed values. 
            If insufficient enough data, the original Dataset will be returned instead.
        """
        #TODO: implement
        raise NotImplementedError()

    #==========    Helper Methods    ==========#
    def _predict(self):
        """
        Computes the expected next value based on the Kalman filter's parameters
        :return: This method mutates internal values, nothing is returned

        """

        #TODO : Project the state pr_x
        #TODO : Project the error covariance
        raise NotImplementedError()

    def _update(self):
        """
        Updates the Kalman filter's parameters based on the predicted computation
        and the actual answers
        :return: This method mutates internal values, nothing is returned
        """
        #TODO : compute Kalman Gain
        #TODO : Update Estimate with measurement z
        #TODO : Update the error covariance
        raise NotImplementedError()


    def _compute_gain(self):
       """
       Computes the gain of the current system.
       :return: This method mutates self.K, and does not return anything
       :rtype: None
       """
       raise NotImplementedError()
        #TODO : pr_P H' (H pr_P H' + R)^-1
