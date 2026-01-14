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
    naming conventions : capital letters are matrices, pr_xxx is priori, and ps_xxx is posterior 
    """

    def __init__(self, df: pd.DataFrame, R : pd.DataFrame):
        """
        init for the Kalman class
        :param df: the dataset to perform the Kalma operation on
        :param R: the
        """
        x, P = self.initialize_system()
        self.x : np.array = x # n by 1 column vector
        self.P : np.ndarray = P # n by n
        self.filtered : np.array = np.array([])
        # see if this is readily available in the sensors
        self.R : np.ndarray= R # m by m
        self.K : np.ndarray= self._compute_gain() # n by m
        self.A : np.ndarray = self.get_A() # n by n
        self.H : np.ndarray = self.get_H() # m by n
        self.Q : np.ndarray = self.get_Q() # n by n
        #----- in between variables -----#
        self._pr_x : np.array = None 
        self._pr_P : np.ndarray= None 
        self.z : np.array = None # m by 1 column vector
    #========== Methods to Implement ==========#
    def get_base_A(self):
        """
        Getter for the State Transition Matrix, whose dimensions are n by n.
        This method is abstract and needs to be defined in the implementing subclass.
        """

        raise NotImplementedError("Method get_base_A() needs to be implemented by the inheriting subclass")
    
    def get_base_H(self):
        """
        Getter for the State to Measurement matrix, whose dimensions are m by m.
        This method is abstract and needs to be defined in the implementing subclass.
        """

        raise NotImplementedError("Method get_base_H() needs to be implemented by the inheriting subclass")
    
    def get_base_Q(self):
        """
        Getter for the Noise Covariance Matrix, whose dimensions are n by n.
        This method is abstract and needs to be defined in the implementing subclass.
        """

        raise NotImplementedError("Method get_base_Q() needs to be implemented by the inheriting subclass")

    def initialize_system(self) -> tuple[np.ndarray, np.ndarray]:
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

        dates = self.df.index
        name = self.df.name
        df = self.df.to_numpy()
        for row in df:
            self.z = row
            self._predict()
            self._update()
            filtered = np.append(filtered, self.x)
        return self.filtered.to_series(index = dates, name = name)

    #==========    Helper Methods    ==========#
    def _predict(self):
        """
        Computes the expected next value based on the Kalman filter's parameters
        :return: This method mutates internal values, nothing is returned
        :rtype: None
        """
        self._pr_x = self.A @ self.x
        self._pr_P = self.A @ self.P @ self.A.T + self.Q

    def _update(self):
        """
        Updates the Kalman filter's parameters based on the predicted computation
        and the actual answers
        :return: This method mutates internal values, nothing is returned
        :rtype: None
        """
        self.K = self._pr_P @ self.H.T @ np.linalg.inv(
            self.H @ self._pr_P @ self.H.T + self.R)
        self.x = self._pr_x + self.K @ (self.z - self.H @ self._pr_x)
        self.P = (np.eye(self.P.shape[0]) - self.K @ self.H) @ self._pr_P


    def _compute_gain(self):
       """
       Computes the gain of the current system.
       :return: This method mutates self.K, and does not return anything
       :rtype: None
       """
       if self._pr_P is None: 
            self._predict()
            self._update() 
       self.K = (self._pr_P @ self.H.T) @ (
           np.linalg.inv(self.H @ self._pr_P @  self.H.T + self.R))
    