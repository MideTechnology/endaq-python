import pandas as pd
from typing import List, Optional, Union, Literal
import numpy as np
import datetime as dt


class KalmanFilterInterface:
    """
    An interface holding the methods used in a Kalman Filter.
    Due to the extensive adaptability of the Kalman Filter, all methods 
    will remain unimplemented and no variables will be set, all left to the super class.
    All implementation should follow the notation below
    - uppercase represents matrix, 
    - lowercase_h represents vector. 
    -  _pr represents priori or prediction
    - _T represents a transposed matrix

    """

    def update_parameters(self, delta_t) -> None:
        """
        Updates any internal parameters based on the given delta time.
        :return : None, all data associated with this method is modified.
        """
        raise NotImplementedError("Method update_parameters(delta_t) " \
                                  "needs to be implemented by the inheriting subclass")
    def initialize_system(self) -> tuple[np.ndarray, np.ndarray]:
        """
        initializes the Kalaman system by defining the initial state and certaintiy.
        This method is abstract and needs to be defined in the implementing subclass.
        :return: a tuple consisting of the state variable `x` 
            and state covariance `P` matrix, index respective.
        """

        raise NotImplementedError("Method initialize_system() " \
                                  "needs to be implemented by the ineriting subclass")


    def get_next_data_point(self) -> Union[tuple[np.array, dt.datetime64], Literal["End"]]:
        """
        gets the next data point (form of a vector) along with the time
        between this point and the previous. "End" is returned if there are no more
        data points.
        This method is abstract and needs to be defined in the implementing subclass.
        :return: (next data point, delta time elapsed) or keyword "End" 
        """
        raise NotImplementedError("Method get_next_data_point()" \
                                  "needs to be implemented by the inheriting subclass")
    
    def run(self) -> pd.Series:
        """
        Runs a Kalman filter using the parameters set in :py:func:`__init__` and the
        data points given in :py:func:`get_next_data_point()`
        :return: a pandas `Series` object, with original timesteps and the computed values. 
        """
        raise NotImplementedError("Method run()" \
                                  "needs to be implemented by the inheriting subclass")



class LinearKF(KalmanFilterInterface):
    """
    An abstract mutable class that performs a Kalman filter, implementation mainly based on 
    https://thekalmanfilter.com/kalman-filter-explained-simply/ with other influences.
    A new `Kalman` class should be instantialized for filter operations on different inputs.
    The following functions need to be implemented by the methods that wish to use this class:
    """

    def __init__(
            self,
            A : np.ndarray,
            H: np.ndarray, 
            Q: np.ndarray, 
            R : np.ndarray,
            ):
        """
        init for the Kalman class. m is the number of input variables, and
        n is the number of prediction variables
        :param df: the dataset to perform the Kalman operation on, an 
            array of z's
        :param A: State Transition Matrix, n by n dimensions
        :param H: State-to-Measurement Matrix, m by n dimensions
        :param Q: Process noise Covariance Matrix, n by n dimensions
        :param R: Measurement Covariance Matrix, m by m dimensions
        """

        #----- prediction variables -----# 
        self._pr_x : np.array = None 
        self._pr_P : np.ndarray = None 

        #-----   system variables   -----#
        self.R : np.ndarray = R 
        self.H : np.ndarray = H
        self.A : np.ndarray = None # This gets set on the first iteration
        self.Q : np.ndarray = Q 
        x, P, start = self.initialize_system()

        self.x : np.array = x # n by 1 column vector
        self.P : np.ndarray = P # n by n


        self.K : np.ndarray = None
        #-----   output variables   -----#
        self.filtered  = []
        self.timestamps = [start]
        


    #==========     Main Methods     ==========#
    def run(self) -> pd.Series:
        self.initialize_system()
        dp = self.get_next_data_point()
        filtered_points = []
        while dp != "End":
            self.A = self.new_A((dp[1] - self.timestamps[-1]).total_seconds())
            self.timestamps.append(dp[1])
            self._predict()
            filtered_points.append(self._update(dp[0]))
            dp = self.get_next_data_point()
        

        return pd.Series(filtered_points, index=self.timestamps[1:])

    #==========    Helper Methods    ==========#
    def _predict(self):
        """
        Computes the expected next value based on the Kalman filter's parameters
        :return: This method mutates internal values, nothing is returned
        :rtype: None
        """
        self._pr_x = self.A @ self.x
        self._pr_P = (self.A @ self.P @ self.A.T + self.Q)


    def _update(self, z) -> np.array:
        """
        Updates the Kalman filter's parameters based on the predicted computation
        and the actual answers
        :param z: contains one or more column measurement column vectors. 
            In the case that `z` is not a nested list, it is assumed to 
            be one singular z value. 
        :return: This method mutates internal values, and returns the adjusted
            value of H @ x (the wanted values of our predicted x) 
        """

        if not isinstance(z[0], list):
            z = [z]
        for z_comp in z:
            S = self.H @ self._pr_P @ self.H.T + self.R
            self.K = self._pr_P @ self.H.T @ np.linalg.pinv(S)
            self.x = self._pr_x + self.K @ (z_comp - self.H @ self._pr_x)
            ikh = (np.eye(self.P.shape[0]) - self.K @ self.H)
            self.P = ikh @ self._pr_P

        return self.H @ self.x  


