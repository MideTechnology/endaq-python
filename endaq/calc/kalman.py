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
    - uppercase Letter represents matrix, 
    - _h represents vector. 
    -  _pr represents priori or prediction
    - _T represents a transposed matrix
    - Unless stated otherwise, all other variable names do not follow specific notation.

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
        self.x_pr : np.ndarray = None 
        self.P_pr : np.ndarray = None 

        #-----   system variables   -----#
        self.A : np.ndarray = A # This gets set on the first iteration
        self.H : np.ndarray = H
        self.R : np.ndarray = R 
        self.Q : np.ndarray = Q 
        x, P, start = self.initialize_system()

        self.x : np.ndarray = x # n by 1 column vector
        self.P : np.ndarray = P # n by n


        self.K : np.ndarray = None
        #-----   output variables   -----#
        self.filtered  = []
        self.timestamps = [start]
        

    #==========     Overriden Methods     ==========#
    def run(self) -> pd.Series:
        self.initialize_system()
        dp = self.get_next_data_point()
        filtered_points = []
        while dp != "End":
            self.update_parameters((dp[1] - self.timestamps[-1]).total_seconds())
            self.timestamps.append(dp[1])

            self.x_pr, self.P_pr = self._calculate_priori(self.x)
            posteriori = self._calculate_posteriori(dp[0])
            filtered_points.append(posteriori)

            dp = self.get_next_data_point()
        

        return pd.Series(filtered_points, index=self.timestamps[1:])

    #==========    Helper Methods    ==========#
    def _calculate_priori(self, x) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the expected next value based on the Kalman filter's parameters
        :return: a tuple consisting of  (x_priori, P_priori)
        """
        x_pr = self.A @ x
        P_pr = (self.A @ self.P @ self.A.T + self.Q)

        return (x_pr, P_pr)


    def _calculate_posteriori(self, z) -> np.array:
        """
        Updates the Kalman filter's parameters based on the predicted computation
        and the actual measurement.
        :param z: contains one or more column measurement column vectors. 
            In the case that `z` is not a nested list, it is assumed to 
            be one singular z value. 
        :return: This method mutates internal values, and returns the adjusted
            value of H @ x (the wanted values of our predicted x) 
        """

        if not isinstance(z[0], list):
            z = [z]
        for z_comp in z:
            S = self.H @ self.P_pr @ self.H.T + self.R
            self.K = self.P_pr @ self.H.T @ np.linalg.pinv(S)
            self.x = self.x_pr + self.K @ (z_comp - self.H @ self.x_pr)
            ikh = (np.eye(self.P.shape[0]) - self.K @ self.H)
            self.P = ikh @ self.P_pr

        return self.H @ self.x  



class UnscentedKF(KalmanFilterInterface): 
    """
    An abstract class for the process of a Unscented Kalman Filter (UKF). This 
    class has additional methods to be implemented: 
    - `self.predict_next_state(x_T)`
    - `self.convert_to_measurement()`
    Implementation mostly follows https://yugu.faculty.wvu.edu/files/d/2cbb566f-9936-4033-bb1c-6d887c30d45a/irl_wvu_online_ukf_implementation_v1-0_06_28_2013.pdf,
    with variable naming convention from a variety of sources.
    The following is Unscented Kalman Filter specific variable notation:
    - cal_ represents an inbetween sigma point calculation. 
    
    """
    
    
    def __init__(self, Q, alpha, *, beta= 2, kappa= 0):
        """
        initialization method for the Unscented Kalman Filter.
        :param Q: proccess noise matrix
        :param alpha: sigma points spread
        :param beta: secondary scaling parameter, default (and most optimal) value is set to 2
        :param kappa: tertiary scaling parameter, default (and most common) value is set to 0
        """
        

        self.Q = Q
        self.v = None
        
        #----- Previous iteration variables -----#
        self.x = None
        self.P = None
        #----- Inbetween variables -----#
        self.S : np.ndarray = None 
        self.K : np.ndarray = None
        #----- Sigma point specifics -----#
        self.scaling = (alpha, beta, kappa)
        #initializng as None for clarity, gets set in `self.unscented_transform()`
        self.eta_c = None #covariance weight vectors
        self.eta_m = None #mean weight vectors
        self.cal_X : np.ndarray = None
        self.cal_Y : np.ndarray = None
        
        #----- Output variables -----#
        #INVARIANT : timestamp will have 2+ timestamps before the first calculation
        #            and will continue to have 2+ for the entire duration
        self.timestamps = np.array([])
        self.filtered_data = np.array([])
        

    def unscented_transform(self, alpha, *, beta = 2, kappa = 0):
        """
        Performs an unscented transform, calculating sigma points and the post-transform covariance
        on an assumption of a zero-mean for our random variable `x`. 
        :param alpha: sigma points spread
        :param beta: secondary scaling parameter, default (and most optimal) value is set to 2
        :param kappa: tertiary scaling parameter, default (and most common) value is set to 0
        :return: a tuple consisting of sigma points and the associated covariance. Values `self.eta_m` and `self.eta_c`
            are also mutated.
        """
        #l represents lambda the variable, not lambda function
        l = alpha ** 2 (self.L + kappa) - self.L 
        self.eta_m = np.array([l / (self.L + l)])
        self.eta_c = np.array([1 / (self.L + l) + 1 - alpha ** 2  + beta])

        self.eta_m += [1/(2(self.L + l))]* (2 * self.L) #this is **not** a np array
        self.eta_c += [1/(2(self.L + l))]* (2 * self.L) #this is **not** a np array
        sigma_mag = np.sqrt(self.L + l) * self.S
        cal_X = np.concat([np.zeros((self.L,1)), sigma_mag, -1 * sigma_mag])
        
        Psi = np.array([self.predict_next_state(cal_X[:,i]) for i in range(cal_X.shape[0])]).T
        
        y_mean = sum([Psi[:, col_idx] * self.eta_m[col_idx] for col_idx in range(Psi.shape[0])])
        P_y = np.sum(
        [self.eta_c[col_idx] * 
        (np.array([Psi[:, col_idx] - y_mean]).T @
         np.array([Psi[:, col_idx] - y_mean]))
         for col_idx in range(Psi.shape[0])], 
        axis = 0)
        return (cal_X, P_y)    
        
    def predict_next_state(self, x_T) -> np.ndarray:
        """
        The state transition function A for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in the only guarenteed parameter, x.
        For efficiency reasons,
        self should be used for all other variables, eg : noise 
        :parameter x_T: the **Transposed** data (x_T is a vector) point. 
        :return: a vector with the same dimensions as x_T
        """
        raise Exception("Method predict_next_state(x_T) " \
        "needs to be implemented by the inheriting subclass")

   
    def convert_to_measurement(self, y_T) -> np.ndarray:
        """
        The measurement conversion function H for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in no parameters. Instead, self should be used
        to find the associated measurement.
        :return: a vector with the same dimensions as y_T

        """
        raise Exception("Method convert_to_measurement()" \
        "needs to be implemented by the inheriting subclass")
    
    
    def run(self) -> pd.Series:
        raise Exception("This method has not been implemented yet,"
        " and should be implemented in the UKF class")
    
    
    def _calculate_priori(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction on what the state should be with the information already known.
        This can be thought of as the "predict" step of a predict and adjust algorithm.
        :return : a tuple consisting of (x_priori, P_priori) 
        """
        S = np.linalg.cholesky(self.P)  
        cal_X, P_y = self.unscented_transform(self.x, self.S)
        cal_X_pr = self.predict_next_state(cal_X) 
        x_pr = sum([cal_X_pr[:, col_idx] * self.eta_m[col_idx] for col_idx in range(cal_X_pr.shape[0])])
        P_pr = self.Q + np.sum(
            [self.eta_c[col_idx] * 
            (np.array([cal_X_pr[:, col_idx]]).T @
             np.array([cal_X_pr[:, col_idx]]))
             for col_idx in range()], 
            axis = 0)
        return (x_pr, P_pr)
        
    def _calculate_posteriori(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Corrects the internal variables based on the accuracy of the calculated priori.
        This can be thought of as the "update" step of a predict and update algorithm.
        :return: a tuple consisting of (x_posteriori, P_posteriori)
        """
        y_pr = None
        P_yy = None
        P_xy = None
        
        K = self.P_xy @ np.linalg.pinv(P_yy)
        x = self.x_pr + self.K @ (self.y - self.y_pr)
        P = self.P_pr - K @ self.P_yy @ self.K.T
        
        return (x, P)