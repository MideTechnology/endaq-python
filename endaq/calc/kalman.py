import pandas as pd
from typing import Union, Literal, Annotated, TypeVar
import numpy as np
import datetime as dt

__all__ = [
    "AccelerationKalmanFilter",
    "OrientationKalmanFilter",
]

class LinearKalmanFilter:
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
        self.x_pr = None 
        self.P_pr = None 
        #-----   system variables   -----#
        self.A = A 
        self.H = H
        self.Q = Q 
        self.R = R 
        x, P, start = self.initialize_system()

        self.x = x 
        self.P = P

        self.K = None
        #-----   output variables   -----#
        self.filtered  = []
        self.timestamps = [start]

    #-----   methods to override   -----#
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


    def _calculate_posteriori(self, z) -> np.ndarray:
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

class UnscentedKalmanFilter: 
    """
    An abstract class for the process of a Unscented Kalman Filter (UKF). This 
    class has additional methods to be implemented: 
    - `self.predict_next_state(x_T)`
    - `self.measurement_to_state(y_T)`
    Implementation mostly follows https://yugu.faculty.wvu.edu/files/d/2cbb566f-9936-4033-bb1c-6d887c30d45a/irl_wvu_online_ukf_implementation_v1-0_06_28_2013.pdf
    and https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb,
    with variable naming convention from a variety of sources.
    The following is Unscented Kalman Filter specific variable notation:
    - cal_ represents an inbetween sigma point calculation. 
    """
    
    def __init__(self, Q, alpha, *, beta= 2, kappa= 0):
        """
        initialization method for the Unscented Kalman Filter.
        All methods with value `None` get set to proper values in methods,
        and are only written out here for clarity.
        :param Q: proccess noise matrix
        :param alpha: sigma points spread
        :param beta: secondary scaling parameter, default (and most optimal) value is set to 2
        :param kappa: tertiary scaling parameter, default (and most common) value is set to 0
        """
        self.Q = Q
        self.n = None 
        #----- Previous iteration variables -----#
        self.x : Annotated[np.ndarray, Literal[(self.n, 1)]] = None
        self.P : Annotated[np.ndarray, Literal[(self.n, self.n)]] = None
        #----- Inbetween variables -----#
        self.S : np.ndarray = None 
        self.K : np.ndarray = None
        #----- Sigma point specifics -----#
        eta_c, eta_m = self._calculate_weights(alpha, beta=beta, kappa=kappa)
        #covariance weight vectors
        self.eta_c : Annotated[np.ndarray, Literal[(1, 2 * self.n)]] = eta_c
        #mean weight vectors
        self.eta_m : Annotated[np.ndarray, Literal[(1, 2 * self.n)]] = eta_m 
        #----- Output variables -----#
        """ INVARIANT : timestamp will have 2+ timestamps before the first calculation
            and will continue to have 2+ for the entire duration, as timestamps are
            never removed. """
        self.timestamps = np.array([])
        self.filtered_data = np.array([])

    #----------- Methods for user to implement -----------#
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

    def predict_next_state(self, x_T) -> np.ndarray:
        """
        The state transition function A for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in the only guarenteed parameter, x.
        For efficiency reasons,
        self should be used for all other variables, eg : noise 
        :parameter x_T: the **Transposed** data (x_T is a vector) point. 
        :return: a column vector (non transposed) with the same dimensions as x
        """
        raise NotImplementedError("Method predict_next_state(x_T) " \
        "needs to be implemented by the inheriting subclass")

    def measurement_to_state(self, y_T) -> np.ndarray:
        """
        The measurement conversion function H for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in no parameters. Instead, self should be used
        to find the associated measurement.
        :return: a vector with the same dimensions as y_T
        """
        raise NotImplementedError("Method convert_to_measurement()" \
        "needs to be implemented by the inheriting subclass")
    
    #----------- Main Methods -----------#
    def run(self) -> pd.Series:
        """
        Runs the Unscented Kalman filter, mutating the internal parameters. 
        The UKF should be re-instantialized every time it is run.
        :return: A series containing timestamps as indecies and the predicted data
        """
        self.x, self.P = self.initialize_system()
        dp = self.get_next_data_point()
        self.n = len(dp[0])
        
        self.timestamps = np.append(self.timestamps, dp[1])
        dp = dp[0]
        
        while dp != 'End':
            self.eta_m, self.eta_c = self._calculate_weights()
        
            #TODO : make sure it's dp for both
            self.x_pr, self.P_pr = self._calculate_priori(dp)
            self.x, self.P = self._calculate_posteriori(dp)
            self.filtered_data = np.append(self.filtered_data, self.x)

            dp = self.get_next_data_point()
            self.timestamps = np.append(self.timestamps, dp[1])
            dp = dp[0]

    #----------- Helper Methods -----------#

    def _calculate_weights(self, alpha, *, beta = 2, kappa = 0) -> tuple[list[float], list[float]]:
        """
        Initializes the mean and covariance weightings of the sigma points
        :param alpha: sigma points spread
        :param beta: secondary scaling parameter, default (and most optimal) value is set to 2
        :param kappa: tertiary scaling parameter, default (and most common) value is set to 0
        :return: a tuple consisting of (mean weightings, covariance weightings), both of
            which are list of floats
        """
        #l represents lambda the variable, not lambda function
        self.l = alpha ** 2 * (self.n + kappa) - self.n 

        eta_m = np.array([self.l / (self.n + self.l)])
        eta_c = np.array([1 / (self.n + self.l) + 1 - alpha ** 2  + beta])

        eta_i = 1/(2* (self.n + self.l))
        #adds 2 * L more eta_i to eta_m and eta_c
        eta_m += [eta_i] * (2 * self.n)
        eta_c += [eta_i] * (2 * self.n)

        return (eta_m, eta_c)

    def _unscented_transform(self, x_T, mu_w, P_w, cov_noise) -> tuple[np.ndarray, np.ndarray]:
        """
        peforms an unscented transform to compute the mean and covariance of the given
        random sample points.
        :param x_T: a numpy array of sample points, who are also numpy arrays.
            Note that this parameter is **transposed**, which is
            a list of column vectors who are represented as rows.
        :param mu_w: mean weight of the sample points.
        :param P_w: mean covariance of the sample points.
        :param cov_noise: the covariance noise associated with the sample points
        :return: a tuple of (mean, covariance)
        """
        if isinstance(x_T, list):
            x_T = np.array(x_T)
        if x_T.size == 0:
            return (np.array([]), np.array([]))
        
        mu_x = np.mean([weight * col for (weight, col) in zip(mu_w, x_T)], axis = 1)
        
        P_x = np.zeros(x_T[0].shape[0])
        for (weight, col) in zip(P_w, x_T):
            zm_col = col - mu_x
            P_x += weight * zm_col
        P_x += cov_noise
        
        return (mu_x, P_x)
    
    def _create_sigma_points(self, x, S) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs an unscented transform, calculating sigma points and the post-transform covariance
        on an assumption of a zero-mean for our random variable `x`. 
        :param alpha: sigma points spread
        :param beta: secondary scaling parameter, default (and most optimal) value is set to 2
        :param kappa: tertiary scaling parameter, default (and most common) value is set to 0
        :return: a tuple consisting of sigma points and the associated covariance. 
            Values `self.eta_m` and `self.eta_c` are also mutated.
        """
        
        sigma_mag = np.sqrt(self.n + self.l) * S
        cal_X = np.concat([np.zeros((self.n,1)), sigma_mag, -1 * sigma_mag])
        
        Psi_T = np.array(
            [self.predict_next_state(cal_X[:,i]) for i in range(cal_X.shape[0])]
            )

        mu_y = 0
        P_y = np.zeros(1 + 2 * self.n)
        
        for col_idx in range(Psi_T.shape[0]):
            mu_y += Psi_T[col_idx] * self.eta_m[col_idx]
            psi_var = [Psi_T[:, col_idx] - mu_y].T @ [Psi_T[:, col_idx] - mu_y]
                
            P_y += self.eta_c[col_idx] * psi_var
        return (cal_X, P_y)    
        
    def _calculate_priori(self, x) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction on what the state should be with the information already known.
        This can be thought of as the "predict" step of a predict and adjust algorithm.
        :return : a tuple consisting of (x_priori, P_priori).
        """
        S = np.linalg.cholesky(self.P)  
        cal_X, self.P_y = self._create_sigma_points(self.x, S)

        cal_X_pr = self.predict_next_state(cal_X) 
        self.cal_Y = cal_X_pr 

        return self._unscented_transform(cal_X_pr, self.eta_m, self.eta_c, self.Q)

    def _calculate_posteriori(self, z) -> tuple[np.ndarray, np.ndarray]:
        """
        Corrects the internal variables based on the accuracy of the calculated priori.
        This can be thought of as the "update" step of a predict and update algorithm.
        :return: a tuple consisting of (x_posteriori, P_posteriori)
        """
        cal_Z = self.measurement_to_state(self.cal_Y)
        mu_z, P_z = self._unscented_transform(cal_Z, self.R) 
        y = z - mu_z
        
        K = self.eta_c * np.sum((self.cal_Y - self.x_pr) @ (cal_Z - mu_z).T, axis = 1)
        K = K @ np.linalg.pinv(P_z)

        x = self.x_pr + K @ y
        P = self.P_pr - K @ P_z @ K.T
        return (x, P)

#------------- CONCRETE IMPLEMENTATIONS -------------#
class AccelerationKalmanFilter(LinearKalmanFilter):
        """
        Concrete implementation a Kalman filter for `refine_acceleration` in 
        filters.py, used to combine multiple acceleration channels on an Endaq Device.
        """ 
        def __init__(self, dfs):
            for df_idx in range(len(dfs)):
                df = dfs[df_idx]
                for col in df.columns:
                    df[col] = df[col] - df[col].mean()
                    if col[0:3] not in ['X (', 'Y (', 'Z (']:
                        dfs[df_idx] = df.drop(col, axis = 1)
            self.df_data = [df.values for df in dfs]
            self.df_timestamps = [df.index for df in dfs]
            self.timestamp_idx = [0 for _ in dfs]

            self._full_dataset = np.concat(self.df_data)
            A = np.eye(6)
            H = np.concat((np.eye(3), np.zeros((3,3))), axis = 1)
            Q = np.diag([0.05] * 3 + [0.0] *3)  #i don't think this is correct.
            R = np.diag(
                [np.var([dp[column_idx] for dp in self.df_data]) 
                    for column_idx in range(len(self.df_data[0][0]))], #HACK : I really don't like this implementation
            )

            self.specs = [df.columns[0][2:] for df in dfs] #gets the rating, eg (40g) 
            
            self.base_rating = {'(8g)' : 0.00002,
                            '(16g)' : 0.004,
                            '(40g)' : 0.00002,
                            '(100g)' : 0.05,
                            } 

            self.hz_rating = {'(8g)' : None,
                            '(16g)' : None,
                            '(40g)' : None,
                            '(100g)' : None,
                            } 
            super().__init__(A, H, Q, R)


        def initialize_system(self):
            dp1 = self.get_next_data_point()
            dp2 = self.get_next_data_point()

            if (isinstance(dp1, str)) or (isinstance(dp2, str)):
                raise StopIteration("Not enough data points")

            deltaT = (dp2[1] - dp1[1]).total_seconds()
            x = np.concatenate((dp2[0], (dp2[0] - dp1[0]) * deltaT))
            P = np.diag(
                np.concatenate([
                    [np.var([dp[column_idx] for dp in self.df_data])
                          for column_idx in range(len(self.df_data[0][0]))],
                    [100,100,100]]))
            #100 is an arbitrary value commonly used in Kalman filter implementations,
            #representing high uncertainty
            return (x,P, dp2[1])
        
        def update_parameters(self, deltaT):
            A = np.eye(6)
            A[0,3] = deltaT
            A[1,4] = deltaT
            A[2,5] = deltaT
            self.A = A

        
        def get_next_data_point(self):
            if self.df_data == []:
                return "End"
            
            candidates = [self.df_timestamps[i][self.timestamp_idx[i]] 
                          for i in range(len(self.timestamp_idx))]
            selected_df_idx = candidates.index(min(candidates)) 
            df_idx = self.timestamp_idx[selected_df_idx]
            next_dp = (
                self.df_data[selected_df_idx][df_idx],
                candidates[selected_df_idx], 
                )
            self._determine_noise(selected_df_idx)
            
            self.timestamp_idx[selected_df_idx] += 1

            if self.timestamp_idx[selected_df_idx] >= len(self.df_timestamps[selected_df_idx]):
                del self.df_data[selected_df_idx]
                del self.df_timestamps[selected_df_idx]
                del self.timestamp_idx[selected_df_idx] 

            return next_dp
    
    
        def _determine_noise(self, selected_df_idx):
            """
            Determines the additional noise 
            :param selected_idx: The index of the selected dataframe in `self.df_data`.

            :return: None, this method mutates `self.noise`.
            """
            name = self.specs[selected_df_idx]
            penultimate_idx = self.timestamp_idx[selected_df_idx] - 2

            if penultimate_idx < 0:
                self.noise = self.base_rating[name]
                return
            
            timestamps = self.df_timestamps[selected_df_idx]
            hz = 1 / (
                timestamps[penultimate_idx + 1] - 
                timestamps[penultimate_idx]
                ).total_seconds()
            #self.noise = self.hz_rating[name](hz) + self.base_rating[name]
            self.noise = self.base_rating[name]

class OrientationKalmanFilter(UnscentedKalmanFilter):
    """
    A concrete implementation of a Unscented Kalman filter for `validate_orientation` in 
    filters.py, by predicting orientation through acceleration and rotation.
    Due to the nature of quaternions, in addition to the the base methods to implement,
    `_create_sigma_points` and `_calculate_weights` have to be overriden.
    """

    def __init__(self, acceleration_df, rotation_df, orientation_df):
        """        
        :param acceleration_df: dataframe representing acceleration. Any ratings of sensors
            are valid, or `refine_acceleration` can be used to combine into one. 
        :param rotation_df: dataframe representing  rotation. 
        :param orientation_df: dataframe representing **relative** orientation. 
        """
        self.cur_idx = 0
        #TODO : align datasets to all have the same time points (and equal number of dps)
        
        aligned_acc = None
        aligned_rot = None
        aligned_ori = None
        #TODO : do I want to keep this as a dataframe or convert to lists (aka ILOC)
        self.acc_df = aligned_acc
        self.rot_df = aligned_rot
        self.ori_df = aligned_ori
        self.timestamps = None #already predetermined from resampling
        self.delta_t = None #already predetermined from resampling

    def update_parameters(self, delta_t) -> None:
        """
        Updates any internal parameters based on the given delta time.
        :return : None, all data associated with this method is modified.
        """
        pass
    
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
        gets the next data point as a 10 index array, [accel, rot, orientation] and the
        time delta between points.
        :return: (next data point, delta time elapsed) or keyword "End" 
        """
        raise NotImplementedError("Method get_next_data_point()" \
                                  "needs to be implemented by the inheriting subclass")

    def predict_next_state(self, x_T) -> np.ndarray:
        """
        The state transition function A for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in the only guarenteed parameter, x.
        For efficiency reasons,
        self should be used for all other variables, eg : noise 
        :parameter x_T: the **Transposed** data (x_T is a vector) point. 
        :return: a column vector (non transposed) with the same dimensions as x
        """
        raise NotImplementedError("Method predict_next_state(x_T) " \
        "needs to be implemented by the inheriting subclass")

    def measurement_to_state(self, y_T) -> np.ndarray:
        """
        The measurement conversion function H for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in no parameters. Instead, self should be used
        to find the associated measurement.
        :return: a vector with the same dimensions as y_T
        """
        raise NotImplementedError("Method convert_to_measurement()" \
        "needs to be implemented by the inheriting subclass")

