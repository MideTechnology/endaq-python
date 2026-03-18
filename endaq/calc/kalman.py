import pandas as pd
from typing import Union, Literal, Annotated, TypeVar
import numpy as np
import datetime as dt
from scipy.spatial.transform import Rotation as R
from endaq.calc.utils import align_datasets


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
    
    def __init__(self, Q, alpha = 0.001, *, beta= 2, kappa= 0):
        """
        initialization method for the Unscented Kalman Filter.
        All methods with value `None` get set to proper values in methods,
        and are only written out here for clarity.
        :param Q: proccess noise matrix
        :param alpha: sigma points spread, default is set to 0.001
        :param beta: secondary scaling parameter, default (and most optimal) value is set to 2
        :param kappa: tertiary scaling parameter, default (and most common) value is set to 0
        """
       
    #----------- Methods for user to implement -----------#
    def update_parameters(self, delta_t) -> None:
        """
        Updates any internal parameters based on the given delta time.
        
        :return: None, all data associated with this method is modified.
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

    def predict_next_state(self, x_T, w_T) -> np.ndarray:
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
       

    #----------- Helper Methods -----------#

    def _calculate_weights(self, **kwargs) -> tuple[list[float], list[float]]:
        """
        Initializes the mean and covariance weightings of the sigma points.
        this method is keyword arguments because weights are dependent on the system.

        :param **kwargs: keyword only arguments, holding anything that is
        """


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

    
    def _create_sigma_points(self, x, P, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Performs an unscented transform, calculating sigma points and the post-transform covariance
        on an assumption of a zero-mean for our random variable `x`. 

        The equation it uses is the following <br>
        $\mathcal{X}^a_{k-1} = \left[\hat{x}^a_{k-1}, \hat{x}^a_{k-1} \pm \sqrt{(L + \lambda) P^a_{k-1}} \right]$

        :param x: the point to create the sigma points around
        :param S: the covariance at the given point


        :param **kwargs: keyword arguemnts for alternative implementations of sigma points. 
            If used, it should be documented in it's docstring
        :return: a tuple consisting of sigma points and the associated covariance. 
        """
        
        return 
        
    def _calculate_priori(self, x) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction on what the state should be with the information already known.
        This can be thought of as the "predict" step of a predict and adjust algorithm.
        :return : a tuple consisting of (x_priori, P_priori).
        """


    def _calculate_posteriori(self, z) -> tuple[np.ndarray, np.ndarray]:
        """
        Corrects the internal variables based on the accuracy of the calculated priori.
        This can be thought of as the "update" step of a predict and update algorithm.
        :return: a tuple consisting of (x_posteriori, P_posteriori)
        """

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
    This method uses scipy.spatial.transform.Rotation for quaternions, which have scalar last
    """

    def __init__(self, 
                acceleration_df, 
                rotation_df, 
                orientation_df):
        """        
        :param acceleration_df: dataframe representing acceleration. Any ratings of sensors
            are valid, or `refine_acceleration` can be used to combine into one. 
        :param rotation_df: dataframe representing  rotation. 
        :param orientation_df: dataframe representing **relative** orientation. 
        """
        self.cur_idx = 0
        
        aligned_acc, aligned_rot, aligned_ori = \
            align_datasets([acceleration_df, rotation_df, orientation_df]) 
        self.acc_iter = aligned_acc.itertuples()
        self.rot_iter = aligned_rot.itertuples()
        self.ori_iter = aligned_ori.itertuples()
        self.timestamps = aligned_acc.index 
        self.delta_t = self.timestamps[1] - self.timestamps[0] 

        self.rot_noise = None
        self.acc_noise = None

    def update_parameters(self, delta_t) -> None:
        """
        Updates any internal parameters based on the given delta time.
        :return : None, all data associated with this method is modified.
        """
        #delta_t is static in this implementation, so nothing needs to be updated
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

        next_acc = next(self.acc_iter, "End")
        if next_acc == "End": 
            return "End"
        #by setup, all 3 iters have the same number of elements
        next_point = np.array([next_acc, next(self.rot_iter), next(self.ori_iter)]).flatten()
        #by definition, delta_t does not change.
        return (next_point, self.delta_t)

    def predict_next_state(self, x_T, w_T) -> np.ndarray:
        """
        The state transition function A for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in the only guarenteed parameter, x.
        For efficiency reasons,
        self should be used for all other variables, eg : noise 
        :parameter x_T: the **Transposed** data (x_T is a vector) point. 
        :return: a column vector (non transposed) with the same dimensions as x
        """
        q = x_T[0:4]
        omega = x_T[4:]
        q_delta = R.from_rotvec(omega)
        q_pr = q * q_delta
        #omega_pr = omega
        return np.array([np.concat([q_pr.as_quat(), omega])]).T


    def measurement_to_state(self, y_T, v_T) -> np.ndarray:
        """
        The measurement conversion function H for a UKF. Due to the variable number of inputs
        in different subclasses, this method takes in no parameters. Instead, self should be used
        to find the associated measurement.
        :return: a vector with the same dimensions as y_T
        """
        q = y_T[0:4]
        omega = y_T[4:]
        z_rot = omega + self.rot_noise
        #TODO : 1 goes in the direction of gravity
        z_acc = q * R.from_quat([0,0,1,0]) * (q ** -1) + self.acc_noise

        return np.array([np.concat([z_rot.as_quat(), z_acc])]).T


    def _create_sigma_points(self, x, P, **kwargs):
        pass

    def _calculate_weights(self, **kwargs) -> tuple[list[float], list[float]]:
        """
        Initializes the mean and covariance weightings of the sigma points
        
        :param **kwargs: For this, we expect the kwargs of "S" and "Q"
        :param S: found in **kwargs, represents the Cholesky of the covariance matrix, with the
            noise already added in

        :return: a tuple consisting of (mean weightings, covariance weightings), both of
            which are list of floats
        """
        if 'S' not in kwargs:
            raise Exception('Expected "S" when calling _calculate_weights')
        S = kwargs['S']
        W = np.sqrt(2 * S.shape[1]) * S.T
        return np.concat([W, -1 * W])
