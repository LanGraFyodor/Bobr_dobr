from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    def __init__(self, initial_x: float, initial_y: float, initial_vx: float = 0.0, initial_vy: float = 0.0):
        # State vector: [x, y, vx, vy]^T
        self.X = np.array([initial_x, initial_y, initial_vx, initial_vy], dtype=np.float64)
        
        # State covariance matrix P
        self.P = np.eye(4, dtype=np.float64) * 100.0  # Initial high uncertainty
        
        # Measurement matrix H (we measure x and y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

    def predict(self, dt: float, process_noise_std: float = 1.0) -> None:
        """
        Propagate the state forward by dt seconds.
        """
        # State transition matrix F
        F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Process noise covariance Q (piecewise constant acceleration model)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q = process_noise_std ** 2
        Q = np.array([
            [dt4/4,   0.0, dt3/2,   0.0],
            [  0.0, dt4/4,   0.0, dt3/2],
            [dt3/2,   0.0,   dt2,   0.0],
            [  0.0, dt3/2,   0.0,   dt2]
        ], dtype=np.float64) * q

        # Predict State
        self.X = F @ self.X
        # Predict Covariance
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_x: float, measured_y: float, measurement_noise_std: float) -> None:
        """
        Update the state based on a new position measurement.
        """
        Z = np.array([measured_x, measured_y], dtype=np.float64)
        
        r = measurement_noise_std ** 2
        R = np.array([
            [r, 0.0],
            [0.0, r]
        ], dtype=np.float64)

        # Innovation (residual)
        Y = Z - (self.H @ self.X)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update State
        self.X = self.X + (K @ Y)
        
        # Update Covariance
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        
    @property
    def x(self) -> float:
        return float(self.X[0])
        
    @property
    def y(self) -> float:
        return float(self.X[1])
        
    @property
    def vx(self) -> float:
        return float(self.X[2])
        
    @property
    def vy(self) -> float:
        return float(self.X[3])
        
    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))
        
    @property
    def azimuth_deg(self) -> float:
        # 0 degrees is North (+Y), 90 degrees is East (+X)
        azimuth = np.rad2deg(np.arctan2(self.vx, self.vy))
        return float(azimuth % 360.0)
