###
# Kalman filter utility function for localization.
###

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        """
        Initialize the Kalman Filter.

        Args:
            process_variance (float): The variance of the process noise.
            measurement_variance (float): The variance of the measurement noise.
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_estimate = 1.0

    def update(self, measurement):
        """
        Update the Kalman Filter with a new measurement.
        """
        # Prediction step
        self.estimate = self.estimate
        self.error_estimate = self.error_estimate + self.process_variance

        # Update step
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_estimate = (1 - kalman_gain) * self.error_estimate

    def get_estimate(self):
        """
        Get the current estimate from the Kalman Filter.

        Returns:
            float: The current estimate.
        """
        return self.estimate

    def reset(self):
        """
        Reset the Kalman Filter to its initial state.
        """
        self.estimate = 0.0
        self.error_estimate = 1.0

def KalmanFilter1D(process_variance, measurement_variance):
    """
    Factory function to create a 1D Kalman Filter.

    Args:
        process_variance (float): The variance of the process noise.
        measurement_variance (float): The variance of the measurement noise.

    Returns:
        KalmanFilter: An instance of the KalmanFilter class.
    """
    return KalmanFilter(process_variance, measurement_variance)