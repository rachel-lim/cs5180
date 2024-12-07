class LinearSchedule(object):
    """Calculates an epsilon value following a linear schedule

    Attributes:
        start_value: starting value of epsilon
        end_value: final value of epsilon
        duration: number of time steps to change epsilon over
    """
    def __init__(self, start_value: float, end_value: float, duration: int) -> None:
        """Initialize LinearSchedule

        Args:
            start_value: starting value of epsilon
            end_value: final value of epsilon
            duration: number of time steps to change epsilon over
        """
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

    def value(self, time):
        return self.start_value + (self.end_value - self.start_value) * min(1.0, time * 1.0 / self.duration)