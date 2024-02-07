# Time series preparation for QuakeBench 


class TimeSeries:
    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = kwargs

    def prepare(self):
        raise NotImplementedError

