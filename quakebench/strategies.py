from abc import ABC, abstractmethod

class TimeSeriesPrepStrategy(ABC):
    @abstractmethod
    def downsample(self):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def split(self):
        pass



SomeClass(X, t, SplitStrat, ScalingStrat, DownsampleStrat, modelArchitecture)
someClass.train()
someClass.test()
someClass.eval()
