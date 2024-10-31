from abc import ABC, abstractmethod

# Abstract Base Class for Configuration
class BaseConfiguration(ABC):
    
    # why abstract methods?
    #   each new class that extends BaseConfiguration
    # will have its own method for save/load, but by
    # making an abstract method, we give a template!
    
    @abstractmethod
    def load(self):  
        pass

    @abstractmethod
    def save(self):
        pass

# Experiment Configuration Class
class ExperimentConfiguration(BaseConfiguration):
    def __init__(self, experiment_name, parameters):
        self._experiment_name = experiment_name
        self._parameters = parameters
    
    @property
    def name(self):
        return self._experiment_name

    @property
    def settings(self):
        return self._parameters

    def load(self):
        print(f"Loading experiment configuration for {self.name}...")

    def save(self):
        print(f"Saving experiment configuration for {self.name}...")


# Instrument Configuration Class
class InstrumentConfiguration(BaseConfiguration):
    def __init__(self, instrument_name, settings):
        self._instrument_name = instrument_name
        self._settings = settings

    @property
    def name(self):
        return self._instrument_name

    @property
    def settings(self):
        return self._settings

    def load(self):
        print(f"Loading instrument configuration for {self.name}...")

    def save(self):
        print(f"Saving instrument configuration for {self.name}...")