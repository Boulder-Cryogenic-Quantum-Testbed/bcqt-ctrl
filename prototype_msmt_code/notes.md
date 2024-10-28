
# BCQT MEASUREMENT CODE

## EXPERIMENT  
--- `ResonatorMeasurement`

  -> uses resonator measurement code

--- `QubitMeasurement`

  -> uses qick and qickutils 

#### EXPERIMENT INPUTS
--- List of `InstrumentDrivers`

--- a single `ExperimentConfig` object

### EXPERIMENT OUTPUTS
--- "Raw Data" in hdf5 or csv formats

## InstrumentDriver
--- All drivers inherit `BaseDriver` class, must implement abstract methods

--- One driver for each instrument we have, e.g. `KeysightPNA`, `AnritsuSigGen`, `KeysightPowerSupply`

Methods include implementations of scpi commands and auxillaries, e.g. `Read`, `Write`, `Query`, `RampPower`, `SweepFrequency`

Attributes include instrument model, 

## DataHandler 
  -> manage data during & between measurements
  -> on-the-fly visualization of data

## DataProcessor 
  -> works with "Raw Data" and packages, labels, and creates metadata, etc
      "Packaged Data" in hdf5, includes metadata and fit results files

  -> most difficult to code, will leave it barebones. Will use QickUtils as reference

## DataAnalysis 
  -> works with "Packaged Data" from DataProcessor




# Programming Notes

    @abstractmethod is for when you:
        Require all children to have a method
        Don't have enough information to define that method in the parent

        !!! Don't need to rewrite the abstract method in BaseClass! You can just call it with super() !!!


