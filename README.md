# bcqt-ctrl
VNA and temperature control software for the measurement of superconducting microwave resonators

# Directories
* instrument\_control    placeholder for generic classes
* pna\_control           control software for Keysight PNA
* temperature\_control   Janis JDry250, BlueFors LD250 sensor reading and
                         temperature control

# Installation
* Start by installing the required packages with pip
```bash
pip install -r requirements.txt
```

# Getting started
* Check out the examples in `pna_control` for connecting to the PNA
* More advanced examples, assuming JDry250, in `user_ctrl_segmented_homophasal.py`
