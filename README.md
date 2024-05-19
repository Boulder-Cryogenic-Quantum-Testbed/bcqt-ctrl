# bcqt-ctrl - modernization branch
Jorge's changes to bcqt-ctrl to allow for easier use with QICK and future instruments


# Directories
* instrument\_control    generic instrument control classes
* pna\_control           control software for Keysight PNA
* temperature\_control   Janis JDry250, BlueFors LD250 sensor reading and
                         temperature control

# Installation
* Start by installing the required packages with pip. May or may not work, yet to be tested.
```bash
pip install -r requirements.txt
```

# Getting started
* Check out the examples in `pna_control` for connecting to the PNA
* More advanced examples, assuming JDry250, in `user_ctrl_segmented_homophasal.py`


