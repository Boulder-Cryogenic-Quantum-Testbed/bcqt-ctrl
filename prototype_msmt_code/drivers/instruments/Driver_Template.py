from BaseDriver import BaseDriver

class Example_Driver(BaseDriver):
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~  Abstract BaseDriver Methods  ~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # run all required abstract methods, and if we don't
    # want to change the original method in BaseDriver,
    # just call it using the super() function
    
    def __init__(self, InstrConfig_Dict, instr_resource=None, instr_address=None, debug=False, **kwargs):
        super().__init__(InstrConfig_Dict, instr_resource, instr_address, debug, **kwargs)
        
    def read_check(self, fmt=...):
        return super().read_check(fmt)
    
    def write_check(self, cmd: str):
        return super().write_check(cmd=cmd)
    
    def query_check(self, cmd, fmt=...):
        return super().query_check(cmd, fmt)
    
    def check_instr_error_queue(self, print_output=False):
        return super().check_instr_error_queue(print_output)
    
    def return_instrument_parameters(self, print_output=False):
        return super().return_instrument_parameters(print_output)
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~  Instr Parameters  ~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~    Just getters and setters!!  ~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # should begin with "get" or "set" so that other
    #   methods can find them easily, for example, 
    #   the `return_instrument_parameters` method 
    #   iterating and calling all "get_" methods

    # examples: 
    
    def get_IF_bandwidth(self, fmt=float):
        return ...
    
    def set_IF_bandwidth(self, IFBW):
        return ...
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~   Instr Methods   ~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~    single responsibility methods; if  ~~~~
    # ~~~    you try to describe its purpose    ~~~~
    # ~~~    it must not use the word "and"!    ~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # more complicated than just editing a parameter,
    #   but should still be as simple as possible,
    #   and ***should not depend on other methods***
    #   these are self contained!!
    
    # examples: 

    def send_marker_to_max(self):
        return ...
    
    def trigger_sweep(self):
        return ...
    
    def toggle_averaging(self, avg_mode):
        return ...


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~   Instr Routines   ~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  a routine is made of several single  ~~~~
    # ~~~  single-responsibility methods, but   ~~~~
    # ~~~  still be as light as possible!       ~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # these essentially are instr methods that must use
    #   other instr methods, but are still too simple
    #   to require an entire Experiment. methods like
    #  "take a trace" or "find peaks" are good examples.
    
    # examples: 
    
    def take_single_trace(self, trace_num=1):
        self.set_IF_bandwidth(IFBW)
        self.toggle_averaging()
        self.trigger_sweep()
        traceData = self.query_check(f'TRAC:DATA? TRACE{trace_num}')
        
        return traceData
        






