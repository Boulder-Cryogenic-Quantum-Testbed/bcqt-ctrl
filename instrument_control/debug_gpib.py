import pyvisa

rm = pyvisa.ResourceManager()


def sweep_gpib_query(sweep_list, query, verbose=False):
    
    if verbose is True:
        print(f"Sweeping GPIB = {sweep_list} for {query = }")
    for idx in sweep_list:
        try:
            gpib_addr = f"GPIB::{idx}::INSTR"
            resource = rm.open_resource(gpib_addr)
            result = resource.query(query)
        except:
            print(f"{gpib_addr} \n    Failed")
            continue
        
        print(f"{gpib_addr} \n    {result}")
        
        