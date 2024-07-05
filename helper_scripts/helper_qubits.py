# %%

'''
    helper_misc.py
'''

# print("    loading helper_misc.py")

# %%
import sys
sys.path.append(r"E:\GitHub\bcqt-helpers")


# %%
def prep_cfgs(all_cfgs):
    """ 
    del_target is the cfg dict that needs to be reset, and then all_dicts will be formed to replace it
    all_dicts should be a list of dictionaries
    """
    # try: del del_target 
    # except: print("ResFreqQubitFreq_config does not exist, proceeding with initializing cfg.")  
    
    # all_dicts is a list of dicts so we unpack 
    # all keywords and values with comprehension
    # final_cfg = {k:v for list_item in args for (k,v) in list_item.items()}
    # for key in all_cfgs:
        # print(key)

    final_cfg = {}
    for cfg_dict in all_cfgs:   
        # print(cfg_dict)    
              
        # add "f_start" to cfg if it's specified by center+span
        for key, val in cfg_dict.items():
            if "center" in key:
                # print(key)
                span_key = key[0:2] + "span"
                start_key = key[0:2] + "start"
                step_key = key[0:2] + "step"
                center_key = key
                # print(start_key)
                start = cfg_dict[center_key] + cfg_dict[span_key] + cfg_dict[step_key]
                final_cfg[start_key] = start

        final_cfg.update(cfg_dict)

    return final_cfg
    