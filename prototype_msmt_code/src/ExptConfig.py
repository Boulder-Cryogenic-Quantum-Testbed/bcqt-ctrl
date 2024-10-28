
# Experiment Configuration Class
class ExptConfig():
    def __init__(self, ExptConfig_Dict):
        print("ExptConfig")
        
        self.experiment_name = ExptConfig_Dict["experiment_name"]
        self.ExptConfig_Dict = ExptConfig_Dict

    def __del__(self):
        """
            Deconstructor to free resources
        """
        if self.rm:
            self.rm.close()
            
    def print_class_members(self):
        """
            Prints all members in the class
        """
        for k, v in self.__dict__.items():
            print(f'{k} : {v}')
            
    def load_config(self):
        print(f"Loading experiment configuration for (placeholder)...")

    def save_config(self):
        print(f"Saving experiment configuration for (placeholder)...")


    def add_parameter(self, parameter):
        """
            provide the new parameter as a dict or tuple with a name and value
                parameter = {"name" : value}
                    or
                parameter = (name, value)
        """
        
        if type(parameter) == dict:
           self.ExptConfig_Dict = self.ExptConfig_Dict | parameter
           
        elif type(parameter) == tuple:
            key, value = parameter
            self.ExptConfig_Dict[key] = value
        else:
            raise TypeError("\n\n    Input parameter must be a dict or tuple.\n\n")
        
        

