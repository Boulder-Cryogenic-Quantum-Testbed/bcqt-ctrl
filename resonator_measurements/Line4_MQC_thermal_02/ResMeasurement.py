import pandas as pd
from glob import glob

# class method: create x ResMeasurement objects from a single directory?
# this code takes the all_datasets dictionary and counts unique resonator folders
#
#   all_resonators = []
#   for dataset_directory, all_res_folders in all_datasets.items():
#       for res_folder in all_res_folders:
#           if res_folder not in all_resonators:
#               all_resonators.append(res_folder)
#

class ResMeasurement:
    def __init__(self, dataset_path):
    
        self.dataset_path = dataset_path
        self.find_csvs(dataset_path)
        self.load_csvs_as_dataframes()
    
    
    def find_csvs(self, dataset_path, glob_string="*.csv"):
        
        # get the filepath of where all the csv files are 
        csv_filenames = sorted([x for x in glob(f"{dataset_path}/{glob_string}") if "qiqc" not in x])
        
        self.all_csvs = csv_filenames
    
    
    def load_csvs_as_dataframes(self):
        
        all_csv_files = self.all_csvs
        all_dataframes = {}
        for csv_file in all_csv_files:
            print(f"        {csv_file}")
            
            power_df = pd.read_csv(csv_file, names=["Frequency", "Magnitude", "Phase"])
            all_dataframes[csv_file] = power_df
            
        self.all_dataframes = all_dataframes
    
    def get_info(self):
        
        info_dict{
            
            "dataframe_keys" : 
            
            
        }
        return info_dict
    
    def load_data_from_metadata(self):
        pass
    
    
    def get_timestamps(self):
        timestamps = [os.path.basename(key) for key in self.all_datasets.keys()]
        return timestamps
    
    
    def add_dataset(self):
        return
        
        
    def get_datasets(self):
        return self.all_datasets
    
    
    def list_datasets(self):
        pass
        
        


if __name__ == "__main__":
    %load_ext autoreload
    %autoreload 2

    dir_path = f"all_datasets"

    # res_test = ResMeasurement()
    # res_test.load_data_from_timestamped_folders(dir_path)
    
    # display(res_test.all_datasets)
    
    # display(res_test.get_timestamps())
    
    
    all_timestamp_paths = sorted([x for x in glob(f"{dir_path}/*") if 'all_plots' not in x])
    
    
    all_datasets = {}
    for timestamp_folder in all_timestamp_paths:
        # all_datasets => "timestamp_folder" : [ array of resonator folders ]
        
        # use os.path.basename() to keep just the folder name, since that is equal to 
        # `timestamp_folder` by construction
        all_datasets[timestamp_folder] = sorted([os.path.basename(x) for x in glob(f"{timestamp_folder}/*GHz")])
        
    
    # for timestamp_folder, all_measurements in all_datasets.items():
    #     print(f"Measurement on {os.path.basename(timestamp_folder)} - {len(all_measurements)} scan(s)")
        
    #     for res_folder in all_measurements:
    #         print(f"    Resonator: {res_folder}")
            
    #         dataset_path = f"{timestamp_folder}/{res_folder}"
    #         power_datasets = sorted([x for x in glob(f"{dataset_path}/*.csv") if "qiqc" not in x])
    #         csv_filenames = [os.path.basename(x) for x in power_datasets]
            
                
    #     print()
    
    all_res_objs = []
    
    for timestamp_folder, all_measurements in all_datasets.items():
        print()
        pass
    
        for res_folder in all_measurements:
            print(f"{timestamp_folder}/{res_folder}")
            
            # get the filepath of where all the csv files are 
            dataset_path = f"{timestamp_folder}/{res_folder}"
            # csv_filenames = sorted([x for x in glob(f"{dataset_path}/*.csv") if "qiqc" not in x])
            
            # create ResMeasurement here
            ResonatorObj = ResMeasurement(dataset_path)
            
            # for csv_file in csv_filenames:
            #     pass
            all_res_objs.append(ResonatorObj)

    # for timestamp_folder, all_measurements in all_datasets.items():

    
    for res_obj in all_res_objs:
        print(res_obj.get_info())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            