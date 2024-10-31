
# DataHandler Class
class DataHandler:
    def __init__(self):
        self.data_storage = []

    def load_data(self, data):
        print("Loading data into storage...")
        self.data_storage.append(data)

    def save_data(self):
        print("Saving data to file or database...")

    def get_data(self):
        return self.data_storage
