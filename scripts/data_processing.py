import pandas as pd

#For data processing





def load_processing(csv_file):
    data = pd.read_csv(csv_file)
    data = data[data['political' != 99]]
    data = data[data['political'].notna()]
    data = data[data['political'].astype(int64)]
    
    if [99, nan] in data['political'].unqiue():
        raise Exception("The processing of political category failed")
    
    data = data[data['domestic' != 99]]
    data = data[data['domestic'].notna()]
    data = data[data['domestic'].astype(int64)]
 
    if [99, nan] in data['domestic'].unqiue():
        raise Exception("The processing of domestic category failed")
  
    data = data['descriptions'].fillna()

    processed_data = data

    return processed_data

