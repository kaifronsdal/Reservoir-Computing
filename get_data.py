import requests
import numpy as np
import os
import wfdb


# Retrieves data for up to 8 axons
def retrieve_data(num_axons=8):
    for i in range(num_axons):
        # Links to raw data and header files
        dat_link = f'https://physionet.org/files/sgamp/1.0.0/raw/a{i + 1}t01.dat?download'
        hea_link = f'https://physionet.org/files/sgamp/1.0.0/raw/a{i + 1}t01.hea?download'

        # Download raw data and header files for all axons
        open(f'raw_data/a{i + 1}t01.dat', "wb").write(requests.get(dat_link).content)
        open(f'raw_data/a{i + 1}t01.hea', "wb").write(requests.get(hea_link).content)

        # record = wfdb.rdrecord(f'raw_data/a{i + 1}t01')
        # Read data and header files into wfdb record and convert to numpy array
        signals, fields = wfdb.rdsamp(f'raw_data/a{i + 1}t01')
        processed_data = np.array(signals)

        # Save processed data in csv files and clean-up data directory
        np.savetxt(f"raw_data/axon_{i + 1}_t01_data.csv", processed_data, delimiter=",")
        os.system(f"rm raw_data/a{i + 1}t01.*")


# Warning: this line will download nearly 600 MB of data!
retrieve_data(num_axons=4)
