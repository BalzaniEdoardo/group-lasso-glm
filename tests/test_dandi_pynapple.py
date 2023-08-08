import pynwb

from pynwb import NWBHDF5IO, TimeSeries

from dandi.dandiapi import DandiAPIClient
import pynapple as nap
import numpy as np
import fsspec
from fsspec.implementations.cached import CachingFileSystem

import pynwb
import h5py

from matplotlib.pyplot import *

#####################################
# Dandi
#####################################

# ecephys, Buzsaki Lab (15.2 GB)
dandiset_id, filepath = "000070", "sub-Jenkins/sub-Jenkins_ses-20090912_behavior+ecephys.nwb"


with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)




# first, create a virtual filesystem based on the http protocol
fs=fsspec.filesystem("http")

# create a cache to save downloaded data to disk (optional)
fs = CachingFileSystem(
    fs=fs,
    cache_storage="nwb-cache",  # Local folder for the cache
)

# next, open the file
file = h5py.File(fs.open(s3_url, "rb"))
io = pynwb.NWBHDF5IO(file=file, load_namespaces=True)


#####################################
# Pynapple 
#####################################

nwb = nap.NWBFile(io.read())

units = nwb["units"]

trials = nwb["trials"]

hand = nwb["Hand"]

print(nwb)

#####################################
# GLM
#####################################
