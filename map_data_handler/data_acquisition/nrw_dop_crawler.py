"""
Create simple dataset from DOP images

TODO: this needs some cleanup
@author Frank Ehebrecht
"""

import os
import time
import argparse

import geopandas as gpd
import h5py
import numpy as np

from map_data_handler.data_acquisition.nrw_wms_geo_patch import NrwWmsPatchProvider

ROOF_ENCODE = {'Spitzdach': 0, 'Flachdach': 1}


def load_geo_dataframe(path, sample_size):
    df = gpd.read_file(path)
    df_sample = df.sample(n=sample_size)
    df_reset = df_sample.reset_index()
    print(len(df_reset))
    return df_reset.to_crs(4326)


def crawl(shape_path, base_out_path, sample_size):
    df = load_geo_dataframe(shape_path, sample_size)
    provider = NrwWmsPatchProvider(resolution=20)
    print('Starting')
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f'Index: {idx}')
        out_path = os.path.join(base_out_path, f'{idx:08d}.h5')
        poly_ = row['geometry']

        try:
            geo_patch = provider.from_polygons_and_size(poly_, (512, 512))
            X = np.concatenate((geo_patch.patch, geo_patch.hull_mask[..., np.newaxis]), axis=2)
            attrs = row.drop('geometry').to_dict()
            with h5py.File(out_path, 'w') as h5f:
                h5f.create_dataset("X", data=X)
                for key_, val_ in attrs.items():
                    h5f['X'].attrs[key_] = val_
        except Exception as e:
            print(f'ERROR at {idx} | {str(repr(e))}')

        time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape_path', type=str)
    parser.add_argument('--out_path', 'str')
    parser.add_argument()
    args = parser.parse_args()
    shape_path_ = args.shape_path
    out_path_ = args.out_path

    sample_size_ = 8000
    crawl(shape_path_, out_path_, sample_size_)
