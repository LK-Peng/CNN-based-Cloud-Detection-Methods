import numpy as np
from osgeo import gdal


def save_img(tiff, out_file, proj=None, geot=(0, 30, 0, 0, 0, 30)):
    """ save tiff image """
    NP2GDAL_CONVERSION = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
    }  # convert np to gdal
    gdal_type = NP2GDAL_CONVERSION[tiff.dtype.name]
    if len(tiff.shape) == 2:
        tiff = np.expand_dims(tiff, axis=0)
    channel, row, col = tiff.shape
    # create data set
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(out_file, col, row, channel, gdal_type)
    if proj is not None:
        out_ds.SetProjection(proj)  # projection
    if geot is not None:
        out_ds.SetGeoTransform(geot)  # geotransform
    # write
    for iband in range(channel):
        out_ds.GetRasterBand(iband+1).WriteArray(tiff[iband, :, :])
    del out_ds
