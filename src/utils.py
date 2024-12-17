import planetary_computer
from pystac_client import Client
import stackstac
import xarray as xr
import rioxarray as rio
import numpy as np




class SentinelDownloader:
    def __init__(self,year,boundary,epsg):
        self.epsg = epsg
        self.year = year
        self.boundary = boundary
        self.cloud_masked_and_scaled = self.download_data()
        self.complete_data = xr.DataArray()
        self.missing_data = list()
        self.indices = xr.DataArray()

    def download_data(self):
        bbox_4326 = tuple(self.boundary.to_crs(4326).total_bounds)
        bbox_utm = tuple(self.boundary.total_bounds)

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        items = catalog.search(
            bbox=bbox_4326,
            collections=["sentinel-2-l2a"],
            datetime=f"{self.year}-01-01/{self.year}-12-31"
        ).item_collection()
        len(items)

        # create xarray
        stack = stackstac.stack(
            items,
            epsg=self.epsg,
            resolution=10,
            bounds=bbox_utm,
            assets=['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']).where(lambda x: x > 0, other=np.nan)

        # filter for images < 20% cloud cover
        lowcloud = stack[stack["eo:cloud_cover"] < 20]
        # print number of filtered images
        print('number of low cloud images:',lowcloud.shape[0])

        scl_band = lowcloud.sel(band='SCL')
        masked = lowcloud.where((scl_band != 9) & (scl_band !=3) & (scl_band !=8) & (scl_band != 10))

        # drop SCL band
        masked = masked.drop_sel(band='SCL')

        if (self.year) >= 2022:
            scaled = (masked - 1000) / 10000
        else:
            scaled = masked / 10000

        drop_duplicates = scaled.drop_duplicates(dim='time',keep='first')
            
        return drop_duplicates

    def plot_initial_data(self):
        self.cloud_masked_and_scaled.isel(band=3).plot(col='time',col_wrap=6,robust=True)

    def drop_missing_data(self):
        to_drop = self.cloud_masked_and_scaled.time.values[self.missing_data]
        self.complete_data = self.cloud_masked_and_scaled.drop_sel({'time':to_drop})
        
    def plot_complete_data(self):
        self.complete_data.isel(band=3).plot(col='time',col_wrap=6,robust=True)

    def save_bands_data(self,root,filename):
        s = self.complete_data.reset_coords(drop=True)
        s.to_netcdf(root / 'sentinel_data' / f'{filename}.nc')
    
    def save_vi_data(self,root,filename):
        self.indices.to_netcdf(root / 'sentinel_data' / f'{filename}_indices.nc')

    def get_indices(self):
        blue = self.complete_data.sel(band='B02')
        red = self.complete_data.sel(band='B04')
        nir = self.complete_data.sel(band='B8A')
        sw1 = self.complete_data.sel(band='B11')
        sw2 = self.complete_data.sel(band='B12')
        re2 = self.complete_data.sel(band='B06')

        ndvi = (nir-red)/(red+nir).expand_dims({'band':['ndvi']})

        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)).expand_dims({'band':['evi']})
        
        lswi = (nir - sw1)/(nir + sw1).expand_dims({'band':['lswi']})
        
        slavi = nir/(red + sw2).expand_dims({'band':['slavi']})
        
        psri = (red - blue)/re2.expand_dims({'band':['psri']})
        
        all = xr.concat([ndvi,evi,lswi,slavi,psri],dim='band').reset_coords(drop=True)
        
        self.indices = all.where(all.where(np.isfinite(all),np.nan))