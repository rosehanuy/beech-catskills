import planetary_computer
from pystac_client import Client
import stackstac
import xarray as xr
import rioxarray as rio
import numpy as np
import pandas as pd
import geopandas as gpd


class SentinelDownloader:
    def __init__(self,root,year,site_name,boundary,epsg):
        self.root = root
        self.site_name = site_name
        self.epsg = epsg
        self.year = year
        self.boundary = boundary
        self.cloud_masked_and_scaled = xr.DataArray()
        self.complete_data = xr.DataArray()
        self.missing_data = list()
        self.indices = xr.DataArray()
        self.indices2 = xr.DataArray()
        

    def download_data(self,drop):
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

        drop_duplicates = scaled.drop_duplicates(dim='time',keep=drop)
            
        self.cloud_masked_and_scaled = drop_duplicates

    def plot_initial_data(self):
        self.cloud_masked_and_scaled.isel(band=3).plot(col='time',col_wrap=6,robust=True)

    def drop_missing_data(self):
        to_drop = self.cloud_masked_and_scaled.time.values[self.missing_data]
        self.complete_data = self.cloud_masked_and_scaled.drop_sel({'time':to_drop})
        
    def plot_complete_data(self):
        self.complete_data.isel(band=3).plot(col='time',col_wrap=6,robust=True)

    def save_bands_data(self):
        s = self.complete_data.reset_coords(drop=True)
        s.to_netcdf(self.root / 'sentinel_data' / f'{self.year}_{self.site_name}.nc')
    
    def save_vi_data(self):
        self.indices.to_netcdf(self.root / 'sentinel_data' / f'{self.year}_{self.site_name}_indices.nc')

    def get_indices(self):
        blue = self.complete_data.sel(band='B02')
        red = self.complete_data.sel(band='B04')
        nir = self.complete_data.sel(band='B8A')
        sw1 = self.complete_data.sel(band='B11')
        sw2 = self.complete_data.sel(band='B12')
        re2 = self.complete_data.sel(band='B06')

        ndvi = (nir-red)/(red+nir).expand_dims({'band':['ndvi']})  # range -1 to 1
        ndvi = self.remove_outliers(ndvi)

        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)).expand_dims({'band':['evi']}) # range -1 - 1
        evi = self.remove_outliers(evi)
        
        lswi = (nir - sw1)/(nir + sw1).expand_dims({'band':['lswi']}) # range -1 to 1
        lswi = self.remove_outliers(lswi)

        slavi = nir/(red + sw2).expand_dims({'band':['slavi']}) # range 0 - 8
        slavi = self.remove_outliers(slavi,norm=False)
        
        psri = (red - blue)/re2.expand_dims({'band':['psri']}) # range -1 to 1
        psri = self.remove_outliers(psri)

        all = xr.concat([ndvi,evi,lswi,slavi,psri],dim='band').reset_coords(drop=True)
        
        self.indices = all

    def get_indices2(self):
        blue = self.complete_data.sel(band='B02')
        red = self.complete_data.sel(band='B04')
        nir = self.complete_data.sel(band='B8A')
        sw1 = self.complete_data.sel(band='B11')
        sw2 = self.complete_data.sel(band='B12')
        re2 = self.complete_data.sel(band='B06')

        ndvi = (nir-red)/(red+nir).expand_dims({'band':['ndvi']})  # range -1 to 1
        ndvi = self.remove_outliers(ndvi)

        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)).expand_dims({'band':['evi']}) # range -1 - 1
        evi = self.remove_outliers(evi)
        
        lswi = (nir - sw1)/(nir + sw1).expand_dims({'band':['lswi']}) # range -1 to 1
        lswi = self.remove_outliers(lswi)

        slavi = nir/(red + sw2).expand_dims({'band':['slavi']}) # range 0 - 8
        slavi = self.remove_outliers(slavi,norm=False)
        
        psri = (red - blue)/re2.expand_dims({'band':['psri']}) # range -1 to 1
        psri = self.remove_outliers(psri)

        all = xr.concat([ndvi,evi,lswi,slavi,psri],dim='band').reset_coords(drop=True)
        
        self.indices = all

    def remove_outliers(self,a,norm=True):
        a = a.where(np.isfinite(a),np.nan)
        if norm == True:
            a = a.where((a >= -1) & (a <= 1))
        else:
            a = a.where((a >= 0) & (a <= 8))
        return a
    
    def check_data_range(self,a):
        for i in range(0,len(a.band),1):
            print(f'Band {i}: min: {np.nanmin(a.isel(band=i))}, max {np.nanmax(a.isel(band=i))}')


def read_neon_trees(root,site,epsg):
    # ref_data_path = root / 'DP3.30006.001' / 'neon-aop-products' / '2019' / 'FullSite' / 'D01' / site_folder / 'L3' / 'Spectrometer' / 'Reflectance'
    # read in tree location data
    
    trees_df = pd.read_csv(root / 'output' / site / f'neon_trees_{site}.csv')
    if site == 'BART':
        trees_df.taxonID = trees_df.taxonID.replace(to_replace='BEPA',value='BEPAP')
    trees_df.taxonID.value_counts()
    geometry = gpd.points_from_xy(trees_df.easting_tree, trees_df.northing_tree, crs=epsg)
    trees_df['geometry'] = geometry
    trees_gdf = gpd.GeoDataFrame(trees_df,geometry='geometry',crs=epsg)
    return trees_gdf
# assign colors for plotting points
# def get_hsv_colors():
#     # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
#     by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) 
#                     for name, color in mcolors.CSS4_COLORS.items())
#     return [name for _, name in by_hsv]

# col_list = get_hsv_colors()[64:]
# r = sample(range(0, len(col_list)),len(np.unique(bart_trees.taxonID)))
# color_list = [col_list[i] for i in r]
# ids = np.unique(bart_trees.taxonID)

# color_map = {id:r for id, r in zip(ids,color_list)}

# color_map['FAGR'] = 'orange'
# color_map['TSCA'] = 'red'