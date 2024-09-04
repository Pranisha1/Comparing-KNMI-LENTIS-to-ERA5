# Comparing-KNMI-LENTIS-to-ERA5


### Unit conversion 
* The KNMI LENTIS data for the precipitation is in the unit units: kg m-2 s-1 which has to be converted to mm/day to make it comparable
* 1 kg of water over 1 square meter (kg m-2) equals 1 mm of water depth and  86400 seconds daily.  Thus, multiply the precipitation rate in kg m-2 s-2 by 86400 to convert it to mm/day.


##### What has been done in this code?
* The NetCDF file for both the ERA5 and LENTIS has been loaded. LENTIS contains 16 members that have to be loaded.
* Several functions were defined to convert LENTIS dataset to same unit as the one in ERA5 (Convert precipitation from kg m-2 s-1 to mm/day and temp from K to Celcius)
* Functions for RMSE and bias were also defined
* Selecting the common lat and lon, the data were stored in df and then seasons were assigned based on the months
* After that long-term monthly mean/sum, yearly sum/mean as such df were compared between two datasets.
* Mainly I have codes produced to compare the results in each grid cell and plot line chart, bar plots, histogram, etc.
