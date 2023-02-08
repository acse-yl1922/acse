import pandas as pd
import numpy as np
import folium
import folium.plugins
import matplotlib
import matplotlib.pyplot as plt
import geojsoncontour
from .live import get_stationReference_and_position
from scipy.interpolate import griddata
from .visualizationpredictions import VisualizationPredictions as vp

def convert_numeric(data):
    """convert string data to float"""
    idx = data['value'][data['value'].isna()].index
    data = data.drop(index = idx)
    data.value = data.value.astype(str)
    idx = data['value'][data.value.str.contains('\|')].index
    for i in idx:
        data.loc[i,'value'] = (float(data.loc[i,'value'].split('|')[0])+float(data.loc[i,'value'].split('|')[1]))/2
    data['value'] = pd.to_numeric(data['value'])
    return data

def transform_unit(data):
    """ transform unite, only necessary for tides data"""
    data['value'] = data.apply(lambda row:row.value/1000 if row.unitName == 'mm' else row.value, axis = 1)
    return data
class Visualisation():
    """ 
    Class providing functions for plotting postcode density, flood risk value, 
    rainfall, river and tides condition in UK.
    """

    def __init__(
        self, household = 'flood_tool/resources/households_per_sector.csv',
        typical_day = 'flood_tool/resources/typical_day.csv',
        wet_day = 'flood_tool/resources/wet_day.csv'):
        
        """
        Store all the needed data for plotting. 
        Call the getReferenceandPosition function to fill missing value of latitude and longitude.
        
        Parameters
        ----------
        household: household dataframe
        typical_day: rainfall data and tide data of typical day
        wet_day: rainfall data and tide data of wet day

        """

        self.household = pd.read_csv(household)
        self.typical_day = pd.read_csv(typical_day)
        self.wet_day = pd.read_csv(wet_day)
        self.wet_day = convert_numeric(self.wet_day)
        self.url = "https://environment.data.gov.uk/flood-monitoring/id/stations/"
        self.station_data = get_stationReference_and_position(self.url)
        self.typical_day["lat"] = self.typical_day["stationReference"].map(dict(zip(self.station_data["stationReference"],self.station_data["lat"])))
        self.typical_day["lon"] = self.typical_day["stationReference"].map(dict(zip(self.station_data["stationReference"],self.station_data["long"])))
        self.wet_day["lat"] = self.wet_day["stationReference"].map(dict(zip(self.station_data["stationReference"],self.station_data["lat"])))
        self.wet_day["lon"] = self.wet_day["stationReference"].map(dict(zip(self.station_data["stationReference"],self.station_data["long"])))

    def plotfile(self,datapath):

        """
        Call all the plotting methods and add plot to the inner map.

        Parameters
        ----------
        datapath: data to predict and plot risk level of flood.

        Example
        -------
        >>> vis = Visualisation()
        >>> vis.plotfile('./resources/postcodes_unlabelled.csv')

        Returns
        -------
        map
            the map that contains all the plot
        """

        self.data = vp(datapath).get_predictions()
        self.lat = self.data['latitude']
        self.lon = self.data['longitude']
        self.map = folium.Map([self.lat.mean(), self.lon.mean()], min_lon = -10, max_lon = 4, min_lat = 40, max_lat = 70, zoom_start = 8, max_bounds = True, min_zoom = 5.5)
        self.draw_postcode()
        self.draw_risk_label()
        self.draw_rainfall_typical()
        self.draw_rainfall_wet()
        self.draw_river_stage_wet()
        self.draw_river_stage_typical()
        self.draw_tide_wet()
        self.draw_tide_typical()
        self.draw_live_data()
        return self


    def rgb2hex(self,color):
        """
        convert colormap to a color string for distinguishing values of different points.

        Parameters
        ----------
        color: array([r,g,b,a]); rgb and alpha value

        Returns
        -------
        string
            color string
        """
        c = [int(i*255) for i in color]
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    def draw_postcode(self):
        """
        draw the density map of postcodes of input data

        """
        density_data = [[i,j] for i, j in zip(self.lat,self.lon)]
        folium.plugins.HeatMap(name = "Postcode density",
                       blur = 5,
                       radius = 5,
                       data = density_data,
                       show=False).add_to(self.map)
        
    def draw_risk_label(self):
        """
        show risk value of flood by circles. color and size of circles represent the level of risk

        """
        prob_map = dict(zip([i for i in range(1,11)],[0.0001,0.0005,0.001,0.005,0.01,0.015,0.02,0.03,0.04,0.05]))
        self.household["sector"] = self.household["postcode sector"].apply(lambda x: " ".join(x.split()))
        self.data["sector"] = self.data["sector"].apply(lambda x: " ".join(x.split()))
        data_merged = pd.merge(self.data,self.household,on = 'sector')
        data_merged['risk'] = data_merged.apply(lambda row:0.05*prob_map[row.riskLabel]*row.medianPrice*(row.households/row['number of postcode units']),axis = 1)
        #data_merged['latitude'] = geo.get_gps_lat_long_from_easting_northing(data_merged['easting'],data_merged['northing'])[0]
        #data_merged['longitude'] = geo.get_gps_lat_long_from_easting_northing(data_merged['easting'],data_merged['northing'])[1]
        featuregroup = folium.FeatureGroup(name = 'Risk of flood')
        norm = matplotlib.colors.Normalize(vmin = data_merged["risk"].min(),vmax = data_merged["risk"].max()) 
        clist = ['green','yellow','orange','red']
        newcmp = matplotlib.colors.LinearSegmentedColormap.from_list("new_cmap",clist)
        for i,j in data_merged.iterrows():
            featuregroup.add_child(folium.Circle([j['latitude'],j['longitude']],radius = norm(j['risk']+0.1)*1000,opacity = 0.4,fill = True,color = self.rgb2hex(newcmp(norm(j['risk']))),fillColor = self.rgb2hex(newcmp(norm(j['risk']))),fill_opacity = 0.6,weight = 1,tooltip = f"risk: {j['risk']:.2f}"))
        featuregroup.add_to(self.map)

    def draw_rainfall_typical(self):
        """
        plot contour map of rainfall condition for a typical day

        """

        rainfall_data = self.typical_day[self.typical_day["parameter"]=="rainfall"]
        rainfall_data = rainfall_data[rainfall_data["lat"].notnull()]
        rainfall_data_sorted = rainfall_data.sort_values(by=["stationReference","dateTime"]).reset_index(drop=True)
        rain_data_mean = rainfall_data_sorted.groupby(by="stationReference").mean(numeric_only = True)
        rain_data_mean["value"] = rain_data_mean["value"]*4
        rain_data_mean = rain_data_mean.reset_index()
        lat = rain_data_mean["lat"]
        lon = rain_data_mean["lon"]
        value = rain_data_mean["value"]
        x = np.arange(min(lon),max(lon),0.05)
        y = np.arange(min(lat),max(lat),0.05)
        X,Y = np.meshgrid(x,y)
        Z = griddata((lon, lat), value, (X, Y), method='cubic')
        Z[Z<0]=0
        contourf = plt.contourf(X,Y,Z,cmap="viridis_r")
        plt.close()
        geojson = geojsoncontour.contourf_to_geojson(
        contourf = contourf,
        min_angle_deg = 0,
        ndigits = 5,
        stroke_width = 1,
        fill_opacity = 0.5)
        folium.GeoJson(
                geojson,
                name="Typical-day rainfall",
                style_function=lambda x: {
                'color':     x['properties']['stroke'],
                'weight':    x['properties']['stroke-width'],
                'fillColor': x['properties']['fill'],
                'opacity': 1,
                'fillOpacity': 0.5},
                tooltip=folium.features.GeoJsonTooltip(
                fields = ['title'],
                aliases=['rainfall:']),
                show=False
                ).add_to(self.map)

    def draw_rainfall_wet(self):
        """
        plot contour map of rainfall condition for a wet day

        """
        self.wet_day = convert_numeric(self.wet_day)
        rainfall_data = self.wet_day[self.wet_day["parameter"]=="rainfall"]
        rainfall_data = rainfall_data[rainfall_data["lat"].notnull()]
        lat = rainfall_data["lat"]
        lon = rainfall_data["lon"]
        value = rainfall_data["value"]
        x = np.arange(min(lon),max(lon),0.05)
        y = np.arange(min(lat),max(lat),0.05)
        X,Y = np.meshgrid(x,y)
        Z = griddata((lon, lat), value, (X, Y), method='cubic')
        Z[Z<0] = 0
        contourf = plt.contourf(X,Y,Z,cmap="viridis_r")
        plt.close()
        geojson = geojsoncontour.contourf_to_geojson(
                contourf = contourf,
                min_angle_deg = 0,
                ndigits = 5,
                stroke_width = 1,
                fill_opacity=0.5)

        folium.GeoJson(
                geojson,
                name = "Wet-day rainfall",
                style_function = lambda x: {
                'color':     x['properties']['stroke'],
                'weight':    x['properties']['stroke-width'],
                'fillColor': x['properties']['fill'],
                'opacity': 1,
                'fillOpacity': 0.5},
                tooltip = folium.features.GeoJsonTooltip(
                fields = ['title'],
                aliases = ['rainfall:']),
                show=False
                ).add_to(self.map)

    def draw_river_stage_wet(self):
        """
        plot river condition for a wet day by circles. Color and size of circles represent the height of river stage.

        """
        self.wet_day = convert_numeric(self.wet_day)
        self.wet_day = transform_unit(self.wet_day)
        river_data = self.wet_day[self.wet_day["qualifier"]=="Stage"]
        river_data = river_data[river_data["lat"].notnull()]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","green","orange","red"])
        norm=matplotlib.colors.Normalize(vmin=river_data["value"].min(),vmax=river_data["value"].max())
        river_g = folium.FeatureGroup(name="Wet-day river stage",show=False)
        for i, row in river_data.iterrows():
            river_g.add_child(folium.Circle(location=(row.lat,row.lon),
                             tooltip = (f"stationReference: {row.stationReference}<br>"
                                        f"value: {row.value} {row.unitName}"),
                             Opacity = 1,
                             weight = 1,
                             radius = (norm(abs(row.value))+0.1)*2000,
                             fill = True,
                             color = self.rgb2hex(cmap(norm(row.value))),
                             fillColor = self.rgb2hex(cmap(norm(row.value))),
                             fill_opacity=0.6))
        river_g.add_to(self.map)

    def draw_river_stage_typical(self):
        """
        draw river condition for a typical day. Color and size of circles represent the height of river stage.

        """
        river_data=self.typical_day[self.typical_day["qualifier"]=="Stage"]
        river_data=river_data[river_data["lat"].notnull()]
        river_data=river_data[river_data["value"]<1000]
        river_data=river_data.sort_values(by="value",ascending=False,key=abs)
        r1=river_data[(river_data["value"]>=2)|(river_data["value"]<=-2)]
        r2=river_data[(river_data["value"]<2)&(river_data["value"]>-2)].sample(5000)
        river_data=pd.concat([r1,r2],axis=0)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","green","orange","red"])
        norm=matplotlib.colors.Normalize(vmin=river_data["value"].min(),vmax=river_data["value"].max())
        from folium.vector_layers import Circle
        river_g = folium.FeatureGroup(name="Typical-day river stage",show=False)
        for i, row in river_data.iterrows():
            river_g.add_child(folium.Circle(location=(row.lat,row.lon),
                             tooltip=(f"stationReference: {row.stationReference}<br>"
                                      f"value: {row.value} {row.unitName}"),
                             Opacity = 0.5,
                             weight = 1,
                             radius = (norm(abs(row.value))+0.1)*2000,
                             fill = True,
                             color = self.rgb2hex(cmap(norm(row.value))),
                             fillColor = self.rgb2hex(cmap(norm(row.value))),
                             fill_opacity = 0.5))
        river_g.add_to(self.map)

    def draw_tide_wet(self):
        """
        draw tide data for a wet day. Color and size of circles represent the level of tide.

        """
        tide_data = self.wet_day[self.wet_day["qualifier"] == "Tidal Level"]
        tide_data = tide_data[tide_data["lat"].notnull()]
        tide_data = transform_unit(tide_data)
        tide_data = tide_data.groupby(by="stationReference").max().reset_index()
        tide_data = tide_data.sort_values(by="value",ascending=False)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","green","orange","red"])
        norm = matplotlib.colors.Normalize(vmin=tide_data["value"].min(),vmax=tide_data["value"].max())
        tide_g = folium.FeatureGroup(name="Wet-day tidal level",show=False)
        for i, row in tide_data.iterrows():
            tide_g.add_child(folium.Marker(location = [row.lat,row.lon],
                                    icon = folium.plugins.BeautifyIcon(
                                    icon_shape ='rectangle-dot',
                                    border_width = 5,
                                    border_color = self.rgb2hex(cmap(norm(row.value))),
                                    background_color = self.rgb2hex(cmap(norm(row.value))),
                                    icon_size = [int(row.value)*2,int(row.value)*2]),
                                    tooltip = (f"stationReference: {row.stationReference}<br>"
                                           f"tidal level: {row.value} {row.unitName}"),
                                    opacity = 0.6
                                  ))
                                     
        tide_g.add_to(self.map)

    def draw_tide_typical(self):
        """
        draw tide data for a typical day. Color and size of circles represent the level of tide.

        """
        tide_data = self.typical_day[self.typical_day["qualifier"] == "Tidal Level"]
        tide_data = tide_data[tide_data["lat"].notnull()]
        tide_data = transform_unit(tide_data)
        tide_data = tide_data.groupby(by="stationReference").max().reset_index()
        tide_data = tide_data.sort_values(by="value",ascending=False)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","green","orange","red"])
        norm = matplotlib.colors.Normalize(vmin=tide_data["value"].min(),vmax=tide_data["value"].max())
        tide_g = folium.FeatureGroup(name="Typical-day tidal level",show=False)
        for i, row in tide_data.iterrows():
            tide_g.add_child(folium.Marker(location = [row.lat,row.lon],
                                    icon = folium.plugins.BeautifyIcon(
                                    icon_shape ='rectangle-dot',
                                    border_width = 5,
                                    border_color = self.rgb2hex(cmap(norm(row.value))),
                                    background_color = self.rgb2hex(cmap(norm(row.value))),
                                    icon_size = [int(row.value)*2,int(row.value)*2]),
                                    tooltip = (f"stationReference: {row.stationReference}<br>"
                                           f"tidal level: {row.value} {row.unitName}"),
                                    opacity = 0.6
                                  ))
        tide_g.add_to(self.map)
    
    def draw_live_data(self):
        """draw the rainfall, river and tide live data on the map

        Returns
        -------
        adding layers to the map
        """
        
        data=pd.read_csv("flood_tool/resources/latest_live_data.csv")
        data=data.dropna()
        
        rainfall_data=data[data["parameter"]=="rainfall"]
        lat=np.array([int(x) for x in rainfall_data["latitude"]])
        lon=np.array([int(x) for x in rainfall_data["longitude"]])
        value=rainfall_data["value"]
        x=np.arange(min(lon),max(lon),0.05)
        y=np.arange(min(lat),max(lat),0.05)
        X,Y=np.meshgrid(x,y)
        Z = griddata((lon, lat), value, (X, Y), method='cubic')
        Z[Z<0]=0
        contourf=plt.contourf(X,Y,Z,cmap="viridis_r")
        plt.close()

        river_data=data[(data["qualifier"]=="Stage")]
        river_units=["m","mASD","mAOD"]
        river_data=transform_unit(river_data)
        river_data=river_data[river_data["unitName"].isin(river_units)]
        
        tide_data=data[(data["qualifier"]=="Tidal Level")]
        tide_data=transform_unit(tide_data)
        
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","green","orange","red"])
        norm_river=matplotlib.colors.Normalize(vmin=river_data["value"].min(),vmax=river_data["value"].max())
        norm_tide=matplotlib.colors.Normalize(vmin=tide_data["value"].min(),vmax=tide_data["value"].max())
        def rgb2hex(color):
            c=[int(i*255) for i in color]
            return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        
        geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=0.5)

        folium.GeoJson(
            geojson,
            name="live rainfall data",
            style_function=lambda x: {
                'color':     x['properties']['stroke'],
                'weight':    x['properties']['stroke-width'],
                'fillColor': x['properties']['fill'],
                'opacity': 1,
                'fillOpacity': 0.5},
            tooltip=folium.features.GeoJsonTooltip(
                    fields=['title'],
                    aliases=['rainfall:']),
            show=False
        ).add_to(self.map)

        river_g = folium.FeatureGroup(name="live river stage",show=False)
        for i, row in river_data.iterrows():
            river_g.add_child(folium.Circle(location=(row.latitude,row.longitude),
                                    tooltip=(f"stationReference: {row.stationReference}<br>"
                                            f"value: {row.value} {row.unitName}"),
                                    Opacity=1,
                                    weight=1,
                                    radius=(norm_river(abs(row.value))+0.1)*1000,
                                    fill=True,
                                    color=rgb2hex(cmap(norm_river(row.value))),
                                    fillColor=rgb2hex(cmap(norm_river(row.value))),
                                    fill_opacity=0.6))
        river_g.add_to(self.map)

        tide_g = folium.FeatureGroup(name="live tidal level",show=False)
        for i, row in tide_data.iterrows():
            tide_g.add_child(folium.Marker(location=[row.latitude,row.longitude],
                                        icon=folium.plugins.BeautifyIcon(
                                            icon_shape='rectangle-dot',
                                            border_width=5,
                                            border_color=rgb2hex(cmap(norm_tide(row.value))),
                                            background_color=rgb2hex(cmap(norm_tide(row.value))),
                                            icon_size=[int(row.value)*2,int(row.value)*2]),
                                        tooltip=(f"stationReference: {row.stationReference}<br>"
                                                f"tidal level: {row.value} {row.unitName}"),
                                        opacity=0.6
                                        ))
                                            
        tide_g.add_to(self.map)
        folium.LayerControl().add_to(self.map)