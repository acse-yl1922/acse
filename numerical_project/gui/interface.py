import tkinter as tk
from armageddon import Planet, damage_zones
import os
import folium
import webbrowser


def plot_circle(lat, lon, radius, map=None, ):
    if not map:
        map = folium.Map(
            location=[lat, lon],
            zoom_start=5
        )
    folium.Circle([lat, lon], radius, fill=True,
                  fillOpacity=0.6).add_to(map)
    return map


class MyWindow:
    def __init__(self, win):
        self.a = tk.Label(window, text="Radius").grid(row=0, column=0)
        self.radius_field = tk.Entry(win)
        self.radius_field.grid(row=0, column=1)
        self.radius_field.insert(tk.END, 30)

        self.b = tk.Label(window, text="angle").grid(row=1, column=0)
        self.angle_field = tk.Entry(win)
        self.angle_field.grid(row=1, column=1)
        self.angle_field.insert(tk.END, 18.3)

        self.c = tk.Label(window, text="strength").grid(row=2, column=0)
        self.strength_field = tk.Entry(win)
        self.strength_field.grid(row=2, column=1)
        self.strength_field.insert(tk.END, 3413448)

        self.d = tk.Label(window, text="velocity").grid(row=3, column=0)
        self.velocity_field = tk.Entry(win)
        self.velocity_field.grid(row=3, column=1)
        self.velocity_field.insert(tk.END, 19.2e3)

        self.e = tk.Label(window, text="latitude").grid(row=4, column=0)
        self.lat_field = tk.Entry(win)
        self.lat_field.grid(row=4, column=1)
        self.lat_field.insert(tk.END, 52.79)

        self.f = tk.Label(window, text="longitude").grid(row=5, column=0)
        self.lon_field = tk.Entry(win)
        self.lon_field.grid(row=5, column=1)
        self.lon_field.insert(tk.END, -2.95)

        self.g = tk.Label(window, text="bearing").grid(row=6, column=0)
        self.bearing_field = tk.Entry(win)
        self.bearing_field.grid(row=6, column=1)
        self.bearing_field.insert(tk.END, 135)

        self.h = tk.Label(window, text="pressure").grid(row=7, column=0)
        self.pressure_field = tk.Entry(win)
        self.pressure_field.grid(row=7, column=1)
        self.pressure_field.insert(tk.END, 1e3)

        self.h = tk.Label(window, text="density").grid(row=8, column=0)
        self.density_field = tk.Entry(win)
        self.density_field.grid(row=8, column=1)
        self.density_field.insert(tk.END, 3300)

        self.b1 = tk.Button(win, text='Calculate', command=self.click).grid(
            row=10, column=1)

    def click(self):
        earth = Planet()
        radius = float(self.radius_field.get())
        angle = float(self.angle_field.get())
        strength = float(self.strength_field.get())
        density = float(self.density_field.get())
        velocity = float(self.velocity_field.get())
        lat = float(self.lat_field.get())
        lon = float(self.lon_field.get())
        bearing = float(self.bearing_field.get())
        pressure = float(self.pressure_field.get())
        res = earth.solve_atmospheric_entry(
            radius, velocity, density, strength, angle)
        res = earth.calculate_energy(res)
        res = earth.analyse_outcome(res)
        print(res)
        outcome = {
            'burst_altitude': 8e3, 'burst_energy': 7e3,
            'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
            'outcome': 'Airburst'}
        print(outcome)
        # blat, blon, rad = damage_zones(
        #     outcome, 52.79, -2.95, 135,
        #     pressures=[1e3, 3.5e3, 27e3, 43e3],
        # )
        blat, blon, rad = damage_zones(
            res, lat, lon, bearing,
            [pressure],
        )
        print(blat, blon, rad)
        # blat, blon, rad = damage_zones(res, lat, lon, bearing, [pressure])
        map = plot_circle(blat, blon, rad[0])
        map_path = os.sep.join((
            os.path.dirname(__file__), 'map.html'))
        map.save(map_path)
        url = 'file:' + map_path
        print(url)
        res = webbrowser.open(url, new=2)  # open in new tab
        print('*', res)


window = tk.Tk()
mywin = MyWindow(window)
window.title('Impact Visualizer')
window.geometry("400x400+10+10")
window.mainloop()
