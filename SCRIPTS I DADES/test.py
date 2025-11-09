import folium
import pandas as pd

# Create Barcelona map with reduced background visibility
barcelona_map = folium.Map(
    location=[41.3851, 2.1734], 
    zoom_start=12,
    tiles='CartoDB positron'
)

# Define metro line colors (TMB Barcelona colors)
line_colors = {
    'L1': '#FF0000',  # Red
    'L2': '#800080',  # Purple
    'L3': '#008000',  # Green
    'L4': '#FFFF00',  # Yellow
    'L5': '#0000FF',  # Blue
    'L9': '#8B4513',  # Brown for L9/L10
}

# Line 1 stations with coordinates
line1_stations = [
    {'name': 'BELLVITGE', 'lat': 41.3456, 'lon': 2.1198, 'line': 'L1'},
    {'name': 'AV. CARRILET', 'lat': 41.3521, 'lon': 2.1263, 'line': 'L1'},
    {'name': 'R.J.OLIVERAS', 'lat': 41.3589, 'lon': 2.1321, 'line': 'L1'},
    {'name': 'CAN SERRA', 'lat': 41.3654, 'lon': 2.1387, 'line': 'L1'},
    {'name': 'FLORIDA', 'lat': 41.3712, 'lon': 2.1445, 'line': 'L1'},
    {'name': 'TORRASSA', 'lat': 41.3768, 'lon': 2.1502, 'line': 'L1'},
    {'name': 'STA. EULÀLIA', 'lat': 41.3821, 'lon': 2.1559, 'line': 'L1'},
    {'name': 'MERCAT NOU', 'lat': 41.3874, 'lon': 2.1612, 'line': 'L1'},
    {'name': 'PL. DE SANTS', 'lat': 41.3758, 'lon': 2.1401, 'line': 'L1'},
    {'name': 'HOSTAFRANCS', 'lat': 41.3762, 'lon': 2.1432, 'line': 'L1'},
    {'name': 'ESPANYA', 'lat': 41.3752, 'lon': 2.1491, 'line': 'L1'},
    {'name': 'ROCAFORT', 'lat': 41.3825, 'lon': 2.1568, 'line': 'L1'},
    {'name': 'URGELL', 'lat': 41.3862, 'lon': 2.1615, 'line': 'L1'},
    {'name': 'UNIVERSITAT', 'lat': 41.3858, 'lon': 2.1642, 'line': 'L1'},
    {'name': 'CATALUNYA', 'lat': 41.3874, 'lon': 2.1686, 'line': 'L1'},
    {'name': 'URQUINAONA', 'lat': 41.3889, 'lon': 2.1732, 'line': 'L1'},
    {'name': 'ARC DEL TRIOMF', 'lat': 41.3912, 'lon': 2.1805, 'line': 'L1'},
    {'name': 'MARINA', 'lat': 41.3945, 'lon': 2.1872, 'line': 'L1'},
    {'name': 'GLÒRIES', 'lat': 41.4032, 'lon': 2.1934, 'line': 'L1'},
    {'name': 'CLOT', 'lat': 41.4108, 'lon': 2.1865, 'line': 'L1'},
    {'name': 'NAVAS', 'lat': 41.4172, 'lon': 2.1821, 'line': 'L1'},
    {'name': 'SAGRERA', 'lat': 41.4238, 'lon': 2.1778, 'line': 'L1'},
    {'name': 'FABRA I PUIG', 'lat': 41.4301, 'lon': 2.1734, 'line': 'L1'},
    {'name': 'ST. ANDREU', 'lat': 41.4365, 'lon': 2.1691, 'line': 'L1'},
    {'name': 'TORRAS I BAGES', 'lat': 41.4428, 'lon': 2.1647, 'line': 'L1'},
    {'name': 'TRINITAT VELLA', 'lat': 41.4492, 'lon': 2.1604, 'line': 'L1'},
    {'name': 'BARÓ DE VIVER', 'lat': 41.4386, 'lon': 2.1934, 'line': 'L1'},
    {'name': 'STA. COLOMA', 'lat': 41.4512, 'lon': 2.2087, 'line': 'L1'},
    {'name': 'FONDO', 'lat': 41.4589, 'lon': 2.2215, 'line': 'L1'}
]

# Line 2 stations with coordinates
line2_stations = [
    {'name': 'PARAL.LEL', 'lat': 41.3751, 'lon': 2.1589, 'line': 'L2'},
    {'name': 'SANT ANTONI', 'lat': 41.3802, 'lon': 2.1634, 'line': 'L2'},
    {'name': 'UNIVERSITAT', 'lat': 41.3858, 'lon': 2.1642, 'line': 'L2'},
    {'name': 'PG. DE GRÀCIA', 'lat': 41.3911, 'lon': 2.1639, 'line': 'L2'},
    {'name': 'TETUAN', 'lat': 41.3952, 'lon': 2.1731, 'line': 'L2'},
    {'name': 'MONUMENTAL', 'lat': 41.3987, 'lon': 2.1802, 'line': 'L2'},
    {'name': 'SGDA. FAMÍLIA', 'lat': 41.4036, 'lon': 2.1744, 'line': 'L2'},
    {'name': 'ENCANTS', 'lat': 41.4078, 'lon': 2.1832, 'line': 'L2'},
    {'name': 'CLOT', 'lat': 41.4108, 'lon': 2.1865, 'line': 'L2'},
    {'name': 'BAC DE RODA', 'lat': 41.4189, 'lon': 2.1991, 'line': 'L2'},
    {'name': 'SANT MARTÍ', 'lat': 41.4256, 'lon': 2.2108, 'line': 'L2'},
    {'name': 'LA PAU', 'lat': 41.4321, 'lon': 2.2215, 'line': 'L2'},
    {'name': 'VERNEDA', 'lat': 41.4289, 'lon': 2.2345, 'line': 'L2'},
    {'name': 'ARTIGUES | S. ADRIÀ', 'lat': 41.4345, 'lon': 2.2412, 'line': 'L2'},
    {'name': 'SANT ROC', 'lat': 41.4412, 'lon': 2.2489, 'line': 'L2'},
    {'name': 'GORG', 'lat': 41.4478, 'lon': 2.2567, 'line': 'L2'},
    {'name': 'PEP VENTURA', 'lat': 41.4545, 'lon': 2.2645, 'line': 'L2'},
    {'name': 'BADALONA | POMPEU FABRA', 'lat': 41.4602, 'lon': 2.2721, 'line': 'L2'}
]

# Combine all stations
all_stations = line1_stations + line2_stations

# Add stations to the map
for station in all_stations:
    folium.CircleMarker(
        location=[station['lat'], station['lon']],
        radius=6,
        popup=f"{station['name']}<br>Line: {station['line']}",
        color='black',
        fillColor=line_colors.get(station['line'], 'gray'),
        fillOpacity=0.8,
        weight=1.5
    ).add_to(barcelona_map)

# Add Line 1 connections
line1_coords = [[station['lat'], station['lon']] for station in line1_stations]
folium.PolyLine(
    line1_coords,
    color=line_colors['L1'],
    weight=4,
    opacity=0.8,
    popup='Line 1 (Red)'
).add_to(barcelona_map)

# Add Line 2 connections
line2_coords = [[station['lat'], station['lon']] for station in line2_stations]
folium.PolyLine(
    line2_coords,
    color=line_colors['L2'],
    weight=4,
    opacity=0.8,
    popup='Line 2 (Purple)'
).add_to(barcelona_map)

# Reduce background visibility
barcelona_map.get_root().html.add_child(folium.Element("""
<style>
    .folium-tile-layer {
        opacity: 0.2 !important;
        filter: grayscale(80%) brightness(1.1) !important;
    }
</style>
"""))

# Add legend
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 90px; 
     background-color: white; border:2px solid grey; z-index:9999; 
     font-size:14px; padding: 10px">
     <p><i style="background: #FF0000; width: 20px; height: 10px; display: inline-block;"></i> Line 1</p>
     <p><i style="background: #800080; width: 20px; height: 10px; display: inline-block;"></i> Line 2</p>
</div>
'''
barcelona_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map
barcelona_map.save('barcelona_metro_lines.html')
barcelona_map
