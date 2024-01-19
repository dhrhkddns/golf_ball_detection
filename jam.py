from shapely.geometry import Polygon

p = Polygon([(0, 0), (1, 1), (1, 0)])
coords = list(p.exterior.coords)

print(coords[1][1])