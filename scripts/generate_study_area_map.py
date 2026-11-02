
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyproj import Transformer


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

GEOJSON_PATH = (
    PROJECT_DIR
    / "data"
    / "external"
    / "mnjoli_w60e_catchment.geojson"
)

# Your actual Natural Earth shapefile location
COUNTRIES_PATH = (
    PROJECT_DIR
    / "data"
    / "external"
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "study_area_map"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GEOGRAPHIC_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32736"


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

if not GEOJSON_PATH.is_file():
    raise FileNotFoundError(
        f"Catchment GeoJSON not found:\n{GEOJSON_PATH}"
    )

if not COUNTRIES_PATH.is_file():
    raise FileNotFoundError(
        f"Natural Earth shapefile not found:\n{COUNTRIES_PATH}"
    )

print("Input files found successfully.")


# ============================================================
# 3. LOAD AND VALIDATE CATCHMENT
# ============================================================

catchment = gpd.read_file(GEOJSON_PATH)

if catchment.empty:
    raise ValueError("The catchment GeoJSON contains no features.")

if catchment.crs is None:
    raise ValueError(
        "The catchment GeoJSON has no CRS defined."
    )

catchment = catchment.to_crs(GEOGRAPHIC_CRS)

if catchment.geometry.is_empty.any():
    raise ValueError(
        "The catchment contains empty geometries."
    )

if not catchment.geometry.is_valid.all():
    raise ValueError(
        "The catchment contains invalid geometries."
    )

# Project to UTM Zone 36S for local distance and area calculations
catchment_utm = catchment.to_crs(PROJECTED_CRS)

# The study area contains one catchment polygon
catchment_geometry = catchment_utm.geometry.iloc[0]

area_km2 = catchment_geometry.area / 1_000_000

print("\nCATCHMENT INFORMATION")
print("---------------------")
print(f"Number of features: {len(catchment)}")
print(f"Geographic CRS: {GEOGRAPHIC_CRS}")
print(f"Projected CRS: {PROJECTED_CRS}")
print(f"Mapped catchment area: {area_km2:.2f} km²")


# ============================================================
# 4. LOAD NATURAL EARTH COUNTRY BOUNDARIES
# ============================================================

countries = gpd.read_file(COUNTRIES_PATH)

if countries.empty:
    raise ValueError(
        "The Natural Earth shapefile contains no features."
    )

if countries.crs is None:
    raise ValueError(
        "The Natural Earth shapefile has no CRS defined."
    )

# Identify the country-name column
possible_name_fields = [
    "ADMIN",
    "NAME",
    "NAME_EN",
    "NAME_LONG",
]

name_field = next(
    (
        field
        for field in possible_name_fields
        if field in countries.columns
    ),
    None,
)

if name_field is None:
    raise ValueError(
        "No recognised country-name field was found "
        "in the Natural Earth dataset."
    )

# Find Eswatini, including older Swaziland naming
country_names = countries[name_field].astype(str)

eswatini = countries[
    country_names.str.contains(
        "Eswatini|Swaziland",
        case=False,
        regex=True,
        na=False,
    )
].copy()

if eswatini.empty:
    raise ValueError(
        "Eswatini was not found in the Natural Earth dataset. "
        f"Country-name field checked: {name_field}"
    )

eswatini = eswatini.to_crs(GEOGRAPHIC_CRS)

print("Natural Earth country boundaries loaded.")
print(f"Country-name field: {name_field}")


# ============================================================
# 5. COORDINATE TRANSFORMERS
# ============================================================

to_utm = Transformer.from_crs(
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    always_xy=True,
)

to_geographic = Transformer.from_crs(
    PROJECTED_CRS,
    GEOGRAPHIC_CRS,
    always_xy=True,
)


# ============================================================
# 6. MAIN MAP EXTENT
# ============================================================

xmin, ymin, xmax, ymax = catchment_geometry.bounds

x_range = xmax - xmin
y_range = ymax - ymin

if x_range <= 0 or y_range <= 0:
    raise ValueError(
        "The catchment has invalid map dimensions."
    )

center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2

# Add space around the catchment
x_padding = x_range * 0.35
y_padding = y_range * 0.35

map_xmin = xmin - x_padding
map_xmax = xmax + x_padding
map_ymin = ymin - y_padding
map_ymax = ymax + y_padding


# ============================================================
# 7. COORDINATE LABEL FORMATTERS
# ============================================================

def longitude_formatter(x, position):
    """Display UTM easting as longitude in degrees East."""
    longitude, _ = to_geographic.transform(
        x,
        center_y,
    )

    return f"{longitude:.3f}°E"


def latitude_formatter(y, position):
    """Display UTM northing as latitude in degrees South."""
    _, latitude = to_geographic.transform(
        center_x,
        y,
    )

    return f"{abs(latitude):.3f}°S"


# ============================================================
# 8. CREATE MAIN MAP
# ============================================================

fig, ax = plt.subplots(figsize=(8, 10))

fig.patch.set_facecolor("white")
ax.set_facecolor("#F7FAFA")

# Plot catchment polygon
catchment_utm.plot(
    ax=ax,
    facecolor="#A8DADC",
    edgecolor="#145A63",
    linewidth=2,
    zorder=3,
)

ax.set_xlim(map_xmin, map_xmax)
ax.set_ylim(map_ymin, map_ymax)

ax.set_aspect("equal", adjustable="box")

ax.set_title(
    "STUDY AREA: MNJOLI W60E CATCHMENT",
    fontsize=15,
    fontweight="bold",
    pad=20,
)

ax.set_xlabel(
    "Longitude",
    fontsize=10,
    labelpad=10,
)

ax.set_ylabel(
    "Latitude",
    fontsize=10,
    labelpad=10,
)


# ============================================================
# 9. DRAW GEOGRAPHIC GRATICULE
# ============================================================

# Longitude lines
longitudes = [
    31.60,
    31.62,
    31.64,
    31.66,
    31.68,
    31.70,
    31.72,
]

# Latitude lines
latitudes = [
    -26.22,
    -26.20,
    -26.18,
    -26.16,
    -26.14,
    -26.12,
    -26.10,
    -26.08,
    -26.06,
]

# Draw longitude lines
for longitude in longitudes:

    lat_values = [
        -26.24 + i * 0.002
        for i in range(111)
    ]

    coordinates = [
        to_utm.transform(longitude, latitude)
        for latitude in lat_values
    ]

    x_values = [point[0] for point in coordinates]
    y_values = [point[1] for point in coordinates]

    ax.plot(
        x_values,
        y_values,
        color="#B8C7C9",
        linewidth=0.5,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )


# Draw latitude lines
for latitude in latitudes:

    lon_values = [
        31.58 + i * 0.002
        for i in range(76)
    ]

    coordinates = [
        to_utm.transform(longitude, latitude)
        for longitude in lon_values
    ]

    x_values = [point[0] for point in coordinates]
    y_values = [point[1] for point in coordinates]

    ax.plot(
        x_values,
        y_values,
        color="#B8C7C9",
        linewidth=0.5,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )


# Format axes as geographic coordinates
ax.xaxis.set_major_locator(
    MaxNLocator(nbins=5)
)

ax.yaxis.set_major_locator(
    MaxNLocator(nbins=6)
)

ax.xaxis.set_major_formatter(
    FuncFormatter(longitude_formatter)
)

ax.yaxis.set_major_formatter(
    FuncFormatter(latitude_formatter)
)

ax.tick_params(
    axis="both",
    labelsize=8,
    direction="out",
    length=4,
)

ax.grid(False)


# ============================================================
# 10. ADD 2 KM SCALE BAR
# ============================================================

scale_length = 2000  # metres

scale_x = xmin + x_range * 0.10
scale_y = ymin - y_range * 0.16

# Main scale bar
ax.plot(
    [
        scale_x,
        scale_x + scale_length,
    ],
    [
        scale_y,
        scale_y,
    ],
    color="black",
    linewidth=2.5,
    solid_capstyle="butt",
    zorder=5,
)

# Left end
ax.plot(
    [scale_x, scale_x],
    [scale_y - 100, scale_y + 100],
    color="black",
    linewidth=1.5,
    zorder=5,
)

# Right end
ax.plot(
    [
        scale_x + scale_length,
        scale_x + scale_length,
    ],
    [
        scale_y - 100,
        scale_y + 100,
    ],
    color="black",
    linewidth=1.5,
    zorder=5,
)

# Scale label
ax.text(
    scale_x + scale_length / 2,
    scale_y - y_range * 0.035,
    "2 km",
    ha="center",
    va="top",
    fontsize=9,
    fontweight="bold",
)


# ============================================================
# 11. ADD NORTH ARROW
# ============================================================

ax.annotate(
    "N",
    xy=(0.93, 0.94),
    xytext=(0.93, 0.84),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    arrowprops={
        "facecolor": "black",
        "edgecolor": "black",
        "width": 2,
        "headwidth": 10,
        "headlength": 12,
    },
    zorder=6,
)


# ============================================================
# 12. ADD ESWATINI LOCATION INSET
# ============================================================

# IMPORTANT: Use a tuple, not a list
inset_ax = ax.inset_axes(
    (0.04, 0.70, 0.30, 0.24)
)

inset_ax.set_facecolor("#F7FAFA")

# Plot Eswatini
eswatini.plot(
    ax=inset_ax,
    facecolor="#E5E5E5",
    edgecolor="#555555",
    linewidth=0.8,
    zorder=1,
)

# Find a representative point inside the catchment
catchment_wgs84 = catchment.to_crs(GEOGRAPHIC_CRS)

study_point = (
    catchment_wgs84.geometry.iloc[0].representative_point()
)

# Mark the study area
inset_ax.scatter(
    study_point.x,
    study_point.y,
    color="#C0392B",
    edgecolor="white",
    linewidth=0.8,
    s=45,
    zorder=5,
    label="MnJoli study area",
)

# Determine country extent
country_bounds = eswatini.total_bounds

country_xmin = country_bounds[0]
country_ymin = country_bounds[1]
country_xmax = country_bounds[2]
country_ymax = country_bounds[3]

inset_x_padding = (
    country_xmax - country_xmin
) * 0.12

inset_y_padding = (
    country_ymax - country_ymin
) * 0.12

inset_ax.set_xlim(
    country_xmin - inset_x_padding,
    country_xmax + inset_x_padding,
)

inset_ax.set_ylim(
    country_ymin - inset_y_padding,
    country_ymax + inset_y_padding,
)

inset_ax.set_title(
    "Eswatini",
    fontsize=9,
    fontweight="bold",
    pad=4,
)

inset_ax.legend(
    loc="lower left",
    fontsize=6,
    frameon=True,
)

inset_ax.set_xticks([])
inset_ax.set_yticks([])

for spine in inset_ax.spines.values():
    spine.set_edgecolor("#555555")
    spine.set_linewidth(0.8)


# ============================================================
# 13. MAP BORDER AND SOURCE NOTE
# ============================================================

for spine in ax.spines.values():
    spine.set_edgecolor("#555555")
    spine.set_linewidth(0.8)

fig.text(
    0.5,
    0.025,
    (
        "Source: MnJoli W60E catchment boundary and "
        "Natural Earth country boundaries. "
        "Projection: WGS 84 / UTM Zone 36S."
    ),
    ha="center",
    fontsize=7,
    color="#444444",
)


# ============================================================
# 14. SAVE MAP AS PNG AND PDF
# ============================================================

fig.subplots_adjust(
    left=0.13,
    right=0.95,
    top=0.91,
    bottom=0.12,
)

png_path = (
    OUTPUT_DIR
    / "mnjoli_map_with_inset.png"
)

pdf_path = (
    OUTPUT_DIR
    / "mnjoli_map_with_inset.pdf"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 15. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 50)
print("MAP GENERATION SUCCESSFUL")
print("=" * 50)

print(f"Catchment area: {area_km2:.2f} km²")
print(f"PNG map: {png_path}")
print(f"PDF map: {pdf_path}")

print("\nTo open the generated map, run:")

print(
    'Invoke-Item ".\\outputs\\study_area_map\\'
    'mnjoli_map_with_inset.png"'
)

print("=" * 50)
