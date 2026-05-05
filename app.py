import io
import time
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import Fullscreen
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely.geometry import Point


st.set_page_config(page_title="Spira CRA Income Mapper", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

SAMPLE_WORKBOOK = DATA_DIR / "assets.xlsx"
FFIEC_FILE = DATA_DIR / "ffiec_tracts.csv"

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "Property": "asset_name",
        "property": "asset_name",
        "Asset": "asset_name",
        "Asset Name": "asset_name",
        "asset": "asset_name",
        "asset_name": "asset_name",
        "Address": "address",
        "address": "address",
        "City": "city",
        "city": "city",
        "State": "state",
        "state": "state",
        "Zip": "zip",
        "ZIP": "zip",
        "zip": "zip",
        "Units": "units",
        "units": "units",
        "Latitude": "latitude",
        "latitude": "latitude",
        "Lat": "latitude",
        "Longitude": "longitude",
        "longitude": "longitude",
        "Lon": "longitude",
        "Lng": "longitude",
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    for col in ["asset_name", "address", "city", "state", "zip"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["units", "latitude", "longitude", "ami_target", "deal_type"]:
        if col not in df.columns:
            df[col] = ""

    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df["zip"] = (
        df["zip"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace("nan", "", regex=False)
        .str.strip()
    )

    return df


@st.cache_data(show_spinner=False)
def geocode_address(full_address: str):
    geolocator = Nominatim(user_agent="spira_cra_income_mapper")
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1.1,
        swallow_exceptions=True,
    )

    result = geocode(full_address)

    if result is None:
        return None, None

    return result.latitude, result.longitude


def geocode_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    for idx, row in df.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            continue

        full_address = f"{row['address']}, {row['city']}, {row['state']} {row['zip']}"
        lat, lon = geocode_address(full_address)

        df.at[idx, "latitude"] = lat
        df.at[idx, "longitude"] = lon

        time.sleep(0.1)

    return df


@st.cache_data(show_spinner=True)
def load_tracts(state_abbr: str) -> gpd.GeoDataFrame:
    statefp = STATE_FIPS[state_abbr.upper()]
    url = f"https://www2.census.gov/geo/tiger/TIGER2025/TRACT/tl_2025_{statefp}_tract.zip"

    tracts = gpd.read_file(url)
    tracts = tracts.to_crs("EPSG:4326")
    tracts["tract_fips"] = tracts["GEOID"].astype(str)

    return tracts


@st.cache_data(show_spinner=False)
def load_ffiec_income() -> pd.DataFrame:
    if not FFIEC_FILE.exists():
        return pd.DataFrame(columns=["tract_fips", "income_level"])

    ffiec = pd.read_csv(FFIEC_FILE, dtype=str)

    rename_map = {
        "GEOID": "tract_fips",
        "geoid": "tract_fips",
        "Tract": "tract_fips",
        "tract": "tract_fips",
        "TRACT": "tract_fips",
        "tract_fips": "tract_fips",
        "Income Level": "income_level",
        "income level": "income_level",
        "Income_Level": "income_level",
        "income_level": "income_level",
        "Tract Income Level": "income_level",
        "tract_income_level": "income_level",
    }

    ffiec = ffiec.rename(columns={c: rename_map.get(c, c) for c in ffiec.columns})

    if "tract_fips" not in ffiec.columns:
        return pd.DataFrame(columns=["tract_fips", "income_level"])

    if "income_level" not in ffiec.columns:
        ffiec["income_level"] = "Unknown"

    ffiec["tract_fips"] = (
        ffiec["tract_fips"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(11)
    )

    ffiec["income_level"] = (
        ffiec["income_level"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

    return ffiec[["tract_fips", "income_level"]]


def attach_income_to_tracts(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ffiec = load_ffiec_income()

    tracts = tracts.merge(ffiec, on="tract_fips", how="left")
    tracts["income_level"] = tracts["income_level"].fillna("Unknown")

    return tracts


def spatial_join_assets(df: pd.DataFrame, tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.copy()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])

    asset_gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        asset_gdf,
        tracts[["tract_fips", "income_level", "geometry"]],
        how="left",
        predicate="within",
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    joined["is_lmi_tract"] = joined["income_level"].isin(["Low", "Moderate"])

    return joined


def build_map(asset_gdf: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, show_all_tracts: bool) -> folium.Map:
    if asset_gdf.empty:
        return folium.Map(location=[37.5, -119.5], zoom_start=6, tiles="CartoDB positron")

    center_lat = asset_gdf.geometry.y.mean()
    center_lon = asset_gdf.geometry.x.mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB positron",
    )

    Fullscreen().add_to(m)

    income_colors = {
        "Low": "#7f0000",
        "Moderate": "#d7301f",
        "Middle": "#fdae6b",
        "Upper": "#f0f0f0",
        "Unknown": "#cccccc",
    }

    if show_all_tracts:
        tract_layer = tracts
    else:
        used_tracts = asset_gdf["tract_fips"].dropna().unique().tolist()
        tract_layer = tracts[tracts["tract_fips"].isin(used_tracts)]

    def tract_style(feature):
        income = feature["properties"].get("income_level", "Unknown")
        return {
            "fillColor": income_colors.get(income, "#cccccc"),
            "color": "#555555",
            "weight": 0.4,
            "fillOpacity": 0.45,
        }

    folium.GeoJson(
        tract_layer[["tract_fips", "income_level", "geometry"]],
        name="Census Tract Income Level",
        style_function=tract_style,
        tooltip=folium.GeoJsonTooltip(fields=["tract_fips", "income_level"]),
    ).add_to(m)

    for _, row in asset_gdf.iterrows():
        income = row.get("income_level", "Unknown")
        is_lmi = bool(row.get("is_lmi_tract", False))
        marker_color = "green" if is_lmi else "gray"

        popup_html = f"""
        <b>{row.get('asset_name', 'Asset')}</b><br>
        {row.get('address', '')}<br>
        {row.get('city', '')}, {row.get('state', '')} {row.get('zip', '')}<br><br>
        Units: {row.get('units', '')}<br>
        Census Tract: {row.get('tract_fips', '')}<br>
        FFIEC Income Level: <b>{income}</b><br>
        LMI Tract: <b>{'Yes' if is_lmi else 'No'}</b>
        """

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=8,
            color=marker_color,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=row.get("asset_name", "Asset"),
        ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 260px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border: 1px solid #999;
        font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
    ">
      <b>CRA Tract Income Legend</b><br>
      <span style="color:#7f0000;">■</span> Low-income tract<br>
      <span style="color:#d7301f;">■</span> Moderate-income tract<br>
      <span style="color:#fdae6b;">■</span> Middle-income tract<br>
      <span style="color:#999999;">■</span> Upper / Unknown tract<br><br>
      <span style="color:green;">●</span> Property in LMI tract<br>
      <span style="color:gray;">●</span> Property not in LMI tract<br>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def make_excel_download(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="CRA Results")

    return output.getvalue()


def load_input_workbook(uploaded_file, use_included_workbook: bool) -> pd.DataFrame | None:
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)

    if use_included_workbook:
        if SAMPLE_WORKBOOK.exists():
            return pd.read_excel(SAMPLE_WORKBOOK)
        st.error(f"Could not find included workbook at: {SAMPLE_WORKBOOK}")
        return None

    return None


# -----------------------------
# UI
# -----------------------------

st.title("Spira Portfolio vs. Census Tract Income Map")

st.write(
    "Maps affordable housing properties against census tract income classifications. "
    "This version does not use bank assessment areas."
)

with st.sidebar:
    st.header("Inputs")

    uploaded_file = st.file_uploader(
        "Upload property workbook",
        type=["xlsx", "csv"],
    )

    use_included_workbook = st.checkbox(
        "Use included assets.xlsx workbook",
        value=True,
    )

    state_to_map = st.selectbox(
        "State to map",
        options=sorted(STATE_FIPS.keys()),
        index=sorted(STATE_FIPS.keys()).index("CA"),
    )

    geocode = st.checkbox(
        "Geocode missing latitude/longitude",
        value=True,
    )

    show_all_tracts = st.checkbox(
        "Show all tracts in selected state",
        value=True,
    )

    run_button = st.button("Run CRA Tract Map", type="primary")


if not run_button:
    st.info("Upload a workbook or use the included assets.xlsx workbook, then click **Run CRA Tract Map**.")
    st.stop()


df = load_input_workbook(uploaded_file, use_included_workbook)

if df is None:
    st.stop()

df = normalize_columns(df)

if geocode:
    with st.spinner("Geocoding missing property coordinates..."):
        df = geocode_missing(df)

with st.spinner("Loading census tracts..."):
    tracts = load_tracts(state_to_map)
    tracts = attach_income_to_tracts(tracts)

with st.spinner("Joining properties to census tracts..."):
    asset_gdf = spatial_join_assets(df, tracts)

if asset_gdf.empty:
    st.error("No properties had usable latitude/longitude. Check addresses or add Latitude/Longitude columns.")
    st.dataframe(df)
    st.stop()

with st.spinner("Building map..."):
    m = build_map(asset_gdf, tracts, show_all_tracts)
    html = m.get_root().render()

st.subheader("Interactive Map")
components.html(html, height=700, scrolling=True)

result_df = pd.DataFrame(asset_gdf.drop(columns="geometry"))

st.subheader("CRA Tract Summary")

summary = {
    "Total Properties Mapped": len(result_df),
    "Properties in Low-Income Tracts": int((result_df["income_level"] == "Low").sum()),
    "Properties in Moderate-Income Tracts": int((result_df["income_level"] == "Moderate").sum()),
    "Properties in LMI Tracts": int(result_df["is_lmi_tract"].sum()),
    "Properties Not in LMI Tracts": int((~result_df["is_lmi_tract"]).sum()),
}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mapped", summary["Total Properties Mapped"])
c2.metric("Low", summary["Properties in Low-Income Tracts"])
c3.metric("Moderate", summary["Properties in Moderate-Income Tracts"])
c4.metric("LMI", summary["Properties in LMI Tracts"])
c5.metric("Non-LMI", summary["Properties Not in LMI Tracts"])

st.dataframe(result_df, use_container_width=True)

excel_bytes = make_excel_download(result_df)

st.download_button(
    label="Download enriched Excel",
    data=excel_bytes,
    file_name="spira_cra_tract_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.download_button(
    label="Download HTML map",
    data=html,
    file_name="spira_cra_income_map.html",
    mime="text/html",
)
