from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely.geometry import Point

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}

INCOME_COLORS = {
    "Low": "#8b0000",
    "Moderate": "#e6550d",
    "Middle": "#fdae6b",
    "Upper": "#f7f7f7",
    "Unknown": "#cccccc",
}

st.set_page_config(page_title="Spira CRA Census Tract Mapper", layout="wide")
st.title("Spira Portfolio vs. Census Tract Income Map")
st.caption("Maps affordable housing properties against FFIEC census tract income classifications. No bank assessment-area layer required.")


def normalize_asset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts the Spira workbook format or the pipeline's assets.csv format."""
    df = df.copy()
    rename_map = {
        "Property": "asset_name",
        "property": "asset_name",
        "Asset Name": "asset_name",
        "Address": "address",
        "City": "city",
        "State": "state",
        "Zip": "zip",
        "ZIP": "zip",
        "Units": "units",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["asset_name", "address", "city", "state", "zip"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    for c in required:
        df[c] = df[c].astype(str).str.strip()

    if "units" not in df.columns:
        df["units"] = ""
    if "latitude" not in df.columns:
        df["latitude"] = pd.NA
    if "longitude" not in df.columns:
        df["longitude"] = pd.NA

    df["full_address"] = df["address"] + ", " + df["city"] + ", " + df["state"] + " " + df["zip"].astype(str)
    return df


@st.cache_data(show_spinner="Loading FFIEC tract income file...")
def load_ffiec_income() -> pd.DataFrame:
    ffiec_path = DATA_DIR / "CensusTractList2026.xlsx"
    df = pd.read_excel(ffiec_path, sheet_name="2024-2026 tracts", dtype=str)
    df = df.rename(columns={
        "FIPS code": "tract_fips",
        "Tract income level": "income_level",
        "Tract income percentage": "tract_income_pct",
        "MSA/MD name": "msa_md_name",
        "County name": "county_name",
        "State": "state_name",
    })
    df["tract_fips"] = df["tract_fips"].astype(str).str.replace(".0", "", regex=False).str.zfill(11)
    df["income_level"] = df["income_level"].fillna("Unknown")
    keep = ["tract_fips", "income_level", "tract_income_pct", "msa_md_name", "county_name", "state_name"]
    return df[keep].drop_duplicates("tract_fips")


@st.cache_data(show_spinner="Loading Census tract shapes...")
def load_tract_shapes(state_abbr: str) -> gpd.GeoDataFrame:
    statefp = STATE_FIPS[state_abbr.upper()]
    url = f"https://www2.census.gov/geo/tiger/TIGER2025/TRACT/tl_2025_{statefp}_tract.zip"
    tracts = gpd.read_file(url)
    tracts = tracts.to_crs("EPSG:4326")
    tracts["tract_fips"] = tracts["GEOID"].astype(str).str.zfill(11)
    return tracts[["tract_fips", "NAME", "NAMELSAD", "geometry"]]


@st.cache_data(show_spinner=False)
def geocode_one_address(address: str) -> tuple[float | None, float | None]:
    geolocator = Nominatim(user_agent="spira_cra_income_mapper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, swallow_exceptions=True)
    result = geocode(address)
    if result is None:
        return None, None
    return result.latitude, result.longitude


def geocode_missing(df: pd.DataFrame, do_geocode: bool) -> pd.DataFrame:
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    missing = df["latitude"].isna() | df["longitude"].isna()
    if do_geocode and missing.any():
        progress = st.progress(0, text="Geocoding missing property coordinates...")
        missing_idx = df.index[missing].tolist()
        for i, idx in enumerate(missing_idx):
            lat, lon = geocode_one_address(df.at[idx, "full_address"])
            df.at[idx, "latitude"] = lat
            df.at[idx, "longitude"] = lon
            progress.progress((i + 1) / len(missing_idx), text=f"Geocoded {i + 1} of {len(missing_idx)}")
        progress.empty()

    return df.dropna(subset=["latitude", "longitude"])


def attach_tracts(assets: pd.DataFrame, tracts: gpd.GeoDataFrame, ffiec: pd.DataFrame) -> gpd.GeoDataFrame:
    asset_gdf = gpd.GeoDataFrame(
        assets,
        geometry=[Point(xy) for xy in zip(assets["longitude"], assets["latitude"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(asset_gdf, tracts, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    joined = joined.merge(ffiec, on="tract_fips", how="left")
    joined["income_level"] = joined["income_level"].fillna("Unknown")
    joined["is_lmi_tract"] = joined["income_level"].isin(["Low", "Moderate"])
    return joined


def build_map(asset_gdf: gpd.GeoDataFrame, tracts_joined: gpd.GeoDataFrame, show_all_tracts: bool) -> folium.Map:
    if len(asset_gdf) == 0:
        return folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

    center = [asset_gdf.geometry.y.mean(), asset_gdf.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    if not show_all_tracts:
        asset_tracts = set(asset_gdf["tract_fips"].dropna().astype(str))
        tracts_to_plot = tracts_joined[tracts_joined["tract_fips"].isin(asset_tracts)].copy()
    else:
        tracts_to_plot = tracts_joined.copy()

    def style_tract(feature):
        level = feature["properties"].get("income_level") or "Unknown"
        return {
            "fillColor": INCOME_COLORS.get(level, INCOME_COLORS["Unknown"]),
            "color": "#444444",
            "weight": 0.45,
            "fillOpacity": 0.50 if level in ["Low", "Moderate"] else 0.28,
        }

    folium.GeoJson(
        tracts_to_plot[["tract_fips", "income_level", "tract_income_pct", "county_name", "msa_md_name", "geometry"]],
        name="Census tracts by FFIEC income level",
        style_function=style_tract,
        tooltip=folium.GeoJsonTooltip(
            fields=["tract_fips", "income_level", "tract_income_pct", "county_name", "msa_md_name"],
            aliases=["Tract", "Income level", "Income %", "County", "MSA/MD"],
            sticky=False,
        ),
    ).add_to(m)

    for _, row in asset_gdf.iterrows():
        level = row.get("income_level", "Unknown")
        dot_color = "green" if level in ["Low", "Moderate"] else "black"
        popup_html = f"""
        <b>{row.get('asset_name', '')}</b><br>
        {row.get('full_address', '')}<br><br>
        Units: {row.get('units', '')}<br>
        Census tract: {row.get('tract_fips', '')}<br>
        FFIEC income level: <b>{level}</b><br>
        Tract income %: {row.get('tract_income_pct', '')}<br>
        County: {row.get('county_name', '')}<br>
        MSA/MD: {row.get('msa_md_name', '')}<br>
        LMI tract: {row.get('is_lmi_tract', False)}
        """
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=8,
            color=dot_color,
            fill=True,
            fill_opacity=0.95,
            popup=folium.Popup(popup_html, max_width=400),
            tooltip=row.get("asset_name", "Property"),
        ).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 35px; left: 35px; z-index:9999; background:white;
                padding:12px; border:1px solid #999; font-size:13px; line-height:1.4;">
      <b>FFIEC Tract Income Level</b><br>
      <span style="color:#8b0000;">■</span> Low<br>
      <span style="color:#e6550d;">■</span> Moderate<br>
      <span style="color:#fdae6b;">■</span> Middle<br>
      <span style="color:#f7f7f7; text-shadow:0 0 1px #555;">■</span> Upper<br>
      <span style="color:#cccccc;">■</span> Unknown<br><br>
      <span style="color:green;">●</span> Property in LMI tract<br>
      <span style="color:black;">●</span> Property in non-LMI tract
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="CRA Tract Results")
    return output.getvalue()


with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload property workbook", type=["xlsx", "csv"])
    default_file = DATA_DIR / "assets.xlsx"
    use_default = st.checkbox("Use included Spira workbook", value=True)
    state = st.selectbox("State to map", sorted(STATE_FIPS.keys()), index=sorted(STATE_FIPS.keys()).index("CA"))
    do_geocode = st.checkbox("Geocode missing latitude/longitude", value=True)
    show_all_tracts = st.checkbox("Show all tracts in selected state", value=True)
    run = st.button("Run CRA Tract Map", type="primary")

st.markdown(
    """
    This simplified version ignores bank assessment areas. It answers one question:
    **which FFIEC income-level census tracts are Spira's properties located in?**
    """
)

if run:
    try:
        if uploaded is not None:
            if uploaded.name.lower().endswith(".csv"):
                raw = pd.read_csv(uploaded, dtype=str)
            else:
                raw = pd.read_excel(uploaded, dtype=str)
        elif use_default and default_file.exists():
            raw = pd.read_excel(default_file, dtype=str)
        else:
            st.error("Upload a workbook or check 'Use included Spira workbook'.")
            st.stop()

        assets = normalize_asset_columns(raw)
        assets_state = assets[assets["state"].str.upper() == state.upper()].copy()
        if assets_state.empty:
            st.warning(f"No properties found for state {state}. Try another state or check the State column.")
            st.stop()

        assets_geo = geocode_missing(assets_state, do_geocode=do_geocode)
        if assets_geo.empty:
            st.error("No properties have usable latitude/longitude. Turn on geocoding or add coordinates to the workbook.")
            st.stop()

        ffiec = load_ffiec_income()
        tracts = load_tract_shapes(state)
        tracts_joined = tracts.merge(ffiec, on="tract_fips", how="left")
        tracts_joined["income_level"] = tracts_joined["income_level"].fillna("Unknown")

        asset_gdf = attach_tracts(assets_geo, tracts, ffiec)

        total = len(asset_gdf)
        lmi_count = int(asset_gdf["is_lmi_tract"].sum())
        low_count = int((asset_gdf["income_level"] == "Low").sum())
        moderate_count = int((asset_gdf["income_level"] == "Moderate").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mapped properties", total)
        c2.metric("LMI tracts", lmi_count)
        c3.metric("Low-income", low_count)
        c4.metric("Moderate-income", moderate_count)

        result_df = asset_gdf.drop(columns="geometry").copy()
        preferred_cols = [
            "asset_name", "address", "city", "state", "zip", "units", "latitude", "longitude",
            "tract_fips", "income_level", "tract_income_pct", "county_name", "msa_md_name", "is_lmi_tract",
            "full_address",
        ]
        result_df = result_df[[c for c in preferred_cols if c in result_df.columns] + [c for c in result_df.columns if c not in preferred_cols]]

        st.subheader("Property CRA tract summary")
        st.dataframe(result_df, use_container_width=True)

        map_obj = build_map(asset_gdf, tracts_joined, show_all_tracts=show_all_tracts)
        map_html = map_obj.get_root().render()

        st.subheader("Interactive map")
        components.html(map_html, height=760, scrolling=True)

        st.download_button(
            "Download enriched Excel",
            data=to_excel_bytes(result_df),
            file_name="spira_properties_with_cra_tract_income.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download HTML map",
            data=map_html.encode("utf-8"),
            file_name="spira_cra_income_tract_map.html",
            mime="text/html",
        )

    except Exception as exc:
        st.exception(exc)
