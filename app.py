from __future__ import annotations

import io
import re
import time
from pathlib import Path
from urllib.parse import quote

import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from streamlit.components.v1 import html

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "output"
CACHE_DIR = APP_DIR / "cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

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
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common workbook columns to the names the app expects."""
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in ["property", "property_name", "asset", "name"]:
            rename_map[col] = "asset_name"
        elif key in ["address", "street_address"]:
            rename_map[col] = "address"
        elif key in ["city"]:
            rename_map[col] = "city"
        elif key in ["state", "st"]:
            rename_map[col] = "state"
        elif key in ["zip", "zipcode", "zip_code"]:
            rename_map[col] = "zip"
        elif key in ["units", "unit_count"]:
            rename_map[col] = "units"
        elif key in ["ami_target", "ami", "ami_targeting"]:
            rename_map[col] = "ami_target"
        elif key in ["deal_type", "type"]:
            rename_map[col] = "deal_type"
        elif key in ["latitude", "lat"]:
            rename_map[col] = "latitude"
        elif key in ["longitude", "lon", "lng"]:
            rename_map[col] = "longitude"
    df = df.rename(columns=rename_map)
    required = ["asset_name", "address", "city", "state", "zip"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after cleanup: {missing}")
    for col in ["units", "ami_target", "deal_type", "latitude", "longitude"]:
        if col not in df.columns:
            df[col] = ""
    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df["zip"] = df["zip"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    return df


def full_address(row: pd.Series) -> str:
    return f"{row['address']}, {row['city']}, {row['state']} {row['zip']}"


def load_geocode_cache() -> pd.DataFrame:
    p = CACHE_DIR / "geocode_cache.csv"
    if p.exists():
        return pd.read_csv(p, dtype=str)
    return pd.DataFrame(columns=["full_address", "latitude", "longitude", "tract_fips", "statefp", "countyfp", "match_status"])


def save_geocode_cache(cache: pd.DataFrame) -> None:
    cache.to_csv(CACHE_DIR / "geocode_cache.csv", index=False)


def census_geocode_one(address: str) -> dict:
    """Use the free Census geocoder. Returns lat/lon and tract GEOID if matched."""
    url = (
        "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
        f"?address={quote(address)}&benchmark=Public_AR_Current&vintage=Current_Current&format=json"
    )
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        data = r.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if not matches:
            return {"match_status": "No match"}
        match = matches[0]
        coords = match.get("coordinates", {})
        geos = match.get("geographies", {})
        tracts = geos.get("Census Tracts", [])
        tract = tracts[0] if tracts else {}
        tract_fips = tract.get("GEOID")
        return {
            "latitude": coords.get("y"),
            "longitude": coords.get("x"),
            "tract_fips": tract_fips,
            "statefp": tract_fips[:2] if tract_fips else None,
            "countyfp": tract_fips[2:5] if tract_fips else None,
            "match_status": "Matched",
        }
    except Exception as e:
        return {"match_status": f"Error: {e}"}


def geocode_assets(df: pd.DataFrame, progress_bar=None) -> pd.DataFrame:
    df = df.copy()
    df["full_address"] = df.apply(full_address, axis=1)
    cache = load_geocode_cache()
    cache_map = {row["full_address"]: row for _, row in cache.iterrows()}
    new_cache_rows = []
    out_rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        addr = row["full_address"]
        if pd.notna(row.get("latitude")) and str(row.get("latitude")).strip() not in ["", "nan"] and pd.notna(row.get("longitude")) and str(row.get("longitude")).strip() not in ["", "nan"]:
            rec = {"latitude": row["latitude"], "longitude": row["longitude"], "tract_fips": row.get("tract_fips", None), "statefp": None, "countyfp": None, "match_status": "Used existing lat/long"}
        elif addr in cache_map:
            rec = cache_map[addr].to_dict()
        else:
            rec = census_geocode_one(addr)
            rec["full_address"] = addr
            new_cache_rows.append(rec)
            time.sleep(0.15)
        merged = row.to_dict()
        merged.update(rec)
        out_rows.append(merged)
        if progress_bar:
            progress_bar.progress(i / len(df), text=f"Geocoding {i} of {len(df)}")

    if new_cache_rows:
        cache = pd.concat([cache, pd.DataFrame(new_cache_rows)], ignore_index=True)
        save_geocode_cache(cache.drop_duplicates(subset=["full_address"], keep="last"))
    return pd.DataFrame(out_rows)


def load_ffiec_tract_list() -> pd.DataFrame:
    path = DATA_DIR / "CensusTractList2026.xlsx"
    if not path.exists():
        url = "https://www.ffiec.gov/sites/default/files/data/census/CensusTractList2026.xlsx"
        content = requests.get(url, timeout=60).content
        path.write_bytes(content)
    ffiec = pd.read_excel(path, sheet_name="2024-2026 tracts", dtype=str)
    ffiec = ffiec.rename(columns={
        "FIPS code": "tract_fips",
        "Tract income level": "income_level",
        "Tract income percentage": "tract_income_pct",
        "MSA/MD name": "msa_md_name",
        "County name": "county_name",
        "State code": "statefp",
        "County code": "countyfp",
    })
    ffiec["tract_fips"] = ffiec["tract_fips"].astype(str).str.zfill(11)
    ffiec["statefp"] = ffiec["statefp"].astype(str).str.zfill(2)
    ffiec["countyfp"] = ffiec["countyfp"].astype(str).str.zfill(3)
    return ffiec[["tract_fips", "income_level", "tract_income_pct", "msa_md_name", "county_name", "statefp", "countyfp"]]


def load_bank_aa() -> pd.DataFrame:
    aa = pd.read_csv(DATA_DIR / "bank_assessment_areas.csv", dtype=str)
    aa["statefp"] = aa["statefp"].astype(str).str.zfill(2)
    aa["countyfp"] = aa["countyfp"].astype(str).str.zfill(3)
    if "aa_weight" not in aa.columns:
        aa["aa_weight"] = 1.0
    aa["aa_weight"] = pd.to_numeric(aa["aa_weight"], errors="coerce").fillna(1.0)
    return aa


def load_tract_shapes(state_abbr: str) -> gpd.GeoDataFrame:
    statefp = STATE_FIPS[state_abbr]
    cache_path = CACHE_DIR / f"tl_2024_{statefp}_tract.zip"
    if not cache_path.exists():
        url = f"https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_{statefp}_tract.zip"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        cache_path.write_bytes(r.content)
    tracts = gpd.read_file(cache_path)
    tracts = tracts.to_crs("EPSG:4326")
    tracts["tract_fips"] = tracts["GEOID"].astype(str)
    tracts["statefp"] = tracts["STATEFP"].astype(str).str.zfill(2)
    tracts["countyfp"] = tracts["COUNTYFP"].astype(str).str.zfill(3)
    return tracts


def add_cra_fields(assets: pd.DataFrame, ffiec: pd.DataFrame, aa_bank: pd.DataFrame) -> pd.DataFrame:
    assets = assets.copy()
    assets["tract_fips"] = assets["tract_fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
    assets = assets.merge(ffiec, on="tract_fips", how="left", suffixes=("", "_ffiec"))
    aa_keys = set(zip(aa_bank["statefp"], aa_bank["countyfp"]))
    assets["asset_county_key"] = list(zip(assets["statefp"].astype(str).str.zfill(2), assets["countyfp"].astype(str).str.zfill(3)))
    assets["in_bank_aa"] = assets["asset_county_key"].isin(aa_keys)
    assets["is_lmi_tract"] = assets["income_level"].isin(["Low", "Moderate"])
    assets["cra_alignment_status"] = "Limited direct alignment"
    assets.loc[assets["in_bank_aa"] & assets["is_lmi_tract"], "cra_alignment_status"] = "Strong: in AA + LMI tract"
    assets.loc[assets["in_bank_aa"] & ~assets["is_lmi_tract"], "cra_alignment_status"] = "In AA, not LMI tract"
    assets.loc[~assets["in_bank_aa"] & assets["is_lmi_tract"], "cra_alignment_status"] = "LMI tract, outside AA"
    assets["cra_alignment_score"] = assets["in_bank_aa"].astype(int) * 60 + assets["is_lmi_tract"].astype(int) * 40
    return assets


def build_map(assets: pd.DataFrame, bank_name: str, state_abbr: str, ffiec: pd.DataFrame, aa_bank: pd.DataFrame) -> folium.Map:
    statefp = STATE_FIPS[state_abbr]
    tracts = load_tract_shapes(state_abbr)
    tracts = tracts.merge(ffiec[["tract_fips", "income_level", "tract_income_pct"]], on="tract_fips", how="left")

    aa_keys = set(zip(aa_bank["statefp"], aa_bank["countyfp"]))
    asset_counties = set(zip(assets["statefp"].astype(str).str.zfill(2), assets["countyfp"].astype(str).str.zfill(3)))
    relevant_counties = aa_keys.union(asset_counties)
    tracts["in_bank_aa"] = tracts.apply(lambda r: (r["statefp"], r["countyfp"]) in aa_keys, axis=1)
    tracts = tracts[tracts.apply(lambda r: (r["statefp"], r["countyfp"]) in relevant_counties or r["statefp"] == statefp, axis=1)]

    lat = pd.to_numeric(assets["latitude"], errors="coerce").mean()
    lon = pd.to_numeric(assets["longitude"], errors="coerce").mean()
    m = folium.Map(location=[lat, lon], zoom_start=8, tiles="CartoDB positron")

    income_colors = {"Low": "#8b0000", "Moderate": "#ef4444", "Middle": "#f59e0b", "Upper": "#e5e7eb", "Unknown": "#9ca3af"}

    def style(feature):
        income = feature["properties"].get("income_level")
        in_aa = feature["properties"].get("in_bank_aa")
        return {
            "fillColor": income_colors.get(income, "#d1d5db"),
            "color": "#111827" if in_aa else "#9ca3af",
            "weight": 1.4 if in_aa else 0.25,
            "fillOpacity": 0.52 if in_aa else 0.10,
        }

    folium.GeoJson(
        tracts[["tract_fips", "income_level", "tract_income_pct", "in_bank_aa", "geometry"]],
        name="FFIEC tract income + bank AA overlay",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["tract_fips", "income_level", "tract_income_pct", "in_bank_aa"]),
    ).add_to(m)

    for _, r in assets.iterrows():
        lat = pd.to_numeric(r.get("latitude"), errors="coerce")
        lon = pd.to_numeric(r.get("longitude"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        status = r.get("cra_alignment_status", "")
        color = "green" if status.startswith("Strong") else "orange" if status.startswith("In AA") else "blue" if status.startswith("LMI") else "gray"
        popup = f"""
        <b>{r.get('asset_name','')}</b><br>
        {r.get('full_address','')}<br><br>
        Units: {r.get('units','')}<br>
        Tract: {r.get('tract_fips','')}<br>
        FFIEC income level: {r.get('income_level','')}<br>
        Tract income %: {r.get('tract_income_pct','')}<br>
        In {bank_name} AA: {r.get('in_bank_aa','')}<br>
        <b>{status}</b>
        """
        folium.CircleMarker([lat, lon], radius=8, color=color, fill=True, fill_opacity=0.9, popup=popup, tooltip=r.get("asset_name", "Asset")).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 40px; left: 40px; width: 280px; z-index:9999;
                background: white; padding: 12px; border: 1px solid #999; font-size: 13px;">
      <b>CRA Alignment Legend</b><br>
      <span style="color:#8b0000;">■</span> Low-income tract<br>
      <span style="color:#ef4444;">■</span> Moderate-income tract<br>
      <span style="color:#f59e0b;">■</span> Middle-income tract<br>
      <span style="color:#e5e7eb;">■</span> Upper-income tract<br><br>
      <span style="color:green;">●</span> Asset in AA + LMI tract<br>
      <span style="color:orange;">●</span> Asset in AA only<br>
      <span style="color:blue;">●</span> Asset in LMI tract only<br>
      <span style="color:gray;">●</span> Limited direct alignment<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


st.set_page_config(page_title="CRA Alignment Mapper", layout="wide")
st.title("CRA Alignment Mapper")
st.caption("Map affordable housing assets against FFIEC tract-income classifications and sample bank CRA assessment areas.")

st.warning("The included bank assessment areas are starter/sample county-level inputs. Verify them against each bank's CRA public file before using investor-facing.", icon="⚠️")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload asset workbook (.xlsx)", type=["xlsx"])
    aa_df = load_bank_aa()
    bank_names = sorted(aa_df["bank_name"].dropna().unique())
    bank_name = st.selectbox("Bank", bank_names, index=bank_names.index("City National Bank") if "City National Bank" in bank_names else 0)
    state_abbr = st.selectbox("State to map", sorted(STATE_FIPS.keys()), index=sorted(STATE_FIPS.keys()).index("CA"))
    run = st.button("Run CRA Mapping", type="primary")

if uploaded is not None:
    assets_raw = pd.read_excel(uploaded)
else:
    assets_raw = pd.read_excel(DATA_DIR / "assets.xlsx")

try:
    assets_preview = normalize_columns(assets_raw)
    st.subheader("Asset input preview")
    st.dataframe(assets_preview.head(20), use_container_width=True)
except Exception as e:
    st.error(f"Input issue: {e}")
    st.stop()

if run:
    with st.spinner("Loading FFIEC tract list..."):
        ffiec = load_ffiec_tract_list()
    aa_bank = aa_df[aa_df["bank_name"] == bank_name].copy()

    progress = st.progress(0, text="Starting geocoding...")
    assets_geo = geocode_assets(assets_preview, progress)
    progress.empty()

    assets_cra = add_cra_fields(assets_geo, ffiec, aa_bank)

    output_xlsx = OUTPUT_DIR / "assets_cra_geocoded.xlsx"
    output_html = OUTPUT_DIR / "cra_alignment_map.html"
    assets_cra.to_excel(output_xlsx, index=False)

    with st.spinner("Building map..."):
        fmap = build_map(assets_cra, bank_name, state_abbr, ffiec, aa_bank)
        fmap.save(output_html)

    st.success("Done. Outputs created in the output folder.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", len(assets_cra))
    c2.metric("In bank AA", int(assets_cra["in_bank_aa"].sum()))
    c3.metric("LMI tracts", int(assets_cra["is_lmi_tract"].sum()))
    c4.metric("Strong alignment", int((assets_cra["in_bank_aa"] & assets_cra["is_lmi_tract"]).sum()))

    st.subheader("Summary")
    summary_cols = ["asset_name", "city", "state", "tract_fips", "income_level", "tract_income_pct", "in_bank_aa", "is_lmi_tract", "cra_alignment_status", "latitude", "longitude"]
    st.dataframe(assets_cra[[c for c in summary_cols if c in assets_cra.columns]], use_container_width=True)

    st.subheader("Map")
    html(fmap.get_root().render(), height=720, scrolling=True)

    with open(output_xlsx, "rb") as f:
        st.download_button("Download enriched workbook", f, file_name="assets_cra_geocoded.xlsx")
    with open(output_html, "rb") as f:
        st.download_button("Download HTML map", f, file_name="cra_alignment_map.html")
else:
    st.info("Click 'Run CRA Mapping' in the sidebar when you're ready.")
