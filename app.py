import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPolygon, Polygon, Point
from pyproj import Transformer
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import numpy as np
import alphashape
import traceback

st.set_page_config(layout="wide", page_title="Field Boundary Detection")

try:
    st.title("Field Boundary Detection from GPS Data")

    uploaded_file = st.file_uploader("Upload CSV file with 'timestamp', 'latitude', 'longitude' columns", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=False)

        required_cols = {'timestamp', 'latitude', 'longitude'}
        if not required_cols.issubset(df.columns):
            st.error(f"Uploaded CSV must contain columns: {required_cols}")
            st.stop()

        df = df[['timestamp', 'latitude', 'longitude']].dropna().reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.drop_duplicates(subset=['latitude', 'longitude'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        st.write(f"Loaded {len(df)} GPS points after cleaning.")

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:24378", always_xy=True)
        df['x'], df['y'] = transformer.transform(df['longitude'].values, df['latitude'].values)
        coords = df[['x', 'y']].values

        if len(coords) < 10:
            st.warning("Not enough data points for clustering. Need at least 10.")
            st.stop()

        eps_m = 3.10
        min_samples = 18
        db = DBSCAN(eps=eps_m, min_samples=min_samples, metric='euclidean')
        df['cluster'] = db.fit_predict(coords)
        df['label'] = df['cluster'].apply(lambda x: 'road' if x == -1 else 'field')

        tree = BallTree(coords, metric='euclidean')
        radius = 10
        indices = tree.query_radius(coords, r=radius)

        labels_corrected = []
        for i, neighbors in enumerate(indices):
            field_count = sum(df.iloc[neighbors]['label'] == 'field')
            road_count = sum(df.iloc[neighbors]['label'] == 'road')
            if df.iloc[i]['label'] == 'road' and field_count > road_count:
                labels_corrected.append('field')
            else:
                labels_corrected.append(df.iloc[i]['label'])
        df['label'] = labels_corrected

        st.write("Clustering and labeling complete.")

        union_hull = None
        alpha = 0.08  # Controls level of concavity (lower = more concave). Try values between 0.01 and 1.0
        hulls_info = []

        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:
                continue

            cluster_df = df[(df['cluster'] == cluster_id) & (df['label'] == 'field')]
            if len(cluster_df) < 150:
                st.write(f"Skipping cluster {cluster_id} (too few points: {len(cluster_df)})")
                continue

            points_itm = list(zip(cluster_df['x'], cluster_df['y']))
            if len(points_itm) < 4:
                st.write(f"Skipping cluster {cluster_id} (less than 4 points)")
                continue

            try:
                hull = alphashape.alphashape(points_itm, alpha)
                if hull.geom_type == 'Polygon':
                    if union_hull is None:
                        union_hull = hull
                    else:
                        union_hull = union_hull.union(hull)
                else:
                    st.write(f"Cluster {cluster_id} hull not polygon (geom_type: {hull.geom_type}), skipped.")
            except Exception as e:
                st.warning(f"Skipping cluster {cluster_id} due to alphashape error: {e}")

        if union_hull is None:
            st.warning("No valid field hulls found.")
        else:
            st.write("Alpha shape hull(s) created and merged.")

        if union_hull is not None:
            for i, row in df[df['label'] == 'field'].iterrows():
                point = Point(row['x'], row['y'])
                inside = False
                if isinstance(union_hull, MultiPolygon):
                    for polygon in union_hull.geoms:
                        if polygon.contains(point):
                            inside = True
                            break
                elif union_hull.contains(point):
                    inside = True
                if not inside:
                    df.at[i, 'label'] = 'road'

            st.write("Relabeled field points outside the union hull as road.")

        center = [df['latitude'].mean(), df['longitude'].mean()]
        m = folium.Map(location=center, zoom_start=17, max_zoom=20, control_scale=True)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=True,
            control=True
        ).add_to(m)

        folium.LayerControl().add_to(m)

        for _, row in df.iterrows():
            color = "blue" if row['label'] == 'field' else "red"
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        hull_count = 0
        transformer_inv = Transformer.from_crs("EPSG:24378", "EPSG:4326", always_xy=True)

        if union_hull is not None:
            polygons = []
            if isinstance(union_hull, MultiPolygon):
                polygons = list(union_hull.geoms)
            elif isinstance(union_hull, Polygon):
                polygons = [union_hull]

            for polygon in polygons:
                hull_count += 1
                hull_latlon = [transformer_inv.transform(x, y)[::-1] for x, y in polygon.exterior.coords]
                area_m2 = polygon.area
                area_gunthas = area_m2 / 101.171367

                folium.Polygon(
                    locations=hull_latlon,
                    color='green',
                    fill=True,
                    fill_opacity=0.3,
                    popup=f"Field Cluster Area: {area_gunthas:.2f} gunthas",
                    tooltip=f"Area: {area_gunthas:.2f} gunthas"
                ).add_to(m)

                centroid = polygon.centroid
                centroid_latlon = transformer_inv.transform(centroid.x, centroid.y)[::-1]
                folium.Marker(
                    location=centroid_latlon,
                    popup=f"Hull_{hull_count} - Area: {area_gunthas:.2f} gunthas"
                ).add_to(m)

                hulls_info.append({
                    "Hull": f"Hull_{hull_count}",
                    "Area (gunthas)": round(area_gunthas, 2)
                })

        st.subheader("Field Map")
        st_data = st_folium(m, width=900, height=600)

        st.subheader("Areas of Detected Field Hulls (in Gunthas)")
        df_hulls = pd.DataFrame(hulls_info)
        if not df_hulls.empty:
            df_hulls = df_hulls.sort_values(by="Area (gunthas)", ascending=False).reset_index(drop=True)
            st.dataframe(df_hulls, use_container_width=True)
        else:
            st.info("No valid field hulls detected to show areas.")

    else:
        st.info("Please upload a CSV file to get started.")

except Exception as e:
    st.error(f"App crashed with error: {e}")
    st.text(traceback.format_exc())
