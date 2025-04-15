# ============================
# IMPORTS
# ============================

import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely import wkt
from shapely.geometry import Point
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import contextily as ctx
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

# ============================
# 1. LOAD DATA & PREPROCESS
# ============================

def load_poi_data(file_path, category_mapping_file):
    df = pd.read_csv(file_path, delimiter="|")
    df['pointx_class'] = df['pointx_class'].astype(str).str.zfill(8)

    group_mapping = pd.read_csv(category_mapping_file, delimiter="|", dtype=str)
    group_dict = group_mapping.set_index("Category Number")["Category Description"].to_dict()

    df["Category Number"] = df["pointx_class"].astype(str).str[2:4]
    df["POI_Category"] = df["Category Number"].map(group_dict)
    df = df.dropna(subset=["POI_Category"])

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.feature_easting, df.feature_northing))
    return gdf


def load_lsoa_data(lsoa_path):
    lsoa_map = pd.read_csv(lsoa_path, delimiter=",")
    lsoa_map['geometry'] = lsoa_map['the_geom'].apply(wkt.loads)
    lsoa_map = gpd.GeoDataFrame(lsoa_map, geometry='geometry')
    return lsoa_map


def add_lsoa_to_poi(gdf, lsoa_map):
    gdf = gdf.set_crs('EPSG:27700')
    lsoa_map = lsoa_map.set_crs('EPSG:27700')
    gdf = gpd.sjoin(gdf, lsoa_map[['LSOA21CD', 'geometry']], how="left", predicate="within")
    return gdf

# ============================
# 2. CREATE POI SEQUENCES
# ============================

def create_poi_sequences(gdf):
    sequences = []
    for lsoa in gdf["LSOA21CD"].unique():
        pois = gdf[gdf["LSOA21CD"] == lsoa]
        if len(pois) > 3:
            sequence = [lsoa] + pois["POI_Category"].tolist()
            sequences.append(sequence)
    return sequences

# ============================
# 3. TRAIN LDA MODEL
# ============================

def train_lda_model(poi_texts, num_topics=6):
    dictionary = Dictionary(poi_texts)
    corpus = [dictionary.doc2bow(text) for text in poi_texts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
    return lda_model, corpus, dictionary


def assign_lda_topics_to_lsoas(corpus, lda_model, poi_sequences):
    lsa_topic_mapping = {}
    for i, bow in enumerate(corpus):
        topic_distribution = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        lsa_topic_mapping[poi_sequences[i][0]] = dominant_topic
    return lsa_topic_mapping

# ============================
# 4. DOC2VEC MODEL & VECTORIZE
# ============================

def train_doc2vec_model(poi_sequences):
    tagged_data = [TaggedDocument(words=seq[1:], tags=[seq[0]]) for seq in poi_sequences]
    model = Doc2Vec(vector_size=20, window=5, min_count=2, workers=4, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def get_lsoa_vectors(model):
    lsoa_vectors = np.array([model.dv[tag] for tag in model.dv.index_to_key])
    lsoa_ids = list(model.dv.index_to_key)
    return lsoa_vectors, lsoa_ids

def get_lsoa_vectors_by_avg_wordvec(model, poi_sequences):
    """
    Get one vector per LSOA by averaging the word vectors of its POI categories.
    Only uses POI categories found in the Doc2Vec vocabulary.
    
    Returns:
        - lsoa_vectors: np.ndarray of shape (n_lsoas, vector_size)
        - lsoa_ids: list of LSOA codes
    """
    lsoa_vectors = []
    lsoa_ids = []

    for seq in poi_sequences:
        lsoa_id = seq[0]
        poi_categories = seq[1:]

        # Filter out POIs not in the vocabulary
        vectors = [model.wv[poi] for poi in poi_categories if poi in model.wv]
        
        if vectors:  # Only include if we have valid POIs
            avg_vector = np.mean(vectors, axis=0)
            lsoa_vectors.append(avg_vector)
            lsoa_ids.append(lsoa_id)

    return np.array(lsoa_vectors), lsoa_ids
# ============================
# 5. CLUSTERING
# ============================

def cluster_lsoas(kmeans, lsoa_vectors):
    return kmeans.fit_predict(lsoa_vectors)


def assign_clusters_to_poi(gdf, cluster_df):
    gdf['Cluster'] = gdf['LSOA21CD'].map(cluster_df.set_index('LSOA')['Cluster'])
    return gdf

# ============================
# 6. ASSIGN FUNCTIONAL LABELS
# ============================

def assign_functional_labels_to_clusters(gdf, cluster_df):
    cluster_labels = {}
    for cluster in cluster_df["Cluster"].unique():
        subset = gdf[gdf["Cluster"] == cluster]
        most_common_topic = subset["LDA_Functional_Area"].mode()[0]
        cluster_labels[cluster] = f"Zone {most_common_topic}"
    gdf["Functional_Zone"] = gdf["Cluster"].map(cluster_labels)
    return gdf


def name_clusters_by_top3_enrichment(gdf, cluster_column="Cluster", category_column="POI_Category"):
    overall_counts = gdf[category_column].value_counts(normalize=True)
    cluster_names = {}

    for cluster in gdf[cluster_column].unique():
        cluster_data = gdf[gdf[cluster_column] == cluster]
        cluster_counts = cluster_data[category_column].value_counts(normalize=True)
        enrichment = (cluster_counts / overall_counts).dropna()

        if enrichment.empty:
            cluster_names[cluster] = "Uncategorized"
        else:
            top3 = enrichment.nlargest(3)
            top3_labels = [f"{category} (x{factor:.2f})" for category, factor in top3.items()]
            cluster_names[cluster] = ", ".join(top3_labels)

    gdf["Cluster_Name"] = gdf[cluster_column].map(cluster_names)
    return gdf, cluster_names

# ============================
# 7. VISUALIZATION
# ============================

def plot_functional_zones(gdf, output_path):
    gdf_web = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(12, 12))

    gdf_web.plot(
        column='Cluster_Name',
        ax=ax,
        legend=True,
        cmap='Set3',
        alpha=0.7,
        legend_kwds={'title': "Urban Functional Area"}
    )

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
    ax.set_axis_off()
    plt.title("Urban Functional Areas Over OS Map", fontsize=16)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Map saved to: {output_path}")
        plt.show()
    else:
        plt.show()

    plt.close()


def plot_pca_clusters(lsoa_vectors, clusters, output_path):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(lsoa_vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='Set3', s=50)
    plt.title("2D PCA of LSOA Vectors (Clustered)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Cluster")

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"PCA plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def display_lda_topics(lda_model, num_topics=6):
    for topic_id in range(num_topics):
        words = lda_model.show_topic(topic_id, topn=5)
        print(f"Topic {topic_id}: {[word[0] for word in words]}")


def plot_lsoa_clusters_as_polygons(gdf_with_clusters, lsoa_map, output_path):
    print(gdf_with_clusters.head())
    lsoa_cluster_df = gdf_with_clusters[['LSOA21CD', 'Cluster', 'Cluster_Name']].drop_duplicates()
    lsoa_polygons = lsoa_map.merge(lsoa_cluster_df, on='LSOA21CD', how='inner')

    if lsoa_polygons.crs is None:
        lsoa_polygons.set_crs(epsg=27700, inplace=True)

    lsoa_polygons = lsoa_polygons.to_crs(epsg=4326)
    center = [lsoa_polygons.geometry.centroid.y.mean(), lsoa_polygons.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    clusters = lsoa_polygons['Cluster'].unique()
    colormap = cm.linear.Set3_09.scale(0, len(clusters)).to_step(n=len(clusters))
    color_dict = {name: colormap(i) for i, name in enumerate(clusters)}

    for _, row in lsoa_polygons.iterrows():
        sim_geo = gpd.GeoSeries(row['geometry']).simplify(0.00000000000000001)
        geo_json = sim_geo.to_json()
        folium.GeoJson(
            data=geo_json,
            style_function=lambda x, color=color_dict[row['Cluster']]: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6
            },
            tooltip=folium.Tooltip(
                f"LSOA: {row['LSOA21CD']}<br>Cluster: {row['Cluster']}<br>Label: {row['Cluster_Name']}"
            )
        ).add_to(m)

    colormap.caption = 'Urban Functional Zones'
    colormap.add_to(m)

    if output_path:
        m.save(output_path)
        print(f"Interactive polygon map saved to: {output_path}")
    else:
        return m

# ============================
# MAIN SCRIPT EXECUTION
# ============================

def main():
    poi_folder = r"C:\Users\c22018931\OneDrive - Cardiff University\Desktop\Urban_Classification\poi-extracts"
    category_mapping_file = r"C:\Users\c22018931\OneDrive - Cardiff University\Desktop\Urban_Classification\poi_docs\POI_CATEGORIES.txt"
    lsoa_path = r"C:\Users\c22018931\OneDrive - Cardiff University\Desktop\Urban_Classification\Doc2VEc\lsoa_2021_w_hwm.csv"

    poi_files = glob.glob(os.path.join(poi_folder, "*.csv"))
    if not poi_files:
        print("No CSV files found in the POI folder.")
        return

    print("\nAvailable POI files:")
    for i, file in enumerate(poi_files):
        print(f"{i + 1}. {os.path.basename(file)}")

    selection = input("\nEnter the number *or* name of the city file you want to process: ").strip()

    file_path = None
    if selection.isdigit():
        index = int(selection) - 1
        if 0 <= index < len(poi_files):
            file_path = poi_files[index]
    else:
        for f in poi_files:
            if selection.lower() in os.path.basename(f).lower():
                file_path = f
                break

    if not file_path:
        print("Invalid selection. Exiting.")
        return

    print(f"\nProcessing: {os.path.basename(file_path)}")

    lsoa_map = load_lsoa_data(lsoa_path)
    gdf = load_poi_data(file_path, category_mapping_file)
    gdf = add_lsoa_to_poi(gdf, lsoa_map)

    poi_sequences = create_poi_sequences(gdf)
    if len(poi_sequences) == 0:
        print("Not enough POIs to process this city.")
        return

    poi_texts = [seq[1:] for seq in poi_sequences]
    lda_model, corpus, dictionary = train_lda_model(poi_texts)
    lsa_topic_mapping = assign_lda_topics_to_lsoas(corpus, lda_model, poi_sequences)
    gdf["LDA_Functional_Area"] = gdf["LSOA21CD"].map(lsa_topic_mapping)

    model = train_doc2vec_model(poi_sequences)
    lsoa_vectors, lsoa_ids = get_lsoa_vectors(model)

    # NEW: Generate vectors using the average word vectors for POI categories
    lsoa_vectors_avg, _ = get_lsoa_vectors_by_avg_wordvec(model, poi_sequences)

    # Perform clustering for both methods
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters_doc2vec = cluster_lsoas(kmeans, lsoa_vectors)
    clusters_avg_poi = cluster_lsoas(kmeans, lsoa_vectors_avg)

    # Create cluster DataFrames for both methods
    cluster_df_doc2vec = pd.DataFrame({"LSOA": lsoa_ids, "Cluster": clusters_doc2vec})
    cluster_df_avg_poi = pd.DataFrame({"LSOA": lsoa_ids, "Cluster": clusters_avg_poi})

    # Assign clusters to POIs for both methods
    gdf_doc2vec = assign_clusters_to_poi(gdf.copy(), cluster_df_doc2vec)
    gdf_avg_poi = assign_clusters_to_poi(gdf.copy(), cluster_df_avg_poi)

    # NEW: Plot maps for both clustering methods
    output_dir = r"C:\Users\c22018931\OneDrive - Cardiff University\Desktop\Urban_Classification\output"
    os.makedirs(output_dir, exist_ok=True)

    city_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a subdirectory for the city
    city_output_dir = os.path.join(output_dir, city_name)
    os.makedirs(city_output_dir, exist_ok=True)

    # Generate cluster names based on top enrichment scores
    gdf_doc2vec, cluster_names_doc2vec = name_clusters_by_top3_enrichment(gdf_doc2vec, cluster_column="Cluster", category_column="POI_Category")
    gdf_avg_poi, cluster_names_avg_poi = name_clusters_by_top3_enrichment(gdf_avg_poi, cluster_column="Cluster", category_column="POI_Category")

    # Optionally, plot PCA of the clusters for both methods
    plot_pca_clusters(lsoa_vectors, clusters_doc2vec, output_path=os.path.join(city_output_dir, f"{city_name}_doc2vec_pca.png"))
    plot_pca_clusters(lsoa_vectors_avg, clusters_avg_poi, output_path=os.path.join(city_output_dir, f"{city_name}_averaged_poi_pca.png"))

    # Display LDA topics
    display_lda_topics(lda_model)

    # Ensure no NaN values in the Cluster column
    gdf_doc2vec['Cluster'] = gdf_doc2vec['Cluster'].fillna('Unassigned')
    gdf_avg_poi['Cluster'] = gdf_avg_poi['Cluster'].fillna('Unassigned')

    # Reproject to a projected CRS before calculating centroids
    if gdf_doc2vec.crs.is_geographic:
        gdf_doc2vec = gdf_doc2vec.to_crs(epsg=3857)  # Example: Web Mercator
    if gdf_avg_poi.crs.is_geographic:
        gdf_avg_poi = gdf_avg_poi.to_crs(epsg=3857)

    # Plot the LSOA clusters as polygons (with functional area labels)
    plot_lsoa_clusters_as_polygons(
        gdf_with_clusters=gdf_doc2vec,
        lsoa_map=lsoa_map,
        output_path=os.path.join(city_output_dir, f"{city_name}_doc2vec_cluster_zones.html")
    )
    plot_lsoa_clusters_as_polygons(
        gdf_with_clusters=gdf_avg_poi,
        lsoa_map=lsoa_map,
        output_path=os.path.join(city_output_dir, f"{city_name}_averaged_poi_cluster_zones.html")
    )

if __name__ == "__main__":
    main()

