import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def transform_texts_to_vectors_improved(texts):
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_df=0.85, min_df=2)
    X = vectorizer.fit_transform(texts)
    return X


def determine_optimal_clusters_using_silhouette(X):
    silhouette_scores = []

    range_n_clusters = range(2, min(60, X.shape[
        0]))
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_clusters = 2 + silhouette_scores.index(max(silhouette_scores))
    return optimal_clusters


def perform_kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans.cluster_centers_


def visualize_clusters_interactive(X, clusters, texts):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X.toarray())
    df = pd.DataFrame(reduced_features, columns=['PCA 1', 'PCA 2'])
    df['Cluster'] = clusters
    df['Text'] = texts

    fig = px.scatter(df, x='PCA 1', y='PCA 2', color='Cluster', hover_data=['Text'],
                     title='Results KMeans')

    output_file_path = "kmeans_clustering_plot.html"
    fig.write_html(output_file_path)

    print(f"Save result in  {output_file_path}.")


def main_cluster_without_lemmatization(texts):
    X = transform_texts_to_vectors_improved(texts)
    optimal_clusters = determine_optimal_clusters_using_silhouette(X)
    cluster_labels, _ = perform_kmeans_clustering(X, optimal_clusters)
    visualize_clusters_interactive(X, cluster_labels, texts)

    clusters_dict = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(texts[i])

    return X, clusters_dict
