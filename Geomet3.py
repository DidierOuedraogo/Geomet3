import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import shap
import io
import base64
import hashlib
from scipy.stats import pearsonr, spearmanr, f_oneway
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.cm as cm
import matplotlib as mpl
import io
from PIL import Image
import joblib
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static
import altair as alt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Tentative d'importation de TensorFlow - gérer l'importation de façon robuste
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow n'est pas disponible. Les fonctionnalités de réseaux de neurones seront désactivées.")

# Fonction pour créer une image de légende colorbar en base64
def cm_to_base64(cmap, min_val, max_val):
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.subplots_adjust(bottom=0.5)
    
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='horizontal')
    
    # Convert figure to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str

# Système d'authentification
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Configuration des utilisateurs et mots de passe avec des mots de passe plus complexes
    users = {
        "didier": "Geo_Metal2025!",
        "admin": "S3cur3P@ssw0rd!"
    }
    
    # Initialiser l'état d'authentification s'il n'existe pas
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
        
    # Si l'utilisateur est déjà authentifié, retourner True
    if st.session_state["authentication_status"]:
        return True
    
    # Sinon, afficher le formulaire de connexion
    st.title("Application de prédiction géométallurgique")
    st.subheader("Connexion")
    
    # Formulaire de connexion
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
    
    if submitted:
        # Vérifier les identifiants
        if username in users and users[username] == password:
            st.session_state["authentication_status"] = True
            st.session_state["username"] = username
            st.success(f"Bienvenue {username}!")
            st.rerun()
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect")
            return False
    
    return False
# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Récupération Métallurgique",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Vérifier l'authentification
if not check_password():
    st.stop()  # Arrêter l'exécution si l'authentification a échoué

# Styles CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .author {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
    .user-info {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #f8f9fa;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        z-index: 1000;
    }
    .logout-btn {
        color: #e74c3c;
        cursor: pointer;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Affichage d'informations sur l'utilisateur connecté et bouton de déconnexion
st.markdown(
    f"""
    <div class="user-info">
        Connecté en tant que: {st.session_state["username"]}
        <span class="logout-btn" onclick="parent.window.location.href='{st.get_option('server.baseUrlPath')}?logout=true'">Déconnexion</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Gérer la déconnexion via paramètre URL
if st.query_params.get("logout", [""])[0] == "true":
    st.session_state["authentication_status"] = False
    st.session_state["username"] = ""
    st.query_params.clear()
    st.rerun()

# Titre de l'application
st.markdown("<h1 class='main-header'>Prédiction de Récupération Métallurgique</h1>", unsafe_allow_html=True)
st.markdown("<p class='author'>Auteur: Didier Ouedraogo, P.Geo, Géologue et Data Scientist</p>", unsafe_allow_html=True)

# Fonction pour télécharger le DataFrame
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Sidebar pour la navigation
st.sidebar.title("Navigation")
pages = ["Accueil", "Importation des données", "Mappage des données", "Exploration des données", "Data Augmentation", "Modélisation", "Prédiction"]
page = st.sidebar.radio("Aller à", pages)

# Variables de session pour stocker les données
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_augmented' not in st.session_state:
    st.session_state.data_augmented = None    
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_pipeline' not in st.session_state:
    st.session_state.model_pipeline = None
if 'geo_data' not in st.session_state:
    st.session_state.geo_data = None
if 'neural_network' not in st.session_state:
    st.session_state.neural_network = None
if 'model_is_fitted' not in st.session_state:
    st.session_state.model_is_fitted = False

# Page d'accueil
if page == "Accueil":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ## Application de prédiction de récupération métallurgique
    
    Cette application vous permet de prédire la récupération métallurgique en utilisant des données de tests géométallurgiques et des algorithmes de machine learning avancés.
    
    ### Fonctionnalités principales
    
    - **Importation de données**: Téléchargez vos données géométallurgiques au format CSV ou Excel
    - **Mappage des données**: Visualisez la distribution spatiale de vos échantillons géométallurgiques
    - **Exploration des données**: Visualisez les corrélations et tendances dans vos données
    - **Data Augmentation**: Augmentez votre jeu de données pour améliorer les performances des modèles
    - **Modélisation**: Entraînez et évaluez différents modèles de machine learning (incluant les réseaux de neurones)
    - **Prédiction**: Utilisez le modèle entraîné pour prédire la récupération métallurgique de nouveaux échantillons
    
    ### Comment utiliser cette application
    
    1. Importez vos données dans l'onglet "Importation des données"
    2. Visualisez la distribution spatiale dans l'onglet "Mappage des données" (si coordonnées disponibles)
    3. Explorez vos données dans l'onglet "Exploration des données"
    4. Utilisez les techniques de data augmentation si nécessaire
    5. Créez un modèle prédictif dans l'onglet "Modélisation"
    6. Faites des prédictions sur de nouveaux échantillons dans l'onglet "Prédiction"
    
    ### Données requises
    
    Pour obtenir les meilleurs résultats, vos données doivent inclure:
    - Propriétés minéralogiques (teneur, taille des grains, etc.)
    - Paramètres de traitement (pH, densité de pulpe, etc.)
    - Résultats des tests (récupération métallurgique)
    - Coordonnées spatiales (optionnel, pour le mappage)
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Exemple de dataset
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Exemple de dataset")
    st.markdown("Si vous n'avez pas de données, vous pouvez utiliser notre jeu de données d'exemple:")
    
    # Création d'un jeu de données d'exemple avec des coordonnées spatiales
    np.random.seed(42)
    n_samples = 100
    
    # Créer des coordonnées spatiales simulant un gisement
    # Centre du gisement
    lat_center = 12.3657
    lon_center = -1.5339
    
    example_data = pd.DataFrame({
        'Teneur_Cu': np.random.uniform(0.5, 3.0, n_samples),
        'Teneur_Au': np.random.uniform(0.1, 1.5, n_samples),
        'Taille_Grain': np.random.uniform(10, 150, n_samples),
        'Durete': np.random.uniform(1, 5, n_samples),
        'pH': np.random.uniform(7, 11, n_samples),
        'Temps_Flottation': np.random.uniform(5, 15, n_samples),
        'Densite_Pulpe': np.random.uniform(20, 40, n_samples),
        'Recuperation': np.random.uniform(60, 95, n_samples),
        'Latitude': lat_center + np.random.normal(0, 0.02, n_samples),
        'Longitude': lon_center + np.random.normal(0, 0.02, n_samples),
        'Profondeur': np.random.uniform(10, 500, n_samples),
        'Type_Minerai': np.random.choice(['Oxyde', 'Sulfure', 'Mixte'], n_samples)
    })
    
    # Ajout d'une relation pour simuler des données réalistes
    example_data['Recuperation'] = 70 + 5 * example_data['Teneur_Cu'] - 3 * example_data['Taille_Grain'] / 100 + 2 * example_data['pH'] + np.random.normal(0, 3, n_samples)
    example_data['Recuperation'] = example_data['Recuperation'].clip(60, 95)
    
    # Ajouter une relation entre le type de minerai et la récupération
    for i, row in example_data.iterrows():
        if row['Type_Minerai'] == 'Oxyde':
            example_data.loc[i, 'Recuperation'] = row['Recuperation'] * 0.9
        elif row['Type_Minerai'] == 'Mixte':
            example_data.loc[i, 'Recuperation'] = row['Recuperation'] * 0.95
    
    st.dataframe(example_data.head())
    
    if st.button("Utiliser cet exemple de données"):
        st.session_state.data = example_data
        st.success("Données d'exemple chargées avec succès!")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
# Page d'importation des données
elif page == "Importation des données":
    st.markdown("<h2 class='sub-header'>Importation des Données</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### Instructions
    
    Téléchargez votre fichier de données géométallurgiques au format CSV ou Excel.
    Assurez-vous que votre fichier contient:
    
    - Une variable cible (récupération métallurgique)
    - Des variables prédictives (propriétés minéralogiques, paramètres de traitement, etc.)
    - Des coordonnées spatiales (optionnel, pour le mappage)
    """)
    
    upload_option = st.radio("Choisissez une option", ["Télécharger un fichier", "Utiliser l'exemple de données"])
    
    if upload_option == "Télécharger un fichier":
        uploaded_file = st.file_uploader("Télécharger votre fichier de données", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.success(f"Fichier chargé avec succès! Dimensions: {data.shape[0]} lignes × {data.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
    else:
        if st.button("Charger les données d'exemple"):
            # Création d'un jeu de données d'exemple (comme dans la page d'accueil)
            np.random.seed(42)
            n_samples = 100
            
            # Centre du gisement
            lat_center = 12.3657
            lon_center = -1.5339
            
            example_data = pd.DataFrame({
                'Teneur_Cu': np.random.uniform(0.5, 3.0, n_samples),
                'Teneur_Au': np.random.uniform(0.1, 1.5, n_samples),
                'Taille_Grain': np.random.uniform(10, 150, n_samples),
                'Durete': np.random.uniform(1, 5, n_samples),
                'pH': np.random.uniform(7, 11, n_samples),
                'Temps_Flottation': np.random.uniform(5, 15, n_samples),
                'Densite_Pulpe': np.random.uniform(20, 40, n_samples),
                'Recuperation': np.random.uniform(60, 95, n_samples),
                'Latitude': lat_center + np.random.normal(0, 0.02, n_samples),
                'Longitude': lon_center + np.random.normal(0, 0.02, n_samples),
                'Profondeur': np.random.uniform(10, 500, n_samples),
                'Type_Minerai': np.random.choice(['Oxyde', 'Sulfure', 'Mixte'], n_samples)
            })
            
            example_data['Recuperation'] = 70 + 5 * example_data['Teneur_Cu'] - 3 * example_data['Taille_Grain'] / 100 + 2 * example_data['pH'] + np.random.normal(0, 3, n_samples)
            example_data['Recuperation'] = example_data['Recuperation'].clip(60, 95)
            
            # Ajouter une relation entre le type de minerai et la récupération
            for i, row in example_data.iterrows():
                if row['Type_Minerai'] == 'Oxyde':
                    example_data.loc[i, 'Recuperation'] = row['Recuperation'] * 0.9
                elif row['Type_Minerai'] == 'Mixte':
                    example_data.loc[i, 'Recuperation'] = row['Recuperation'] * 0.95
            
            st.session_state.data = example_data
            st.success("Données d'exemple chargées avec succès!")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Affichage des données si elles sont disponibles
    if st.session_state.data is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Aperçu des Données")
        
        data = st.session_state.data
        st.dataframe(data.head())
        
        st.markdown("### Statistiques Descriptives")
        st.dataframe(data.describe())
        
        st.markdown("### Informations sur les Données")
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        # Vérification des valeurs manquantes
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            st.warning("Votre jeu de données contient des valeurs manquantes:")
            st.dataframe(missing_values[missing_values > 0])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Option pour télécharger les données nettoyées
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Préparation des Données")
        
        # Nettoyage des données
        if st.checkbox("Nettoyer automatiquement les données"):
            # Supprimer les lignes avec valeurs manquantes
            data_cleaned = data.dropna()
            
            # Afficher les résultats du nettoyage
            st.write(f"Données avant nettoyage: {data.shape[0]} lignes")
            st.write(f"Données après nettoyage: {data_cleaned.shape[0]} lignes")
            
            # Option pour remplacer le jeu de données original
            if st.button("Utiliser les données nettoyées"):
                st.session_state.data = data_cleaned
                st.success("Données nettoyées appliquées avec succès!")
                
            # Lien de téléchargement pour les données nettoyées
            st.markdown(get_download_link(data_cleaned, "donnees_nettoyees.csv", "Télécharger les données nettoyées"), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Page de mappage des données
elif page == "Mappage des données":
    st.markdown("<h2 class='sub-header'>Mappage des Données</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez aller à la page 'Importation des données' pour charger vos données.")
    else:
        data = st.session_state.data
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Visualisation Spatiale des Données")
        
        # Vérifier la présence de coordonnées spatiales
        has_coordinates = False
        lat_col = None
        lon_col = None
        
        # Détection automatique des colonnes de latitude et longitude
        possible_lat_columns = ['lat', 'latitude', 'y', 'northing', 'north']
        possible_lon_columns = ['lon', 'long', 'longitude', 'x', 'easting', 'east']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(lat_name in col_lower for lat_name in possible_lat_columns):
                lat_col = col
            if any(lon_name in col_lower for lon_name in possible_lon_columns):
                lon_col = col
        
        # Interface utilisateur pour sélectionner les colonnes de coordonnées
        st.markdown("#### Sélection des Coordonnées Spatiales")
        
        if lat_col and lon_col:
            st.success(f"Coordonnées détectées automatiquement: Latitude = '{lat_col}', Longitude = '{lon_col}'")
        else:
            st.info("Coordonnées non détectées automatiquement. Veuillez sélectionner manuellement les colonnes de latitude et longitude.")
        
        lat_col = st.selectbox("Colonne de latitude", options=[None] + list(data.columns), index=0 if lat_col is None else list(data.columns).index(lat_col) + 1)
        lon_col = st.selectbox("Colonne de longitude", options=[None] + list(data.columns), index=0 if lon_col is None else list(data.columns).index(lon_col) + 1)
        
        has_coordinates = lat_col is not None and lon_col is not None
        
        if has_coordinates:
            # Sélectionner variable pour la coloration
            color_var = st.selectbox("Variable pour la coloration des points", 
                                    options=data.select_dtypes(include=['float64', 'int64']).columns,
                                    index=0)
            
            # Créer la carte
            st.markdown("#### Carte des Échantillons")
            
            # Vérifier que les coordonnées sont numériques
            if pd.api.types.is_numeric_dtype(data[lat_col]) and pd.api.types.is_numeric_dtype(data[lon_col]):
                # Filtrer les données valides pour la carte
                map_data = data.dropna(subset=[lat_col, lon_col, color_var])
                
                # Vérifier que les valeurs sont dans les plages valides
                map_data = map_data[(map_data[lat_col] >= -90) & (map_data[lat_col] <= 90) & 
                                    (map_data[lon_col] >= -180) & (map_data[lon_col] <= 180)]
                
                if not map_data.empty:
                    # Calculer le centre de la carte
                    center_lat = map_data[lat_col].mean()
                    center_lon = map_data[lon_col].mean()
                    
                    # Créer la carte
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                    
                    # Normaliser la variable de couleur
                    norm = plt.Normalize(map_data[color_var].min(), map_data[color_var].max())
                    cmap = plt.cm.viridis
                    
                    # Ajouter les marqueurs groupés
                    marker_cluster = MarkerCluster().add_to(m)
                    
                    for idx, row in map_data.iterrows():
                        color = plt.cm.viridis(norm(row[color_var]))
                        hex_color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                        
                        # Création du popup avec les informations sur l'échantillon
                        popup_text = f"""
                        <b>ID:</b> {idx}<br>
                        <b>{color_var}:</b> {row[color_var]:.2f}<br>
                        """
                        
                        # Ajouter d'autres variables
                        for col in data.columns:
                            if col not in [lat_col, lon_col] and col != color_var:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    popup_text += f"<b>{col}:</b> {row[col]:.2f}<br>"
                                else:
                                    popup_text += f"<b>{col}:</b> {row[col]}<br>"
                        
                        folium.CircleMarker(
                            location=[row[lat_col], row[lon_col]],
                            radius=8,
                            color=hex_color,
                            fill=True,
                            fill_color=hex_color,
                            fill_opacity=0.7,
                            popup=folium.Popup(popup_text, max_width=300)
                        ).add_to(marker_cluster)
                    
                    # Ajouter la légende (en tant qu'image de base64)
                    colormap = cm_to_base64(cmap, min_val=map_data[color_var].min(), max_val=map_data[color_var].max())
                    
                    # Afficher la carte
                    folium_static(m)
                    
                    # Afficher la légende
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 10px;">
                        <p style="margin-bottom: 5px;"><b>{color_var}</b></p>
                        <img src="data:image/png;base64,{colormap}" style="width: 200px;">
                        <div style="display: flex; justify-content: space-between; width: 200px; margin: 0 auto;">
                            <span>{map_data[color_var].min():.2f}</span>
                            <span>{map_data[color_var].max():.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Analyse spatiale
                    st.markdown("#### Analyse Spatiale")
                    
                    # Option pour effectuer une analyse par clustering spatial
                    if st.checkbox("Effectuer une analyse par clustering spatial"):
                        n_clusters = st.slider("Nombre de clusters", 2, 10, 3)
                        
                        # Préparer les données pour le clustering
                        cluster_data = map_data[[lat_col, lon_col]].values
                        
                        # Appliquer KMeans
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        map_data['Cluster'] = kmeans.fit_predict(cluster_data)
                        
                        # Créer une nouvelle carte avec les clusters
                        m_cluster = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        
                        # Palette de couleurs pour les clusters
                        cluster_colors = plt.cm.tab10.colors
                        
                        # Ajouter les marqueurs avec la couleur du cluster
                        for idx, row in map_data.iterrows():
                            cluster_id = int(row['Cluster'])
                            color = cluster_colors[cluster_id % len(cluster_colors)]
                            hex_color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                            
                            folium.CircleMarker(
                                location=[row[lat_col], row[lon_col]],
                                radius=8,
                                color=hex_color,
                                fill=True,
                                fill_color=hex_color,
                                fill_opacity=0.7,
                                popup=f"ID: {idx}, Cluster: {cluster_id}, {color_var}: {row[color_var]:.2f}"
                            ).add_to(m_cluster)
                        
                        # Ajouter les centres des clusters
                        for i, center in enumerate(kmeans.cluster_centers_):
                            color = cluster_colors[i % len(cluster_colors)]
                            hex_color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                            
                            folium.CircleMarker(
                                location=[center[0], center[1]],
                                radius=12,
                                color='black',
                                fill=True,
                                fill_color=hex_color,
                                fill_opacity=0.9,
                                popup=f"Centre du cluster {i}"
                            ).add_to(m_cluster)
                        
                        # Afficher la carte des clusters
                        st.markdown("##### Carte des Clusters Spatiaux")
                        folium_static(m_cluster)
                        
                        # Statistiques par cluster
                        st.markdown("##### Statistiques par Cluster")
                        cluster_stats = map_data.groupby('Cluster')[color_var].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
                        st.dataframe(cluster_stats)
                        
                        # Visualiser la distribution de la variable cible par cluster
                        fig = px.box(map_data, x='Cluster', y=color_var, 
                                    title=f"Distribution de {color_var} par Cluster")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stocker les données géospatiales pour les autres pages
                        st.session_state.geo_data = map_data
                else:
                    st.error("Aucune donnée valide pour créer la carte. Vérifiez que vos coordonnées sont valides.")
            else:
                st.error("Les colonnes sélectionnées ne sont pas numériques. Veuillez sélectionner des colonnes numériques pour les coordonnées.")
        else:
            st.warning("Veuillez sélectionner les colonnes de latitude et longitude pour afficher la carte.")
            
            # Option pour générer des coordonnées simulées
            if st.button("Générer des coordonnées factices pour test"):
                # Générer des coordonnées centrées autour de Ouagadougou, Burkina Faso
                n_samples = len(data)
                lat_center = 12.3657
                lon_center = -1.5339
                
                data['Latitude'] = lat_center + np.random.normal(0, 0.02, n_samples)
                data['Longitude'] = lon_center + np.random.normal(0, 0.02, n_samples)
                
                st.session_state.data = data
                st.success("Coordonnées fictives générées. Veuillez recharger cette page pour voir la carte.")
                st.markdown("*Note: Ces coordonnées sont fictives et générées uniquement à des fins de test.*")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Statistiques spatiales avancées
        if has_coordinates and 'map_data' in locals() and not map_data.empty:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Analyses Spatiales Avancées")
            
            analysis_type = st.selectbox(
                "Type d'analyse spatiale", 
                ["Carte de chaleur", "Analyse de tendance", "Variogramme simplifié"]
            )
            
            if analysis_type == "Carte de chaleur":
                st.markdown("#### Carte de Chaleur de la Récupération Métallurgique")
                
                target_var = st.selectbox("Variable à visualiser", 
                                         options=data.select_dtypes(include=['float64', 'int64']).columns,
                                         index=0)
                
                # Créer une carte de chaleur
                m_heat = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                
                # Préparer les données pour la carte de chaleur
                heat_data = [[row[lat_col], row[lon_col], row[target_var]] for _, row in map_data.iterrows() if not pd.isna(row[target_var])]
                
                # Ajouter la carte de chaleur
                HeatMap(heat_data, min_opacity=0.5, max_zoom=18, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 0.9: 'red'}).add_to(m_heat)
                
                # Afficher la carte
                folium_static(m_heat)
                
            elif analysis_type == "Analyse de tendance":
                st.markdown("#### Analyse de Tendance Spatiale")
                
                target_var = st.selectbox("Variable à analyser", 
                                         options=data.select_dtypes(include=['float64', 'int64']).columns,
                                         index=0)
                
                # Créer un graphique 3D de tendance
                fig = go.Figure(data=[go.Scatter3d(
                    x=map_data[lon_col],
                    y=map_data[lat_col],
                    z=map_data[target_var],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=map_data[target_var],
                        colorscale='Viridis',
                        colorbar=dict(title=target_var),
                        opacity=0.8
                    )
                )])
                
                fig.update_layout(
                    title=f'Analyse de tendance spatiale pour {target_var}',
                    scene=dict(
                        xaxis_title='Longitude',
                        yaxis_title='Latitude',
                        zaxis_title=target_var
                    ),
                    width=800,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Régression de surface
                if st.checkbox("Afficher la régression de surface"):
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    from sklearn.pipeline import Pipeline
                    
                    # Créer un grid pour la prédiction
                    lon_range = np.linspace(map_data[lon_col].min(), map_data[lon_col].max(), 20)
                    lat_range = np.linspace(map_data[lat_col].min(), map_data[lat_col].max(), 20)
                    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
                    
                    # Préparer le modèle
                    degree = st.slider("Degré du polynôme", 1, 5, 2)
                    model = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree)),
                        ('linear', LinearRegression())
                    ])
                    
                    # Entraîner le modèle
                    X = map_data[[lon_col, lat_col]]
                    y = map_data[target_var]
                    model.fit(X, y)
                    
                    # Prédire sur la grille
                    grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
                    try:
                        predictions = model.predict(grid_points).reshape(lon_grid.shape)
                        
                        # Graphique 3D avec surface de régression
                        fig = go.Figure()
                        
                        # Ajouter les points de données
                        fig.add_trace(go.Scatter3d(
                            x=map_data[lon_col],
                            y=map_data[lat_col],
                            z=map_data[target_var],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=map_data[target_var],
                                colorscale='Viridis',
                                opacity=0.8
                            ),
                            name='Données'
                        ))
                        
                        # Ajouter la surface de régression
                        fig.add_trace(go.Surface(
                            x=lon_range,
                            y=lat_range,
                            z=predictions,
                            colorscale='Viridis',
                            opacity=0.7,
                            name='Surface de régression'
                        ))
                        
                        fig.update_layout(
                            title=f'Régression de surface pour {target_var} (Polynôme degré {degree})',
                            scene=dict(
                                xaxis_title='Longitude',
                                yaxis_title='Latitude',
                                zaxis_title=target_var
                            ),
                            width=800,
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur lors de la création de la surface de régression: {e}")
                
            elif analysis_type == "Variogramme simplifié":
                st.markdown("#### Variogramme Simplifié")
                
                target_var = st.selectbox("Variable à analyser", 
                                         options=data.select_dtypes(include=['float64', 'int64']).columns,
                                         index=0)
                
                # Calcul simplifié d'un variogramme expérimental
                max_distance = st.slider("Distance maximale (°)", 0.001, 0.1, 0.05)
                n_bins = st.slider("Nombre de bins", 5, 50, 20)
                
                # Calculer les distances entre les points
                from scipy.spatial.distance import pdist, squareform
                
                coords = map_data[[lon_col, lat_col]].values
                distances = squareform(pdist(coords))
                
                # Calculer les différences de valeurs au carré
                values = map_data[target_var].values
                value_diff_squared = np.zeros_like(distances)
                
                for i in range(len(values)):
                    for j in range(len(values)):
                        value_diff_squared[i, j] = (values[i] - values[j]) ** 2
                
                # Créer les bins
                bins = np.linspace(0, max_distance, n_bins + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Calcul du variogramme
                variogram = np.zeros(n_bins)
                counts = np.zeros(n_bins)
                
                for i in range(len(distances)):
                    for j in range(i+1, len(distances)):
                        dist = distances[i, j]
                        if dist <= max_distance:
                            bin_idx = np.digitize(dist, bins) - 1
                            if 0 <= bin_idx < n_bins:
                                variogram[bin_idx] += value_diff_squared[i, j]
                                counts[bin_idx] += 1
                
                # Normaliser
                for i in range(n_bins):
                    if counts[i] > 0:
                        variogram[i] /= (2 * counts[i])
                
                # Graphique du variogramme
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=bin_centers,
                    y=variogram,
                    mode='markers+lines',
                    name='Variogramme empirique'
                ))
                
                fig.update_layout(
                    title=f'Variogramme simplifié pour {target_var}',
                    xaxis_title='Distance (°)',
                    yaxis_title='Semi-variance',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Note sur l'interprétation du variogramme:**
                - La semi-variance indique la variabilité spatiale de la variable à différentes distances
                - Un plateau indique la distance au-delà de laquelle les échantillons ne sont plus corrélés
                - La "portée" est la distance où le variogramme atteint le plateau
                - Le "palier" est la valeur de semi-variance au plateau
                - L'"effet pépite" est la semi-variance à distance nulle (indique la variabilité à très petite échelle)
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Page d'exploration des données
elif page == "Exploration des données":
    st.markdown("<h2 class='sub-header'>Exploration des Données</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez aller à la page 'Importation des données' pour charger vos données.")
    else:
        data = st.session_state.data
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Sélection de la Variable Cible")
        
        # Sélection de la variable cible avec "Recuperation" comme valeur par défaut si elle existe
        default_target_idx = data.columns.get_loc("Recuperation") if "Recuperation" in data.columns else 0
        target_col = st.selectbox("Sélectionnez la variable cible (récupération métallurgique)", data.columns, index=default_target_idx)
        st.session_state.target = target_col
        
        # Sélection des features
        st.markdown("### Sélection des Variables Prédictives")
        feature_cols = st.multiselect("Sélectionnez les variables prédictives", 
                                     [col for col in data.columns if col != target_col],
                                     default=[col for col in data.columns if col != target_col 
                                              and pd.api.types.is_numeric_dtype(data[col])])
        st.session_state.features = feature_cols
        
        if not feature_cols:
            st.error("Veuillez sélectionner au moins une variable prédictive.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if feature_cols:
            # Visualisation des distributions
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Distribution des Variables")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.subheader("Distribution de la Variable Cible")
                fig = px.histogram(data, x=target_col, nbins=20, 
                                   title=f"Distribution de {target_col}",
                                   labels={target_col: target_col},
                                   color_discrete_sequence=['#3498db'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats de la variable cible
                st.markdown(f"**Statistiques pour {target_col}:**")
                st.write(f"Moyenne: {data[target_col].mean():.2f}")
                st.write(f"Médiane: {data[target_col].median():.2f}")
                st.write(f"Écart-type: {data[target_col].std():.2f}")
                st.write(f"Min: {data[target_col].min():.2f}")
                st.write(f"Max: {data[target_col].max():.2f}")
                
            with viz_col2:
                # Graphique pour les distributions des variables explicatives
                var_to_plot = st.selectbox("Sélectionnez une variable à visualiser", feature_cols)
                fig = px.histogram(data, x=var_to_plot, nbins=20,
                                  title=f"Distribution de {var_to_plot}",
                                  labels={var_to_plot: var_to_plot},
                                  color_discrete_sequence=['#2ecc71'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Matrice de corrélation
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Analyse des Corrélations")
            
            # Sélection des colonnes numériques uniquement
            numeric_cols = [col for col in feature_cols + [target_col] if pd.api.types.is_numeric_dtype(data[col])]
            
            if len(numeric_cols) >= 2:
                try:
                    # Calcul de la matrice de corrélation
                    corr_matrix = data[numeric_cols].corr()
                    
                    # Visualisation de la matrice de corrélation avec plotly
                    fig = px.imshow(corr_matrix, 
                                   text_auto=True, 
                                   color_continuous_scale="RdBu_r",
                                   title="Matrice de Corrélation",
                                   zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Corrélations avec la variable cible
                    if target_col in numeric_cols:
                        st.subheader(f"Corrélations avec {target_col}")
                        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
                        
                        fig = px.bar(x=target_corr.index, y=target_corr.values,
                                    labels={'x': 'Variables', 'y': 'Coefficient de Corrélation'},
                                    title=f"Corrélations avec {target_col}",
                                    color=target_corr.values,
                                    color_continuous_scale="RdBu_r")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors du calcul des corrélations: {e}")
                    st.info("Conseil: Vérifiez s'il y a des valeurs manquantes dans les données et nettoyez-les dans la page 'Importation des données'.")
            else:
                st.warning("Il faut au moins deux variables numériques pour calculer les corrélations.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Graphiques de dispersion
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Graphiques de Dispersion")
            
            # Filtrer uniquement les colonnes numériques pour le graphique de dispersion
            numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
            
            if numeric_features:
                # Sélectionner les variables pour le graphique de dispersion
                x_var = st.selectbox("Sélectionnez la variable pour l'axe X", numeric_features)
                
                # Graphique de dispersion avec régression
                try:
                    fig = px.scatter(data, x=x_var, y=target_col, 
                                    title=f"{target_col} vs {x_var}",
                                    trendline="ols",
                                    labels={x_var: x_var, target_col: target_col})
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la création du graphique de dispersion: {e}")
                
                # Variables catégorielles pour coloration
                categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
                
                if categorical_cols:
                    st.subheader("Analyse par catégorie")
                    cat_var = st.selectbox("Variable catégorielle pour coloration", options=[None] + categorical_cols)
                    
                    if cat_var:
                        try:
                            fig = px.scatter(data, x=x_var, y=target_col, 
                                            color=cat_var,
                                            title=f"{target_col} vs {x_var} par {cat_var}",
                                            labels={x_var: x_var, target_col: target_col, cat_var: cat_var})
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Boxplot par catégorie
                            fig = px.box(data, x=cat_var, y=target_col,
                                        title=f"Distribution de {target_col} par {cat_var}",
                                        color=cat_var)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur lors de la création des graphiques par catégorie: {e}")
                
                # Graphique à variables multiples
                st.subheader("Graphique à variables multiples")
                col_subset = st.multiselect("Sélectionnez 2 à 4 variables à visualiser", 
                                           numeric_features, 
                                           default=numeric_features[:min(3, len(numeric_features))])
                
                if len(col_subset) >= 2:
                    if len(col_subset) <= 4:
                        color_var = st.selectbox("Variable pour la couleur", [None] + col_subset + categorical_cols)
                        
                        try:
                            if color_var is not None:
                                fig = px.scatter_matrix(data, 
                                                      dimensions=col_subset,
                                                      color=color_var,
                                                      title="Graphique de dispersion multiple")
                            else:
                                fig = px.scatter_matrix(data, 
                                                      dimensions=col_subset,
                                                      title="Graphique de dispersion multiple")
                                
                            fig.update_layout(height=700)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur lors de la création de la matrice de dispersion: {e}")
                    else:
                        st.warning("Veuillez sélectionner au maximum 4 variables pour la visualisation.")
            else:
                st.warning("Aucune variable numérique sélectionnée. Veuillez sélectionner au moins une variable numérique.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analyse avancée
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Analyse Avancée")
            
            advanced_analysis = st.selectbox(
                "Type d'analyse avancée",
                ["PCA (Analyse en Composantes Principales)", "Détection d'outliers", "Analyse bivariée (variable par variable)"]
            )
            
            if advanced_analysis == "PCA (Analyse en Composantes Principales)":
                st.markdown("#### Analyse en Composantes Principales")
                
                # Sélection des variables pour la PCA
                numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
                
                if len(numeric_features) >= 2:
                    pca_features = st.multiselect(
                        "Sélectionnez les variables pour la PCA",
                        numeric_features,
                        default=numeric_features[:min(5, len(numeric_features))]
                    )
                    
                    if len(pca_features) >= 2:
                        try:
                            # Préparation des données
                            X = data[pca_features].dropna()
                            
                            # Normalisation
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # PCA
                            n_components = min(len(pca_features), 10)
                            pca = PCA(n_components=n_components)
                            X_pca = pca.fit_transform(X_scaled)
                            
                            # Variance expliquée
                            explained_variance_ratio = pca.explained_variance_ratio_
                            cumulative_variance = np.cumsum(explained_variance_ratio)
                            
                            # Graphique de variance expliquée
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=list(range(1, n_components + 1)),
                                y=explained_variance_ratio,
                                name='Variance expliquée par composante'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(1, n_components + 1)),
                                y=cumulative_variance,
                                name='Variance expliquée cumulative',
                                mode='lines+markers'
                            ))
                            
                            fig.update_layout(
                                title='Variance expliquée par les composantes principales',
                                xaxis_title='Composante',
                                yaxis_title='Ratio de variance expliquée',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Projection des données sur les deux premières composantes
                            if n_components >= 2:
                                st.markdown("#### Projection des données sur les deux premières composantes")
                                
                                # Créer un dataframe pour la visualisation
                                pca_df = pd.DataFrame({
                                    'PC1': X_pca[:, 0],
                                    'PC2': X_pca[:, 1]
                                })
                                
                                # Ajouter la variable cible ou autres variables pour coloration
                                available_vars = [target_col] + [col for col in data.columns if col not in pca_features]
                                color_var = st.selectbox("Variable pour coloration des points", available_vars)
                                
                                # Copier les indices pour joindre avec les données originales
                                pca_df.index = X.index
                                
                                # Joindre avec la variable de coloration
                                pca_df[color_var] = data.loc[X.index, color_var]
                                
                                # Graphique
                                fig = px.scatter(
                                    pca_df,
                                    x='PC1',
                                    y='PC2',
                                    color=color_var,
                                    title='Projection PCA',
                                    labels={'PC1': f'PC1 ({explained_variance_ratio[0]:.2%})', 
                                           'PC2': f'PC2 ({explained_variance_ratio[1]:.2%})'}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Biplot (loading plot)
                                st.markdown("#### Biplot (Loading Plot)")
                                
                                # Coefficients de pondération
                                loadings = pca.components_.T
                                
                                # Créer un graphique avec les projections des données
                                fig = go.Figure()
                                
                                # Scatter plot des individus
                                fig.add_trace(go.Scatter(
                                    x=X_pca[:, 0],
                                    y=X_pca[:, 1],
                                    mode='markers',
                                    name='Observations',
                                    marker=dict(
                                        color=pca_df[color_var] if pd.api.types.is_numeric_dtype(data[color_var]) else None,
                                        colorscale='Viridis',
                                        showscale=pd.api.types.is_numeric_dtype(data[color_var]),
                                        colorbar=dict(title=color_var) if pd.api.types.is_numeric_dtype(data[color_var]) else None,
                                        opacity=0.7
                                    )
                                ))
                                
                                # Flèches pour les variables
                                scaling_factor = np.max(np.abs(X_pca[:, :2])) / np.max(np.abs(loadings[:, :2])) * 0.8
                                
                                for i, feature in enumerate(pca_features):
                                    fig.add_trace(go.Scatter(
                                        x=[0, loadings[i, 0] * scaling_factor],
                                        y=[0, loadings[i, 1] * scaling_factor],
                                        mode='lines+markers+text',
                                        name=feature,
                                        line=dict(color='red'),
                                        text=['', feature],
                                        textposition='top center'
                                    ))
                                
                                fig.update_layout(
                                    title='Biplot (Loading Plot)',
                                    xaxis_title=f'PC1 ({explained_variance_ratio[0]:.2%})',
                                    yaxis_title=f'PC2 ({explained_variance_ratio[1]:.2%})',
                                    xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
                                    yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **Comment interpréter un biplot:**
                                - Les points représentent les observations projetées dans l'espace des deux premières composantes principales
                                - Les flèches représentent les variables d'origine
                                - La direction d'une flèche indique la direction de variance maximale de cette variable
                                - La longueur d'une flèche indique l'importance de la contribution de cette variable aux composantes
                                - Des flèches proches indiquent des variables corrélées positivement
                                - Des flèches opposées indiquent des variables corrélées négativement
                                - Des flèches perpendiculaires indiquent des variables non corrélées
                                """)
                        except Exception as e:
                            st.error(f"Erreur lors de l'analyse PCA: {e}")
                    else:
                        st.warning("Veuillez sélectionner au moins 2 variables pour la PCA.")
                else:
                    st.warning("Il faut au moins 2 variables numériques pour effectuer une PCA.")
                
            elif advanced_analysis == "Détection d'outliers":
                st.markdown("#### Détection des Valeurs Aberrantes (Outliers)")
                
                # Sélection de la méthode
                method = st.selectbox(
                    "Méthode de détection d'outliers",
                    ["Z-score", "IQR (Écart Interquartile)", "Isolation Forest"]
                )
                
                # Sélection des variables pour la détection
                numeric_features = [col for col in feature_cols + [target_col] if pd.api.types.is_numeric_dtype(data[col])]
                
                if numeric_features:
                    outlier_features = st.multiselect(
                        "Sélectionnez les variables pour la détection d'outliers",
                        numeric_features,
                        default=[target_col]
                    )
                    
                    if outlier_features:
                        try:
                            # Préparation des données
                            X_outlier = data[outlier_features].dropna()
                            
                            if method == "Z-score":
                                # Z-score
                                threshold = st.slider("Seuil Z-score", 1.0, 5.0, 3.0, 0.1)
                                
                                # Calcul des Z-scores
                                z_scores = pd.DataFrame(index=X_outlier.index)
                                for col in outlier_features:
                                    z_scores[col] = np.abs((X_outlier[col] - X_outlier[col].mean()) / X_outlier[col].std())
                                
                                # Identification des outliers
                                outliers = (z_scores > threshold).any(axis=1)
                                
                                # Affichage du nombre d'outliers
                                st.write(f"Nombre d'outliers détectés (Z-score > {threshold}): {outliers.sum()} sur {len(X_outlier)} ({outliers.sum() / len(X_outlier):.2%})")
                                
                            elif method == "IQR (Écart Interquartile)":
                                # IQR
                                factor = st.slider("Facteur IQR", 1.0, 3.0, 1.5, 0.1)
                                
                                # Calcul des quartiles et IQR
                                outliers = pd.Series(False, index=X_outlier.index)
                                
                                for col in outlier_features:
                                    Q1 = X_outlier[col].quantile(0.25)
                                    Q3 = X_outlier[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    
                                    lower_bound = Q1 - factor * IQR
                                    upper_bound = Q3 + factor * IQR
                                    
                                    col_outliers = (X_outlier[col] < lower_bound) | (X_outlier[col] > upper_bound)
                                    outliers = outliers | col_outliers
                                
                                # Affichage du nombre d'outliers
                                st.write(f"Nombre d'outliers détectés (IQR × {factor}): {outliers.sum()} sur {len(X_outlier)} ({outliers.sum() / len(X_outlier):.2%})")
                                
                            else:  # Isolation Forest
                                from sklearn.ensemble import IsolationForest
                                
                                # Paramètres
                                contamination = st.slider("Contamination estimée", 0.01, 0.5, 0.1, 0.01)
                                
                                # Normalisation des données
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X_outlier)
                                
                                # Application de Isolation Forest
                                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                                outlier_pred = iso_forest.fit_predict(X_scaled)
                                
                                # Interprétation des résultats (-1 pour outlier, 1 pour inlier)
                                outliers = pd.Series(outlier_pred == -1, index=X_outlier.index)
                                
                                # Affichage du nombre d'outliers
                                st.write(f"Nombre d'outliers détectés (Isolation Forest): {outliers.sum()} sur {len(X_outlier)} ({outliers.sum() / len(X_outlier):.2%})")
                            
                            # Visualisation des outliers
                            if len(outlier_features) >= 2:
                                # Sélection de deux variables pour la visualisation
                                x_var = st.selectbox("Variable X", outlier_features, index=0)
                                y_var = st.selectbox("Variable Y", outlier_features, index=min(1, len(outlier_features) - 1))
                                
                                # Créer un dataframe pour la visualisation
                                viz_df = pd.DataFrame({
                                    'x': X_outlier[x_var],
                                    'y': X_outlier[y_var],
                                    'outlier': outliers
                                })
                                
                                # Graphique
                                fig = px.scatter(
                                    viz_df,
                                    x='x',
                                    y='y',
                                    color='outlier',
                                    color_discrete_map={True: 'red', False: 'blue'},
                                    labels={'x': x_var, 'y': y_var, 'outlier': 'Outlier'},
                                    title=f"Détection d'outliers: {x_var} vs {y_var}"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Option pour filtrer les outliers
                            if st.checkbox("Exclure les outliers du jeu de données"):
                                # Filtrer les données
                                data_filtered = data.loc[~outliers].copy()
                                
                                # Afficher les statistiques avant/après
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Statistiques avant filtrage:**")
                                    st.dataframe(data[outlier_features].describe())
                                
                                with col2:
                                    st.markdown("**Statistiques après filtrage:**")
                                    st.dataframe(data_filtered[outlier_features].describe())
                                
                                # Option pour appliquer le filtrage
                                if st.button("Appliquer le filtrage"):
                                    st.session_state.data = data_filtered
                                    st.success(f"Outliers supprimés. Nombre d'observations restantes: {len(data_filtered)}")
                                    st.rerun()  # Recharger la page pour mettre à jour les visualisations
                        except Exception as e:
                            st.error(f"Erreur lors de la détection d'outliers: {e}")
                    else:
                        st.warning("Veuillez sélectionner au moins une variable pour la détection d'outliers.")
                else:
                    st.warning("Aucune variable numérique disponible pour la détection d'outliers.")
                
            elif advanced_analysis == "Analyse bivariée (variable par variable)":
                st.markdown("#### Analyse Bivariée")
                
                # Sélection de la variable explicative
                explanatory_var = st.selectbox("Sélectionnez une variable explicative", feature_cols)
                
                try:
                    # Vérifier le type de la variable
                    if pd.api.types.is_numeric_dtype(data[explanatory_var]):
                        # Variable numérique
                        st.markdown(f"**Analyse de la relation entre {explanatory_var} et {target_col}**")
                        
                        # Graphique de dispersion avec régression
                        fig = px.scatter(
                            data,
                            x=explanatory_var,
                            y=target_col,
                            trendline="ols",
                            title=f"Relation entre {explanatory_var} et {target_col}",
                            labels={explanatory_var: explanatory_var, target_col: target_col}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Analyse statistique
                        from scipy.stats import pearsonr, spearmanr
                        
                        # Filtrer les valeurs manquantes
                        valid_data = data[[explanatory_var, target_col]].dropna()
                        
                        # Corrélations
                        pearson_corr, p_value_pearson = pearsonr(valid_data[explanatory_var], valid_data[target_col])
                        spearman_corr, p_value_spearman = spearmanr(valid_data[explanatory_var], valid_data[target_col])
                        
                        # Statistiques de régression
                        import statsmodels.api as sm
                        X = sm.add_constant(valid_data[explanatory_var])
                        model = sm.OLS(valid_data[target_col], X).fit()
                        
                        # Affichage des résultats
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Corrélations:**")
                            st.write(f"Corrélation de Pearson: {pearson_corr:.4f} (p-value: {p_value_pearson:.4f})")
                            st.write(f"Corrélation de Spearman: {spearman_corr:.4f} (p-value: {p_value_spearman:.4f})")
                        
                        with col2:
                            st.markdown("**Résumé de la régression:**")
                            st.write(f"R²: {model.rsquared:.4f}")
                            st.write(f"Coefficient: {model.params[1]:.4f}")
                            st.write(f"Intercept: {model.params[0]:.4f}")
                            st.write(f"P-value: {model.pvalues[1]:.4f}")
                        
                        # Visualisation avancée
                        if st.checkbox("Afficher des visualisations avancées"):
                            # Regrouper par bins pour voir les tendances
                            st.markdown("**Analyse par bins:**")
                            n_bins = st.slider("Nombre de bins", 3, 20, 10)
                            
                            # Créer les bins
                            data['bin'] = pd.qcut(data[explanatory_var], n_bins, duplicates='drop')
                            
                            # Calculer les statistiques par bin
                            bin_stats = data.groupby('bin')[target_col].agg(['mean', 'std', 'count']).reset_index()
                            bin_stats['bin_center'] = bin_stats['bin'].apply(lambda x: x.mid)
                            
                            # Graphique
                            fig = go.Figure()
                            
                            # Barres d'erreur
                            fig.add_trace(go.Scatter(
                                x=bin_stats['bin_center'],
                                y=bin_stats['mean'],
                                error_y=dict(
                                    type='data',
                                    array=bin_stats['std'] / np.sqrt(bin_stats['count']),
                                    visible=True
                                ),
                                mode='markers',
                                name='Moyenne par bin'
                            ))
                            
                            # Ligne de tendance
                            x_range = np.linspace(data[explanatory_var].min(), data[explanatory_var].max(), 100)
                            y_pred = model.params[0] + model.params[1] * x_range
                            
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_pred,
                                mode='lines',
                                name='Régression linéaire',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f"Tendance de {target_col} par bins de {explanatory_var}",
                                xaxis_title=explanatory_var,
                                yaxis_title=f"Moyenne de {target_col}"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Recherche de relations non linéaires
                            st.markdown("**Exploration de relations non linéaires:**")
                            
                            # Polynomiale
                            degree = st.slider("Degré du polynôme", 1, 5, 2)
                            
                            from sklearn.preprocessing import PolynomialFeatures
                            from sklearn.linear_model import LinearRegression
                            from sklearn.pipeline import Pipeline
                            
                            # Modèle polynomial
                            poly_model = Pipeline([
                                ('poly', PolynomialFeatures(degree=degree)),
                                ('linear', LinearRegression())
                            ])
                            
                            # Entraînement
                            X_poly = valid_data[explanatory_var].values.reshape(-1, 1)
                            y_poly = valid_data[target_col].values
                            
                            poly_model.fit(X_poly, y_poly)
                            
                            # Prédictions pour le graphique
                            X_plot = np.linspace(data[explanatory_var].min(), data[explanatory_var].max(), 100).reshape(-1, 1)
                            y_plot = poly_model.predict(X_plot)
                            
                            # Score R²
                            r2_poly = poly_model.score(X_poly, y_poly)
                            
                            # Graphique
                            fig = px.scatter(
                                valid_data,
                                x=explanatory_var,
                                y=target_col,
                                title=f"Régression polynomiale (degré {degree}, R²={r2_poly:.4f})"
                            )
                            
                            fig.add_trace(go.Scatter(
                                x=X_plot.ravel(),
                                y=y_plot,
                                mode='lines',
                                name=f'Polynôme degré {degree}',
                                line=dict(color='red')
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                    else:
                        # Variable catégorielle
                        st.markdown(f"**Analyse de l'impact de {explanatory_var} sur {target_col}**")
                        
                        # Boxplot par catégorie
                        fig = px.box(
                            data,
                            x=explanatory_var,
                            y=target_col,
                            title=f"Distribution de {target_col} par {explanatory_var}",
                            color=explanatory_var
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistiques par catégorie
                        stats_by_cat = data.groupby(explanatory_var)[target_col].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
                        
                        st.markdown("**Statistiques par catégorie:**")
                        st.dataframe(stats_by_cat)
                        
                        # Test ANOVA
                        from scipy.stats import f_oneway
                        
                        # Préparer les groupes pour ANOVA
                        groups = []
                        categories = []
                        
                        for cat in data[explanatory_var].unique():
                            group_data = data[data[explanatory_var] == cat][target_col].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data)
                                categories.append(cat)
                        
                        if len(groups) >= 2:
                            # Effectuer ANOVA
                            f_stat, p_value = f_oneway(*groups)
                            
                            st.markdown("**Résultats ANOVA:**")
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"p-value: {p_value:.4f}")
                            
                            if p_value < 0.05:
                                st.success(f"Il existe une différence significative de {target_col} entre les différentes catégories de {explanatory_var} (p-value < 0.05).")
                            else:
                                st.info(f"Il n'y a pas de différence significative de {target_col} entre les différentes catégories de {explanatory_var} (p-value >= 0.05).")
                        
                        # Graphique à barres pour la moyenne par catégorie
                        fig = px.bar(
                            stats_by_cat,
                            x=explanatory_var,
                            y='mean',
                            error_y='std',
                            title=f"Moyenne de {target_col} par {explanatory_var}",
                            color=explanatory_var
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse bivariée: {e}")
                
            st.markdown("</div>", unsafe_allow_html=True)

# Page de data augmentation
elif page == "Data Augmentation":
    st.markdown("<h2 class='sub-header'>Data Augmentation</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez aller à la page 'Importation des données' pour charger vos données.")
    elif st.session_state.features is None or st.session_state.target is None:
        st.warning("Variables prédictives et/ou variable cible non définies. Veuillez aller à la page 'Exploration des données'.")
    else:
        data = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Pourquoi augmenter les données ?")
        
        st.markdown("""
        L'augmentation de données est une technique qui permet d'accroître artificiellement la taille d'un jeu de données 
        d'entraînement en créant des variations modifiées des données existantes. Elle est particulièrement utile pour:
        
        - Améliorer les performances des modèles quand les données sont limitées
        - Réduire le risque de surapprentissage
        - Équilibrer les classes sous-représentées
        - Augmenter la robustesse des modèles face à la variabilité des données
        
        Dans le contexte géométallurgique, l'augmentation de données peut aider à simuler différentes conditions 
        de traitement ou variations minéralogiques qui pourraient être rencontrées en production.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Méthodes d'augmentation de données")
        
        augmentation_method = st.selectbox(
            "Sélectionnez une méthode d'augmentation",
            [
                "Perturbation gaussienne", 
                "SMOTE (Synthetic Minority Over-sampling Technique)",
                "Interpolation linéaire",
                "Combinaison de caractéristiques",
                "Bootstrap"
            ]
        )
        
        # Nombre de données originales
        n_original = len(data)
        st.write(f"Nombre d'observations originales: {n_original}")
        
        # Filtre pour obtenir uniquement les caractéristiques numériques (pour l'augmentation)
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(data[f])]
        
        if numeric_features:
            # Sélection des caractéristiques pour l'augmentation
            features_for_augmentation = st.multiselect(
                "Sélectionnez les caractéristiques numériques à augmenter",
                numeric_features,
                default=numeric_features
            )
            
            if features_for_augmentation:
                # Permettre de générer jusqu'à 1000 échantillons
                max_samples = max(1000, n_original * 5)
                
                # Options spécifiques à chaque méthode
                if augmentation_method == "Perturbation gaussienne":
                    st.markdown("#### Perturbation gaussienne")
                    st.markdown("""
                    Cette méthode ajoute un bruit gaussien aux caractéristiques numériques. 
                    L'amplitude du bruit est contrôlée par le paramètre sigma (écart-type).
                    """)
                    
                    # Paramètres
                    n_samples = st.slider("Nombre d'échantillons à générer", 10, max_samples, min(1000, n_original))
                    sigma_percent = st.slider("Amplitude du bruit (% de l'écart-type)", 1, 50, 10)
                    
                    if st.button("Générer les données augmentées"):
                        try:
                            # Filtrer les données pour ne garder que les lignes complètes
                            complete_data = data.dropna(subset=features_for_augmentation + [target])
                            
                            # Préparer les caractéristiques à augmenter
                            X = complete_data[features_for_augmentation].values
                            y = complete_data[target].values
                            
                            # Calculer l'écart-type de chaque caractéristique
                            stds = np.std(X, axis=0)
                            
                            # Générer les données augmentées
                            augmented_data = []
                            
                            for _ in range(n_samples):
                                # Sélectionner un échantillon aléatoire
                                idx = np.random.randint(0, len(X))
                                sample_X = X[idx].copy()
                                sample_y = y[idx]
                                
                                # Ajouter du bruit
                                noise = np.random.normal(0, stds * sigma_percent / 100, size=sample_X.shape)
                                sample_X = sample_X + noise
                                
                                # Stocker l'échantillon augmenté
                                augmented_sample = {target: sample_y}
                                for i, feature in enumerate(features_for_augmentation):
                                    augmented_sample[feature] = sample_X[i]
                                
                                augmented_data.append(augmented_sample)
                            
                            # Créer un DataFrame avec les données augmentées
                            augmented_df = pd.DataFrame(augmented_data)
                            
                            # Ajouter les autres colonnes avec des valeurs NaN
                            for col in data.columns:
                                if col not in augmented_df.columns:
                                    augmented_df[col] = np.nan
                            
                            # Stocker les données augmentées
                            st.session_state.data_augmented = augmented_df
                            
                            # Afficher un échantillon des données augmentées
                            st.success(f"Génération réussie! {len(augmented_df)} nouvelles observations ont été créées.")
                            st.markdown("#### Échantillon des données augmentées")
                            st.dataframe(augmented_df.head())
                        except Exception as e:
                            st.error(f"Erreur lors de la génération des données augmentées: {e}")
                
                elif augmentation_method == "SMOTE (Synthetic Minority Over-sampling Technique)":
                    st.markdown("#### SMOTE (Synthetic Minority Over-sampling Technique)")
                    st.markdown("""
                    SMOTE génère de nouveaux échantillons synthétiques en interpolant entre des instances proches 
                    dans l'espace des caractéristiques. Cette méthode est particulièrement utile lorsque vos données 
                    sont déséquilibrées.
                    """)
                    
                    try:
                        # Tentative d'importation de SMOTE
                        try:
                            from imblearn.over_sampling import SMOTE
                            smote_available = True
                        except ImportError:
                            smote_available = False
                            st.warning("La bibliothèque imbalanced-learn n'est pas installée. Une implémentation simplifiée de SMOTE sera utilisée.")
                        
                        # Discrétiser la variable cible pour SMOTE
                        target_bins = st.slider("Nombre de bins pour discrétiser la variable cible", 2, 10, 5)
                        
                        # Option pour le ratio d'augmentation
                        sampling_strategy = st.slider("Ratio d'échantillonnage", 0.5, 2.0, 1.0, 0.1)
                        
                        # Option pour k_neighbors
                        k_neighbors = st.slider("Nombre de voisins", 1, 10, 5)
                        
                        # Nombre d'échantillons
                        n_samples = st.slider("Nombre d'échantillons à générer", 10, max_samples, min(1000, n_original))
                        
                        if st.button("Générer les données augmentées"):
                            try:
                                # Filtrer les données pour ne garder que les lignes complètes
                                complete_data = data.dropna(subset=features_for_augmentation + [target])
                                
                                # Préparer les caractéristiques et la variable cible
                                X = complete_data[features_for_augmentation].values
                                y = complete_data[target].values
                                
                                # Discrétiser la variable cible
                                y_binned = pd.qcut(y, target_bins, labels=False)
                                
                                if smote_available:
                                    # Appliquer SMOTE
                                    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
                                    X_resampled, y_resampled = smote.fit_resample(X, y_binned)
                                else:
                                    # Implémentation simplifiée de SMOTE
                                    # Pour chaque classe minoritaire, sélectionnez des paires d'échantillons et interpolez entre eux
                                    bin_counts = np.bincount(y_binned)
                                    max_bin_count = np.max(bin_counts)
                                    
                                    X_list = [X[y_binned == i] for i in range(len(bin_counts))]
                                    y_list = [y_binned[y_binned == i] for i in range(len(bin_counts))]
                                    
                                    X_resampled = []
                                    y_resampled = []
                                    
                                    for i in range(len(bin_counts)):
                                        X_bin = X_list[i]
                                        y_bin = y_list[i]
                                        
                                        # Ajouter les échantillons originaux
                                        X_resampled.append(X_bin)
                                        y_resampled.append(y_bin)
                                        
                                        # Si la classe est minoritaire, générer des échantillons synthétiques
                                        if bin_counts[i] < max_bin_count:
                                            n_samples_to_add = int((max_bin_count - bin_counts[i]) * sampling_strategy)
                                            
                                            if len(X_bin) >= 2:  # Nécessite au moins 2 échantillons pour interpoler
                                                synthetic_samples = []
                                                
                                                for _ in range(n_samples_to_add):
                                                    # Sélectionner un échantillon aléatoire
                                                    idx1 = np.random.randint(0, len(X_bin))
                                                    
                                                    # Sélectionner un voisin aléatoire
                                                    neighbors_idx = np.random.choice(len(X_bin), min(k_neighbors, len(X_bin)), replace=False)
                                                    neighbors_idx = neighbors_idx[neighbors_idx != idx1]  # Exclure l'échantillon lui-même
                                                    
                                                    if len(neighbors_idx) > 0:
                                                        idx2 = neighbors_idx[0]
                                                        
                                                        # Interpoler
                                                        alpha = np.random.random()
                                                        synthetic_sample = X_bin[idx1] * alpha + X_bin[idx2] * (1 - alpha)
                                                        synthetic_samples.append(synthetic_sample)
                                                
                                                if synthetic_samples:
                                                    X_resampled.append(np.array(synthetic_samples))
                                                    y_resampled.append(np.full(len(synthetic_samples), i))
                                    
                                    # Concaténer les listes
                                    X_resampled = np.vstack(X_resampled)
                                    y_resampled = np.hstack(y_resampled)
                                
                                # Prendre seulement les n_samples premiers échantillons ou tous si moins disponibles
                                n_samples_generated = min(n_samples, len(X_resampled) - len(X))
                                if n_samples_generated > 0:
                                    synthetic_indices = np.random.choice(range(len(X), len(X_resampled)), 
                                                                        size=n_samples_generated, 
                                                                        replace=False)
                                else:
                                    synthetic_indices = np.arange(len(X), len(X_resampled))
                                
                                # Regénérer les valeurs continues pour la variable cible
                                # Pour chaque bin, calculer la moyenne des valeurs originales
                                bin_means = {}
                                for bin_idx in range(target_bins):
                                    if np.sum(y_binned == bin_idx) > 0:
                                        bin_means[bin_idx] = np.mean(y[y_binned == bin_idx])
                                    else:
                                        bin_means[bin_idx] = np.mean(y)  # Fallback
                                
                                # Ajouter du bruit pour éviter que toutes les valeurs soient identiques
                                y_continuous = np.array([bin_means[bin_idx] for bin_idx in y_resampled])
                                y_noise = np.random.normal(0, np.std(y) * 0.1, size=len(y_continuous))
                                y_continuous = y_continuous + y_noise
                                
                                # Créer un DataFrame avec les données augmentées (uniquement les nouvelles)
                                augmented_data = []
                                for idx in synthetic_indices:
                                    sample = {target: y_continuous[idx]}
                                    for i, feature in enumerate(features_for_augmentation):
                                        sample[feature] = X_resampled[idx, i]
                                    augmented_data.append(sample)
                                
                                augmented_df = pd.DataFrame(augmented_data)
                                
                                # Ajouter les autres colonnes avec des valeurs NaN
                                for col in data.columns:
                                    if col not in augmented_df.columns:
                                        augmented_df[col] = np.nan
                                
                                # Stocker les données augmentées
                                st.session_state.data_augmented = augmented_df
                                
                                # Afficher un échantillon des données augmentées
                                st.success(f"Génération réussie! {len(augmented_df)} nouvelles observations ont été créées.")
                                st.markdown("#### Échantillon des données augmentées")
                                st.dataframe(augmented_df.head())
                                
                                # Visualisation de la distribution
                                fig = go.Figure()
                                
                                fig.add_trace(go.Histogram(
                                    x=data[target],
                                    name='Données originales',
                                    opacity=0.7,
                                    marker_color='blue'
                                ))
                                
                                fig.add_trace(go.Histogram(
                                    x=augmented_df[target],
                                    name='Données augmentées',
                                    opacity=0.7,
                                    marker_color='red'
                                ))
                                
                                fig.update_layout(
                                    title='Distribution de la variable cible',
                                    xaxis_title=target,
                                    yaxis_title='Fréquence',
                                    barmode='overlay'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Erreur lors de l'application de SMOTE: {e}")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de l'initialisation de SMOTE: {e}")
                
                elif augmentation_method == "Interpolation linéaire":
                    st.markdown("#### Interpolation linéaire")
                    st.markdown("""
                    Cette méthode génère de nouveaux échantillons en interpolant linéairement entre des paires 
                    d'échantillons existants. Cela permet de créer des points intermédiaires plausibles.
                    """)
                    
                    # Paramètres
                    n_samples = st.slider("Nombre d'échantillons à générer", 10, max_samples, min(1000, n_original))
                    
                    if st.button("Générer les données augmentées"):
                        try:
                            # Filtrer les données pour ne garder que les lignes complètes
                            complete_data = data.dropna(subset=features_for_augmentation + [target])
                            
                            # Préparer les caractéristiques à augmenter
                            X = complete_data[features_for_augmentation].values
                            y = complete_data[target].values
                            
                            # Générer les données augmentées
                            augmented_data = []
                            
                            for _ in range(n_samples):
                                # Sélectionner deux échantillons aléatoires
                                idx1, idx2 = np.random.choice(len(X), 2, replace=False)
                                
                                # Générer un coefficient d'interpolation aléatoire
                                alpha = np.random.random()
                                
                                # Interpoler les caractéristiques et la variable cible
                                interp_X = X[idx1] * alpha + X[idx2] * (1 - alpha)
                                interp_y = y[idx1] * alpha + y[idx2] * (1 - alpha)
                                
                                # Stocker l'échantillon interpolé
                                interpolated_sample = {target: interp_y}
                                for i, feature in enumerate(features_for_augmentation):
                                    interpolated_sample[feature] = interp_X[i]
                                
                                augmented_data.append(interpolated_sample)
                            
                            # Créer un DataFrame avec les données augmentées
                            augmented_df = pd.DataFrame(augmented_data)
                            
                            # Ajouter les autres colonnes avec des valeurs NaN
                            for col in data.columns:
                                if col not in augmented_df.columns:
                                    augmented_df[col] = np.nan
                            
                            # Stocker les données augmentées
                            st.session_state.data_augmented = augmented_df
                            
                            # Afficher un échantillon des données augmentées
                            st.success(f"Génération réussie! {len(augmented_df)} nouvelles observations ont été créées.")
                            st.markdown("#### Échantillon des données augmentées")
                            st.dataframe(augmented_df.head())
                        except Exception as e:
                            st.error(f"Erreur lors de l'interpolation linéaire: {e}")
                
                elif augmentation_method == "Combinaison de caractéristiques":
                    st.markdown("#### Combinaison de caractéristiques")
                    st.markdown("""
                    Cette méthode crée de nouveaux échantillons en combinant les caractéristiques 
                    de différents échantillons existants et en estimant la variable cible à l'aide 
                    d'un modèle prédictif préliminaire.
                    """)
                    
                    # Paramètres
                    n_samples = st.slider("Nombre d'échantillons à générer", 10, max_samples, min(1000, n_original))
                    
                    # Sélection du modèle pour l'estimation
                    model_type = st.selectbox(
                        "Modèle pour estimer la variable cible",
                        ["Random Forest", "Gradient Boosting", "Linear Regression"]
                    )
                    
                    if st.button("Générer les données augmentées"):
                        try:
                            # Filtrer les données pour ne garder que les lignes complètes
                            complete_data = data.dropna(subset=features_for_augmentation + [target])
                            
                            # Préparer les caractéristiques et la variable cible
                            X = complete_data[features_for_augmentation]
                            y = complete_data[target]
                            
                            # Créer et entraîner le modèle prédictif
                            if model_type == "Random Forest":
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            elif model_type == "Gradient Boosting":
                                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                            else:
                                model = LinearRegression()
                            
                            model.fit(X, y)
                            
                            # Générer de nouveaux échantillons
                            augmented_data = []
                            
                            for _ in range(n_samples):
                                new_sample = {}
                                
                                # Pour chaque caractéristique, prendre une valeur aléatoire des données originales
                                for feature in features_for_augmentation:
                                    idx = np.random.randint(0, len(X))
                                    new_sample[feature] = X.iloc[idx][feature]
                                
                                # Prédire la variable cible
                                X_new = pd.DataFrame([new_sample])
                                y_pred = model.predict(X_new)[0]
                                
                                # Ajouter un peu de bruit à la prédiction
                                y_noise = np.random.normal(0, np.std(y) * 0.05)
                                new_sample[target] = y_pred + y_noise
                                
                                augmented_data.append(new_sample)
                            
                            # Créer un DataFrame avec les données augmentées
                            augmented_df = pd.DataFrame(augmented_data)
                            
                            # Ajouter les autres colonnes avec des valeurs NaN
                            for col in data.columns:
                                if col not in augmented_df.columns:
                                    augmented_df[col] = np.nan
                            
                            # Stocker les données augmentées
                            st.session_state.data_augmented = augmented_df
                            
                            # Afficher un échantillon des données augmentées
                            st.success(f"Génération réussie! {len(augmented_df)} nouvelles observations ont été créées.")
                            st.markdown("#### Échantillon des données augmentées")
                            st.dataframe(augmented_df.head())
                        except Exception as e:
                            st.error(f"Erreur lors de la combinaison de caractéristiques: {e}")
                
                else:  # Bootstrap
                    st.markdown("#### Bootstrap")
                    st.markdown("""
                    Le bootstrap génère de nouveaux échantillons en tirant aléatoirement avec remise dans 
                    les données existantes et en ajoutant un bruit aléatoire contrôlé.
                    """)
                    
                    # Paramètres
                    n_samples = st.slider("Nombre d'échantillons à générer", 10, max_samples, min(1000, n_original))
                    noise_level = st.slider("Niveau de bruit (% de l'écart-type)", 0, 30, 5)
                    
                    if st.button("Générer les données augmentées"):
                        try:
                            # Filtrer les données pour ne garder que les lignes complètes
                            complete_data = data.dropna(subset=features_for_augmentation + [target])
                            
                            # Calculer les écarts-types pour le bruit
                            stds = {col: data[col].std() for col in features_for_augmentation + [target]}
                            
                            # Générer les échantillons bootstrap
                            augmented_data = []
                            
                            for _ in range(n_samples):
                                # Sélectionner un échantillon aléatoire avec remise
                                idx = np.random.randint(0, len(complete_data))
                                sample = complete_data.iloc[idx].copy()
                                
                                # Ajouter du bruit aux caractéristiques numériques
                                for col in features_for_augmentation + [target]:
                                    noise = np.random.normal(0, stds[col] * noise_level / 100)
                                    sample[col] += noise
                                
                                augmented_data.append(sample)
                            
                            # Créer un DataFrame avec les données augmentées
                            augmented_df = pd.DataFrame(augmented_data)
                            
                            # Stocker les données augmentées
                            st.session_state.data_augmented = augmented_df
                            
                            # Afficher un échantillon des données augmentées
                            st.success(f"Génération réussie! {len(augmented_df)} nouvelles observations ont été créées.")
                            st.markdown("#### Échantillon des données augmentées")
                            st.dataframe(augmented_df.head())
                        except Exception as e:
                            st.error(f"Erreur lors du bootstrap: {e}")
                
                # Options pour combiner avec les données originales
                if st.session_state.data_augmented is not None:
                    st.markdown("### Utilisation des données augmentées")
                    
                    # Visualiser la distribution des données originales vs augmentées
                    if st.checkbox("Visualiser la comparaison des distributions"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Distribution de la variable cible
                            fig = go.Figure()
                            
                            fig.add_trace(go.Histogram(
                                x=data[target],
                                name='Données originales',
                                opacity=0.7,
                                marker_color='blue'
                            ))
                            
                            fig.add_trace(go.Histogram(
                                x=st.session_state.data_augmented[target],
                                name='Données augmentées',
                                opacity=0.7,
                                marker_color='red'
                            ))
                            
                            fig.update_layout(
                                title=f'Distribution de {target}',
                                xaxis_title=target,
                                yaxis_title='Fréquence',
                                barmode='overlay'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Sélection d'une caractéristique à visualiser
                            feature_to_viz = st.selectbox(
                                "Sélectionnez une caractéristique à visualiser",
                                features_for_augmentation
                            )
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Histogram(
                                x=data[feature_to_viz],
                                name='Données originales',
                                opacity=0.7,
                                marker_color='blue'
                            ))
                            
                            fig.add_trace(go.Histogram(
                                x=st.session_state.data_augmented[feature_to_viz],
                                name='Données augmentées',
                                opacity=0.7,
                                marker_color='red'
                            ))
                            
                            fig.update_layout(
                                title=f'Distribution de {feature_to_viz}',
                                xaxis_title=feature_to_viz,
                                yaxis_title='Fréquence',
                                barmode='overlay'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Option pour combiner les données
                    combine_strategy = st.radio(
                        "Comment utiliser les données augmentées ?",
                        ["Remplacer les données originales", "Ajouter aux données originales"]
                    )
                    
                    if st.button("Appliquer les données augmentées"):
                        if combine_strategy == "Remplacer les données originales":
                            st.session_state.data = st.session_state.data_augmented
                            st.success("Les données originales ont été remplacées par les données augmentées.")
                            st.rerun()  # Recharger la page
                        else:
                            # Combiner les données originales et augmentées
                            combined_data = pd.concat([data, st.session_state.data_augmented], ignore_index=True)
                            st.session_state.data = combined_data
                            st.success(f"Les données augmentées ont été ajoutées aux données originales. Nouveau nombre d'observations: {len(combined_data)}")
                            st.rerun()  # Recharger la page
            else:
                st.warning("Veuillez sélectionner au moins une caractéristique à augmenter.")
        else:
            st.warning("Aucune caractéristique numérique disponible pour l'augmentation de données.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Page de modélisation
elif page == "Modélisation":
    st.markdown("<h2 class='sub-header'>Modélisation</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez aller à la page 'Importation des données' pour charger vos données.")
    elif st.session_state.features is None or st.session_state.target is None:
        st.warning("Variables prédictives et/ou variable cible non définies. Veuillez aller à la page 'Exploration des données'.")
    else:
        data = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Configuration du Modèle")
        
        # Division train/test
        test_size = st.slider("Pourcentage des données pour le test", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Seed aléatoire", 0, 1000, 42)
        
        # Prétraitement
        preprocessing = st.checkbox("Normaliser les données", True)
        
        # Sélection du modèle
        model_options = ["Random Forest", "Gradient Boosting", "Régression Linéaire"]
        if TENSORFLOW_AVAILABLE:
            model_options.append("Réseau de Neurones")
        
        model_type = st.selectbox("Sélectionnez le type de modèle", model_options)
        
        # Configuration spécifique au modèle
        if model_type == "Random Forest":
            n_estimators = st.slider("Nombre d'arbres", 10, 500, 100)
            max_depth = st.slider("Profondeur maximale", 2, 30, 10)
            
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state
            }
            
            model = RandomForestRegressor(**model_params)
            
        elif model_type == "Gradient Boosting":
            n_estimators = st.slider("Nombre d'estimateurs", 10, 500, 100)
            learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01)
            max_depth = st.slider("Profondeur maximale", 2, 10, 3)
            
            model_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "random_state": random_state
            }
            
            model = GradientBoostingRegressor(**model_params)
            
        elif model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
            st.markdown("#### Configuration du Réseau de Neurones")
            
            # Paramètres du réseau
            n_layers = st.slider("Nombre de couches cachées", 1, 5, 2)
            layer_sizes = []
            
            for i in range(n_layers):
                layer_size = st.slider(f"Nombre de neurones dans la couche {i+1}", 4, 128, 32, 4)
                layer_sizes.append(layer_size)
            
            dropout_rate = st.slider("Taux de dropout (prévention du surapprentissage)", 0.0, 0.5, 0.2, 0.05)
            learning_rate = st.slider("Taux d'apprentissage", 0.0001, 0.01, 0.001, 0.0001)
            epochs = st.slider("Nombre d'époques", 10, 500, 100)
            batch_size = st.slider("Taille du batch", 4, 128, 32, 4)
            activation = st.selectbox("Fonction d'activation", ["relu", "tanh", "sigmoid"])
            
            # Création du modèle
            def create_nn_model(input_dim, layer_sizes, dropout_rate, learning_rate, activation):
                model = Sequential()
                
                # Première couche cachée
                model.add(Dense(layer_sizes[0], input_dim=input_dim, activation=activation))
                model.add(Dropout(dropout_rate))
                
                # Couches cachées supplémentaires
                for size in layer_sizes[1:]:
                    model.add(Dense(size, activation=activation))
                    model.add(Dropout(dropout_rate))
                
                # Couche de sortie (régression)
                model.add(Dense(1))
                
                # Compilation
                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='mse',
                    metrics=['mae']
                )
                
                return model
            
            # Early stopping pour éviter le surapprentissage
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            # Le modèle sera créé plus tard, après avoir déterminé la dimension d'entrée
            model = None
            st.session_state.neural_network = {
                'layer_sizes': layer_sizes,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'batch_size': batch_size,
                'activation': activation,
                'early_stopping': early_stopping,
                'create_model': create_nn_model
            }
        else:  # Régression Linéaire
            model = LinearRegression()
        
        # Préparation des données
        X = data[features].copy()
        y = data[target].copy()
        
        # Gestion des variables catégorielles
        categorical_features = [f for f in features if not pd.api.types.is_numeric_dtype(data[f])]
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(data[f])]
        
        if categorical_features:
            st.markdown("#### Variables catégorielles détectées")
            st.write("Les variables suivantes seront encodées: " + ", ".join(categorical_features))
            
            encoding_method = st.selectbox(
                "Méthode d'encodage pour les variables catégorielles",
                ["One-Hot Encoding", "Label Encoding"]
            )
        
        # Construction des preprocesseurs
        preprocessors = []
        
        if numeric_features:
            if preprocessing:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
            else:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median'))
                ])
            
            preprocessors.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            if encoding_method == "One-Hot Encoding":
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])
            else:  # Label Encoding
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', LabelEncoder())
                ])
            
            preprocessors.append(('cat', categorical_transformer, categorical_features))
        
        # Création du préprocesseur complet
        preprocessor = ColumnTransformer(transformers=preprocessors)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Construction finale du pipeline
        if model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
            # Pour les réseaux de neurones, le pipeline est géré différemment
            # Nous appliquons d'abord le préprocesseur, puis créons le modèle
            nn_pipeline = {
                'preprocessor': preprocessor,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            st.session_state.neural_network.update(nn_pipeline)
        else:
            # Pipeline standard pour les autres modèles
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.model = model
            st.session_state.model_pipeline = pipeline
        
        # Entraînement du modèle
        if st.button("Entraîner le modèle"):
            with st.spinner("Entraînement du modèle en cours..."):
                try:
                    if model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                        # Prétraitement des données
                        X_train_processed = preprocessor.fit_transform(X_train)
                        X_test_processed = preprocessor.transform(X_test)
                        
                        # Récupération des paramètres du réseau
                        nn_params = st.session_state.neural_network
                        
                        # Création du modèle
                        input_dim = X_train_processed.shape[1]
                        nn_model = nn_params['create_model'](
                            input_dim,
                            nn_params['layer_sizes'],
                            nn_params['dropout_rate'],
                            nn_params['learning_rate'],
                            nn_params['activation']
                        )
                        
                        # Création d'une barre de progression pour l'entraînement
                        progress_bar = st.progress(0)
                        epochs_status = st.empty()
                        
                        # Callback pour mise à jour de la barre de progression
                        class ProgressCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / nn_params['epochs']
                                progress_bar.progress(progress)
                                epochs_status.text(f"Époque {epoch+1}/{nn_params['epochs']}")
                        
                        # Entraînement du modèle
                        history = nn_model.fit(
                            X_train_processed, y_train,
                            epochs=nn_params['epochs'],
                            batch_size=nn_params['batch_size'],
                            validation_split=0.2,
                            callbacks=[nn_params['early_stopping'], ProgressCallback()],
                            verbose=0
                        )
                        
                        # Évaluation du modèle
                        y_pred = nn_model.predict(X_test_processed).flatten()
                        
                        # Stocker le modèle et le préprocesseur
                        st.session_state.neural_network['model'] = nn_model
                        st.session_state.neural_network['preprocessor'] = preprocessor
                        st.session_state.model_is_fitted = True
                        
                        # Graphiques d'entraînement
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(history.history['loss']) + 1)),
                            y=history.history['loss'],
                            mode='lines',
                            name='Perte (entraînement)'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(history.history['val_loss']) + 1)),
                            y=history.history['val_loss'],
                            mode='lines',
                            name='Perte (validation)'
                        ))
                        
                        fig.update_layout(
                            title='Évolution de la perte pendant l\'entraînement',
                            xaxis_title='Époque',
                            yaxis_title='Perte (MSE)',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Entraînement des modèles standards
                        pipeline.fit(X_train, y_train)
                        st.session_state.model_is_fitted = True
                        
                        # Évaluation du modèle
                        y_pred = pipeline.predict(X_test)
                    
                    # Métriques
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Validation croisée (sauf pour les réseaux de neurones)
                    if model_type != "Réseau de Neurones":
                        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        cv_mean = "N/A"
                        cv_std = "N/A"
                    
                    st.success("Modèle entraîné avec succès!")
                    
                    # Affichage des métriques dans un tableau
                    metrics_data = {
                        "Métrique": ["MSE", "RMSE", "MAE", "R²", "R² CV (moyenne)", "R² CV (écart-type)"],
                        "Valeur": [mse, rmse, mae, r2, cv_mean, cv_std]
                    }
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df)
                    
                    # Visualisation des prédictions vs valeurs réelles
                    fig = px.scatter(
                        x=y_test, y=y_pred, 
                        labels={"x": "Valeurs Réelles", "y": "Valeurs Prédites"},
                        title="Prédictions vs Valeurs Réelles"
                    )
                    
                    # Ajout de la ligne de référence (y=x)
                    fig.add_trace(
                        go.Scatter(
                            x=[y_test.min(), y_test.max()], 
                            y=[y_test.min(), y_test.max()],
                            mode="lines", 
                            line=dict(color="red", dash="dash"),
                            name="Ligne de référence"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Importance des variables pour les modèles qui le supportent
                    if model_type in ["Random Forest", "Gradient Boosting"]:
                        try:
                            # Pour les pipelines avec prétraitement, obtenir les noms des features après transformation
                            if categorical_features and encoding_method == "One-Hot Encoding":
                                # Pour One-Hot Encoding, les noms des features doivent être reconstruits
                                feature_names = []
                                
                                # Récupérer les features numériques
                                feature_names.extend(numeric_features)
                                
                                # Récupérer les features catégorielles encodées
                                cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                                cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
                                feature_names.extend(cat_features_encoded)
                                
                                # Obtenir l'importance des features
                                importances = pipeline.named_steps['model'].feature_importances_
                                
                                # S'assurer que les dimensions correspondent
                                if len(importances) == len(feature_names):
                                    importance_df = pd.DataFrame({
                                        'Variable': feature_names,
                                        'Importance': importances
                                    }).sort_values('Importance', ascending=False)
                                    
                                    fig = px.bar(
                                        importance_df, 
                                        x='Variable', 
                                        y='Importance',
                                        title="Importance des Variables"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Impossible d'afficher l'importance des variables: dimensions incompatibles.")
                            else:
                                # Pour Label Encoding ou sans variables catégorielles
                                importances = pipeline.named_steps['model'].feature_importances_
                                
                                importance_df = pd.DataFrame({
                                    'Variable': features,
                                    'Importance': importances
                                importance_df = pd.DataFrame({
                                    'Variable': features,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance_df, 
                                    x='Variable', 
                                    y='Importance',
                                    title="Importance des Variables"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Impossible d'afficher l'importance des variables: {e}")
                        
                        # SHAP values pour l'interprétabilité
                        if st.checkbox("Afficher l'analyse SHAP (interprétabilité avancée)"):
                            try:
                                with st.spinner("Calcul des valeurs SHAP en cours..."):
                                    # Création de l'explainer SHAP
                                    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)
                                    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
                                    shap_values = explainer.shap_values(X_test_processed)
                                    
                                    # Conversion du plot SHAP en figure matplotlib
                                    st.subheader("Graphique de résumé SHAP")
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    # Tenter de reconstruire les noms des features pour SHAP
                                    try:
                                        if categorical_features and encoding_method == "One-Hot Encoding":
                                            # Reconstruire les noms des features pour One-Hot Encoding
                                            feature_names = []
                                            feature_names.extend(numeric_features)
                                            cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                                            cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
                                            feature_names.extend(cat_features_encoded)
                                            
                                            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
                                        else:
                                            shap.summary_plot(shap_values, X_test_processed, feature_names=features, show=False)
                                    except:
                                        # En cas d'échec, procéder sans noms de features
                                        shap.summary_plot(shap_values, X_test_processed, show=False)
                                    
                                    st.pyplot(fig)
                                    plt.clf()
                            except Exception as e:
                                st.error(f"Erreur lors du calcul des valeurs SHAP: {e}")
                    
                    # Pour les réseaux de neurones, option de sauvegarde du modèle
                    if model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                        if st.button("Sauvegarder le modèle"):
                            # Sauvegarder le modèle et le préprocesseur
                            model_path = "modele_reseau_neuronal.h5"
                            preprocessor_path = "preprocesseur.joblib"
                            
                            nn_model.save(model_path)
                            joblib.dump(preprocessor, preprocessor_path)
                            
                            # Créer un lien de téléchargement pour le modèle
                            with open(model_path, "rb") as f:
                                model_bytes = f.read()
                                b64_model = base64.b64encode(model_bytes).decode()
                                href_model = f'<a href="data:file/h5;base64,{b64_model}" download="{model_path}">Télécharger le modèle de réseau neuronal</a>'
                                st.markdown(href_model, unsafe_allow_html=True)
                            
                            # Créer un lien de téléchargement pour le préprocesseur
                            with open(preprocessor_path, "rb") as f:
                                preprocessor_bytes = f.read()
                                b64_preprocessor = base64.b64encode(preprocessor_bytes).decode()
                                href_preprocessor = f'<a href="data:file/joblib;base64,{b64_preprocessor}" download="{preprocessor_path}">Télécharger le préprocesseur</a>'
                                st.markdown(href_preprocessor, unsafe_allow_html=True)
                            
                            st.success("Modèle et préprocesseur sauvegardés avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement du modèle: {e}")
                    st.info("Conseil: Vérifiez vos données et assurez-vous qu'il n'y a pas de valeurs manquantes ou de problèmes de type de données.")
        st.markdown("</div>", unsafe_allow_html=True)

# Page de prédiction
elif page == "Prédiction":
    st.markdown("<h2 class='sub-header'>Prédiction de Récupération Métallurgique</h2>", unsafe_allow_html=True)
    
    # Vérifier si un modèle a été entraîné
    model_trained = (st.session_state.model_pipeline is not None and st.session_state.model_is_fitted) or \
                   (st.session_state.neural_network is not None and 'model' in st.session_state.neural_network)
    
    if not model_trained:
        st.warning("Aucun modèle n'a été entraîné. Veuillez aller à la page 'Modélisation' pour entraîner un modèle.")
        if st.button("Aller à la page Modélisation"):
            st.session_state.page = "Modélisation"
            st.rerun()
    elif st.session_state.features is None:
        st.warning("Variables prédictives non définies. Veuillez aller à la page 'Exploration des données'.")
    else:
        # Vérifier quel type de modèle est disponible
        if st.session_state.model_pipeline is not None and st.session_state.model_is_fitted:
            pipeline = st.session_state.model_pipeline
            model_type = "standard"
        else:
            nn_data = st.session_state.neural_network
            model_type = "neural_network"
        
        features = st.session_state.features
        target = st.session_state.target
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Prédiction pour un nouvel échantillon")
        
        # Méthode pour rentrer les valeurs
        input_method = st.radio("Méthode d'entrée des données", ["Saisie manuelle", "Importer un fichier"])
        
        if input_method == "Saisie manuelle":
            # Création d'un formulaire pour entrer les valeurs des features
            input_data = {}
            
            for feature in features:
                # Obtenir la plage des valeurs dans les données d'entraînement pour définir les min/max
                if st.session_state.X_train is not None:
                    if pd.api.types.is_numeric_dtype(st.session_state.X_train[feature]):
                        min_val = float(st.session_state.X_train[feature].min())
                        max_val = float(st.session_state.X_train[feature].max())
                        mean_val = float(st.session_state.X_train[feature].mean())
                        
                        # Ajuster légèrement les min/max pour éviter les problèmes de types
                        min_val = min_val * 0.9 if min_val > 0 else min_val * 1.1
                        max_val = max_val * 1.1 if max_val > 0 else max_val * 0.9
                        
                        input_data[feature] = st.slider(
                            f"{feature}", 
                            min_val, 
                            max_val, 
                            mean_val
                        )
                    else:
                        # Pour les variables catégorielles
                        categories = st.session_state.X_train[feature].dropna().unique()
                        input_data[feature] = st.selectbox(f"{feature}", options=categories)
                else:
                    # Fallback si X_train n'est pas disponible
                    input_data[feature] = st.number_input(f"{feature}")
            
            # Bouton pour faire la prédiction
            if st.button("Prédire"):
                try:
                    # Préparation des données d'entrée
                    input_df = pd.DataFrame([input_data])
                    
                    # Faire la prédiction
                    if model_type == "standard":
                        prediction = pipeline.predict(input_df)[0]
                    else:
                        # Pour le réseau de neurones
                        preprocessor = nn_data['preprocessor']
                        nn_model = nn_data['model']
                        
                        # Prétraitement
                        input_processed = preprocessor.transform(input_df)
                        
                        # Prédiction
                        prediction = nn_model.predict(input_processed)[0][0]
                    
                    # Afficher la prédiction
                    st.success(f"Prédiction de la récupération métallurgique: **{prediction:.2f}%**")
                    
                    # Jauge pour visualiser la prédiction
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Prédiction de {target}"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#2ecc71"},
                            'steps': [
                                {'range': [0, 50], 'color': "#e74c3c"},
                                {'range': [50, 80], 'color': "#f39c12"},
                                {'range': [80, 100], 'color': "#2ecc71"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")
                    st.info("Il semble que le modèle n'a pas été correctement entraîné ou qu'il y a un problème avec les données d'entrée. Veuillez retourner à la page Modélisation.")
                
        else:  # Importation d'un fichier
            st.markdown("""
            Téléchargez un fichier CSV ou Excel contenant les données pour lesquelles vous souhaitez faire des prédictions.
            Le fichier doit contenir les colonnes suivantes:
            """)
            
            # Afficher les features requises
            for feature in features:
                st.markdown(f"- {feature}")
            
            # Upload du fichier
            uploaded_file = st.file_uploader("Télécharger votre fichier de données", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    # Chargement des données
                    if uploaded_file.name.endswith('.csv'):
                        predict_data = pd.read_csv(uploaded_file)
                    else:
                        predict_data = pd.read_excel(uploaded_file)
                    
                    # Vérifier que toutes les colonnes requises sont présentes
                    missing_cols = [col for col in features if col not in predict_data.columns]
                    
                    if missing_cols:
                        st.error(f"Colonnes manquantes dans le fichier: {', '.join(missing_cols)}")
                    else:
                        # Extraire seulement les colonnes nécessaires
                        predict_data = predict_data[features]
                        
                        # Faire les prédictions
                        try:
                            if model_type == "standard":
                                predictions = pipeline.predict(predict_data)
                            else:
                                # Pour le réseau de neurones
                                preprocessor = nn_data['preprocessor']
                                nn_model = nn_data['model']
                                
                                # Prétraitement
                                input_processed = preprocessor.transform(predict_data)
                                
                                # Prédiction
                                predictions = nn_model.predict(input_processed).flatten()
                            
                            # Ajouter les prédictions au dataframe
                            results = predict_data.copy()
                            results[f"{target}_predit"] = predictions
                            
                            # Afficher les résultats
                            st.subheader("Résultats des prédictions")
                            st.dataframe(results)
                            
                            # Histogramme des prédictions
                            fig = px.histogram(
                                results, 
                                x=f"{target}_predit", 
                                nbins=20,
                                title=f"Distribution des prédictions de {target}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Option pour télécharger les résultats
                            csv = results.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="resultats_predictions.csv">Télécharger les résultats des prédictions</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erreur lors de la prédiction: {e}")
                            st.info("Il semble que le modèle n'a pas été correctement entraîné ou qu'il y a un problème avec les données d'entrée. Veuillez retourner à la page Modélisation.")
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Section pour la simulation et l'optimisation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Optimisation des Paramètres")
        
        st.markdown("""
        Vous pouvez utiliser cette section pour explorer l'effet de différents paramètres sur la récupération métallurgique 
        et identifier les paramètres optimaux pour maximiser la récupération.
        """)
        
        # Sélection des paramètres à optimiser
        numeric_features = [f for f in features if st.session_state.X_train is not None and pd.api.types.is_numeric_dtype(st.session_state.X_train[f])]
        
        params_to_optimize = st.multiselect(
            "Sélectionnez les paramètres à optimiser",
            numeric_features,
            default=numeric_features[:min(2, len(numeric_features))]
        )
        
        if len(params_to_optimize) >= 1:
            # Valeurs par défaut pour les paramètres qui ne sont pas optimisés
            default_values = {}
            for feature in features:
                if feature not in params_to_optimize:
                    if st.session_state.X_train is not None:
                        if pd.api.types.is_numeric_dtype(st.session_state.X_train[feature]):
                            default_values[feature] = float(st.session_state.X_train[feature].mean())
                        else:
                            # Pour les variables catégorielles, prendre la valeur la plus fréquente
                            default_values[feature] = st.session_state.X_train[feature].mode()[0]
                    else:
                        default_values[feature] = 0.0
            
            # Créer des sliders pour les plages des paramètres à optimiser
            param_ranges = {}
            
            for param in params_to_optimize:
                if st.session_state.X_train is not None:
                    min_val = float(st.session_state.X_train[param].min())
                    max_val = float(st.session_state.X_train[param].max())
                    
                    # Ajuster légèrement les min/max pour éviter les problèmes de types
                    min_val = min_val * 0.9 if min_val > 0 else min_val * 1.1
                    max_val = max_val * 1.1 if max_val > 0 else max_val * 0.9
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        param_min = st.number_input(f"Min pour {param}", value=min_val)
                    with col2:
                        param_max = st.number_input(f"Max pour {param}", value=max_val)
                    
                    param_ranges[param] = (param_min, param_max)
                else:
                    param_ranges[param] = (0, 10)  # Valeurs par défaut
            
            # Bouton pour lancer la simulation
            if st.button("Lancer la simulation"):
                with st.spinner("Simulation en cours..."):
                    try:
                        if len(params_to_optimize) == 1:
                            # Simulation 1D
                            param = params_to_optimize[0]
                            param_range = np.linspace(param_ranges[param][0], param_ranges[param][1], 100)
                            
                            sim_results = []
                            for val in param_range:
                                input_data = default_values.copy()
                                input_data[param] = val
                                input_df = pd.DataFrame([input_data])
                                
                                if model_type == "standard":
                                    prediction = pipeline.predict(input_df)[0]
                                else:
                                    # Pour le réseau de neurones
                                    preprocessor = nn_data['preprocessor']
                                    nn_model = nn_data['model']
                                    
                                    # Prétraitement
                                    input_processed = preprocessor.transform(input_df)
                                    
                                    # Prédiction
                                    prediction = nn_model.predict(input_processed)[0][0]
                                
                                sim_results.append(prediction)
                            
                            # Visualisation
                            fig = px.line(
                                x=param_range, 
                                y=sim_results,
                                labels={"x": param, "y": f"Prédiction de {target}"},
                                title=f"Effet de {param} sur {target}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Trouver la valeur optimale
                            optimal_idx = np.argmax(sim_results)
                            optimal_val = param_range[optimal_idx]
                            optimal_result = sim_results[optimal_idx]
                            
                            st.success(f"Valeur optimale de {param}: {optimal_val:.2f} → {optimal_result:.2f}% de récupération")
                            
                        elif len(params_to_optimize) == 2:
                            # Simulation 2D
                            param1, param2 = params_to_optimize
                            param1_range = np.linspace(param_ranges[param1][0], param_ranges[param1][1], 30)
                            param2_range = np.linspace(param_ranges[param2][0], param_ranges[param2][1], 30)
                            
                            # Créer une grille de valeurs
                            param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
                            sim_results = np.zeros_like(param1_grid)
                            
                            # Calculer les prédictions pour chaque combinaison
                            for i in range(len(param1_range)):
                                for j in range(len(param2_range)):
                                    input_data = default_values.copy()
                                    input_data[param1] = param1_grid[j, i]
                                    input_data[param2] = param2_grid[j, i]
                                    input_df = pd.DataFrame([input_data])
                                    
                                    if model_type == "standard":
                                        sim_results[j, i] = pipeline.predict(input_df)[0]
                                    else:
                                        # Pour le réseau de neurones
                                        preprocessor = nn_data['preprocessor']
                                        nn_model = nn_data['model']
                                        
                                        # Prétraitement
                                        input_processed = preprocessor.transform(input_df)
                                        
                                        # Prédiction
                                        sim_results[j, i] = nn_model.predict(input_processed)[0][0]
                            
                            # Visualisation de la surface de réponse
                            fig = go.Figure(data=[go.Surface(z=sim_results, x=param1_range, y=param2_range)])
                            fig.update_layout(
                                title=f"Surface de réponse pour {target}",
                                scene=dict(
                                    xaxis_title=param1,
                                    yaxis_title=param2,
                                    zaxis_title=target
                                ),
                                width=700,
                                height=700
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Graphique de contour pour une visualisation plus claire
                            fig = px.contour(
                                x=param1_range, 
                                y=param2_range, 
                                z=sim_results,
                                labels=dict(x=param1, y=param2, z=target),
                                title=f"Contours de {target}",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Trouver les valeurs optimales
                            optimal_idx = np.unravel_index(np.argmax(sim_results), sim_results.shape)
                            optimal_val1 = param1_range[optimal_idx[1]]
                            optimal_val2 = param2_range[optimal_idx[0]]
                            optimal_result = sim_results[optimal_idx]
                            
                            st.success(f"Valeurs optimales: {param1}={optimal_val1:.2f}, {param2}={optimal_val2:.2f} → {optimal_result:.2f}% de récupération")
                    except Exception as e:
                        st.error(f"Erreur lors de la simulation: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# Gestion des paramètres d'URL pour la déconnexion
if st.query_params.get("logout", [""])[0] == "true":
    st.session_state["authentication_status"] = False
    st.session_state["username"] = ""
    st.query_params.clear()
    st.rerun()

# Footer
st.markdown("<div class='footer'>Développé par Didier Ouedraogo, P.Geo, Géologue et Data Scientist - Application de Prédiction de Récupération Métallurgique © 2025</div>", unsafe_allow_html=True)