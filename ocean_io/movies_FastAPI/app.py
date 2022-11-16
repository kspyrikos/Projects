from multiprocessing.sharedctypes import Value
from fastapi import FastAPI,APIRouter,Query,Response,BackgroundTasks
import uvicorn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import pairwise_distances_argmin_min
import ast # transform strings to list

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello Ocean"}

def get_categories():
    all_categories = []
    data = pd.read_csv('movies.csv')
    for i,genre in enumerate(data['categories']):
        list_categories = ast.literal_eval(genre)
        for categories in list_categories:
            all_categories.append(categories['name'])
    return list(set(all_categories))

def get_features(i):
    if i == 1:
        return ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
    else:
        return ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'genre']

def data_preprocessing(data):
    data.drop(['belongs_to_collection', 'homepage', 'tagline', 'poster_path', 
    'overview', 'imdb_id', 'spoken_languages'], inplace=True, axis=1)
    column_changes = ['production_companies', 'production_countries', 'categories']
    json_shrinker_dict = dict({'production_companies': list(), 'production_countries': list(), 'categories': list()})
    
    for col in column_changes:
        if col == 'production_companies':
            for i in data[col]:
                i = ast.literal_eval(i)
                if len(i) < 1:
                    json_shrinker_dict['production_companies'].append(None)

                for element in i:
                    json_shrinker_dict['production_companies'].append(element['name'])
                    break
        elif col == 'production_countries':
            for i in data[col]:
                i = ast.literal_eval(i)
                if len(i) < 1:
                    json_shrinker_dict['production_countries'].append(None)
                for element in i:
                    json_shrinker_dict['production_countries'].append(element['iso_3166_1'])
                    break
        else:
            for i in data[col]:
                i = ast.literal_eval(i)
                if len(i) < 1:
                    json_shrinker_dict['categories'].append(None)

                for element in i:
                    json_shrinker_dict['categories'].append(element['name'])
                    break

    for i in column_changes:
        data[i] = json_shrinker_dict[i]

    data.dropna(inplace=True)
    data['budget'] = data['budget'].astype(int)
    data.dropna(inplace=True)
    return data

def kmeans(scaled_data, clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    cluster_labels = kmeans.fit(scaled_data).labels_
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, scaled_data)

    string_labels = ["c{}".format(i) for i in cluster_labels]
    scaled_data['cluster_label'] = cluster_labels
    scaled_data['cluster_string'] = string_labels
    return scaled_data, closest

def clustering(data, clusters, features = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']):
    scalar = MinMaxScaler()
    scaled_data = data[features]
    title_data = scaled_data.copy()
    if 'genre' in features: # I am using label encoding to genre
        labelencoder = LabelEncoder() 
        scaled_data['genre']=labelencoder.fit_transform(scaled_data['genre'])
        data['genre']=labelencoder.fit_transform(data['genre'])
    scaled = scalar.fit_transform(data[features])
    scaled_data = pd.DataFrame(scaled, index=scaled_data.index, columns=scaled_data.columns)
    scaled_data, closest = kmeans(scaled_data, clusters) # calling 
    title_data = title_data.join(scaled_data[['cluster_label', 'cluster_string']])
    title_data = title_data.join(data[['title', 'categories']])
    
    return scaled_data, title_data, closest


  
def movies(category:str, clusters:int, feature1:str,feature2:str, feature3:str, 
feature4:str, feature5:str, feature6:str):
    movies_genres = []
    data = pd.read_csv('movies.csv')
    for i,genre in enumerate(data['categories']):
        list_categories = ast.literal_eval(genre)
        for categories in list_categories:
            if category == categories['name']:
                movies_genres.append(data['title'][i])
    data = data_preprocessing(data)
    data = data[data.categories==category].reset_index(False)
    
    if clusters == 1:
        return {"List of all {} movies:".format(category) : movies_genres}

    features = list(set([feature1, feature2, feature3, feature4, feature5, feature6]))
    features.remove('')
    if clusters != len(features):
        clusters = len(features)
    print(features)
    scaled_data, title_data, closest = clustering(data, clusters, features)
    
    res = []
    for c in title_data.cluster_label.unique():
        titles = title_data.loc[title_data.cluster_label ==  c, 'title'].to_list()
        movies_center = title_data.loc[closest[c],'title']
        res.append({"Cluster with movies like {0}".format(movies_center) : titles})

    return tuple(res)
    
     
    
def movies2(clusters:int, feature1:str,feature2:str, feature3:str, 
feature4:str, feature5:str, feature6:str, feature7:str, category:str, language:str, adult:str):
    data = pd.read_csv('movies.csv')
    data = data_preprocessing(data)
    data_clust = data.copy()
    data['genre'] = data['categories'].astype('category')
    data_clust['genre'] = data_clust['categories'].astype('category')
    # PREFERENCES

    if category != 'All':
        data = data[data.categories==category]

    if language == 'English':
        data = data[data.original_language=='en']
    elif language == 'Foreign':
        data = data[data.original_language!='en']
    
    if adult == 'Yes':
        data = data[data.adult==1]
    elif adult == 'No':
        data = data[data.adult==0]

    res2 = data['title'] # MOVIES WITH THE GIVEN PREFERENCES

    data = data.reset_index(False)
    data_clust = data_clust.reset_index(False)
    features = list(set([feature1, feature2, feature3, feature4, feature5, feature6, feature7]))
    features.remove('')
    if clusters != len(features):
        clusters = len(features)
    scaled_data, title_data, closest = clustering(data_clust, clusters, features)
    
    res = []
    movies = {}
    for c in title_data.cluster_label.unique():
        titles = title_data.loc[title_data.cluster_label ==  c, 'title'].to_list()
        movies_center = title_data.loc[closest[c],'title']
        res.append({"This cluster contains {0} different movies which are like '{1}'".format(len(titles), movies_center) : titles})
    
    movies["Here are all the movies, based on the selected preferences:"]=list(res2)
    return tuple(res), movies



# Router
movies_router=APIRouter()
@movies_router.post("/",
    summary='Endpoint to clustering',
    description='Endpoint to clustering'
)
def movies_databases(Category: str=Query("",enum=get_categories()), Clusters:int=Query(1,enum=[i for i in range (1, 31)]), Feature1:str=Query("",enum=get_features(1)),
Feature2:str=Query("",enum=get_features(1)), Feature3:str=Query("",enum=get_features(1)), Feature4:str=Query("",enum=get_features(1)), Feature5:str=Query("",enum=get_features(1)), 
 Feature6:str=Query("",enum=get_features(1))):
    
    return movies(category=Category, clusters=Clusters, feature1=Feature1, 
    feature2=Feature2, feature3=Feature3, feature4=Feature4,feature5=Feature5, 
    feature6=Feature6)

@movies_router.post("/clustering_with_genre_and_filtering/",
    summary='Endpoint to Cluster',
    description='Endpoint to Cluster'
)
def clustering_movies(Clusters:int=Query(2,enum=[i for i in range (2, 31)]), Feature1:str=Query("",enum=get_features(2)),
Feature2:str=Query("",enum=get_features(2)), Feature3:str=Query("",enum=get_features(2)), Feature4:str=Query("",enum=get_features(2)), Feature5:str=Query("",enum=get_features(2)), 
 Feature6:str=Query("",enum=get_features(2)), Feature7:str=Query("",enum=get_features(2)), Genre:str=Query("",enum=get_categories()+['all']), 
Language:str=Query("",enum=['English','Foreign','All']), Adults:str=Query("",enum=['All','No','Yes'])):
    
    return movies2(clusters=Clusters, feature1=Feature1, 
    feature2=Feature2, feature3=Feature3, feature4=Feature4,feature5=Feature5, 
    feature6=Feature6, feature7=Feature7, category=Genre, language=Language, adult=Adults)

app.include_router(
    movies_router,
    prefix="/api/movies",
    tags=["movies"],
    responses={404: {"description": "Not found"}},
)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
