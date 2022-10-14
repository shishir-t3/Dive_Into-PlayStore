# importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers, regularizers

# reading all the files neccesary
ASSETS_DIRECTORY = "assets"

# has genres and appId's for the entirely fetched apps.
apps_data = pd.read_csv(f"{ASSETS_DIRECTORY}/apps_data.csv")
# to have a cleaned data.
engineered_apps_data = pd.read_csv(
    f"{ASSETS_DIRECTORY}/engineered_apps_data.csv")
# has all the standardised data which goes into the model training.
model_df = pd.read_csv(f"{ASSETS_DIRECTORY}/Model_df.csv")
# importing the reviews file
reviews = pd.read_csv(f"{ASSETS_DIRECTORY}/reviews.csv")

# helper counter variable to time the workflow
counter = 0
function_call = False
split = False

# Function to provide recommendation based on user-Id.
apps_to_take = pd.read_csv(f"{ASSETS_DIRECTORY}/apps_to_take.csv", index_col=0)
apps_dict = {app: i for (app, i) in zip(
    apps_to_take.appId.unique(), range(len(apps_to_take.appId.unique())))}

# Giving a title for the App.
st.title('A dive into the Google Playstore!!')

# Get the caption
st.caption('Get your app recommendations by answering few questions.')

# writing the input fields to get the information from the user

# selecting genres
options = st.multiselect(
    'What are your favorite genres?',
    list(apps_data.genre.unique()))

# radio button to choose between free or paid apps --> try using Model_df for reference
amount_bool = st.radio(
    "Preferred type of app",
    ('Free', 'Paid'),
    horizontal=True)
if amount_bool == 'Free':
    amount_bool = 1
else:
    amount_bool = 0

# give the range for number of installs
min_installs = st.radio(
    "What do you want the minimum number of installs to be?",
    ('100K', '10M', '100M', '1B'),
    horizontal=False)
if min_installs == '100K':
    min_installs = 100000
elif min_installs == '10M':
    min_installs = 10000000
elif min_installs == '100M':
    min_installs = 100000000
else:
    min_installs = 10000000000

# list of apps to select
genre_list, free, installs = [], 0, 0
if options:
    genre_list = options
    free = amount_bool
    installs = min_installs
# empty temporary dataframe
temp_df = pd.DataFrame([], columns=engineered_apps_data.columns)
for i in range(len(genre_list)):
    a = engineered_apps_data[engineered_apps_data['genre'] == genre_list[i]]
    b = a[a['free'] == free]
    c = b[b['installs'] >= installs]
    d = c.sort_values('score', ascending=False).head(5)
    temp_df = pd.concat([temp_df, d], ignore_index=True, axis=0)

titles_list = st.multiselect(
    'Select few apps from the below list?',
    list(temp_df.title))

if st.button('Recommend Me!!'):
    userid = 1000
    app_vals = apps_data['appId'][apps_data['title'].isin(titles_list)]
    app_idx = [val for key, val in apps_dict.items() if key in app_vals]
    X_df = model_df[model_df['appId'].isin(app_idx)][:len(app_vals)]
    X_df['user'] = userid
    split = True
    counter += 1
else:
    st.write('Click on the Button above to get your recommendations')


# Now we scale the dataset and split it before building the model.
scaler = StandardScaler()
df_to_std = model_df.drop(['appId', 'user', 'y'], axis=1)

df_std = scaler.fit_transform(df_to_std)
df_std = pd.DataFrame(df_std, columns=df_to_std.columns)

df_std = model_df[['appId', 'user', 'y']].join(df_std)

# Splitting the dataset.
if split:
    train, test = train_test_split(
        df_std, train_size=0.75, random_state=40, stratify=df_std['y'])

    train = pd.concat([train, X_df])
    split = False

# Also Making a list of app features.

features = df_to_std.columns
n_features = len(features)
n_users = 1001
n_apps = 1130

# HElPER FUNCTIONS


def compute_score(user_val, app_val, measure='dot'):

    u = user_val
    a = app_val

    if measure == 'cosine':

        u = u / (np.linalg.norm(u))
        a = a / (np.linalg.norm(a, keepdims=True))

    scores = u.dot(a.T)
    return scores


# Making a user dictionary.
unique_list = reviews['userImage'].value_counts().nlargest(1001)[1:]
unique_names = unique_list.index
user_dict = {user: i for (user, i) in zip(
    unique_names, range(len(unique_names)))}


def Top_Recommendation(id, app_vals, measure, n=5):

    scores = []

    vals = app_vals

    # Retrieving apps which are not in user's rated apps.
    select_apps = [app for app in apps_dict.keys() if app not in vals]
    # Getting Indexes for apps in app-embedding.
    select_apps = list(map(lambda x: apps_dict.get(x), select_apps))

    for idx in select_apps:

        # Calculating the top scores based on the measure
        score = compute_score(
            user_embedding[id], App_embedding[idx], measure=measure)
        scores.append(score)

    best_idx = np.argsort(scores)[-n:]
    best_scores = np.sort(scores)[-n:]
    # Taking the top 5 scores and returning the recommended apps.
    pos = [list(apps_dict.values()).index(i) for i in best_idx]
    app_id = [list(apps_dict.keys())[i] for i in pos]

    return app_id, best_scores


# Function to calculate score of an user-app pair based on dot product or cosine similarity.


if counter == 1:
    # preparing the model.
    # Building Model Input layers.

    users_in = layers.Input(shape=(1, ), name='User Input')
    apps_in = layers.Input(shape=(1, ), name='App Input')

    features_in = layers.Input(shape=(n_features, ), name='Feature Input')

    # Creating embeddings, flattening the output and getting Dot product.
    embed_size = 50
    reg = regularizers.L1(0.001)

    users_embed = layers.Embedding(input_dim=n_users, output_dim=embed_size,
                                   embeddings_regularizer=reg, name='User-Embedding')(users_in)
    apps_embed = layers.Embedding(input_dim=n_apps, output_dim=embed_size,
                                  embeddings_regularizer=reg, name='App-Embedding')(apps_in)

    users_flat = layers.Flatten()(users_embed)
    apps_flat = layers.Flatten()(apps_embed)

    dot_prd = layers.Dot(normalize=True, axes=1)([users_flat, apps_flat])

    # Dense layer for features input.

    feat_1 = layers.Dense(units=n_features, activation='relu',
                          kernel_regularizer=reg)(features_in)

    # Merging the layers to get final output and compiling the model.

    merged = layers.Concatenate()([dot_prd, feat_1])

    dense_1 = layers.Dense(units=256, activation='relu',
                           kernel_regularizer=reg)(merged)
    dropout_1 = layers.Dropout(0.2)(dense_1)

    dense_2 = layers.Dense(units=128, activation='relu',
                           kernel_regularizer=reg)(dropout_1)
    dropout_2 = layers.Dropout(0.2)(dense_2)

    dense_3 = layers.Dense(units=64, activation='relu',
                           kernel_regularizer=reg)(dropout_2)

    output = layers.Dense(1, activation='linear')(dense_3)

    hyb_model = models.Model(
        inputs=[users_in, apps_in, features_in], outputs=output)

    hyb_model.compile(optimizer=optimizers.Adam(0.001),
                      loss='mean_absolute_error', metrics=['mse'])

    # Fitting the model on the train data.

    history = hyb_model.fit(x=[train['user'], train['appId'], train[features]],
                            y=train['y'], epochs=50, batch_size=32,
                            validation_split=0.3, verbose=0)

    # Getting the Apps and user embeddings.
    App_embedding = hyb_model.get_layer(name="App-Embedding").get_weights()[0]
    user_embedding = hyb_model.get_layer(
        name="User-Embedding").get_weights()[0]
    counter = 0
    function_call = True

if function_call:
    app_id, best_score = Top_Recommendation(
        1000, app_vals=app_vals, measure="cosine")
    names = apps_data['title'][apps_data['appId'].isin(app_id)].values
    temp = []
    for i in best_score:
        temp.append(f"{i*100:.2f} %")
    best_score = temp[::-1]
    names = names[::-1]
    final = {'App Name': names, 'How likely we recommend to you': best_score}
    st.subheader('These are the top 5 apps we reccomend you...!')
    final_df = pd.DataFrame(final)
    st.dataframe(final_df, use_container_width=True)
    function_call = False
