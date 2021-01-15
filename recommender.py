import turicreate as tc
import codecs
import pandas as pd
from turicreate import SFrame
Books_path = "Data/"

df_books = pd.read_csv(Books_path + 'BX-Books.csv', sep = ';', encoding = "latin-1", error_bad_lines = False, verbose = False)
df_users = pd.read_csv(Books_path + 'BX-Users.csv', sep = ';', encoding = "latin-1", error_bad_lines = False)
df_ratings = pd.read_csv(Books_path + 'BX-Book-Ratings.csv', sep = ';', encoding = "latin-1", error_bad_lines = False)
df_ratings['Book-Rating'].unique()

df_books.to_csv(Books_path+"books.csv")
df_users.to_csv(Books_path+"users.csv")
df_ratings.to_csv(Books_path+"ratings.csv")

sf_books = tc.SFrame.read_csv(Books_path+"books.csv", verbose = False)
sf_users = tc.SFrame.read_csv(Books_path+"users.csv", verbose = False)
sf_ratings = tc.SFrame.read_csv(Books_path+"ratings.csv", verbose = False)

sf_books.shape
df_books.shape
sf_users.shape
df_users.shape
sf_ratings.shape
df_ratings.shape


df_books.head(2)

sf_users.shape
df_users
df_ratings.head(2)

df_ratings.groupby(['ISBN']).count()
df_temp = df_ratings[0:1000]

grp = df_temp.groupby(['ISBN'])

grp['User-ID'].count()

grp.get_group('0971880107')
len(df_ratings['ISBN'].unique())
len(df_ratings['Book-Rating'].unique())
df_ratings['Book-Rating'].unique()
df_ratings['ISBN'].count()

p = df_temp['ISBN'].value_counts(sort = True).rename_axis('ISBN').reset_index(name = 'counts')
p
df_ratings['Book-Rating'].value_counts()
# Only the explicity rating, so drop the rating value = 0
df_ratings_explicit = df_ratings[df_ratings['Book-Rating'] != 0]
df_ratings_explicit.shape
df_ratings.shape

df_ratings_explicit['Book-Rating'].describe()

#most of the ratings are high > 8
df_ratings_explicit["Book-Rating"].hist(bins=10)


#Map the users and the Books

df_ratings_explicit.head(3)

users = df_ratings_explicit['User-ID'].unique()
len(users)

user_map = {x:user_x for x,user_x in enumerate(users)}
i_user_map =   {user_x:x for x,user_x in enumerate(users)}

books = df_ratings_explicit['ISBN'].unique()
len(books)


book_map = {x:book_x for x,book_x in enumerate(books)}
i_book_map =   {book_x:x for x,book_x in enumerate(books)}

df_ratings_explicit["User-ID"] = df_ratings_explicit["User-ID"].map(i_user_map)
df_ratings_explicit.head(10)


df_ratings_explicit["old_book_id"] = df_ratings_explicit["ISBN"] # copying for join with metadata

df_ratings_explicit["ISBN"] = df_ratings_explicit["ISBN"].map(i_book_map)

users_nb = df_ratings_explicit['User-ID'].value_counts().reset_index()
users_nb.columns = ['User-ID', 'nb_lines']
users_nb
users_nb['nb_lines'].describe()

import seaborn
users_nb['nb_lines'].hist()



books_nb = df_ratings_explicit['old_book_id'].value_counts().reset_index()
books_nb.columns= ['old_book_id','nb_lines']
books_nb['nb_lines'].describe()
books_nb['nb_lines'].hist()

books_nb
# Let's find a few popular books
df_ratings_explicit.isnull().values.any()

df_ratings_explicit.head(10)
df_books.head(10)
df_ratings_explicit.head(2)

books_merged = pd.merge(df_books, books_nb, how='left',left_on='ISBN', right_on='old_book_id')
books_merged.shape
books_merged.sort_values(by = 'nb_lines', ascending = True).head(20)
import sklearn
from sklearn.model_selection import train_test_split
import pickle
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import pydot
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout
from tensorflow.keras.layers import Dot
from tensorflow.keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Dropout, Reshape
import keras
from keras.models import Model
from keras import optimizers


ratings_train, ratings_test = train_test_split(df_ratings_explicit, test_size = 0.2, random_state = 10)

user_id_input = Input(shape=[1], name = 'user')
item_id_input = Input(shape=[1], name= 'item')

embedding_size = 30

user_embedding = Embedding()
item_embedding = Embedding()
user_vecs =
item_vecs =
