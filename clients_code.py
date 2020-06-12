#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
import umap
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.figure_factory as ff
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_regression


# In[2]:


#Read Csv
clients_ds=pd.read_csv("C:/Users/bhara/OneDrive/Desktop/sem2/visual/Clients.csv")
#to check if nay null value in dataset
clients_ds.isnull().sum()/len(clients_ds)*100


# In[3]:



# Categorical boolean mask
clients_categorical_feature = clients_ds.dtypes==object
# filter categorical columns using mask and turn it into a list
clients_categorical_cols = clients_ds.columns[clients_categorical_feature].tolist()


# In[4]:


#Converting Categorical Data to Numerical data
le = LabelEncoder()
clients_ds[clients_categorical_cols] = clients_ds[clients_categorical_cols].apply(lambda col: le.fit_transform(col))
clients_ds[clients_categorical_cols].head(10)


# In[5]:



df=pd.get_dummies(clients_ds)


# In[6]:


#to normalize the data
feature_scaler = StandardScaler()
df = feature_scaler.fit_transform(df)


# In[130]:


#Get the HeatMap 
corrs = clients_ds.corr()
print(corrs)
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True,colorscale='Viridis')
offline.plot(figure,filename='corrheatmap_clients.html')


# In[9]:


# We first find the right number of clusters (K), by using the Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[81]:


# PCA on Full Data set:
pca = PCA(n_components = 2)
pca.fit(df)
x_pca = pca.transform(df)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[82]:


# PCA Dimensionality Reduction using k-label on Full-Client Dataset
kmeans = KMeans(n_clusters = 5)
kmeans.fit(df)
age= list(clients_ds["age"])
job = list(clients_ds["job"])
marital = list(clients_ds["marital"])
education = list(clients_ds["education"])
default= list(clients_ds["default"])
balance = list(clients_ds["balance"])
personal = list(clients_ds["personal"])
housing = list(clients_ds["housing"])
term=list(clients_ds["term"])
data = [go.Scatter(x=x_pca[:,0], y=x_pca[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='spectral', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on Full-Client Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA.html')


# In[10]:


#Umap on All dataset
client_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_client_fit = client_reducer.fit_transform(df)
umap_client_fit.shape


# In[17]:


# UMAP Dimensionality Reduction using k-label on Full-Client Dataset
kmeans = KMeans(n_clusters = 5)
kmeans.fit(df)
age= list(clients_ds["age"])
job = list(clients_ds["job"])
marital = list(clients_ds["marital"])
education = list(clients_ds["education"])
default= list(clients_ds["default"])
balance = list(clients_ds["balance"])
personal = list(clients_ds["personal"])
housing = list(clients_ds["housing"])
term=list(clients_ds["term"])
data = [go.Scatter(x=umap_client_fit[:,0], y=umap_client_fit[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='spectral', opacity=0.5),
                                text=[f'age:{a};job:{b};marital:{c},education:{d},default:{e},balance:{f},housing:{g},personal:{h},Term:{i}' for a,b,c,d,e,f,g,h,i in list(zip(age,job,marital,education,default,balance,housing,personal,term))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on Full-Client Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap.html')


# In[ ]:


data = [go.Scatter(x=umap_client_fit[:,0], y=umap_client_fit[:,1], mode='markers',
                    marker = dict(color=marital, colorscale='spectral', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction ColorBy Marital on Full-Client Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap.html')


# In[110]:


# Implementing t-SNE to visualize dataset
from sklearn.manifold import TSNE
kmeans = KMeans(n_clusters = 5)
kmeans.fit(df)
tsne_client = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_client_fit = tsne_client.fit_transform(df)


data = [go.Scatter(x=tsne_client_fit[:,0], y=tsne_client_fit[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='spectral', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Full-Client Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Client.html')


# In[125]:


data = [go.Scatter(x=tsne_client_fit[:,0], y=tsne_client_fit[:,1], mode='markers',
                    marker = dict(color=marital, colorscale='spectral', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using marital on Full-Client Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Client-ColorBy-Marital.html')


# In[20]:


clients_ds_subset1=clients_ds[['job', 'marital', 'education', 'default', 'balance','housing','personal']]
df_subset1=pd.get_dummies(clients_ds_subset1)
feature_scaler = StandardScaler()
df_subset1 = feature_scaler.fit_transform(df_subset1)


# In[21]:


# We first find the right number of clusters (K), by using the Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(df_subset1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[83]:


# PCA on Subset1:
pca = PCA(n_components = 2)
pca.fit(df_subset1)
x_pca1 = pca.transform(df_subset1)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[132]:


#  pca Dimensionality Reduction using k-label on Subset 1
#'job', 'marital', 'education', 'default', 'balance','housing','personal'
kmeans1 = KMeans(n_clusters = 4)
kmeans1.fit(df_subset1)
job = list(clients_ds_subset1["job"])
marital = list(clients_ds_subset1["marital"])
education = list(clients_ds_subset1["education"])
default= list(clients_ds_subset1["default"])
balance = list(clients_ds_subset1["balance"])
personal = list(clients_ds_subset1["personal"])
housing = list(clients_ds_subset1["housing"])

data = [go.Scatter(x=x_pca1[:,0], y=x_pca1[:,1], mode='markers',
                    marker = dict(color=kmeans1.labels_, colorscale='jet', opacity=0.5),
                                text=[f' job: {a}; marital:{b}, education:{c}, default : {d} , balance : {e},housing : {f},personal :{g}' for a,b,c,d,e,f,g in list(zip(job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on Subset-1', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA-subset1.html')


# In[32]:


client_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_clientsubset1__fit = client_reducer.fit_transform(df_subset1)
umap_clientsubset1__fit.shape


# In[133]:


# umap Dimensionality Reduction using k-label on Subset 1
#'job', 'marital', 'education', 'default', 'balance','housing','personal'
kmeans1 = KMeans(n_clusters = 4)
kmeans1.fit(df_subset1)
job = list(clients_ds_subset1["job"])
marital = list(clients_ds_subset1["marital"])
education = list(clients_ds_subset1["education"])
default= list(clients_ds_subset1["default"])
balance = list(clients_ds_subset1["balance"])
personal = list(clients_ds_subset1["personal"])
housing = list(clients_ds_subset1["housing"])

data = [go.Scatter(x=umap_clientsubset1__fit[:,0], y=umap_clientsubset1__fit[:,1], mode='markers',
                    marker = dict(color=kmeans1.labels_, colorscale='jet', opacity=0.5),
                                text=[f' job: {a}; marital:{b}, education:{c}, default : {d} , balance : {e},housing : {f},personal :{g}' for a,b,c,d,e,f,g in list(zip(job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on Subset-1', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap-subset1.html')


# In[128]:


# Implementing t-SNE to visualize subset1
kmeans1 = KMeans(n_clusters = 5)
kmeans1.fit(df_subset1)
tsne_client_subset1 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_client_fit1 = tsne_client_subset1.fit_transform(df_subset1)


data = [go.Scatter(x=tsne_client_fit1[:,0], y=tsne_client_fit1[:,1], mode='markers',
                    marker = dict(color=kmeans1.labels_, colorscale='jet', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset1', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Subset1.html')


# In[34]:


clients_ds_subset2=clients_ds[['age', 'job', 'education', 'balance', 'personal','housing']]
df_subset2=pd.get_dummies(clients_ds_subset2)
feature_scaler = StandardScaler()
df_subset2 = feature_scaler.fit_transform(df_subset2)


# In[35]:


# We first find the right number of clusters (K), by using the Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(df_subset2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[86]:


# PCA on Subset2:
pca = PCA(n_components = 2)
pca.fit(df_subset2)
x_pca2 = pca.transform(df_subset2)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[87]:


# PCA Dimensionality Reduction using k-label on subset2
kmeans2 = KMeans(n_clusters = 4)
kmeans2.fit(df_subset2)

age= list(clients_ds_subset2["age"])
job = list(clients_ds_subset2["job"])
education = list(clients_ds_subset2["education"])
balance = list(clients_ds_subset2["balance"])
housing = list(clients_ds_subset2["housing"])
data = [go.Scatter(x=x_pca2[:,0], y=x_pca2[:,1], mode='markers',
                    marker = dict(color=kmeans2.labels_, colorscale='rainbow', opacity=0.5),
                                text=[f'age: {a}; job: {b}; education:{c}, balance:{d}, personal : {e} , housing : {f}' for a,b,c,d,e,f in list(zip(age,job,education,balance,personal,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on subset2', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA-subset2.html')


# In[37]:


client_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_clientsubset2__fit = client_reducer.fit_transform(df_subset2)
umap_clientsubset2__fit.shape


# In[43]:


# UMAP Dimensionality Reduction using k-label on subset2
kmeans2 = KMeans(n_clusters = 4)
kmeans2.fit(df_subset2)

age= list(clients_ds_subset2["age"])
job = list(clients_ds_subset2["job"])
education = list(clients_ds_subset2["education"])
balance = list(clients_ds_subset2["balance"])
housing = list(clients_ds_subset2["housing"])
data = [go.Scatter(x=umap_clientsubset2__fit[:,0], y=umap_clientsubset2__fit[:,1], mode='markers',
                    marker = dict(color=kmeans2.labels_, colorscale='rainbow', opacity=0.5),
                                text=[f'age: {a}; job: {b}; education:{c}, balance:{d}, personal : {e} , housing : {f}' for a,b,c,d,e,f in list(zip(age,job,education,balance,personal,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on subset2', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap-subset2.html')


# In[129]:


# Implementing t-SNE to visualize subset2
kmeans2 = KMeans(n_clusters = 5)
kmeans2.fit(df_subset2)
tsne_client_subset2 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_client_fit2 = tsne_client_subset2.fit_transform(df_subset2)


data = [go.Scatter(x=tsne_client_fit2[:,0], y=tsne_client_fit2[:,1], mode='markers',
                    marker = dict(color=kmeans2.labels_, colorscale='rainbow', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset2', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Subset2.html')


# In[45]:


clients_ds_subset3=clients_ds[['age', 'job', 'marital', 'balance', 'housing', 'personal','education']]
df_subset3=pd.get_dummies(clients_ds_subset3)
feature_scaler = StandardScaler()
df_subset3 = feature_scaler.fit_transform(df_subset3)


# In[46]:


# We first find the right number of clusters (K), by using the Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(df_subset3)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[88]:


# PCA on Subset3:
pca = PCA(n_components = 2)
pca.fit(df_subset3)
x_pca3 = pca.transform(df_subset3)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[89]:


# PCA Dimensionality Reduction using k-label on subset3
#['age', 'job', 'marital', 'balance', 'housing', 'personal','education']
kmeans3 = KMeans(n_clusters = 4)
kmeans3.fit(df_subset3)

age= list(clients_ds_subset3["age"])
job = list(clients_ds_subset3["job"])
education = list(clients_ds_subset3["education"])
balance = list(clients_ds_subset3["balance"])
housing = list(clients_ds_subset3["housing"])
marital = list(clients_ds_subset3["marital"])

data = [go.Scatter(x=x_pca3[:,0], y=x_pca3[:,1], mode='markers',
                    marker = dict(color=kmeans3.labels_, colorscale='magenta', opacity=0.5),
                                text=[f'age: {a}; job: {b}; education:{c}, balance:{d}, personal : {e} , housing : {f},marital:{g}' for a,b,c,d,e,f,g in list(zip(age,job,education,balance,personal,housing,marital))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on subset3', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA-subset3.html')


# In[48]:


client_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_clientsubset3__fit = client_reducer.fit_transform(df_subset3)
umap_clientsubset3__fit.shape


# In[126]:


# UMAP Dimensionality Reduction using k-label on subset3
kmeans3 = KMeans(n_clusters = 3)
kmeans3.fit(umap_clientsubset3__fit)

age= list(clients_ds_subset3["age"])
job = list(clients_ds_subset3["job"])
marital = list(clients_ds_subset3["marital"])
education = list(clients_ds_subset3["education"])
balance = list(clients_ds_subset3["balance"])
personal = list(clients_ds_subset3["personal"])
housing = list(clients_ds_subset3["housing"])
marital = list(clients_ds_subset3["marital"])


data = [go.Scatter(x=umap_clientsubset3__fit[:,0], y=umap_clientsubset3__fit[:,1], mode='markers',
                    marker = dict(color=kmeans3.labels_, colorscale='magenta', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, balance : {e} , personal : {f},housing : {g}' for a,b,c,d,e,f,g in list(zip(age,job,marital,education,balance,personal,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on subset-3', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap-subset3.html')


# In[131]:


# Implementing t-SNE to visualize subset3
kmeans3 = KMeans(n_clusters = 4)
kmeans3.fit(df_subset3)

tsne_client_subset3 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_client_fit3 = tsne_client_subset3.fit_transform(df_subset3)


data = [go.Scatter(x=tsne_client_fit3[:,0], y=tsne_client_fit3[:,1], mode='markers',
                    marker = dict(color=kmeans3.labels_, colorscale='magenta', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset3', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Subset3.html')


# In[54]:


clients_ds_subset4=clients_ds[['job', 'marital', 'education', 'default', 'balance', 'housing','age']]
df_subset4=pd.get_dummies(clients_ds_subset4)
feature_scaler = StandardScaler()
df_subset4 = feature_scaler.fit_transform(df_subset4)


# In[55]:


# We first find the right number of clusters (K), by using the Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(df_subset4)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[90]:


pca = PCA(n_components = 2)
pca.fit(df_subset4)
x_pca4 = pca.transform(df_subset4)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[91]:


# PCA Dimensionality Reduction using k-label on subset4
kmeans4 = KMeans(n_clusters = 4)
kmeans4.fit(df_subset4)
#['job', 'marital', 'education', 'default', 'balance', 'housing','age']]
age= list(clients_ds_subset4["age"])
job = list(clients_ds_subset4["job"])
education = list(clients_ds_subset4["education"])
balance = list(clients_ds_subset4["balance"])
housing = list(clients_ds_subset4["housing"])
data = [go.Scatter(x=x_pca4[:,0], y=x_pca4[:,1], mode='markers',
                    marker = dict(color=kmeans4.labels_, colorscale='tropic', opacity=0.5),
                                text=[f'age: {a}; job: {b}; education:{c}, balance:{d}, default : {e} , housing : {f},marital:{g}' for a,b,c,d,e,f,g in list(zip(age,job,education,balance,personal,housing,marital))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on subset4', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA-subset4.html')


# In[56]:


client_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_clientsubset4__fit = client_reducer.fit_transform(df_subset4)
umap_clientsubset4__fit.shape


# In[58]:


# UMAP Dimensionality Reduction using k-label on subset4

kmeans4 = KMeans(n_clusters = 4)
kmeans4.fit(df_subset4)
age= list(clients_ds_subset4["age"])
job = list(clients_ds_subset4["job"])
marital = list(clients_ds_subset4["marital"])
education = list(clients_ds_subset4["education"])
default= list(clients_ds_subset4["default"])
balance = list(clients_ds_subset4["balance"])
housing = list(clients_ds_subset4["housing"])
data = [go.Scatter(x=umap_clientsubset4__fit[:,0], y=umap_clientsubset4__fit[:,1], mode='markers',
                    marker = dict(color=kmeans4.labels_, colorscale='tropic', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g}' for a,b,c,d,e,f,g, in list(zip(age,job,marital,education,default,balance,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on subset4', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap-subset4.html')


# In[114]:


# Implementing t-SNE to visualize subset4
kmeans4 = KMeans(n_clusters = 4)
kmeans4.fit(df_subset4)

tsne_client_subset4 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_client_fit4 = tsne_client_subset4.fit_transform(df_subset4)


data = [go.Scatter(x=tsne_client_fit4[:,0], y=tsne_client_fit4[:,1], mode='markers',
                    marker = dict(color=kmeans4.labels_, colorscale='tropic', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset4', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Subset4.html')


# In[59]:


clients_ds_subset5=clients_ds[['age', 'marital', 'education', 'balance', 'housing', 'personal','job']]
df_subset5=pd.get_dummies(clients_ds_subset5)
feature_scaler = StandardScaler()
df_subset5 = feature_scaler.fit_transform(df_subset5)


# In[60]:


# We first find the right number of clusters (K), by using the Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(df_subset5)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[92]:


pca = PCA(n_components = 2)
pca.fit(df_subset5)
x_pca5 = pca.transform(df_subset5)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[95]:


# PCA Dimensionality Reduction using k-label on subset5

kmeans5 = KMeans(n_clusters = 2)
kmeans5.fit(df_subset5)
age= list(clients_ds_subset5["age"])
job = list(clients_ds_subset5["job"])
marital = list(clients_ds_subset5["marital"])
education = list(clients_ds_subset5["education"])
personal= list(clients_ds_subset5["personal"])
balance = list(clients_ds_subset5["balance"])
housing = list(clients_ds_subset5["housing"])
data = [go.Scatter(x=x_pca5[:,0], y=x_pca5[:,1], mode='markers',
                    marker = dict(color=kmeans5.labels_, colorscale='reds', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, personal : {e} , balance : {f},housing : {g}' for a,b,c,d,e,f,g, in list(zip(age,job,marital,education,personal,balance,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on subset5', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA-subset5.html')


# In[62]:


client_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_clientsubset5__fit = client_reducer.fit_transform(df_subset5)
umap_clientsubset5__fit.shape


# In[64]:


# UMAP Dimensionality Reduction using k-label on subset5

kmeans5 = KMeans(n_clusters = 2)
kmeans5.fit(df_subset5)
age= list(clients_ds_subset5["age"])
job = list(clients_ds_subset5["job"])
marital = list(clients_ds_subset5["marital"])
education = list(clients_ds_subset5["education"])
personal= list(clients_ds_subset5["personal"])
balance = list(clients_ds_subset5["balance"])
housing = list(clients_ds_subset5["housing"])
data = [go.Scatter(x=umap_clientsubset5__fit[:,0], y=umap_clientsubset5__fit[:,1], mode='markers',
                    marker = dict(color=kmeans5.labels_, colorscale='reds', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, personal : {e} , balance : {f},housing : {g}' for a,b,c,d,e,f,g, in list(zip(age,job,marital,education,personal,balance,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on subset5', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap-subset5.html')


# In[115]:


# Implementing t-SNE to visualize subset5
kmeans5 = KMeans(n_clusters = 2)
kmeans5.fit(df_subset5)
tsne_client_subset5 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_client_fit5 = tsne_client_subset5.fit_transform(df_subset5)


data = [go.Scatter(x=tsne_client_fit5[:,0], y=tsne_client_fit5[:,1], mode='markers',
                    marker = dict(color=kmeans5.labels_, colorscale='reds', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}, education:{d}, default : {e} , balance : {f},housing : {g},personal :{h}' for a,b,c,d,e,f,g,h in list(zip(age,job,marital,education,default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset5', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Subset5.html')


# In[70]:


clients_ds_subset6=clients_ds[['default','balance','housing','personal']]
df_subset6=pd.get_dummies(clients_ds_subset6)
feature_scaler = StandardScaler()
df_subset6 = feature_scaler.fit_transform(df_subset6)


# In[71]:


# We first find the right number of clusters (K), by using the Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(df_subset6)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[96]:


pca = PCA(n_components = 2)
pca.fit(df_subset6)
x_pca6 = pca.transform(df_subset6)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[98]:


# PCA Dimensionality Reduction using k-label on subset6
#'default','balance','housing','personal'
kmeans6 = KMeans(n_clusters = 5)
kmeans6.fit(df_subset6)
default= list(clients_ds_subset6["default"])
personal= list(clients_ds_subset6["personal"])
balance = list(clients_ds_subset6["balance"])
housing = list(clients_ds_subset6["housing"])
data = [go.Scatter(x=x_pca6[:,0], y=x_pca6[:,1], mode='markers',
                    marker = dict(color=kmeans6.labels_, colorscale='ylorrd', opacity=0.5),
                                text=[f'default: {a}, personal : {b} , balance : {c},housing : {d}' for a,b,c,d in list(zip(default,personal,balance,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on subset6', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA-subset6.html')


# In[73]:


client_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_clientsubset6__fit = client_reducer.fit_transform(df_subset6)
umap_clientsubset6__fit.shape


# In[80]:


# UMAP Dimensionality Reduction using k-label on subset6
default= list(clients_ds_subset6["default"])
personal= list(clients_ds_subset6["personal"])
balance = list(clients_ds_subset6["balance"])
housing = list(clients_ds_subset6["housing"])
data = [go.Scatter(x=umap_clientsubset6__fit[:,0], y=umap_clientsubset6__fit[:,1], mode='markers',
                    marker = dict(color=kmeans6.labels_, colorscale='ylorrd', opacity=0.5),
                                text=[f'default: {a}, personal : {b} , balance : {c},housing : {d}' for a,b,c,d in list(zip(default,personal,balance,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on subset6', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='umap-subset6.html')


# In[116]:


# Implementing t-SNE to visualize subset6
kmeans6 = KMeans(n_clusters = 5)
kmeans6.fit(df_subset6)
tsne_client_subset6 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_client_fit6 = tsne_client_subset6.fit_transform(df_subset6)


data = [go.Scatter(x=tsne_client_fit6[:,0], y=tsne_client_fit6[:,1], mode='markers',
                    marker = dict(color=kmeans6.labels_, colorscale='ylorrd', opacity=0.5),
                                text=[f'default : {a} , balance : {b},housing : {c},personal :{d}' for a,b,c,d in list(zip(default,balance,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset6', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Subset6.html')

