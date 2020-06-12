#!/usr/bin/env python
# coding: utf-8

# In[63]:




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


# In[64]:



patches_ds=pd.read_csv("Patches.csv")

patches_ds.isnull().sum()/len(patches_ds)*100


# In[127]:





# In[65]:





# Categorical boolean mask
patches_categorical_feature = patches_ds.dtypes==object
# filter categorical columns using mask and turn it into a list
patches_categorical_cols = patches_ds.columns[patches_categorical_feature].tolist()


# In[66]:




le = LabelEncoder()
patches_ds[patches_categorical_cols] = patches_ds[patches_categorical_cols].apply(lambda col: le.fit_transform(col))
patches_ds[patches_categorical_cols].head(10)


# In[67]:



df=pd.get_dummies(patches_ds)


# In[126]:



feature_scaler = StandardScaler()
df = feature_scaler.fit_transform(df)
print(df)


# In[69]:




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


# In[70]:




# PCA on Full Data set:
pca = PCA(n_components = 2)
pca.fit(df)
x_pca = pca.transform(df)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[71]:



# PCA Dimensionality Reduction using k-label on Full-patches Dataset
kmeans = KMeans(n_clusters = 2)
kmeans.fit(df)

Elevation = list(patches_ds["Elevation"])
Slope = list(patches_ds["Slope"])
Horizontal_Distance_To_Hydrology = list(patches_ds["Horizontal_Distance_To_Hydrology"])
Vertical_Distance_To_Hydrology = list(patches_ds["Vertical_Distance_To_Hydrology"])
Horizontal_Distance_To_Roadways = list(patches_ds["Horizontal_Distance_To_Roadways"])
Horizontal_Distance_To_Fire_Points = list(patches_ds["Horizontal_Distance_To_Fire_Points"])
Tree = list(patches_ds["Tree"])

data = [go.Scatter(x=x_pca[:,0], y=x_pca[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='spectral', opacity=0.5),
                                text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{g}' for a,b,c,d,e,f,g in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on Full-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA_all.html')


# In[72]:


patches_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_patches_fit = patches_reducer.fit_transform(df)
umap_patches_fit.shape



data = [go.Scatter(x=umap_patches_fit[:,0], y=umap_patches_fit[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='spectral', opacity=0.5),
                                text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{g}' for a,b,c,d,e,f,g in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on Full-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PatchesUmapAll.html')


# In[96]:


kmeans = KMeans(n_clusters = 2)
kmeans.fit(df)

tsne_patches = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_patches_fit = tsne_patches.fit_transform(df)


data = [go.Scatter(x=tsne_patches_fit[:,0], y=tsne_patches_fit[:,1], mode='markers',
                   marker = dict(color=kmeans.labels_, colorscale='spectral', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{}'
                         for a,b,c,d,e,f in list(zip(Elevation, Slope, Horizontal_Distance_To_Hydrology , Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,tree))],
                   hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Full-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Patches_all.html')


# In[73]:


# Subset1

patches_ds_subset1=patches_ds[['Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']]
df_subset1=pd.get_dummies(patches_ds_subset1)
feature_scaler = StandardScaler()
df_subset1 = feature_scaler.fit_transform(df_subset1)


# In[74]:




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


# In[75]:



# PCA on Subset1:
pca1 = PCA(n_components = 2)
pca1.fit(df_subset1)
x_pca1 = pca1.transform(df_subset1)
print("Variance explained by each of the n_components: ",pca1.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca1.explained_variance_ratio_))


# In[97]:


kmeans1 = KMeans(n_clusters = 4)
kmeans1.fit(df_subset1)

#['Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']

data = [go.Scatter(x=x_pca1[:,0], y=x_pca1[:,1], mode='markers',
                    marker = dict(color=kmeans1.labels_, colorscale='magenta', opacity=0.5),
                                text=[f'Slope:{a};HorizontalHydro:{b},VerticalHydro:{c},HorizontalRoad:{d},HorizontalFirepts:{e}' for a,b,c,d,e in list(zip(Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on Subset1-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='Patches_PCA_subset1.html')


# In[98]:


patches_reducer1 = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_patches_fit1 = patches_reducer1.fit_transform(df_subset1)
umap_patches_fit1.shape



data = [go.Scatter(x=umap_patches_fit1[:,0], y=umap_patches_fit1[:,1], mode='markers',
                    marker = dict(color=kmeans1.labels_, colorscale='magenta', opacity=0.5),
                                text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{g}' for a,b,c,d,e,f,g in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on Subset1-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PatchesUmap_subset1.html')


# In[99]:


tsne_patches_subset1 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_patches_fit1 = tsne_patches_subset1.fit_transform(df_subset1)

data = [go.Scatter(x=tsne_patches_fit1[:,0], y=tsne_patches_fit1[:,1], mode='markers',
                   marker = dict(color=kmeans1.labels_, colorscale='magenta', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{g}' for a,b,c,d,e,f,g in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,Tree))],
                   hoverinfo='text')]


layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on subset1-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Patches_subset1.html')
                   


# In[100]:




patches_ds_subset2=patches_ds[['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Tree']]
df_subset2=pd.get_dummies(patches_ds_subset2)
feature_scaler = StandardScaler()
df_subset2 = feature_scaler.fit_transform(df_subset2)


# In[101]:




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


# In[102]:




# PCA on Subset2:
pca = PCA(n_components = 2)
pca.fit(df_subset2)
x_pca2 = pca.transform(df_subset2)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[103]:


# PCA Dimensionality Reduction using k-label on subset2
kmeans2 = KMeans(n_clusters = 3)
kmeans2.fit(df_subset2)

# PCA Dimensionality Reduction using k-label on subset2
data = [go.Scatter(x=x_pca2[:,0], y=x_pca2[:,1], mode='markers',
                   marker = dict(color=kmeans2.labels_, colorscale='rainbow', opacity=0.5),# PCA Dimensionality Reduction using k-label on subset2
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c}Tree:{d}' for a,b,c,d in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Tree))],
                   hoverinfo='text')]


layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on subset2-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='Patches_PCA_subset2.html')                   


# In[104]:


# UMAP Dimensionality Reduction using k-label on subset2


patches_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_patches_subset2__fit = patches_reducer.fit_transform(df_subset2)
umap_patches_subset2__fit.shape


data = [go.Scatter(x=umap_patches_subset2__fit[:,0], y=umap_patches_subset2__fit[:,1], mode='markers',
                    marker = dict(color=kmeans2.labels_, colorscale='rainbow', opacity=0.5),
                                text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{g}' for a,b,c,d,e,f,g in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on subset2-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PatchesUmapSubset2.html')


# In[105]:


tsne_patches_subset2 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_patches_fit2 = tsne_patches_subset2.fit_transform(df_subset2)


data = [go.Scatter(x=tsne_patches_fit2[:,0], y=tsne_patches_fit2[:,1], mode='markers',
                   marker = dict(color=kmeans2.labels_, colorscale='rainbow', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f}'
                         for a,b,c,d,e,f in list(zip(Elevation, Slope, Horizontal_Distance_To_Hydrology , Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points))],
                   hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on subset2-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-Patches_subset2.html')
                   


# In[106]:



patches_ds_subset3=patches_ds[['Elevation', 'Slope', 'Vertical_Distance_To_Hydrology', 'Tree']]
df_subset3=pd.get_dummies(patches_ds_subset3)
feature_scaler = StandardScaler()
df_subset3 = feature_scaler.fit_transform(df_subset3)


# In[107]:




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


# In[108]:




# PCA on Subset3:
pca = PCA(n_components = 2)
pca.fit(df_subset3)
x_pca3 = pca.transform(df_subset3)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[109]:




# PCA Dimensionality Reduction using k-label on subset3
kmeans3 = KMeans(n_clusters = 3)
kmeans3.fit(df_subset3)

data = [go.Scatter(x=x_pca3[:,0], y=x_pca3[:,1], mode='markers',
                   marker = dict(color=kmeans3.labels_, colorscale='jet', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{g}' for a,b,c,d,e,f,g in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on subset3-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA_patches_subset3.html')
                   


# In[110]:



patches_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_patchessubset3__fit = patches_reducer.fit_transform(df_subset3)
umap_patchessubset3__fit.shape




data = [go.Scatter(x=umap_patchessubset3__fit[:,0], y=umap_patchessubset3__fit[:,1], mode='markers',
                    marker = dict(color=kmeans3.labels_, colorscale='jet', opacity=0.5),
                                text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f},Tree:{g}' for a,b,c,d,e,f,g in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on Full-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PatchesUmap_subset3.html')



# In[112]:



tsne_patches_subset3 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_patches_fit3 = tsne_patches_subset3.fit_transform(df_subset3)


data = [go.Scatter(x=tsne_patches_fit3[:,0], y=tsne_patches_fit3[:,1], mode='markers',
                   marker = dict(color=kmeans3.labels_, colorscale='jet', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},VerticalHydro:{d},HorizontalRoad:{e},HorizontalFirepts:{f}'
                         for a,b,c,d,e,f in list(zip(Elevation, Slope, Horizontal_Distance_To_Hydrology , Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Horizontal_Distance_To_Fire_Points))],
                   hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset3', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-PatchesSubset3.html')


# In[113]:




patches_ds_subset4=patches_ds[['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Tree']]
df_subset4=pd.get_dummies(patches_ds_subset4)
feature_scaler = StandardScaler()
df_subset4 = feature_scaler.fit_transform(df_subset4)


# In[114]:




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


# In[115]:




pca = PCA(n_components = 2)
pca.fit(df_subset4)
x_pca4 = pca.transform(df_subset4)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))


# In[116]:


# PCA Dimensionality Reduction using k-label on subset4
kmeans4 = KMeans(n_clusters = 2)
kmeans4.fit(df_subset4)

data = [go.Scatter(x=x_pca4[:,0], y=x_pca4[:,1], mode='markers',
                   marker = dict(color=kmeans4.labels_, colorscale='tropic', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},HorizontalRoad:{d},Tree:{e}' for a,b,c,d,e in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Tree))],
                   hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction using k-label on Full-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PCA_subset4.html')
                   
                   


# In[117]:


#Umap for subset 4 

patches_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2)
umap_patchessubset4__fit = patches_reducer.fit_transform(df_subset4)
umap_patchessubset4__fit.shape

data = [go.Scatter(x=umap_patchessubset4__fit[:,0], y=umap_patchessubset4__fit[:,1], mode='markers',
                   marker = dict(color=kmeans4.labels_, colorscale='tropic', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},HorizontalRoad:{d},Tree:{e}' for a,b,c,d,e in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Tree))],
                   hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction using k-label on subset4-patches Dataset', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='PatchesUmap_subset4.html')




# In[119]:


tsne_patches_subset4 = TSNE(n_components = 2, perplexity = 50 ,n_iter=3000)
tsne_patches_fit4 = tsne_patches_subset4.fit_transform(df_subset4)



data = [go.Scatter(x=tsne_patches_fit4[:,0], y=tsne_patches_fit4[:,1], mode='markers',
                   marker = dict(color=kmeans4.labels_, colorscale='tropic', opacity=0.5),
                   text=[f'Elevation:{a};Slope:{b};HorizontalHydro:{c},HorizontalRoad:{d},Tree:{e}' for a,b,c,d,e in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Tree))],
                   hoverinfo='text')]

layout = go.Layout(title = 'T-SNE Dimensionality Reduction using k-label on Subset4', width = 700, height = 700,template='plotly_dark',
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='T-SNE-PatchesSubset4.html')

