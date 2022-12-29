#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('beer_reviews.csv')
df = df.iloc[:,1:]
df.head()


# In[4]:


df.shape


# # Data Preprocessing

# In[5]:


#brewery_name, review_profilename, and beer_abv don't match the shape
df.info('include all')


# In[6]:


df[df['beer_abv'].isna()]


# In[7]:


med = df.groupby('beer_style')['beer_abv'].transform('median')
med


# In[8]:


#replace NaN values with the beer_style average
df['beer_abv'] = df['beer_abv'].fillna(med)
df.info()


# In[9]:


#keep data without NAs => only consists of .02% of data
df = df[df['brewery_name'].notna()]
df = df[df['review_profilename'].notna()]
df.info()


# In[10]:


#check rating values and counts
sorted(df['review_overall'].unique())


# In[11]:


#zero is not a valid rating, only 7 entries
df['review_overall'].value_counts()


# In[12]:


#instances where overall review = 0
df[df['review_overall']==0]


# In[13]:


#ignore these values as well => only 7 entries
df = df[df['review_overall']!=0]
df.info()


# # Data Exploration

# In[15]:


#distribution of numerical variables
fig = plt.figure(figsize=(25,12))
ax1 = fig.add_subplot(331)
ax1.set_title('Histogram - Review Time')
df["review_time"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(332)
ax1.set_title('Histogram - Review Overall')
df["review_overall"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(333)
ax1.set_title('Histogram - Review Aroma')
df["review_aroma"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(334)
ax1.set_title('Histogram - Review Appearance')
df["review_appearance"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(335)
ax1.set_title('Histogram - Review Palate')
df["review_palate"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(336)
ax1.set_title('Histogram - Review Taste')
df["review_taste"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(337)
ax1.set_title('Histogram - Beer Abv')
df["beer_abv"].plot(kind="hist", bins=15)


# In[16]:


#correlation matrix => numerical data
df_nums = df[['review_time','review_overall','review_aroma','review_appearance','review_palate','review_taste','beer_abv']]
corr = df_nums.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[17]:


df_cats = pd.DataFrame(df[['brewery_name','review_profilename','beer_style','beer_name']])
df_cats.head()


# In[18]:


cat_cts = pd.DataFrame([[len(df['brewery_name'].value_counts()),
                             len(df['review_profilename'].value_counts()),
                             len(df['beer_style'].value_counts()),
                        len(df['beer_name'].value_counts())]], columns=df_cats.columns)
cat_cts.head()


# In[19]:


#graph unique vals
cat_cts.plot(kind="bar",title="Unique Counts by Category", ylabel="Count", logy=True).legend(bbox_to_anchor=(1.5, 1))
print(cat_cts)


# In[20]:


beer_style_df = pd.DataFrame(df['beer_style'].value_counts())
beer_style_df.head()


# # Transformation - Review Counts Dataframe

# In[21]:


df.head()


# In[22]:


#average reviews by beer df
df_avgrev = df.groupby(['beer_beerid'], as_index=False)[['review_overall','review_aroma',
                                                                       'review_appearance','review_palate','review_taste',
                                                                       'beer_abv']].mean()


# In[23]:


df_avgrev.head()


# In[24]:


df_avgrev.columns = ['beer_beerid','avgrev_overall','avgrev_aroma',
                        'avgrev_appearance','avgrev_palate','avgrev_taste','beer_abv']
df_avgrev.head()


# In[25]:


#add pertinent categoricals (can't add user since we are aggregating user scores)
df_cat = df.groupby(['beer_beerid'],as_index = False)[['brewery_name','beer_style','review_time']].min()
df_cat.head()


# In[26]:


#merge categoricals with df_avgrev
df_transform = pd.merge(df_avgrev, df_cat, on='beer_beerid', how="inner")
df_transform.rename(columns={'review_time':'firstrev_time'}, inplace=True)
df_transform.head()


# In[27]:


#count number of reviews, merge to new df
df_ctrev = df.groupby(['beer_beerid'], as_index=False)['review_overall'].count()
df_ctrev.head()


# In[28]:


#merge tables, update col names
df_transform = pd.merge(df_transform,df_ctrev, on='beer_beerid',how="inner")
df_transform.rename(columns={'review_overall':'ctrev'}, inplace=True)
df_transform.head()


# In[30]:


#distribution of numerical variables
fig = plt.figure(figsize=(25,12))
ax1 = fig.add_subplot(331)
ax1.set_title('Histogram - First Review Time')
df_transform["firstrev_time"].plot(kind="hist", bins=15)

ax1 = fig.add_subplot(332)
ax1.set_title('Histogram - Average Review Overall')
df_transform["avgrev_overall"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(333)
ax1.set_title('Histogram - Average Review Aroma')
df_transform["avgrev_aroma"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(334)
ax1.set_title('Histogram - Average Review Appearance')
df_transform["avgrev_appearance"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(335)
ax1.set_title('Histogram - Average Review Palate')
df_transform["avgrev_palate"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(336)
ax1.set_title('Histogram - Average Review Taste')
df_transform["avgrev_taste"].plot(kind="hist", bins=30)

ax1 = fig.add_subplot(337)
ax1.set_title('Histogram - Beer Abv')
df_transform["beer_abv"].plot(kind="hist", bins=15)

ax1 = fig.add_subplot(338)
ax1.set_title('Histogram - Count of Reviews')
df_transform["ctrev"].plot(kind="hist", bins=15)


# In[31]:


#divide target from data
X = pd.DataFrame(np.array(df_transform)[:,1:10],columns=df_transform.columns[1:-1])
X.head()


# In[32]:


y = pd.DataFrame(df_transform['ctrev'])
y.head()


# # Unsupervised Learning - Cluster Analysis w/ PCA

# In[33]:


#normalize data matrix, turn categoricals into dummies
X.shape


# In[34]:


#get dummies for categoricals
X_dum = pd.get_dummies(X, columns=['beer_style'])
X_dum.head()


# In[35]:


#drop brewery_name => computationally too much
X_dum.drop('brewery_name',axis=1,inplace=True)


# In[36]:


X_dum.shape


# In[55]:


#min-max norm, scale to range of 0-1
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler().fit(X_dum)

X_norm = min_max_scaler.transform(X_dum)
X_norm = pd.DataFrame(X_norm, columns=X_dum.columns, index=X_dum.index)
X_norm.head()


# In[38]:


#dimensionality reduction => PCA
from sklearn import decomposition


# In[39]:


#create pca, fit to df_norm
pca = decomposition.PCA()
X_pca = pca.fit_transform(X_norm)


# In[40]:


np.set_printoptions(suppress=True, precision=2, linewidth=120)
print(pca.explained_variance_ratio_)


# In[41]:


#78 dimensions capture 95% of the data, reduces data by 23 dimensions
res = 0
for i in enumerate(pca.explained_variance_ratio_):
    res += i[1]
    print(i[0]+1,res)
        


# In[42]:


#scree plot
varPercentage = pca.explained_variance_ratio_*100

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(111), varPercentage[:], marker='^')
plt.title('Scree Plot - Components vs % Var Explained')
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')


# In[43]:


#create reduced dataframe with 95% explained variance
X_reduced = pd.DataFrame(np.array(X_pca[:,:78]))
X_reduced.head()


# In[44]:


#import clustering package
from sklearn.cluster import KMeans


# In[45]:


#create function to check best mean silhouette value for given k
def silhouette_calc(data,N):
    '''takes in data and a value for N and checks that number of k and returns max silhouette val w/ k'''
    from sklearn import metrics
    
    res = {}
    
    for i in range(2,N+1):
        kmeans = KMeans(n_clusters=i, max_iter=100) #kmeans for i# of clusters
        kmeans.fit(data) #fit data
        clusters = kmeans.predict(data) #assign clusters
        silhouettes = metrics.silhouette_samples(data, clusters) #calc silhouette scores
        res[i] = silhouettes.mean() #stores silhouette score
    print(res)
    return max(res, key=res.get), max(res.values())


# In[46]:


nclust = silhouette_calc(X_reduced, 100)
nclust


# In[48]:


#From silhouette results, we get a total of 66 clusters for the optimal value based on the silhouette
#Set K=20, probably won't be a meaningful col in our regression
kmeans = KMeans(n_clusters=66, max_iter=500)
kmeans.fit(X_reduced)
clusters = kmeans.predict(X_reduced)


# In[49]:


def cluster_sizes(clusters):
    size = {}
    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]
    
    for c in cluster_labels:
        size[c] = len(X_reduced[clusters == c])
    return size


# In[50]:


size = cluster_sizes(clusters)

for c in size.keys():
    print("Size of Cluster", c,"- ",size[c])


# In[51]:


clusters


# In[56]:


#add back to X_norm
X_norm['Cluster'] = clusters


# In[57]:


#Make it a dummy variable
X_norm = pd.get_dummies(X_norm, columns=['Cluster'])
X_norm.head(20)


# In[58]:


#transform y to log2
y_trans = np.log2(y)
y_trans.head()


# # Linear Regression

# In[59]:


#train test split size 80/20, random_state=33
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm,y_trans, test_size = .2, random_state=33)


# In[60]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[61]:


#import linear regression module, fit training data
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

linreg.fit(X_train, y_train)


# In[65]:


#RMSE calculation on training data
from sklearn.metrics import mean_squared_error
p_train = linreg.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(p_train, y_train))
print("RMSE on Train Data: ", rmse_train)


# In[66]:


#Correlation between predicted and actual values of target attr
plt.plot(p_train, y_train, 'ro', markersize=4)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Target Train vs Predicted Train")


# In[68]:


#RMSE calculation on test data - shockingly high
p_test = linreg.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(p_test, y_test))
print("RMSE on Test Data: ",rmse_test)


# In[73]:


X_names = X_train.columns
X_names


# In[74]:


#find the best features based on RMSE values
def feature_selection(X, y, model, K=5):
    '''takes in the training/target variable and returns optimal percentage of the most informative features to use'''
    #import necessary packages
    from sklearn import feature_selection
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_score
    
    err = {}
    attrs = {}
    #cross validation / feature selection / mean squared error
    for i in range(1,101):
        #convert data to np arrays
        X = np.array(X)
        y = np.array(y)
        
        #SelectPercentile, fit model, perform cv RMSE
        fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=i)
        X_train_fs = fs.fit_transform(X, y)
        scores = abs(cross_val_score(model, X_train_fs, y, scoring = 'neg_root_mean_squared_error', cv=K))
        #print(scores)
        #store percentile and avg cross val score
        err[i] = scores.mean()
        
        #store optimal features
        X = pd.DataFrame(X, columns=X_names)
        attrs[i] = X.columns[fs.get_support()].values
        #calculate RMSE w/ percentile
    
    
    plt.plot(err.keys(), err.values())
    plt.title("Root Mean Squared Error with Percent of Features Selected")
    plt.xlabel("Percentage of Features Selected")
    plt.ylabel("Cross Validation Root Mean Squared Error")
    
    return min(err, key=err.get), attrs


# In[75]:


#Return Optimal Index and Plot of RMSE, keep 100% of features selected for lowest RMSE
ind,attrs =feature_selection(X_train, y_train, linreg)
ind


# In[77]:


#Create subset train/test data w/ optimal features, 18% fs
X_train_fs = X_train[attrs[ind]]
X_test_fs = X_test[attrs[ind]]


# In[210]:


attrs[ind]


# In[80]:


#identify features with most impact => reviews, beer_abv, first review time, beer styles, then some clusters
X_train_fs.head()


# In[82]:


#Use 18% attrs to retrain new fs model, calc the RMSE test train => Wow what an improvement
from sklearn.metrics import mean_absolute_error, mean_squared_error
linreg_fs = LinearRegression()
linreg_fs.fit(X_train_fs, y_train)
p_train = linreg_fs.predict(X_train_fs)
p_test = linreg_fs.predict(X_test_fs)
rmse_train = mean_squared_error(p_train, y_train)
rmse_test = mean_squared_error(p_test, y_test)
print(f"RMSE train: {rmse_train}")
print(f"RMSE test: {rmse_test}")


# # Summary
# Removing 82% of the data clearly reduced the noise that took place in the original model, which increased our RMSE train, but more importantly, significantly decreased our RMSE test value. You can see from our features selected graph that the more features seem to make our RMSE value more volatile, and with cross validation, that tells us a lower # of features is optimal.

# # Ridge & Lasso Regression

# In[83]:


#find optimal alpha/weight for regularized regression techniques
def ridge_lasso(X, y, a, model, K=5):
    '''performs Ridge or Lasso regression to find best alpha value and plots RMSE'''
    
    #import pertinent packages
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    
    #set range for alpha hyperparameter
    if model == "L":
        
        alpha = np.linspace(.0001, a, 100)
    elif model == "R":
        alpha = np.linspace(.001, a, 100)
        
    
    cv = {}
    train = {}
    
    #Train model, cv, store error terms
    for i in alpha:
        
        #determine type (Ridge/Lasso) and fit model w/ i alpha level
        if model == "L":
            m = Lasso(alpha=i)
            m.fit(X, y)
            p = m.predict(X)
            rmse_train = np.sqrt(mean_squared_error(p, y))
        elif model == "R":
            m = Ridge(alpha=i)
            m.fit(X, y)
            p = m.predict(X)
            rmse_train = np.sqrt(mean_squared_error(p, y))
        #perform cv, store rmse scores
        scores = abs(cross_val_score(m, X, y, scoring = 'neg_root_mean_squared_error', cv=K))
        
        cv[i] = scores.mean()
        train[i] = rmse_train
       
    #plot error values
    plt.plot(cv.keys(), cv.values(), label="RMSE-XVal")
    plt.plot(train.keys(), train.values(), label="RMSE-Train")
    if model == "L":
        plt.title("Lasso Regression RMSE values for Given Alpha")
    else:
        plt.title("Ridge Regression RMSE values for Given Alpha")
    plt.legend(('RMSE-XVal', 'RMSE-Train'))
    plt.xlabel("Alpha")
    plt.ylabel("RMSE")
    
    return min(cv, key=cv.get), min(cv.values())


# In[84]:


#Lasso Regression
l = ridge_lasso(X_train, y_train, .01 , "L")
l


# In[85]:


#Ridge Regression
r = ridge_lasso(X_train, y_train, 10, "R")
r


# In[86]:


#Lasso Regression - optimal alpha
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=l[0])
lasso.fit(X_train, y_train)
p_test = lasso.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(p_test, y_test))
print(f"Lasso RMSE test: {rmse_test}")


# In[87]:


#Ridge Regression - optimal alpha
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=r[0])
ridge.fit(X_train, y_train)
p_test = ridge.predict(X_test)

RMSE_test = np.sqrt(mean_squared_error(p_test, y_test))
print(f"Ridge RMSE test: {RMSE_test}")


# In[88]:


#predict X_test, graph results
pred_lasso = lasso.predict(X_test)
pred_ridge = ridge.predict(X_test)


# In[90]:


import seaborn as sns
sns.distplot(y_test['ctrev']-pred_lasso)


# In[91]:


sns.distplot(y_test['ctrev']-pred_ridge[:,0])


# In[92]:


lasso.coef_


# In[93]:


#lasso regression feature importance
X_names = X_train.columns.values
plt.figure(figsize=(8, 25), dpi=80)
plt.barh(range(len(X_names)), lasso.coef_, align='center')
plt.yticks(np.arange(len(X_names)), X_names)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.ylim(-1, len(X_names))
plt.title("Values per Regression Coefficient")


# # KNN

# In[94]:


#discretize the target variable y
y_trans.head()


# In[119]:


#divide data in 2 bins, low/high ~90/10
bins = pd.DataFrame(pd.qcut(y['ctrev'], q=[0,.91,1], labels=["low","high"]))


# In[120]:


bins.value_counts()


# In[121]:


y_check = y
y_check['label'] = bins['ctrev']


# In[122]:


#highest value for "low"
y_check[y_check['label']=='low'].max()


# In[123]:


#Train/Test split again w/ classfier
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_norm,bins, test_size = .2, random_state=44)


# In[124]:


#import KNeighbors Classifier
from sklearn import neighbors


# In[125]:


def knnClassifier(K, w):
    '''takes in a K value and whether there should be distance weighting an returns the accuracies'''
    from sklearn import neighbors
    acc = []
    if w == 0:
        w = 'uniform'
    elif w == 1:
        w = 'distance'
    for i in range(1, K+1,5):
        knnclf = neighbors.KNeighborsClassifier(i, weights=w)
        knnclf.fit(X_train2, y_train2['ctrev'])
        acc.append(knnclf.score(X_test2, y_test2))
    return acc


# In[126]:


#uniform/distance knn acc
uniknn = knnClassifier(50,0)
distknn = knnClassifier(50,1)


# In[127]:


uniknn


# In[128]:


distknn


# In[129]:


#weighted vs uniform graph
K = list(range(1,51,5))
plt.figure(figsize=(10,5))
plt.plot(K, distknn, label = 'Weighted')
plt.plot(K, uniknn, label = 'Uniform')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Weighted vs Uniform Accuracy Across K Neighbors')
plt.xticks(range(5,51,5))
plt.legend(loc='center left',bbox_to_anchor=(1,.5))


# In[130]:


#create best knn model k=21, weights = distance
knnclf = neighbors.KNeighborsClassifier(21, weights='distance')
knnclf.fit(X_train2, y_train2['ctrev'])


# In[131]:


#acc score
knnclf.score(X_test2, y_test2)


# In[132]:


knnpreds_test = knnclf.predict(X_test2)
knnpreds_test


# In[133]:


#how good at classifying smaller category "high?"
from sklearn.metrics import confusion_matrix
knncm = confusion_matrix(y_test2, knnpreds_test)
print(knncm)


# In[134]:


y_test2.value_counts()


# In[135]:


#Classification Report, recall .28 for high, .98 for low, not good at classifying the high class
from sklearn.metrics import classification_report
print(classification_report(y_test2, knnpreds_test))


# In[136]:


from sklearn.metrics import recall_score
print(recall_score(y_test2,knnpreds_test, average="macro"))


# In[137]:


#run model for optimal recall score instead of accuracy
def knnClassifier2(K, w):
    '''takes in a K value and whether there should be distance weighting an returns the accuracies'''
    from sklearn import neighbors
    from sklearn.metrics import recall_score
    rec = []
    if w == 0:
        w = 'uniform'
    elif w == 1:
        w = 'distance'
    for i in range(1, K+1,5):
        knnclf = neighbors.KNeighborsClassifier(i, weights=w)
        knnclf.fit(X_train2, y_train2['ctrev'])
        knnpreds_test = knnclf.predict(X_test2)
        rec.append(recall_score(y_test2,knnpreds_test, average="macro"))
    return rec


# In[138]:


#uniform/distance knn acc
uniknn = knnClassifier2(50,0)
distknn = knnClassifier2(50,1)


# In[139]:


uniknn


# In[140]:


distknn


# In[141]:


#weighted vs uniform graph
K = list(range(1,51,5))
plt.figure(figsize=(10,5))
plt.plot(K, distknn, label = 'Weighted')
plt.plot(K, uniknn, label = 'Uniform')
plt.xlabel('K')
plt.ylabel('Recall')
plt.title('Weighted vs Uniform Recall Across K Neighbors')
plt.xticks(range(5,51,5))
plt.legend(loc='center left',bbox_to_anchor=(1,.5))


# In[142]:


#create model for K=6, uniform
knnclf = neighbors.KNeighborsClassifier(6, weights='uniform')
knnclf.fit(X_train2, y_train2['ctrev'])


# In[143]:


#acc score, still pretty high but with better recall
knnclf.score(X_test2, y_test2)


# In[144]:


#predict test labels
knnpreds_test = knnclf.predict(X_test2)


# In[145]:


#confusion matrix
knncm = confusion_matrix(y_test2, knnpreds_test)
print(knncm)


# In[146]:


#classification report
print(classification_report(y_test2, knnpreds_test))


# # Summary
# Adjusting from K=26, distance to K=6, uniform I believe is a better generalization for this model. This is because the recall score got 19% better for the high, with the overall accruacy only getting ~2% worse. While the recall is still low, this is the best we can do with the dataset when we try and predict the smaller category of the target labels.

# # Decision Trees

# In[147]:


#utilize grid search to find optimal parameters for our decision tree
from sklearn import tree
from sklearn.model_selection import GridSearchCV


# In[148]:


#set parameters, create gs/treeclf
parameters = {
    'criterion': ['gini','entropy'],
    'max_depth': np.linspace(1, 30, 15, dtype=int),
    'min_samples_split': np.linspace(2,20,10, dtype=int)
}
treeclf = tree.DecisionTreeClassifier()
gs = GridSearchCV(treeclf, parameters, verbose=1, cv=5)


# In[149]:


#run gs on tree
get_ipython().run_line_magic('time', '_ = gs.fit(X_train2, y_train2)')

params, score = gs.best_params_, gs.best_score_
params, score


# In[150]:


#best model: criterion=>entropy, max_depth=>13, min_samples_split=>18
treeclf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=13, min_samples_split=18)
treeclf.fit(X_train2, y_train2)


# In[151]:


#Train Score
treeclf.score(X_train2, y_train2)


# In[152]:


#Test Score
treeclf.score(X_test2, y_test2)


# In[153]:


treepreds_test = treeclf.predict(X_test2)


# In[154]:


#confusion matrix
treecm = confusion_matrix(y_test2, treepreds_test)
print(treecm)


# In[155]:


#classification report
print(classification_report(y_test2, treepreds_test))


# In[156]:


import graphviz


# In[157]:


from sklearn.tree import export_graphviz


# In[158]:


#accuracy optimized decision tree
export_graphviz(treeclf,out_file='tree.dot', feature_names=X_train2.columns, class_names=["high","low"],
                rotate=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[159]:


#create a gs with optimizing recall
parameters = {
    'criterion': ['gini','entropy'],
    'max_depth': np.linspace(1, 30, 15, dtype=int),
    'min_samples_split': np.linspace(2,20,10, dtype=int)
}
treeclf2 = tree.DecisionTreeClassifier()
gs2 = GridSearchCV(treeclf2, parameters, verbose=1, cv=5, scoring='recall_macro')


# In[160]:


#run new gs on tree
get_ipython().run_line_magic('time', '_ = gs2.fit(X_train2, y_train2)')

#params, score = gs.best_params_, gs.best_score_
#params, score


# In[161]:


gs2.best_params_, gs2.best_score_


# In[162]:


#fit new model
treeclf2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_split=10)
treeclf2.fit(X_train2, y_train2)


# In[163]:


#Train Score
treeclf2.score(X_train2, y_train2)


# In[164]:


#Test Score, overfitting some
treeclf2.score(X_test2, y_test2)


# In[165]:


treepreds2_test = treeclf2.predict(X_test2)


# In[166]:


#confusion matrix
treecm2 = confusion_matrix(y_test2, treepreds2_test)
print(treecm2)


# In[167]:


#classification report
print(classification_report(y_test2, treepreds2_test))


# In[168]:


import graphviz


# In[169]:


from sklearn.tree import export_graphviz


# In[170]:


#full decision tree, max_depth 30
export_graphviz(treeclf2,out_file='tree.dot', feature_names=X_train2.columns, class_names=["high","low"],
                rotate=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# # Summary
# As seen from the decision tree graph, the first time at which a beer is reviewed matters the most in our model to predict how much a beer has been reviewed. Other important features revolve around the average review scores for each beer, sprinkled in with some of the cluster/dummy data as well. Since this tree has a max depth of 30, it is a little more difficult to interpret, but in terms of accuracy, it performs better than our KNN model. We get a macro recall score of 75%, compared to the 70% of our best KNN model. However, it does appear that we are overfitting due to the disparity between the train and test accuracy scores. Overall, I think I still would take the latter decision tree model just due to the fact that it is actually able to classify more than have of the high reviewed beer correctly.

# # Ensemble Method - Random Forest

# In[171]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, random_state=55)
rf = rf.fit(X_train2, y_train2['ctrev'])


# In[172]:


#basic rf to see results
rfpreds_test = rf.predict(X_test2)


# In[173]:


#classification report, worse than both of our prior models for KNN/Decision Trees
print(classification_report(y_test2, rfpreds_test))


# In[174]:


print(rf.get_params())


# In[175]:


#use calc_params function to test different parameter optimizations
from sklearn.model_selection import KFold

def calc_params(X, y, clf, param_values, param_name, K):
    
    # Convert input to Numpy arrays
    X = np.array(X)
    y = np.array(y)

    # initialize training and testing score arrays with zeros
    train_scores = np.zeros(len(param_values))
    test_scores = np.zeros(len(param_values))
    
    # iterate over the different parameter values
    for i, param_value in enumerate(param_values):

        # set classifier parameters
        clf.set_params(**{param_name:param_value})
        
        # initialize the K scores obtained for each fold
        k_train_scores = np.zeros(K)
        k_test_scores = np.zeros(K)
        
        # create KFold cross validation
        cv = KFold(n_splits=K, shuffle=True, random_state=0)
        
        # iterate over the K folds
        j = 0
        for train, test in cv.split(X):
            # fit the classifier in the corresponding fold
            # and obtain the corresponding accuracy scores on train and test sets
            clf.fit(X[train], y[train])
            k_train_scores[j] = clf.score(X[train], y[train])
            k_test_scores[j] = clf.score(X[test], y[test])
            j += 1
            
        # store the mean of the K fold scores
        train_scores[i] = np.mean(k_train_scores)
        test_scores[i] = np.mean(k_test_scores)
        print(param_name, '=', param_value, "Train =", train_scores[i], "Test =", test_scores[i])
       
    # plot the training and testing scores in a log scale
    plt.plot(param_values, train_scores, label='Train', alpha=0.4, lw=2, c='b')
    plt.plot(param_values, test_scores, label='X-Val', alpha=0.4, lw=2, c='g')
    plt.legend(loc=7)
    plt.xlabel(param_name + " values")
    plt.ylabel("Mean cross validation accuracy")

    # return the training and testing scores on each parameter value
    return train_scores, test_scores


# In[176]:


ms = range(1,20)


# In[177]:


#test values for min_samples_leaf: 8 works best
rf = RandomForestClassifier(n_estimators=10, random_state=55)
train_scores, test_scores = calc_params(X_train2, y_train2['ctrev'], rf, ms, 'min_samples_leaf',5)


# In[178]:


m_depth = range(1,20)


# In[179]:


#max_depth has a good value at 12 so we are more accurate but not overfitting the training data
rf = RandomForestClassifier(n_estimators=10, random_state=55)
train_scores, test_scores = calc_params(X_train2, y_train2['ctrev'], rf, m_depth, 'max_depth',5)


# In[180]:


nest = range(5,101,5)


# In[181]:


#number of estimators, 20 looks pretty good!
rf = RandomForestClassifier(n_estimators=10, random_state=55)
train_scores, test_scores = calc_params(X_train2, y_train2['ctrev'], rf, nest, 'n_estimators',5)


# In[182]:


#final rf model with selected parameters 20, 8, 12
rf = RandomForestClassifier(n_estimators=20, min_samples_leaf=8, max_depth=12, random_state=55)
rf.fit(X_train2, y_train2['ctrev'])


# In[183]:


rfpreds_test = rf.predict(X_test2)


# In[184]:


#classification report, good at predicting low, but terrible recall score for high
print(classification_report(y_test2, rfpreds_test))


# In[185]:


#let's try grid search now

rf2 = RandomForestClassifier()
parameters = {
    'min_samples_leaf': range(1,22,5),
    'max_depth': range(1,22,5),
    'n_estimators': range(5,51,5)
}

gs = GridSearchCV(rf2, parameters, verbose=1, cv=5, scoring='recall_macro')


# In[186]:


#run gs on rf
get_ipython().run_line_magic('time', "_ = gs.fit(X_train2, y_train2['ctrev'])")

#params, score = gs.best_params_, gs.best_score_
#params, score


# In[187]:


#retreive best parameteres for optimal recall
gs.best_params_, gs.best_score_


# In[188]:


#train model for optimal recall
rf2 = RandomForestClassifier(n_estimators=5, min_samples_leaf=1, max_depth=21, random_state=55)
rf2.fit(X_train2, y_train2['ctrev'])


# In[191]:


#check train and test scores
from sklearn import metrics
from sklearn.metrics import accuracy_score
rf2_predtrain = accuracy_score(y_train2, rf2.predict(X_train2))
rf2_predtest = accuracy_score(y_test2, rf2.predict(X_test2))
rf2_predtrain, rf2_predtest


# In[192]:


#classification report, good at predicting low, but terrible recall score for high
rf2preds_test = rf2.predict(X_test2)
print(classification_report(y_test2, rf2preds_test))


# In[193]:


#confusion matrix
rfcm2 = confusion_matrix(y_test2, rf2preds_test)
print(rfcm2)


# In[194]:


#most effective attrs
def plot_feature_importances(model, n_features, feature_names):
    plt.figure(figsize=(8, 25), dpi=80)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

features = X_train2.columns
plot_feature_importances(rf2, len(features), features)


# ## Summary
# From our random forest model, we get a pretty good accuracy score of 85%, but see that our recall scores for high and medium # of review's is significantly worse. Therefore, this would probably not be the model we would want to use, since it classifies the smaller labels a lot worse than the decision tree and Knn. However, we do get to see which features have the most importance in the model, which can be useful. Firstrev_time is again considered the most useful like in the decision tree, and then the ratings, beer abv, and a little sprinkle of the clusters and dummy variables.

# # Ada Boost

# In[195]:


from sklearn.ensemble import AdaBoostClassifier


# In[196]:


#grid search AdaBoost
ab = AdaBoostClassifier()


# In[197]:


parameters = {
    'learning_rate': [.01, .05, .1, .3, .5, 1.0, 1.3, 1.5, 1.8, 2.0],
    'n_estimators': range(5,51,5)
}


# In[198]:


gs = GridSearchCV(ab, parameters, verbose=1, cv=5)


# In[199]:


#run gs on ada, accuracy optimization
get_ipython().run_line_magic('time', "_ = gs.fit(X_train2, y_train2['ctrev'])")

params, score = gs.best_params_, gs.best_score_
params, score


# In[200]:


#train adaboost model with grid search optimal parameters n_estimators=50, learning_rate = 1.3
ab = AdaBoostClassifier(n_estimators = 50, learning_rate = 1.3)
ab.fit(X_train2, y_train2['ctrev'])


# In[201]:


abtest_pred = ab.predict(X_test2)


# In[202]:


#classification report, better than random forest, still worse than tree
print(classification_report(y_test2, abtest_pred))


# In[203]:


from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score


# In[204]:


#use recall_macro as the optimal scoring method in gs2
ab2 = AdaBoostClassifier()
gs2 = GridSearchCV(ab2, parameters,scoring='recall_macro', verbose=1, cv=5)


# In[205]:


#run new model
get_ipython().run_line_magic('time', "_ = gs2.fit(X_train2, y_train2['ctrev'])")


# In[206]:


gs2.best_params_, gs2.best_score_


# In[207]:


#ada model, recall optimized n_estimators=50, learning_rate = 1.5
ab2 = AdaBoostClassifier(n_estimators=50, learning_rate = 1.5)
ab2.fit(X_train2, y_train2['ctrev'])


# In[208]:


ab_test_pred2 = ab2.predict(X_test2)


# In[209]:


#classification report, better than random forest, still a little worse than tree
print(classification_report(y_test2, ab_test_pred2))


# # Summary
# Overall, our Ada boost performed a lot better than our random forest model, in terms of accuracy and recall. We got our highest recorded recall score for high (.24) but still 20% off our best for medium. This model could be considered if we were particularly interested in classifying the high reviewed beers better, but overall our decision tree model is still better.
