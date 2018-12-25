from pprint import pprint

'''
该程序可以对语料进行简单处理，使用TFIDF特征及逻辑回归算法进行三个任务的预测
将输出文件temp.csv，将该文件提交到评测系统中，可得到baseline的结果
该程序运行需要4个文件
train/train_status.txt
train/train_labels.txt
valid/valid_status.txt
valid/valid_nolabel.txt
'''
provsString = '''
    东北,辽宁,吉林,黑龙江
    华北,河北,山西,内蒙古,北京,天津
    华东,山东,江苏,安徽,浙江,台湾,福建,江西,上海
    华中,河南,湖北,湖南
    华南,广东,广西,海南,香港,澳门
    西南,云南,重庆,贵州,四川,西藏
    西北,新疆,陕西,宁夏,青海,甘肃
    境外,其他,海外
    None,None
    '''
provs = {}  # 将地域按照[省份: 地域】区分出来
for line in provsString.split('\n'):
    items = line.split(',')
    for item in items[1:]:
        provs[item] = items[0].strip()


def map_age(x):
    x = int(x)
    if x >= 1990:
        return '1990+'
    elif x < 1980:
        return '1979-'
    else:
        return '1980-1989'


def map_location(x):
    x = x.split(' ')[0]
    return provs[x]


# # 从文件中读取语料并整理，将同一个人发的微博连在一起

# In[2]:

import pandas as pd
# 读取训练集
train_file = map(lambda x: x.split(',', maxsplit=5),
                 open('train/train_status.txt', encoding='utf8'))
valid_labels = set(map(lambda x: x.strip(), open('valid/valid_nolabel.txt')))
valid_file = filter(lambda x: x[0] in valid_labels,
                    map(lambda x: x.split(',', maxsplit=5),
                        open('valid/valid_status.txt', encoding='utf8')))
df = pd.DataFrame(data=list(train_file)+list(valid_file),
                  columns='id,review,forward,source,time,content'.split(','),)

# In[3]:


# 读取训练集标注
labels = pd.read_csv('train/train_labels.txt', sep='\|\|', encoding='utf8', engine='python',
                     names='id,gender,age,location'.split(','))

labels.age = labels.age.apply(map_age)
labels.location = labels.location.apply(map_location)


# In[4]:


# 按id进行合并微博内容
X = pd.DataFrame(df.groupby(by='id', sort=False).content.sum()).reset_index()
X.id = X.id.astype(int)

data = pd.merge(X, labels, on='id', how='left')


# # 从文本中抽取TFIDF特征

# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000)

f_tfidf = tfidf.fit_transform(data.content)


# # 训练逻辑回归分类器，并进行预测

# In[6]:


from sklearn.linear_model.logistic import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.neural_network import MLPClassifier as mlp

valid_data = data[3200:].reset_index()

clf_gender = mlp(hidden_layer_sizes=(2,1),
                 verbose=0,
                 activation='tanh')
clf_gender.fit(f_tfidf[:3200], data.gender[:3200])
valid_data.gender = clf_gender.predict(f_tfidf[3200:])

# clf_age_pre = LR()
# clf_age_pre.fit(f_tfidf[:3200], data.age[:3200])

clf_age = GBDT(n_estimators=300,
               verbose=1)
clf_age.fit(f_tfidf[:3200], data.age[:3200])
valid_data.age = clf_age.predict(f_tfidf[3200:])

clf_location = GBDT(n_estimators=300,
                    verbose=1)
clf_location.fit(f_tfidf[:3200], data.location[:3200])
valid_data.location = clf_location.predict(f_tfidf[3200:])


# # 输出到temp.csv

# In[7]:


valid_data.loc[:, ['id', 'age', 'gender', 'location']].to_csv(
    'result/gender_mlp_2_1_age_gbdt_n_est_300_loc_gbdt_n_est_300.csv',
    index=False)
