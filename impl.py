import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import  webbrowser
import folium
from folium import plugins
import json
import requests
from folium.plugins import HeatMap
import warnings
import jieba
from wordcloud import WordCloud
from pylab import mpl



warnings.filterwarnings('ignore')#忽略警告消息

mpl.rcParams['font.sans-serif'] = ['SimHei']#防止title显示方框


pd.set_option('display.max_columns', None)#输出全部信息，避免省略号
df = pd.read_csv('./listings.csv',index_col=0)        #加载数据，使用第一列为默认索引


print("--------------------数据信息·开始--------------------")
print(df.describe())
print("--------------------数据信息·结束--------------------")


df.number_of_reviews.fillna(df.number_of_reviews.mean(),inplace=True)#填充空值为均值


print("--------------------特征数据·开始-------------------")
def get_con(df):
    subsets=['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']
    data={}
    for i in subsets:
        data.setdefault(i,[])
        data[i].append(df[i].skew())#偏度，偏度正向左偏
        data[i].append(df[i].kurt())#峰度，峰度大于三，峰的形状比正态分布峰要陡峭
        data[i].append(df[i].mean())#均值
        data[i].append(df[i].std())#标准差
        data[i].append(df[i].std()/df[i].mean())#变异系数，离散程度，越大越离散
        data[i].append(df[i].max()-df[i].min())#极差
        data[i].append(df[i].quantile(0.75)-df[i].quantile(0.25))#四分位距，所有数值由小到大排列并分成四等份，处于三个分割点位置的数值就是四分位数。
        data[i].append(df[i].median())#中位数
    data_df=pd.DataFrame(data,index=['偏度','峰度','均值','标准差','变异系数','极差','四分位距','中位数'],columns=subsets)
    return data_df.T
df2=get_con(df)
print(df2)
print("--------------------特征数据·结束-------------------")


print("--------------------绘制房价分布图·开始-------------------")
h_price=df[(df.price<2000)&(df.price>0)]#做图时去掉价格大于2000和等于0的房子
h_price=np.array(h_price['price'])#nparray化
X=np.linspace(0,2500,15)#0~2500划分15个区间
Y=np.linspace(0,12000,12)
fig1=plt.figure()#新建画板
ax1=plt.axes()#新建轴集
ax1.set(xticks=X,yticks=Y,title='房价分布图',xlabel='House price',ylabel='Numbers')#设定x轴y轴映射，title
ax1.hist(h_price)#绘制柱状图
plt.show()
print("--------------------绘制房价分布图·结束-------------------")



print("--------------------绘制评价分布图·开始-------------------")
h_review=df[(df.number_of_reviews>-1)]
h_review=np.array(h_review['number_of_reviews'])
print("评论数为零的有："+str(sum(h_review==0)))
h_review=df[(df.number_of_reviews>0)&(df.number_of_reviews<200)]
h_review=np.array(h_review['number_of_reviews'])
X=np.linspace(0,200,10)
Y=np.linspace(0,12000,12)
fig2=plt.figure()
ax2=plt.axes()
ax2.set(xticks=X,yticks=Y,title='评价分布图',xlabel='House reviews',ylabel='Numbers')
ax2.hist(h_review)
plt.show()
print("--------------------绘制评价分布图·开始-------------------")


print("--------------------绘制地区分布饼图·开始-------------------")
lis_dis=df.neighbourhood.value_counts()#地区-数量
labels=lis_dis.index
sns.set(font_scale=1.5)
plt.rcParams['font.sans-serif']='SimHei'#设置正常显示字符
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(9,20))
plt.title('各地区分布占比',fontdict={'fontsize':18})
plt.pie(lis_dis,labels=labels,autopct='%.2f%%',explode=[0.1 if i in ['东城区','朝阳区 / Chaoyang','海淀区'] else 0 for i in labels],startangle=90,counterclock=False,textprops={'fontsize':12,'color':'black'},colors=sns.color_palette('RdBu',n_colors=18)) 
#饼图
plt.show()
print("--------------------绘制地区分布饼图·结束-------------------")



print("--------------------绘制房型分布饼图·开始-------------------")
lis_dis=df.room_type.value_counts()#地区-数量
labels=lis_dis.index
sns.set(font_scale=1.5)
plt.rcParams['font.sans-serif']='SimHei'#设置正常显示字符
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(9,20))
plt.title('各房型分布占比',fontdict={'fontsize':18})
plt.pie(lis_dis,labels=labels,autopct='%.2f%%',startangle=90,counterclock=False,textprops={'fontsize':12,'color':'black'},colors=sns.color_palette('RdBu',n_colors=18)) 
#饼图
plt.show()
print("--------------------绘制房型分布饼图·结束-------------------")



print("--------------------绘制地区-房型价格图·开始-------------------")

pd.options.display.precision=2#精度为2
plt.figure(figsize=(10,10))
feature_df=pd.pivot_table(df,index='neighbourhood',values=['price'],columns='room_type',aggfunc=np.mean)#透视表
sns.heatmap(feature_df.price,cmap=sns.color_palette('RdBu_r',n_colors=32),annot=True,fmt='.0f')#热力图，调色盘红蓝对比32格，在格子上显示数字
plt.show()
print("--------------------绘制地区-房型价格图·结束-------------------")



feature=['names','minimum_nights','price','room_type']
label=['number_of_reviews']

def get_review_tb(df,num):#获取各区评论数量top/bottom的数量，根据参数，num为正表示top，为负数表示bottom
    result=[]
    groups=df.groupby('neighbourhood')
    for x,group in groups:
        if num>0:
            result.append(group.sort_values(by='number_of_reviews',ascending=False)[:num])
        if num<0:
            result.append(group.sort_values(by='number_of_reviews',ascending=False)[num:])
    result=pd.concat(result)
    return result


reviews_top50=get_review_tb(df,50)#获取各区评论数top50的listing信息
reviews_bottom50=get_review_tb(df,-50)#获取各区评论数bottom50的listing信息


def get_words(df):#利用jieba对房屋名称进行拆词分析，获取高频词汇
    s=[]
    wordsdic={}
    with open('./stopwords.txt',encoding='utf8') as f:#根据停用词过滤词汇
        result=f.read().split()
    for i in df:
        words=jieba.lcut(i)
        word=[x for x in words if x not in result]
        s.extend(word)
    for word in s:
        wordsdic.setdefault(word,0)
        wordsdic[word]+=1
    return wordsdic
top_words=get_words(reviews_top50.name.astype('str'))
bottom_words=get_words(reviews_bottom50.name.astype('str'))
top_words_df=pd.Series(top_words).sort_values(ascending=False)[1:21]#转换成Series格式，方面绘图可视化
bottom_words_df=pd.Series(bottom_words).sort_values(ascending=False)[1:21]#从1开始是为了过滤空值




plt.figure(figsize=(10,5))
plt.title('评论较多listing，name中数量较多的词汇分布')
top_words_df.plot(kind='bar',ylim=[0,200])

plt.figure(figsize=(10,5))
plt.title('评论较少listing，name中数量较少的词汇分布')
bottom_words_df.plot(kind='bar',ylim=[0,200])
plt.show()






#############################################


url = 'anfran.json'#北京行政区划分http://datav.aliyun.com/tools/atlas/#&lat=31.769817845138945&lng=104.29901249999999&zoom=3

bj_geo = f'{url}'
'''
对原始json数据做一个处理，把['features'][index]['properties']['adcode']挪入上层['features'][index]['id']
f = open("san.json", encoding='utf-8')
setting = json.load(f)
for index in np.arange(len(setting['features'])):
    setting['features'][index]['id'] = setting['features'][index]['properties']['adcode']
with open("anfran.json",'w') as file_obj:
    json.dump(setting,file_obj)
'''

latitude,longitude = 116.40,39.90#经纬度


incidents = folium.map.FeatureGroup()
for lat, lng, in zip(df.latitude, df.longitude):#元组

    incidents.add_child(
        folium.CircleMarker(
            [lat, lng],
            radius=1, # 半径
            color='red',
            fill=True,
            fill_color='red',

        )
    )

bj_map1 = folium.Map(location=[longitude, latitude], zoom_start=12)#建立地图，按经纬度定位，zoostart缩放越大视角越低
bj_map1.add_child(incidents)#将点锚在地图上
bj_map1.save('位置总览1.html')
webbrowser.open('位置总览1.html')

####################################################

bj_map2 = folium.Map(location=[longitude, latitude], zoom_start=9)
folium.GeoJson(#划分行政区
    bj_geo,
    style_function=lambda feature: {
        'fillColor': '#ffff00',
        'color': 'black',
        'weight': 1,

    }
).add_to(bj_map2)


incidents = plugins.MarkerCluster().add_to(bj_map2)#点数量聚簇

for lat, lng, label, in zip(df.latitude, df.longitude, df.neighbourhood):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# add incidents to map
bj_map2.add_child(incidents)
bj_map2.save('位置分布2.html')
webbrowser.open('位置分布2.html')

####################################################
bj_map3 = folium.Map(location=[longitude, latitude], zoom_start=9)

disdata = pd.DataFrame(df['neighbourhood'].value_counts())
disdata.reset_index(inplace=True)
disdata.rename(columns={'index':'Neighborhood','neighbourhood':'Count'},inplace=True)
print(disdata)
disdata = disdata.replace("朝阳区 / Chaoyang",110105)#标注行政区号
disdata = disdata.replace("东城区",110101)
disdata = disdata.replace("海淀区",110108)
disdata = disdata.replace("丰台区 / Fengtai",110106)
disdata = disdata.replace("西城区",110102)
disdata = disdata.replace("通州区 / Tongzhou",110112)
disdata = disdata.replace("昌平区",110114)
disdata = disdata.replace("密云县 / Miyun",110228)
disdata = disdata.replace("顺义区 / Shunyi",110113)
disdata = disdata.replace("怀柔区 / Huairou",110116)
disdata = disdata.replace("大兴区 / Daxing",110115)
disdata = disdata.replace("延庆县 / Yanqing",110229)
disdata = disdata.replace("房山区",110111)
disdata = disdata.replace("石景山区",110107)
disdata = disdata.replace("门头沟区 / Mentougou",110109)
disdata = disdata.replace("平谷区 / Pinggu",110117)


disdata['Neighborhood']=disdata['Neighborhood'].astype('int')#转换类型，让数据符合标准


folium.Choropleth(#按数量给区划填色
    geo_data=bj_geo,
    data=disdata,
    columns=['Neighborhood','Count'],
    key_on='feature.id',

    fill_color='YlOrRd',#红色

).add_to(bj_map3)
bj_map3.save('数量分布涂色3.html')
webbrowser.open('数量分布涂色3.html')
################################################


bj_map4 = folium.Map(location=[longitude, latitude], zoom_start=9)

folium.GeoJson(#行政区划
    bj_geo,
    style_function=lambda feature: {
        'fillColor': '#ffff00',
        'color': 'black',
        'weight': 1,

    }
).add_to(bj_map4)
heatdata = df[['latitude','longitude']].values.tolist()

HeatMap(heatdata,radius=11).add_to(bj_map4)#添加热力点

bj_map4.save('数量分布热力4.html')
webbrowser.open('数量分布热力4.html')
################################################


bj_map5 = folium.Map(location=[longitude, latitude], zoom_start=9)
price_df=pd.pivot_table(df,index='neighbourhood',values=['price'],aggfunc=np.mean)

price_df.reset_index(inplace=True)

print(price_df)
price_df = price_df.replace("朝阳区 / Chaoyang",110105)
price_df = price_df.replace("东城区",110101)
price_df = price_df.replace("海淀区",110108)
price_df = price_df.replace("丰台区 / Fengtai",110106)
price_df = price_df.replace("西城区",110102)
price_df = price_df.replace("通州区 / Tongzhou",110112)
price_df = price_df.replace("昌平区",110114)
price_df = price_df.replace("密云县 / Miyun",110228)
price_df = price_df.replace("顺义区 / Shunyi",110113)
price_df = price_df.replace("怀柔区 / Huairou",110116)
price_df = price_df.replace("大兴区 / Daxing",110115)
price_df = price_df.replace("延庆县 / Yanqing",110229)
price_df= price_df.replace("房山区",110111)
price_df = price_df.replace("石景山区",110107)
price_df= price_df.replace("门头沟区 / Mentougou",110109)
price_df = price_df.replace("平谷区 / Pinggu",110117)


folium.Choropleth(
    geo_data=bj_geo,
    data=price_df,
    columns=['neighbourhood','price'],
    key_on='feature.id',
    #fill_color='red',
    fill_color='YlOrRd',

).add_to(bj_map5)
bj_map5.save('价格分布涂色5.html')
webbrowser.open('价格分布涂色5.html')
################################################
bj_map6 = folium.Map(location=[longitude, latitude], zoom_start=9)
# Convert data format
heatdata = df[['latitude','longitude','price']].values.tolist()

folium.GeoJson(
    bj_geo,
    style_function=lambda feature: {
        'fillColor': '#ffff00',
        'color': 'black',
        'weight': 2,

    }
).add_to(bj_map6)

# add incidents to map
HeatMap(heatdata,radius=9).add_to(bj_map6)

# display world map
bj_map6.save('价格分布热力6.html')
webbrowser.open('价格分布热力6.html')
################################################

print("ok")
