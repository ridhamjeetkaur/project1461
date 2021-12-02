import pandas as pd
import numpy as np
import ipywidgets as ipy
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import time
from   sklearn.model_selection import train_test_split

st.session_state['answer'] = ''!

st.write(st.session_state)

realans = ['', 'abc', 'edf']

if  st.session_state['answer'] in realans:
    answerStat = "correct"
elif st.session_state['answer'] not in realans:
    answerStat = "incorrect"

st.write(st.session_state)
st.write(answerStat)



aptitude = pd.read_excel('aptitude_file.xlsx')


aptitude = pd.read_excel('aptitude_file.xlsx')
image = Image.open('images.jpg')
st.title('Predict the stream')
st.image(image,use_column_width=True)


#checking the aptitude
st.write('This is an application that helps you to know which stream to chose.......')
#check_box=st.checkbox('View the aptitude')
#if check_box:
#    st.write(aptitude.head(4))
#st.write("Let's check the prediction with other parameters")

# splitting your aptitude
aptitude['Accounts'].fillna(0,inplace=True)
aptitude['BusinessStudies'].fillna(1,inplace=True)
aptitude['Economics'].fillna(2,inplace=True)
aptitude['Mathematics'].fillna(3,inplace=True)
aptitude['English'].fillna(4,inplace=True)
aptitude['Computer_science'].fillna(5,inplace=True)
aptitude['Physics'].fillna(6,inplace=True)
aptitude['Chemistry'].fillna(0,inplace=True)
aptitude['Biology'].fillna(1,inplace=True)
aptitude['History'].fillna(2,inplace=True)
aptitude['Geography'].fillna(3,inplace=True)
aptitude['Political_science'].fillna(4,inplace=True)
aptitude['observation_experiment'].fillna(5,inplace=True)
aptitude['sports'].fillna(6,inplace=True)
aptitude['study_hours'].fillna(6,inplace=True)


#X=aptitude.iloc[:,:7]
#y=aptitude.iloc[:,-1]


from   sklearn.model_selection import train_test_split

y=aptitude['streams']
X=aptitude.drop('streams',axis=1)
X_train,X_test,y_train ,y_test = train_test_split(X,y,random_state=0,test_size=0.4)

#input the no.
a=st.slider('Accounts:',int(X.Accounts.min()),int(X.Accounts.max()),int(X.Accounts.mean()))
bs=st.slider('Business Studies:',int(X.BusinessStudies.min()),int(X.BusinessStudies.max()),int(X.BusinessStudies.mean()))
e=st.slider('Economics:',int(X.Economics.min()),int(X.Economics.max()),int(X.Economics.mean()))
m=st.slider('Mathematics:',int(X.Mathematics.min()),int(X.Mathematics.max()),int(X.Mathematics.mean()))
eng=st.slider('English:',int(X.English.min()),int(X.English.max()),int(X.English.mean()))
cs=st.slider('Computer Science:',int(X.Computer_science.min()),int(X.Computer_science.max()),int(X.Computer_science.mean()))
ph=st.slider('Physics:',int(X.Physics.min()),int(X.Physics.max()),int(X.Physics.mean()))

c=st.slider('Chemistry:',int(X.Chemistry.min()),int(X.Chemistry.max()),int(X.Chemistry.mean()))
b=st.slider('Biology:',int(X.Biology.min()),int(X.Biology.max()),int(X.Biology.mean()))
h=st.slider('History:',int(X.History.min()),int(X.History.max()),int(X.History.mean()))
g=st.slider('Geography:',int(X.Geography.min()),int(X.Geography.max()),int(X.Geography.mean()))
ps=st.slider('Political science:',int(X.Political_science.min()),int(X.Political_science.max()),int(X.Political_science.mean()))
oe=st.slider('Observation or experiment:',int(X.observation_experiment.min()),int(X.observation_experiment.max()),int(X.observation_experiment.mean()))
sp=st.slider('Sports:',int(X.sports.min()),int(X.sports.max()),int(X.sports.mean()))
sh=st.slider('Study hour:',int(X.study_hours.min()),int(X.study_hours.max()),int(X.study_hours.mean()))


model=LogisticRegression()
model.fit(X_train,y_train)
#model.fit(X,y)

prediction = model.predict((np.array([[a,bs,e,m,eng,cs,ph,c,b,h,g,ps,oe,sp,sh]])))

#checking the prediction
if st.button('Predict the stream:'):
    st.header("You can choose {}".format(prediction))
