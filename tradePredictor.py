import pandas as pd
import numpy as np
import ipywidgets as ipy
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import time
from   sklearn.model_selection import train_test_split




#image = Image.open('images.jpg')


tradeDataset = pd.read_excel('tradePredict.xlsx')

#check the missing values in dataset
#tradeDataset.isnull().sum()




def prediction(comm,math,lead,technical,creative,wUP,probSol,wrCS,circuit,wires,cleanMach,prog,car,phy,desVeh,decision,draw,cons,build,newTech,swDev,digiDes,sec,mach,hack):   
 
    # Pre-processing user input    
    if comm == "No":
        comm = 0
    else:
        comm = 1
 
    if math == "No":
        math = 0
    else:
        math = 1
 
    if lead == "No":
        lead = 0
    else:
        lead = 1  
   
    if technical == "No":
        technical= 0
    else:
        technical = 1 
            
    if creative == "No":
         creative = 0
    else:
         creative= 1         

    if  wUP== "No":
         wUP= 0
    else:
         wUP= 1 
 
    if  probSol== "No":
         probSol= 0
    else:
         probSol= 1 
            
    if wrCS== "No":
          wrCS= 0
    else:
          wrCS= 1         

    if circuit == "No":
         circuit= 0
    else:
         circuit= 1 

    if  wires== "No":
         wires= 0
    else:
         wires= 1 
            
    if  cleanMach== "No":
         cleanMach= 0
    else:
         cleanMach= 1         

    if  prog== "No":
         prog= 0
    else:
         prog= 1 
 
    if  car== "No":
         car= 0
    else:
         car= 1 
            
    if phy==  "No":
         phy= 0
    else:
         phy= 1         

    if  desVeh== "No":
         desVeh= 0
    else:
         desVeh= 1           

    if  decision== "No":
         decision= 0
    else:
         decision= 1         

    if  draw== "No":
         draw= 0
    else:
         draw= 1   
  
    if cons== "No":
         cons= 0
    else:
         cons= 1         

    if  build== "No":
         build= 0
    else:
         build= 1   
            
    if  newTech== "No":
         newTech= 0
    else:
         newTech= 1         

    if  swDev== "No":
         swDev= 0
    else:
         swDev= 1   
   
    if  digiDes== "No":
         digiDes= 0
    else:
         digiDes= 1         

    if  sec== "No":
         sec= 0
    else:
         sec= 1   
            
    if  mach== "No":
         mach= 0
    else:
         mach= 1         

    if  hack== "No":
         hack= 0
    else:
         hack= 1  
            
            
            



st.title('Predict your trade')


comm=st.radio('Do you have good communication skills:',('Yes','No'))
math=st.radio('Do you like mathematics:',('Yes','No'))
lead=st.radio('Do you have leadership skills:',('Yes','No'))
technical=st.radio('Do you have technical skills:',('Yes','No'))
creative=st.radio('Creative:',('Yes','No'))
wUP=st.radio('Can you work under pressure:',('Yes','No'))
probSol=st.radio('Do you have problem solving skill:',('Yes','No'))
wrCS=st.radio('Can you communicate easily through writings:',('Yes','No'))
circuit=st.radio('Do you like circuits:',('Yes','No'))
wires=st.radio('Do you like wires:',('Yes','No'))
cleanMach=st.radio('Do you like managing machines:',('Yes','No'))
prog=st.radio('Do you like programming:',('Yes','No'))
car=st.radio('Have you ever thought that how a care works?:',('Yes','No'))
phy=st.radio('Do you like physics:',('Yes','No'))
desVeh=st.radio('Have you ever wished to design a vehicle?',('Yes','No'))
decision=st.radio('Do you have the ability to make good decisions?',('Yes','No'))
draw=st.radio('Do you like drawing:',('Yes','No'))
cons=st.radio('Have you ever wished to supervise on construction site?',('Yes','No'))
build=st.radio('Do you wanna build a building?',('Yes','No'))
newTech=st.radio('Do you find new technologies ?',('Yes','No'))
swDev=st.radio('Do you wanna develop a software?',('Yes','No'))
digiDes=st.radio('Do you like digital designing?',('Yes','No'))
sec=st.radio('Do you wanna make your system security best?',('Yes','No'))
mach=st.radio('Have you ever opened a machine for experiment or for your wish?',('Yes','No'))
hack=st.radio('Have you ever thought that who removes the unauthorized access from computer system?',('Yes','No'))



#lets create training ad testing sets for validation of results
from   sklearn.model_selection import train_test_split

X=tradeDataset.drop('trade',axis=1)
y=tradeDataset['trade']

X_train,X_test,y_train ,y_test = train_test_split(X,y,random_state=0,test_size=0.2)


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X_train,y_train)


array = [comm,math,lead,technical,creative,wUP,probSol,wrCS,circuit,wires,cleanMach,prog,car,phy,desVeh,decision,draw,cons,build,newTech,swDev,digiDes,sec,mach,hack
]
arr = np.array(array)
arr1 = arr.reshape(1, 25)


prediction = model.predict(arr1)


if st.button('Predict the trade'):
    st.header("You can choose {}".format(prediction))