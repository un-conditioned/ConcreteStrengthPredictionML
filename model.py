import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("compresive_strength_concrete.csv")
df.columns


df_new = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'Cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'BFS',
       'Fly Ash (component 3)(kg in a m^3 mixture)':'Fly_Ash',
       'Water  (component 4)(kg in a m^3 mixture)':'Water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)':'Superplasticizer',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'Coarser_agg',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'Fine_agg',
       'Age (day)':'Days',
       'Concrete compressive strength(MPa, megapascals) ':'Comp_str'})

df_new.columns



from sklearn.model_selection import train_test_split, cross_val_score
features = ['Cement', 'BFS', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarser_agg',
       'Fine_agg', 'Days']
targets = ['Comp_str']

X_train, X_test, y_train, y_test = train_test_split(df_new[features], df_new[targets], test_size=0.20, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

params_RFR = [{'n_estimators':[200, 250, 300, 350, 400, 450, 500, 550, 600]}]

RFR = RandomForestRegressor()
grid_RFR = GridSearchCV(RFR, params_RFR, cv=3, scoring='r2')
model_RFR = grid_RFR.fit(X_train, y_train)


RFR_1 = RandomForestRegressor(n_estimators=300, criterion='mse')




model_RFR_1 = RFR_1.fit(X_train,y_train)




y_RFR1 = model_RFR_1.predict(X_test)



import pickle

pickle.dump(RFR_1,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

# New Section
