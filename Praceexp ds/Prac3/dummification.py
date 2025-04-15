# 2.Perform feature dummification to convert categorical variables into numerical representations.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data={
    'Color':['Red','Blue','Green','Red','Blue'],
    'Size':['Small','Large','Medium','Medium','Small'],
    'Label':[1,0,1,0,1]
}

df=pd.DataFrame(data)

df_encoded=pd.get_dummies(df,columns=['Color','Size'],drop_first=True)

print("Original DataFrame:")
print(df)
print("\nDataFrame after Feature Dummification:")
print(df_encoded)

X= df_encoded.drop('Label', axis=1)
y=df_encoded['Label']

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("\nModel Accuracy:",accuracy)