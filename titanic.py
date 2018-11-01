import pandas as pd
import xgboost as xgb

# this changes gender to a boolean so xgb can actually handle it
def fix_sex(data):
	data.loc[data.Sex=='male','Sex'] = True
	data.loc[data.Sex=='female','Sex'] = False
	
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

# exclude the attributes that dont give us much info or are too confusing to use
attributes_to_exclude = ['Survived','Name','PassengerId','Cabin','Ticket']

# create a training matrix but also exclude the class value
x_train = train_raw.drop(attributes_to_exclude,axis=1)
# remove the class value because its never shows up in the testing set
attributes_to_exclude.remove('Survived')
# the testing csv doesnt even have a class value
x_test = test_raw.drop(attributes_to_exclude,axis=1)

# at this point, only two columns are non-numeric/non-boolean: Sex and Embarked

# turn sex into a boolean so xgboost can deal with it
fix_sex(x_train)
fix_sex(x_test)

# dummify Embarked (ie make one column for each category)
x_train = pd.get_dummies(x_train,columns=['Embarked'])
x_test = pd.get_dummies(x_test,columns=['Embarked'])


# make the vector of correct values for the training set
y_train = train_raw['Survived']

# create a classifier object with some parameters
# most of the parameters are set to default, which is fine
classifier = xgb.XGBClassifier(n_estimators=300,n_jobs=4,silent=1,eta=0.1)
# now actually run the classifier on the data and get a prediction vector
classifier.fit(x_train,y_train)
pred = classifier.predict(x_test)

# now turn the predictions into a pandas dataframe and place that into a csv file
submission = pd.DataFrame({ 'PassengerId': test_raw['PassengerId'],
                            'Survived': pred })
submission.to_csv("submission.csv", index=False)