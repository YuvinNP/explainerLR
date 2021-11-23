# explainerLR - Local Rule Based Explanations for black box models

explainerLR is a local rule based explanation method which provides rules for given instance that needs to be predicted
As inputs, it needs,
Neighbhourhood size
Encoded instances: encoded features 
Encoders: used encoders for training data (both categorical features and the target column)

For categorical features only situations:
Needs to pass all the column names at once

For mix situations:
Needs to pass all the catergorical and numerical column names seperately,
Needs to pass categorical column values and numerical column values seperately

## Installation
```pip install -i https://test.pypi.org/simple/ explainerlr ```

## Pre-requisites
```pip install pandas ```
```pip install pickle-mixin ```
```pip install chefboost ```


## Usage
```explainerlr.fit_data(n, <Encoded instance that needs to be predicted>, X_train, y_train, <columns>, <target>, <onehot encoder>, <label encoder>) ```

## Individual Instance Prediction
```for key in explainerlr.decode_data(ohe, X_test[1:2], ['buying','maint','doors','persons','lug_boot','safety'])[1].columns:
        explanation_rule[key] = explainerlr.decode_data(ohe, X_test[1:2], ['buying','maint','doors','persons','lug_boot','safety'])[1][key].values[0]
    
explanation_rule['class'] = 'None'
predicted_result = explainerlr.model_predict(explanation_rule, 'class')```
