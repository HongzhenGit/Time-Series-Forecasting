# Time-Series-Forecasting
Here is a time series forecasting model constructed by Prophet (developed by Meta). With this framework, one can easily build a fantastic time series model for data with significant seasonality. Compared with traditional time series models like ARIMA and VAR, this Prophet-based model could easily capture the seasonality features within a time series, which makes it possible to make long-run predictions. <br>
For more information, please refer its official site:<br>
https://facebook.github.io/prophet/<br>
and the paper:<br>
https://peerj.com/preprints/3190/<br>
## A first glance at data
We would like to forecast the future demanding for electricity of Australia. Here we leverage the dataset from Kaggle datasets:<br>
https://www.kaggle.com/datasets/aramacus/electricity-demand-in-victoria-australia<br>
Before starting, one should look into what the data look like and do some preprocessing works to remove outliers and fill NULL values.<br> 
![Glance at Data](https://github.com/HongzhenGit/Time-Series-Forecasting/blob/main/Assets/line_data_glance.png)<br>
## Construct a time series model without extra-regressors
At the very beginning, it would be easy to build a model on a single time series(here it should the demangding time series data) without any independent variables. In this case, the training set contains the demanding data before 2018-06-06 and the validation set is between 2018-06-06 and 2019-06-06. The rest is treated as testing set(after 2019-06-06). With a Grid Search method, we could check which combination of parameters would lead to the best performance on the validation set.<br>
Here is the fittings and predictions made by this uni-variate model:<br>
![Unimodel Fits and Predicts](https://github.com/HongzhenGit/Time-Series-Forecasting/blob/main/Assets/unimodel_fits_predictions.png)<br>
It is illustrated that, the seasonality features of this electricity demanding time series could be well modeled. However, the model performance is not good enough for some values that are out of a normal range(not good at describing the fluctuations).It would be more significant on a scatter plot between actual demandings and model estimates. Here our MAE(Mean Absolute Error) is 6.5 and the correlation coeff is 0.77.<br>
![Unimodel Correlations](https://github.com/HongzhenGit/Time-Series-Forecasting/blob/main/Assets/unimodel_corr.png)<br>
## Construct a time series model with extra-regressors
Look into the dataset, we could observe that there is V-shape relationship between the electricity demanding and the max temperature. It would be easy to explain this V-shape relation: During cold or hot weather, peoplre are always having high demanding of electricity energy. Here we made a transformation on the data of max temperature:
$$y_i = ABS(x_i-x_{mean})$$
Then we would have a new variable named max temperature 1 that has a almost linear relation with electricity demanding.<br>
![Scatter Plot](https://github.com/HongzhenGit/Time-Series-Forecasting/blob/main/Assets/variable_scatters.png)<br>
Based on the analysis above, the new varibale max temperature 1 would be selected as the extra regressor for electricity demanding. Here a Grid Search method is also leveraged to help tune the parameters.<br>
With max temperature 1 embedded in as an extra regressor, this new model would have a stronger capability in capturing larger fluctuations in the time series:<br>
![Multimodel Fits and Predicts](https://github.com/HongzhenGit/Time-Series-Forecasting/blob/main/Assets/multimodel_fits_predictions.png)<br>
Thereby it will have a lower MAE and a higher correlation:<br>
![Multimodel Correlations](https://github.com/HongzhenGit/Time-Series-Forecasting/blob/main/Assets/multimodel_corr.png)<br>
***For more details regarding parameter tuning and compoents with seasonality in different lever, please check my Jupyter Notebook***
