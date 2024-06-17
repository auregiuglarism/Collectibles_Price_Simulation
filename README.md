# Collectibles_Price_Simulation

## The project

As trading between peers in the collectibles market becomes facilitated through technology, the data generated increases. New indices emerge in the space aggregating data from different assets to track the market. It is thus most interesting for actors in the space to have a forecast on this new kind of time series data to anticipate the future. This project evaluates univariate and multivariate (S)ARIMA(X) models on a wine, watch, and art index showing promising results with a 90\% accuracy improvement from the worst to the best-tuned (S)ARIMA(X) models. However, ARIMA models cannot predict sudden spikes (fat-tails) with a higher probability than the normal distribution suggests. As such due to the complexity of the data, the best ARIMA models cannot perform better than the arithmetic mean $\mu$ in their categories. This highlights the predictive power of simple models on complex time series in the collectibles market and the potential loss in accuracy and reliability that comes with complex models for in and out-of-sample forecasts.

## Files

preprocessing.py extracts, tabularizes and adjusts data for inflation 

univariate.py handles the univariate model creation, validation and forecasting for strategy 1. Univariate strategies 2 and 3 are handled in univariate_2.py and univariate_3.py respectively.

multivariate.py handles the multivariate model creation, validation and forecasting 

To get outputs, uncomment functions to display model outputs and accuracies as well as visualizations of the data. Modify the outputs by changing the function parameters of each forecast function in the main section of each files.

All models are already optimized with parameters explicitely written in the corresponding multivariate or univariate files.

