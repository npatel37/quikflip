### QuikFlip Flask App

This is a Flask app built off the [standard bootstrap template](https://getbootstrap.com/docs/3.3/) that includes python scripts built for making back-end data analysis and modeling. The final web app product is deplayed using amzon AWS. Please visit my app at quikflip.xyz

### Dataset 
The city of Boston provides [tax-assessment data](https://data.boston.gov/dataset/property-assessment/) for all the commercial and residential properties from 2010 to 2019 (current) years. This is the main dataset used for building this app. 

### Modeling
A linear regression model is used with L1 and L2 regularization (Elasticnet) to predict current and future market-value of residential properties. Additionally, cross-validation was performed for hyperparameter tuning and to control overfitting. The model is used to extract the feature parameters that positively and negatively affect the market evaluation of a house. The negatively weighted features are used to suggest renovation, and market-value is predicted after this renovation to estimate the total return on investment (ROI). 

### Outcome and Product
The final product is deployed as a web app at [quikflip.xyz](http://quikflip.xyz). This app provides a recommendations for houses that has the largest ROI prediction based on the regression house evaluation model. This app allow house flippers to make quantitavely backed decisions for their bussiness and saves time that is often spent looking for real-estate investments. 


#

### Disclaimer
The quikflip was built as part of an independent fun project. quikflip app or anyone associated with this app is not responsible for any decisions made based on this project/app. 