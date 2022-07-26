# Classification of Productivity
A project implementing simple neural network on Keras to classify productivity of a garment team on given conditions.

# Dataset
The project is based on a UCI Machine Learning Repository, prepared by Abdullah Al Imran, which can be found at the link: https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees. This dataset includes important attributes of the garment manufacturing process and the productivity of the employees which had been collected manually and also been validated by the industry experts. The attribute information is as follows:

01 date : Date in MM-DD-YYYY

02 day : Day of the Week*

03 quarter : A portion of the month. A month was divided into four quarters

04 department : Associated department with the instance

05 team_no : Associated team number with the instance

06 no_of_workers : Number of workers in each team

07 no_of_style_change : Number of changes in the style of a particular product

08 targeted_productivity : Targeted productivity set by the Authority for each team for each day.

09 smv : Standard Minute Value, it is the allocated time for a task

10 wip : Work in progress. Includes the number of unfinished items for products

11 over_time : Represents the amount of overtime by each team in minutes

12 incentive : Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action.

13 idle_time : The amount of time when the production was interrupted due to several reasons

14 idle_men : The number of workers who were idle due to production interruption

15 actual_productivity : The actual % of productivity that was delivered by the workers. It ranges from 0-1.

* Please note that the 'day' column takes all values of weekday except 'Friday' as it is probably a rest day for the company.

# Data Preprocessing
We first drop the 'date' column as it is unrelevant to our project. We then convert the categorical columns, namely 'quarter' and 'department', to dummy columns. Please note that an extra step of preprocessing on the column 'department' is necessary as 'finishing' is mistakenly noted as 'finishing ' with an extra space in the dataset. In addition, the missing values in 'smv' column is replaced with 0 as these represent an already completed task with 0 number of remaining unfinished items.

Last, we scale the data with StandardScaler.

# Model
We then implement a NN with 2 hidden layers with relu activation and l1 regularisation. We also include Dropout layers to avoid overfitting.

# Result
With 50 epochs and a batch size of 32, our model attains a training accuracy of 90.39% and validation accuracy of 88.75%



