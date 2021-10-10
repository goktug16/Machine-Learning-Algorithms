# Machine-Learning-Algorithms

In this project, the dataset was created through a survey opened on Google forms. 
The purpose of the form is to find the person's favorite shopping type based on the information provided. In this context, 13 questions were asked to the user.
As a result of these questions, the estimation of the shopping type, which is a classification problem, will be carried out with 5 different algorithms.

These algorithms;

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine
- K Neighbors
- Decision Tree

algorithms will have a total of 12 parameters

A total of 219 people participated in the survey and the answers given to this form were used in the training of the algorithm.

Target variables to be estimated;

- Clothing 
- Technology 
- Home/Life 
- Book/Magazine

The questions asked to make the estimation are as follows:
- Gender
- Age
- Which store would you prefer to go to?
- Which store would you prefer to go to?
- Which store would you prefer to go to?
- What is your favorite season?
- What is the importance of the dollar exchange rate for your shopping?
- What is your satisfaction level with your budget for shopping?
- How would you rate your social life?
- Which of the online shopping sites do you prefer?
- How often do you go shopping?
- What is your average sleep time per day?
- What is your favorite type of shopping? // target

The dataset, which is in the form of a csv file, is read to the system as a dataframe. 
And the column of information in which hour and minute the user filled out the form, which does not make sense for our algorithm, is removed.

Since the numbers in some columns is way more different than the others before the PCA operation is performed,
the standardization process is applied to the columns so that they do not have a greater effect than the combination of these columns during the PCA operation.

The features and target columns to be used during the export of the dataset to the algorithms are determined.

In order to fit the resulting algorithms, the initial state of the dataset, 
its normalized state and the pca applied states are kept separately. The generated data is divided into parts as train = 0.8 and test = 0.2. 
Cross Validation process will be applied on 0.8 train data.

Before giving the dataset to the 5 algorithms, 
the answers written in the text in the dataset and the text in the other questions are encoded and the dataset is converted into numbers.


The 5 algorithms are functions from the sklearn library. The Cross Validation process was performed using the GridSearchCV() function, excluding the Logistic Regression algorithm. 
In the Logistic regression algorithm, since it is possible to do Cross Validation with the logistic regression function it is not necessary to use GridSearchCV().

GridSearchCV() applies K-Fold Cross Validation by trying the parameters I gave for the function, 
the number of K for my project is 10. By dividing the cross validation process parameters and the train data we provide, it is determined at which values we can get the best result.

An algorithm is created using the determined parameters and the algorithm is tested with the test data to be fitted with the train data.

Detailed information about dataset can be found in the report.
