
#   cancer Prediction System

##  Project Overview
This project is an end-to-end **cancer**  built using **Machine Learning**, **MLflow**, and **FastAPI**.  
It predicts whether a loan application should be ** type and here state  ** based on various applicant and financial attributes.
 

## ⚙️ Steps Followed

### 1️⃣ Data Exploration & Preprocessing
- Handled missing values and categorical encoding in my data its be cleand nor have any missing values in it and data more cleand bit in project we in sure  from this in a begining in preprocessing .
- Normalized numerical features for better model performance.
- Split dataset into `train` and `test` sets (80/20).
- Used `ColumnTransformer` and `OneHotEncoder` for mixed-type preprocessing.
- Saved preprocessing pipeline using **joblib**.

### 2️⃣ Model Training & Evaluation
- Trained  one classification models:  
  - XGBoost  
  before that i used 2 classification model three give me  accuracy > 90% but i preeferd use  LinearRegression() ,
  becuse he give me most height performance 
- Tuned hyperparameters using GridSearchCV.  
- Selected the best model with **accuracy > 90%** .  
- Saved the trained model and preprocessor pipeline using **joblib**.

### 3️⃣ Experiment Tracking (MLflow)
- Used **MLflow** to track experiments, model parameters, and metrics.
- Each run logged:
  - Model type  
  - Accuracy,   
  - Serialized model artifacts  

### 4️⃣ Model Serving (FastAPI)
- Built a **FastAPI** app with two endpoints:
  - `/regression cancer.joblib` → Uses locally saved joblib model
 
- Used **Pydantic** models for input validation and schema enforcement.

### 5️⃣ Front-End (HTML)
- Created a simple one-page modern UI using HTML, CSS, and JS.
- Allows users to input loan application data and get real-time prediction results from the FastAPI backend.

### 6️⃣ Dockerization
- Containerized the app with **Docker** for easy deployment.
- Configured the container to expose the FastAPI API on port `8000`.

---

## 🚀 How to Run the Project Locally anouther way by using docker in line   

## most important line here for run 

1-57 for build local server and run on cancer.html direct 
2- 80 run on docker and take yrl and applay in html page or test by postman or any think 

   - `/regression cancer.joblib` → Uses locally saved joblib model in beging  you will run file **app_joblib.py**
after that run code to run local surver using  joilib **uvicorn app:app --host 127.0.0.1 --port 8080**
 after that console output your local URL   **http://127.0.0.1:8080**
can run in my page **cancer.html** 
by inter localurl and API in any file as you need app_joilib.py  **http://127.0.0.1:8000/**
or 
from mlflow.tracking import MlflowClient
using mlflow RUN_ID
in cmd uvicorn  app:app --reload for running 

app_mlflow.py  **http://127.0.0.1:8080/predict_severity**
 if uou want inter data as backend for test on test program like postman 
 method POST 
  ##  url   http://127.0.0.1:8080/predict_severity 
  
and he give you result 
to get more information go ## **http://localhost:8000/docs** ##
 
####  anouther way for deployment ############## using docker  ############
deploy and inference a machine learning model (built on the iris dataset) using Docker and FastAPI.

1. With terminal navigate to the root of this repository
--------------------------------------------------------

2. Build docker image
---------------------
.. code-block::

    docker build -t image_name .
   ##   ex : docker build -t regressioncancer:v1 .
  ##

3. Run container

----------------
.. code-block::

    docker run --name container_name -p 8000:8000 image_name
 ##   ex :   docker run -d -p 8000:8000   regressioncancer:v1   ##

 all think must to be run on http://localhost:8000
 for api 
 ## http://localhost:8000/predict_severity ##
 

 ########## if  you want to sheck on koan applovel page on html ################
 change api in start page with 

 ## http://localhost:8000/predict_severity ##
 
 
4. Output will contain
----------------------
INFO:     Uvicorn running on http://0.0.0.0:8000

http://localhost:8000/docs for check  on web 
 

### 🔧 1. Clone the Repository

