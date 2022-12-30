source: https://github.com/GokuMohandas/mlops-course
# 0 Getting It To Run
```terminal
Setup:
    cd /Users/chang/Documents/dev/git/ml/mlops
    source venv/bin/activate
    
    
ML WorkFlows:
    Train Model: 
        python tagifai/main.py  --help
        python tagifai/main.py train-model
    
    Create Documentation:
        python3 -m pip install -e ".[docs]"
            * setup.py defines docs_packages= ["mkdocs==1.3.0", "mkdocstrings==0.18.1"]   
        manual
            * python3 -m mkdocs new .
            * mkdocs.yml 
                defines the docs structure
            * python3 -m mkdocs serve
        
    Start MLFlow server
        mlflow server -h 0.0.0.0 -p 8001 --backend-store-uri $PWD/experiments/
    
    
Dev Workflow
    Setup pyton version
        pyenv local 3.8.10
        
    Setup python virtual env 
        make venv
    
    Lint + Format files 
        make style
    
    Run tests
        make test
    
    Version your data
        make dvc

Start Web service
    gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app
    http://0.0.0.0:8000/docs

    curl -X POST "http://localhost:8000/models" \   # method and URI
          * -H  "accept: application/json" \            # client accepts JSON 
          * -H  "Content-Type: application/json" \      # client sends JSON 
          * -d "{'name': 'RoBERTa', ...}"               # request body

# Convert markdown to docx --> upload to google doc
    pandoc -o mlops-notes.docx -f markdown -t docx notes.md

#

git remote rm origin
git remote set-url origin https://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@github.com/thomaschangsf/mlops.git
git push -u origin main

```

# 1 Design

# 2 Data
## Labeling (Skipped)
- WORKFLOW
  - Establish data pipelines:
  - [IMPORT] new data for annotation
  - [EXPORT] annotated data for QA, testing, modeling, etc. 
  - Create a quality assurance (QA) workflow:
  - separate from labeling workflow (no bias)
  - communicates with labeling workflow to escalate errors
- LIBRARIES
```commandline
GENERAL:
    Labelbox: the data platform for high quality training and validation data for AI applications.
    Scale AI: data platform for AI that provides high quality training data.
    Label Studio: a multi-type data labeling and annotation tool with standardized output format.
    Universal Data Tool: collaborate and label any type of data, images, text, or documents in an easy web interface or desktop app.
    Prodigy: recipes for the Prodigy, our fully scriptable annotation tool.
    Superintendent: an ipywidget-based interactive labelling tool for your data to enable active learning.

NLP:
    Doccano: an open source text annotation tool for text classification, sequence labeling and sequence to sequence tasks.
    BRAT: a rapid annotation tool for all your textual annotation needs.

CV:
    LabelImg: a graphical image annotation tool and label object bounding boxes in images.
    CVAT: a free, online, interactive video and image annotation tool for computer vision.
    VoTT: an electron app for building end-to-end object detection models from images and videos.
    makesense.ai: a free to use online tool for labelling photos.
    remo: an app for annotations and images management in computer vision.
    Labelai: an online tool designed to label images, useful for training AI models.

AUDIO:
    Audino: an open source audio annotation tool for voice activity detection (VAD), diarization, speaker identification, automated speech recognition, emotion recognition tasks, etc.
    audio-annotator: a JavaScript interface for annotating and labeling audio files.
    EchoML: a web app to play, visualize, and annotate your audio files for machine learning.


MISCELLANEOUS:
    MedCAT: a medical concept annotation tool that can extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS.


```
### Active Learning
- Active Learning:
  - Given a partially labeled dataset, active learning returns unlabeled datasets that needs to be labeled.
  ```
    1. Label a small, initial dataset to train a model.
    2. Ask the trained model to predict on some unlabeled data.
    3. Determine which new data points to label from the unlabeled data based on:
         - entropy over the predicted class probabilities
         - samples with lowest predicted, calibrated, confidence (uncertainty sampling)
         - discrepancy in predictions from an ensemble of trained models
         - Repeat until the desired performance is achieved.
  ```
  - Libraries
    - modAL:  a modular active learning framework for Python.
    - libact:  pool-based active learning in Python.
    - ALiPy: active learning python toolbox, which allows users to conveniently evaluate, compare and analyze the performance of active learning methods.
    
  - Vs Data Augmentation (with weak supervision, ie SNORKEL)
       * Data augmentation increases the size of training dataset
         * Via weak supervision (ie SNORKEL)
           * Use noisy sources (non humnan, non manual) to create more training data
         * Via programatic functions (slicing, transforming)
         * See Modeling section for more details
       * If we had samples that needed labeling or if we simply wanted to validate existing labels, we can use weak supervision to generate labels as opposed to hand labeling all of them. We could utilize weak supervision via labeling functions to label our existing and new data, where we can create constructs based on keywords, pattern expressions, knowledge bases, etc. And we can add to the labeling functions over time and even mitigate conflicts amongst the different labeling functions.
    

## Exploration
```python
from collections import Counter
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud, STOPWORDS
sns.set_theme()
warnings.filterwarnings("ignore") 

# TAG DISTRIBUTION:
tags, tag_counts = zip(*Counter(df.tag.values).most_common())
plt.figure(figsize=(10, 3))
ax = sns.barplot(list(tags), list(tag_counts))
plt.title("Tag distribution", fontsize=20)
plt.xlabel("Tag", fontsize=16)
ax.set_xticklabels(tags, rotation=90, fontsize=14)
plt.ylabel("Number of projects", fontsize=16)
plt.show()

# WORD CLOUD:
# Most frequent tokens for each tag
@widgets.interact(tag=list(tags))
def display_word_cloud(tag="natural-language-processing"):
    # Plot word clouds top top tags
    plt.figure(figsize=(15, 5))
    subset = df[df.tag==tag]
    text = subset.title.values
    cloud = WordCloud(
        stopwords=STOPWORDS, background_color="black", collocations=False,
        width=500, height=300).generate(" ".join(text))
    plt.axis("off")
    plt.imshow(cloud)
```

## Preprocessing

### Preparation
- Organizing and cleaning data
```commandline
JOINS:
    SELECT * FROM A
    INNER JOIN B on A.id == B.id

MISSING VALUES:
    # Drop a row (sample) by index
    df.drop([4, 10, ...])
    # Conditionally drop rows (samples)
    df = df[df.value > 0]
    # Drop samples with any missing feature
    df = df[df.isnull().any(axis=1)]
    
    # Fill in missing values with mean
    df.A = df.A.fillna(df.A.mean())

    # Replace zeros to NaNs
    import numpy as np
    df.A = df.A.replace({"0": np.nan, 0: np.nan})

OUTLIERS:
    # Drop samples with outliers
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    
    # Feature value must be within 2 standard deviations
    df[np.abs(df.A - df.A.mean()) <= (2 * df.A.std())]
    
    # CAVEATS
        - Be careful not to remove important outliers (ex. fraud)
        - Values may not be outliers when we apply a transformation (ex. power law)
        - Anomalies can be global (point), contextual (conditional) or collective (individual points are not anomalous and the collective group is an outlier)

CLEANING:
    use domain expertise and EDA
    apply constraints via filters
    ensure data type consistency
    removing data points with certain or null column values
```

### Transformation
Feature engineering + encoding

```commandline
SCALING (Features)
    Caveats
        required for models where the scale of the input affects the processes
        learn constructs from train split and apply to other splits (local)
        don't blindly scale features (ex. categorical features)

    Standardization: Rescale values between 0 and 1
        x = np.random.random(4) # values between 0 and 1
        x_standardized = (x - np.mean(x)) / np.std(x)

    Min-max: rescale values between min and max
        x = np.random.random(4) # values between 0 and 1
        x_scaled = (x - x.min()) / (x.max() - x.min())
        
    Binning: convert continuous features into categorical using bins 
        x = np.random.random(4) # values between 0 and 1
        x_binned = np.digitize(x, bins=[0.25, 0.5, 0.75])
        
        bins = np.linspace(0, 1, 5) # bins between 0 and 1
        binned = np.digitize(x, bins)

 
ENCODING
    Represent categorical features as numerical values
    
    # One-hot encoding
    pd.get_dummies(df, columns=["A", "B"])
    
    # Label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df.A = le.fit_transform(df.A)
    
    # Ordinal encoding
    df.A = df.A.map({"A": 0, "B": 1, "C": 2})
    
    # Binary encoding
    from category_encoders import BinaryEncoder
    be = BinaryEncoder()
    df = be.fit_transform(df)
    
    # Hash encoding
    from category_encoders import HashingEncoder
    he = HashingEncoder()
    df = he.fit_transform(df)
    
    # Target encoding
    from category_encoders import TargetEncoder
    te = TargetEncoder()
    df = te.fit_transform(df.A, df.B)
    
    # Frequency encoding
    df.A = df.A.map(df.A.value_counts())
    
    # Count encoding
    df.A = df.A.map(df.A.value_counts(dropna=False))
    
    # Mean encoding
    df.A = df.A.map(df.groupby("A").B.mean())
    
    # Weight of evidence encoding
    df.A = df.A.map(df.groupby("A").B.mean() / (1 - df.groupby("A").B.mean()))
    
    # Leave one out encoding
    df.A = df.A.map(df.groupby("A").B.mean() * df.A.value_counts() / (df.A.value_counts() - 1))
    

Extraction
    Extract singals from existing features
        transfer learning: using a pretrained model as a feature extractor and finetuning on it's results
        auto-encoders: learn to encode inputs for compressed knowledge representation
        PCA: linear dimensionality reduction to project data in a lower dimensional space.    
```

## Splitting : Training, Validation, Testing
Intuition:
- Training:
  - used to train model for one epoch A
  - update model weights based on result

- Validation:
  - used to evaluate model at end of epoch A 
  - update algorithm hyper parameters based on result

- Test:
  - After all the epochs, we use test to determine the final results
  
Caveats:
- Make sure we have similar # of classes across the 3 datasets
- Code:
  - Naive:
    - ```python
      from sklearn.model_selection import train_test_split
      import pandas as pd
      
      # Split (train)
      X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
      
      # Split (test)
      X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)

      # Get counts for each class
      counts = {}
      counts["train_counts"] = {tag: label_encoder.decode(y_train).count(tag) for tag in label_encoder.classes}
      counts["val_counts"] = {tag: label_encoder.decode(y_val).count(tag) for tag in label_encoder.classes}
      counts["test_counts"] = {tag: label_encoder.decode(y_test).count(tag) for tag in label_encoder.classes}
      
      # View distributions
      pd.DataFrame({
        "train": counts["train_counts"],
        "val": counts["val_counts"],
        "test": counts["test_counts"]
      }).T.fillna(0)

      # Skipped; more code at https://madewithml.com/courses/mlops/splitting/
      ```

## Augmentation
- Intuition:
  - Split the dataset first, we only augment the training set
  - Analyze the augmented dataset to confirm it makes sense and reflects our task
- Libraries 
  - Depending on the feature types and tasks, there are many data augmentation libraries 
  - NLP
    - NLPAug: data augmentation for NLP. 
    - TextAttack: a framework for adversarial attacks, data augmentation, and model training in NLP. 
    - TextAugment: text augmentation library.
  - Computer Vision
    - Imgaug: image augmentation for machine learning experiments. 
    - Albumentations: fast image augmentation library. 
    - Augmentor: image augmentation library in Python for machine learning. 
    - Kornia.augmentation: a module to perform data augmentation in the GPU. 
    - SOLT: data augmentation library for Deep Learning, which supports images, segmentation masks, labels and key points.
  - Other
    - Snorkel: system for generating training data with weak supervision. 
      - weak supervision: labeling data with imperfect or noisy signals
    - DeltaPy⁠⁠: tabular data augmentation and feature engineering. 
    - Audiomentations: a Python library for audio data augmentation. 
    - Tsaug: a Python package for time series augmentation.

- [Snorkel](https://github.com/snorkel-team/snorkel)
  - [Tutorial](https://www.snorkel.org/get-started/)
  - Key concepts
    - Labeling function: Labeling unlabeled training data
    - Transformation functions: data augmentation
    - Slicing functions: Monitor critical subsets of the data
  - Steps
  - ``` 
    1. Writing Labeling Functions (LFs): First, rather than hand-labeling any training data, we’ll programmatically label our unlabeled dataset with LFs.
    2. Modeling & Combining LFs: Next, we’ll use Snorkel’s LabelModel to automatically learn the accuracies of our LFs and reweight and combine their outputs into a single, confidence-weighted training label per data point.
    3. Writing Transformation Functions (TFs) for Data Augmentation: Then, we’ll augment this labeled training set by writing a simple TF.
    4. Writing Slicing Functions (SFs) for Data Subset Selection: We’ll also preview writing an SF to identify a critical subset or slice of our training set.
    5. Training a final ML model: Finally, we’ll train an ML model with our training set.
    ```
- Example: [NLPAug with Snorkel](https://madewithml.com/courses/mlops/augmentation/#Application)
   

# 3 Modeling
## Evaluations
Refer to evaluations [opus](https://docs.google.com/document/d/1UGyQ2SNkExHz1_MAOklSBFx-_ey95PyLKQBpXHrwWIw/edit?usp=sharing) evaluations
### Confusion Matrix
### Confident Learning
### Manual Slices
###  Generated Slices
###  Interpretability
### CounterFactuals
### Behavorial Testing
### Online Testing
####  A/B Testing
####  Canary Tests

## Experiment Tracking

- To Execute:
    * mlflow server -h 0.0.0.0 -p 8001 --backend-store-uri $PWD/experiments/
    * python tagifai/main.py train-model
          * this runs, and updates config/run_id.txt with the run id, which is used by app.py to load the model in the app

## Optimization
### Tools
- Options
   - [Optuna](https://optuna.org/)
   - [Hyperopt](https://github.com/hyperopt/hyperopt)
   - [RayeTune]
- Code (Optuna)
  - Objective
- Code (Hyperopt)
  - Objective

  
    def objective(args, trial):
    """Objective function for optimization trials."""
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))
    trial.set_user_attr("precision", performance["precision"])
    trial.set_user_attr("recall", performance["recall"])
    trial.set_user_attr("f1", performance["f1"])

    return performance["f1"]

  
   - Code


        from numpyencoder import NumpyEncoder
        from optuna.integration.mlflow import MLflowCallback
        NUM_TRIALS = 20
        # Optimize
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
        study.optimize(lambda trial: objective(args, trial),
                    n_trials=NUM_TRIALS,
                    callbacks=[mlflow_callback])


# 4 DEVELOPING
## Packaging
    Pyenv
      - Manage your local python version
      - Workflow
         - pyenv list
         - pyenv install 3.7.13
         - pyenv local 3.7.13  # use this verion of python
         - python --versions
      - Note:  ~/.python-version will override the command above.
        - pyenv local/global 3.7.13 will update the .python-version; conversely, one can manually change it.
    Makefile
      - Purpose1: Command shortcuts
        - Examples: 
          - make venv
          - make test
          - 
      - Purpose 2: build individual files
    Setup.py
      - Used by pip install -e 
        - python3 -m pip install -e ".[dev]"
      - Can refer to rquirements.txt
    
    Requirements.txt
      - used to install python environment
      - python3 -m pip install -r requirements.txt
  
  ## Organization
    Configuration
      - config.py
        - Contains all the configuration variables
        - Can be used to load environment variables
        - Can be used to make directories, ie logs, stores
      - args.json
        - Contains argument for model training
    Project (tagifai)
      ├── data.py       - data processing utilities
      ├── evaluate.py   - evaluation components
      ├── main.py       - training/optimization pipelines
      ├── predict.py    - inference utilities
      ├── train.py      - training utilities
      └── utils.py      - supplementary utilities

## Logging
    Components
      - Logger: emits the log messages from our application.
      - Handler: sends log records to a specific location: console, .
      - Formatter: formats and styles the log records.
      
      - config/config.py sets up the handler, formatters
    Levels
      - debug, info, warning, error, critical

    Examples
      - config.py actually defines the log config and create logger
          logging.config.dictConfig(logging_config)
          logger = logging.getLogger()
          logger.handlers[0] = RichHandler(markup=True)
   

## Documentation
    Comments vs docsstrings vs docs
        - Comments: short descriptions as to why a piece of code exists.
        
        - Docsstrings: meaningful descriptions for functions and classes that describe overall utility, arguments, returns, etc
            * with mkdocs || sphinx, can be used to generate documentation

        - Docs: rendered webpage that summarizes all the functions, classes, workflows, examples, etc.
    
    To create documentation
        - python3 -m pip install -e ".[docs]"
            * setup.py defines docs_packages= ["mkdocs==1.3.0", "mkdocstrings==0.18.1"]   
        - manual
            * python3 -m mkdocs new .
            * mkdocs.yml 
                defines the docs structure
            * python3 -m mkdocs serve

## Styling
    Tools
      - Black: an in-place reformatter that (mostly) adheres to PEP8.
          * uses pyproject.toml to define the style
          * To run: black .
	
      - isort: sorts and formats import statements inside Python scripts.
  	      * To run: python3 -m isort .

      - flake8: a code linter with stylistic conventions that adhere to PEP8.
          * To run: flake8

## MakeFile
    Purposes: 
        Shortcuts for commands
        Build individual files
    
    Examples: 
        make venv
        make test
    
    Details: 
        https://madewithml.com/courses/mlops/makefile/

# 5 SERVING

## Command line [Typer]
    Use typer to create a command line interface
        Typer uses the functions docstring and input parameter to create the cli command
    
    Code: (main.py)
        @app.command()
        def predict_tag(text: str = "", run_id: str = None) -> None:
            """Predict tag for text.
        
            Args:
                text (str): input text to predict label for.
                run_id (str, optional): run id to load artifacts for prediction. Defaults to None.

    To Execute:
        python tagifai/main.py train-model
        
        python tagifai/main.py predict-tag --help
        python tagifai/main.py predict-tag --text="Transfer learning with transformers for text classification."

          

## REST Api [Via FastAPI]

### Batch (offline, cached) vs Real Time Serving
  - Batch: 
    - Prediction is preprocessed and stored in a database
    - Prediction is retrieved from the database and served
    - Pros: 
      - Fast
      - Scalable
    - Cons:
      - Stale features, cold starts
      - Requires a database
      - Input cardinality is limited (ie users, items, etc)
  - Real Time Serving:
    - Prediction is made on the fly
    - Pros:
      - Fresh features
      - No database
      - Input cardinality is unlimited, but may be less robust bc need to guard against input space
    - Cons:
      - Slow
      - Not scalable

### Request
  - Uniform Resource Identifier
    - Ex: 
      - https://localhost:8000/models/{modelId}/?filter=passed#details
      - |SCHEME|://|DOMAIN|:|PORT|/|PATH|/|QUERY|#ANCHOR
        - path: location of the resource
        - query: parameters to identify the resource
        - anchor: location on webpage
  - Curl Command
      - Options:
        - -X, --request : HTTP method (ie. GET)
        - -H, --header   headers to be sent to the request (ex. authentication)
        - -d, --data     data to POST, PUT/PATCH, DELETE (usually JSON)
    
      - Examples: 
       
      
        - curl -X GET "http://localhost:8000/models/1/?filter=passed#details" -H "accept: application/json"

        - curl -X POST "http://localhost:8000/models" \   # method and URI
          * -H  "accept: application/json" \            # client accepts JSON 
          * -H  "Content-Type: application/json" \      # client sends JSON 
          * -d "{'name': 'RoBERTa', ...}"               # request body

  - Methods
    - GET: get a resource. 
    - POST: create or update a resource. 
    - PUT/PATCH: create or update a resource. 
    - DELETE: delete a resource.
      
### FastAPI (~Play)
#### Getting it up and running
    gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app
    http://0.0.0.0:8000/docs

#### FastAPI Vs Flask
  - Flask uses starlette
    - Starlette is a lightweight ASGI framework/toolkit, which is ideal for building high performance asyncio services.
    - Flask is more like Starlette; FastAPI adds more features to 
    
#### Benefits
Comes with multiple features
        
    Swagger UI
    Pydantic: data validation and improves robustness
            https://docs.pydantic.dev/#rationale
    Dependency Injection
    Security with OAuth2
    Performant


#### UVIGorn vs Gunicorn
  - FastAPI is a web application framework. Web frameworks needs to run on a web server like WSGI or ASGI

  - ASGI vs WSGI (Web Server Gateway Interface)
    - Both defines inferfance between python web app and web server
    - Unlike WSGI, ASGI allows multiple asynchronous events per application

  - Uvicorn
    - FastAPI uses Uvicorn as the ASGI server
    - Uvicorn is a lightning-fast ASGI server implementation, using uvloop and httptools.
    - Uvicorn is a pure Python ASGI server, so it has no dependencies other than Python itself.
    - Cmd:

       
          uvicorn app.api:app \       # location of app (`app` directory > `api.py` script > `app` object)
          --host 0.0.0.0 \        # localhost
          --port 8000 \           # port 8000
          --reload \              # reload every time we update
          --reload-dir tagifai \  # only reload on updates to `tagifai` directory
          --reload-dir app        # and the `app` directory

  - Gunicorn
    - Gunicorn is a Python WSGI HTTP Server for UNIX.
    - Gunicorn 'Green Unicorn' is a Python WSGI HTTP Server for UNIX. It's a pre-fork worker model ported from Ruby's Unicorn project. The Gunicorn server is broadly compatible with various web frameworks, simply implemented, light on server resources, and fairly speedy.
    - Gunicorn is a mature, stable, and well-tested solution.
    - Gunicorn is a great choice for production deployment.
    - If we want multiple uvicorn workers, use gunicorn
    - Cmd        

        
        gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app

#### Taking It Further
A full app also requires:

    Database (SQL, Redis, Couch)
    Authenticaton with JWT
    Asynchronous Task Queue with Celery
    Customizatble fronetend with Vue.js or React.js
    Docker, Docker-compose, K8s
    etc..

A good template is here: [FastAPI Template](https://fastapi.tiangolo.com/project-generation/)

#### Model Server
Deals with how to host, download multiple models, and serve them from docker containers

Candidates: MLFlow, BentoML

# 6 Testing
## Code
## Data
## Models

# 7 Reproducibility
## Git
## Pre-commit
## Versioning Code, Data, and Models
- Tools: Data Version Control (DVC), GitLFS, Dolt

## Docker

# 8 Production (Skipped)
## Dashboard
## CI/CD
## Monitoring
### System Health (CPU, GPU, memory)
### Model Performance
- Tool: [monitoring-ml](https://github.com/GokuMohandas/monitoring-ml)
### Drift (Data, Target, Concept)
### Locating Drift
### Measuring Drift
### Detecting Drift Online
### Outliers
## System Design

# 9 Data Engineering
## Data Stack
- Data System
  - Data Base
    - Is an online transaction processing (OLTP) system, which support CRUD operations
  - Data Lake: 
    - stores unstructured (images, videos, etc) and structured data
  - Data Warehouse
    - stores structured data
    - Online Analytical Processing (OLAP) system, which is optmized to perform aggregating columns rather than rows
  - Data LakeHouse
    - Salesforce CDP
- ETL (Extract and Load)
  - DB|Streams|Apps --> Extract Transform pipelines --> Lakes, Warehouses, 
  - Extract
    - Tools: Fivetran, AirByte, Stitch
  - Transform
    - Tools: dbt, Apache Spark, Apache Beam
        
- Observability
  - Data quality:
    - testing and monitoring data quality after every step, ie schema, completeness, recency
  - Data lineage
    - tracking the flow of data from source to destination
    - Tools: dbt, Apache Atlas (catalog), Amundsen
  - Privacy and security
    - Tools: Apache Ranger, Apache Atlas, Amundsen
  - Discoverability
    - Enable discovery of different data sources and features for downstream features
    - Tools: Apache Atlas, Amundsen

## Orchestration
- Previously, we used Typer CLI to run our workflow locally.
    - Next step is to scale across wfs, users, complexity
    - Orchestration tools like Airflow help us do this by providing connectors (airbyte, dbt, google cloud) and wrap it in a GUI with scheduler.

- Intuition:
  - ````commandline
    - PURPOSE
       * Schedule WF 
       * Scale WF
       * Monitor WF
    
    - TOOLS
        * Airflow
        * Prefect
        * Luigi
        * KubeFlow  


### Airflow
- Setup
    - ````commandline
        # Configurations
        export AIRFLOW_HOME=${PWD}/airflow
        AIRFLOW_VERSION=2.3.3
        CONSTRAINT_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt
        
        # Install Airflow (may need to upgrade pip)
        pip install apache-airflow==${AIRFLOW_VERSION} --constraint ${CONSTRAINT_URL}
        
        # Initialize DB (SQLite by default)
        airflow db init

        # Update Airflow config at airflog/airflow.cfg
        
        airflow db init
        
        # We'll be prompted to enter a password
        airflow users create \
            --username admin \
            --firstname thomas \
            --lastname chang \
            --role Admin \
            --email thomas.w.chang@gmail.com
        
            username=admin
            pwd=Airflow123$
        
        # Launch webserver
        export AIRFLOW_HOME=${PWD}/airflow
        airflow webserver --port 8082  # http://localhost:8080 
        localhost:8082
        
        # Launch scheduler
        Scheduler runs in the background and is responsible for monitoring the DAGs folder and triggering tasks as needed.
                
        source venv/bin/activate
        export AIRFLOW_HOME=${PWD}/airflow
        airflow scheduler

- Airflow Code
    ```python
    # airflow/dags/workflows.py
    from pathlib import Path
    from airflow.decorators import dag
    from airflow.utils.dates import days_ago
    
    # Default DAG args
    default_args = {
        "owner": "airflow",
        "catch_up": False,
    }
    BASE_DIR = Path(__file__).parent.parent.parent.absolute()
    
    @dag(
        dag_id="dataops",
        description="DataOps workflows.",
        default_args=default_args,
        schedule_interval=None,
        start_date=days_ago(2),
        tags=["dataops"],
    )
    def dataops():
        """DataOps workflows."""
        # Fill in below
        pass
    
    # Run DAG
    do = dataops()
    ```
    
- AF Integration with Airbyte (Extract)
```python
      # Start Airbyte
      git clone https://github.com/airbytehq/airbyte.git  # skip if already create in data-stack lesson
      cd airbyte
      
      pip install apache-airflow-providers-airbyte==3.1.0

      # Add Airbyte connection on Airflow Connection UI
        Connection ID: airbyte
        Connection Type: HTTP
        Host: localhost
        Port: 8000
        
      # Airflow code using AirbyteTriggerSyncOperator  
      @dag(...)
        def dataops():
            """Production DataOps workflows."""
            # Extract + Load
            extract_and_load_projects = AirbyteTriggerSyncOperator(
                task_id="extract_and_load_projects",
                airbyte_conn_id="airbyte",
                connection_id="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",  # REPLACE
                asynchronous=False,
                timeout=3600,
                wait_seconds=3,
            )
            extract_and_load_tags = AirbyteTriggerSyncOperator(
                task_id="extract_and_load_tags",
                airbyte_conn_id="airbyte",
                connection_id="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",  # REPLACE
                asynchronous=False,
                timeout=3600,
                wait_seconds=3,
            )
        
            # Define DAG
            extract_and_load_projects
            extract_and_load_tags  
    # Go to Airbyte webserver, connections, to find the CONNECTION ID
    # NOTE: Airbyte server defines the extract job, AF is just communicated with it
```      


- AF Integration with  Validate (Great Expectation (ge))
  - Use Great Expectation to validate data on Google BigQuery DW  
  ```commandline
    # Setup
    pip install airflow-provider-great-expectations==0.1.1 great-expectations==0.15.19
    great_expectations init
        tests/great_expectations/
              ├── checkpoints/
              ├── expectations/
              ├── plugins/
              ├── uncommitted/
              ├── .gitignore
              └── great_expectations.yml
    #  Big Query Data Source setup 
    pip install pybigquery==0.10.2 sqlalchemy_bigquery==1.4.4
    export GOOGLE_APPLICATION_CREDENTIALS=/Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json  # REPLACE
    great_expectations datasource new
        Enter values on terminal ...
  
    find ge checkpoint
        great_expectations checkpoint new projects

    # Airflow Code
    GE_ROOT_DIR = Path(BASE_DIR, "great_expectations")
    @dag(...)
    def dataops():
        ...
        validate_projects = GreatExpectationsOperator(
            task_id="validate_projects",
            checkpoint_name="projects",
            data_context_root_dir=GE_ROOT_DIR,
            fail_task_on_validation_failure=True,
        )
        validate_tags = GreatExpectationsOperator(
            task_id="validate_tags",
            checkpoint_name="tags",
            data_context_root_dir=GE_ROOT_DIR,
            fail_task_on_validation_failure=True,
        )
    
        # Define DAG
        extract_and_load_projects >> validate_projects
        extract_and_load_tags >> validate_tags
  ```
  
- AF Integration with  dbt (Transform)
    ```commandline
    Skipped ...
    ```

## Feature Store
- Feature store is a centralized repository of features that can be shared across teams and projects
- Do we need a feature store ?
  - Ideal when an entity's (user) feature needs to up to date
    - example
      - Entity: User  
      - Features: clicks, purchases
  - Factors
    - Feature Compute Duplication
      - if there are only a couple of ML applications where ML features is not shared across apps
    - Skew 
      - is the feature updated frequently?
    - Benefit
      - Feature Store can simplify app/pipeline by having a singular interaction source, vs interacting with stream, batch, db, etc..

### Tool: Feast

- Data Flow + Feast components 
  - [doc](https://madewithml.com/courses/mlops/feature-store/#feast)
  - Components
    - Registry : metadata store for feature definitions
    - online store db (redis, sql) that stores the LATEST features for defined entities
  - Data Flow
    - ```terminal
      Spark|DBT|app writes --> Data Warehouse (blob, DB, etc..)
          --> Offline Store (BigQuery)
                --> materialize --> Online Store (REdis/SqlLite) --> get_online_features() + registry --> serving data
                --> get_historical_features() + registry --> training data
     
- Setup 
  - ```terminal
    # Install Feast and dependencies
    pip install feast==0.10.5 PyYAML==5.3.1 -q

    mkdir -p stores/feature
    mkdir -p data
    feast init --minimal --template local features
    cd features
    touch features.py
        features/
                ├── feature_store.yaml  - configuration
                └── features.py         - feature definitions
    
    # Update feature_store.yaml
        project: features
        registry: ../stores/feature/registry.db
        provider: local
        online_store:
          path: ../stores/feature/online_store.db

  
- Define Feast Data Source
  - Options: 
    - file (parquet only)
    - data warehouse (Big Query)
    - data stream (Kafka, Kinesis)
  - Code: ingest from parquet
    - ```python  
      import os
      import pandas as pd
      # Load labeled projects
      projects = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv")
      tags = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv")
      df = pd.merge(projects, tags, on="id")
      df["text"] = df.title + " " + df.description
      df.drop(["title", "description"], axis=1, inplace=True)
      df.head(5)

      # Format timestamp
      df.created_on = pd.to_datetime(df.created_on)
   
      # Convert to parquet
      DATA_DIR = Path(os.getcwd(), "data")
      df.to_parquet(
         Path(DATA_DIR, "features.parquet"),
         compression=None,
         allow_truncated_timestamps=True)
)

- Define Feast Feature Definition
  - ```python
    from datetime import datetime
    from pathlib import Path
    from feast import Entity, Feature, FeatureView, ValueType
    from feast.data_source import FileSource
    from google.protobuf.duration_pb2 import Duration
  
    # Step1: Define location of feature (in our case FilreSource parquet file) and timestemp column
    START_TIME = "2020-02-17"
    project_details = FileSource(
        path=str(Path(DATA_DIR, "features.parquet")),
        event_timestamp_column="created_on",)
        
    # Step2: Define Entity
    project = Entity(
       name="id",
       value_type=ValueType.INT64,
       description="project id",)

    # Step3: create FeatureView that loads specifc features
    project_details_view = FeatureView(
       name="project_details", entities=["id"],
       ttl=Duration(seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60),
       features=[
         Feature(name="text", dtype=ValueType.STRING).
         Feature(name="tag", dtype=ValueType.STRING),],
       online=True,
       input=project_details,
       tags={})
  

  
- Use case: fetch historical feature values
  - ```python
    import pandas as pd
    from feast import FeatureStore

    # Identify entities
    project_ids = df.id[0:3].to_list()
    now = datetime.now()
    timestamps = [datetime(now.year, now.month, now.day)]*len(project_ids)
    entity_df = pd.DataFrame.from_dict({"id": project_ids, "event_timestamp": timestamps})
    entity_df.head()

    # Get historical features
    store = FeatureStore(repo_path="features")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        feature_refs=["project_details:text", "project_details:tag"],
    ).to_df()
    training_df.head()

- Use Case: Materialize : updates the online store
  - ```commandline
    cd features
    CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
    feast materialize-incremental $CURRENT_TIME

- Use Case: Fetch Online Feature Values
  - ```python
    # Get online features
    store = FeatureStore(repo_path="features")
    feature_vector = store.get_online_features(
        feature_refs=["project_details:text", "project_details:tag"],
        entity_rows=[{"id": 6}],
    ).to_dict()
    feature_vector
