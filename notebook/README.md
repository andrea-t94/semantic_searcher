# Exploratory Data Analysis
This analysis aims to extract meaningful insights to optimise model training and service

## Insights
I will summarise here the main insights from the analysis:
1. the dataset is huge to be run locally, therefore the need to down-sample it
2. the dataset format contains triplets of (query, positive, negative), which requires *MultipleNegativesRankingLoss* or *TripletLoss*
3. even considering only passage titles, the dataset is meant for asymmetric semantic search, where you usually have a short query and you want to find a longer paragraph answering the query. **I will use pre-trained MS-MARCO models**
4. the text lenght is short: the 99th percentile requires less than 64 tokens. Since I am running locally I will set the maax sequence lenght to 64 for both training and inference
5. the test dataset doesn't contain a positive or negative passage. I will split the dev dataset into validation (for training) and test (for evaluation)

## How to run the notebook
Assuming you have set-up your local environment described in the README of the parent directory, you only need to run
```make jupyter ```.