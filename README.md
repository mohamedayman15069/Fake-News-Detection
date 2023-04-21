# Fake News Detection

## Introduction

The proliferation of fake news has become a significant issue in today's digital age. In response, we sought to develop an automatic machine learning classification model to combat its dissemination. Through a rigorous experimentation process, we found that neural networks had the highest accuracy in identifying fake news. We explored different hyperparameters, and achieved 69% accuracy in our implementation and 72% accuracy using the Sklearn MLP classifier. To enable word training in neural networks, we utilized Skip gram Word2Vec to obtain numerical representations for each word. Additionally, we used web applications to help communicate our understanding of fake news to the general public.

## Dataset
For this project, we used the Fakeddit dataset, a multimodal dataset with over 1 million samples, consisting of fine-grained categorization, metadata, and comment data. Although the dataset includes both text and image data, we relied solely on textual data in this project. The dataset is available for download at https://fakeddit.netlify.app/.

## Phases Explanation
The project consisted of five phases, each with a specific objective in mind:

### Phase 1: Literature Review
In this phase, we explored various state-of-the-art models and their use in improving the accuracy of fake news detection. We conducted a literature review on this problem to inform our approach.

### Phase 2: Dataset Preparation
We checked the consistency of the Fakeddit dataset, cleaned and preprocessed it, and made it ready for training. This phase ensured that the dataset was ready for use in our experiments.

### Phase 3: Model Comparison
In this phase, we experimented with different machine learning models and compared their accuracies, recalls, and precisions. We used this comparison to identify the most effective model for fake news detection.

### Phase 4: Model Development
We implemented a complete Skip gram Word2Vec and Neural Network from scratch in this phase. The neural network was trained on the preprocessed Fakeddit dataset to detect fake news.

### Phase 5: Implementation
In this final phase, we developed a frontend, backend, and a database system to make the model available for use. With this system, users can input text to be evaluated by the model, which outputs the likelihood of the input being fake news.


## Usage
To use our fake news detection model, download the code from Section 4, and run it on your local machine. Input the text you want to evaluate, and the model will output the likelihood of the text being fake news.

## Conclusion
We believe that our machine learning model, trained on the Fakeddit dataset and utilizing neural networks and Skip gram Word2Vec, can be used to combat the widespread dissemination of fake news. Our work in this project represents a significant contribution to the fight against fake news and its negative effects on society.

### Please note that the primary objective of this project is to implement various model blocks from scratch, without relying on any external libraries.
