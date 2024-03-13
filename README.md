Identifying Gender Bias in Social Media Posts
Daniel Atallah, Jayme Fisher, Courtney Gibson

1. INTRODUCTION
Researchers have observed implicit biases—unconscious attitudes and behaviors associated with specific groups—in a number of contexts, from education to health to housing (Staats et al., 2017). Unlike explicit biases, which are deliberate and conscious, implicit biases are subtle, operating without awareness. Nonetheless, these biases have tangible, real-world consequences. Audit studies have revealed gender-based discrimination in high-priced restaurants and law firms (Bertrand & Duflo, 2017; Rivera & Tilcsik, 2016). Similarly, correspondence studies have revealed differential treatment in the labor market (Bertrand & Duflo, 2017). 
Though effective in identifying bias, audit and correspondence studies are less practical for mitigation due to the significant time and resources required. Language offers a practical alternative to audit and correspondence studies. Research has shown that linguistic analyses can effectively reveal biases, and that these analyses correlate with traditional implicit bias measurements (Bhatia & Walasek, 2023). However, most existing research focuses on bias at the level of text corpora. Our objective is to identify document-level bias, facilitating targeted mitigation strategies. Specifically, we have employed three distinct methods to explore gender-related implicit bias. First, we used supervised learning algorithms—k-Nearest Neighbors, Naive Bayes, and Extreme Gradient Boosting—to predict the gender of a poster based on response text. Then we identified the words and phrases that influenced classification. Second, we developed word embedding models to assess the spatial relationships between vectors representing men and women. Finally, we used dimensionality reduction to determine whether topics, rather than words, are better indicators of gender bias.
All three methods yielded words and phrases that are potentially indicative of gender bias, showing promise for future bias mitigation efforts. However, additional research and refinement are needed before it is possible to deploy a solution addressing gender bias.
2. RELATED WORK
Using machine learning and natural language processing to mitigate individual bias is not an entirely new concept. Wayfair, an e-commerce company, developed the Bias Analyzer, a tool for mitigating bias in performance reviews (Schmiedl, 2021). This tool, which highlights biased words and phrases to provide real-time feedback to managers, has contributed to an increase in women in leadership positions. However, the tool is only available to Wayfair, and the absence of published methodologies limits broader evaluation and application. 
At Facebook, researchers have used large language models to detect bias. For example, Dinan et al. (2020) trained a transformer model to identify three distinct forms of gender bias. While effective in identifying bias, the complexity of these models impedes explainability. Our project seeks to bridge this gap by employing supervised learning models that offer clearer insights into the underlying reasoning, providing the potential for education alongside identification.
The downside of supervised learning methods is that these methods require annotated datasets. Garg et al. (2018) propose an alternative approach: word embeddings. Instead of relying on annotated data, Garg et al. measure the distance between gendered nouns and pronouns in vector space, providing a scalable and efficient approach to bias detection. Building on this foundation, our project identifies a broader range of words and phrases that are potentially indicative of bias.
In short, while there have been promising developments in the field of bias identification, limitations remain. Our project addresses these gaps by broadening the application of bias detection methodologies and enhancing their accessibility and explainability. Through this work, we aim to contribute a valuable resource to the ongoing efforts to understand and mitigate bias in various contexts.
3. DATA
Annotated data is a necessary prerequisite for any supervised learning task. We will be using the social media dataset developed by Voigt et al. (2018), in which each post is flagged as having been written by a man or woman.The RtGender dataset (https://nlp.stanford.edu/robvoigt/rtgender/) contains data from various social networking sources. Separate “posts” and “responses” files exist for each of these sources. The “posts” files contain indices to identify the original posters, the original posters’ genders, indices to identify the posts, and the text within the posts. The “responses” files contain indices to identify the original posters, the original posters’ genders, indices to identify the original posts, indices to identify the responders, the text within the responses, and the responders’ genders. The methodology for labeling the original poster’s gender can be found in the dataset documentation (https://nlp.stanford.edu/robvoigt/rtgender/rtgender.pdf) – different methodologies exist for each source. Only two genders are represented with M indicating men and W indicating women. There is no specified date range for when these posts were recorded however we can assume, based on the platforms studied, that they were recorded somewhat recently (since the 2010s). 
The social networking sources included are Facebook, TED, Fitocracy, and Reddit. There are two sources for Facebook – Facebook wiki contains posts by public figures and Facebook congress contains posts by U.S. politicians. Lastly, there is an “annotations” file that contains about fifteen thousand posts and responses with each source represented. We used the annotations file as a lightweight dataset to familiarize ourselves with the data and test some of our preprocessing steps in Google Colab.
The original posters’ genders and the response text were the two most valuable fields for our analysis. A majority of preprocessing needed to be done to the response text columns as they are collections of strings. It is standard practice in a natural language processing (NLP) project to align the case of the text, remove stopwords (words that offer little information) from the text, and lemmatize (convert to base form) of the words in the text. We used variations of stopword removal and tokenization throughout our project.
4. SUPERVISED LEARNING
4.1. Features
The steps in our feature engineering pipeline for the supervised learning models included the following. The team created source IDs by tagging each response data source with an identifier that signified the original social media platform the data was generated from. Then we merged the source data for the five files related to user responses and mapped the gender labels to binary values by converting ‘W’ and ‘M’ labels to the binary values, 0 & 1 respectively. The full RtGender responses dataset accumulated to twenty six million records. We identified a class imbalance between the gender labels and a data source imbalance between the five files that create RtGender. For these reasons we implemented a sampling process. Our methods included balancing the data source stratification to reduce overfitting on a particular underlying data source (e.g. Facebook, Fitocracy, Reddit). Then, we balanced the gender labels by upsampling the minority class of original posters, which happened to be women. 
After splitting the sampled data into training and testing buckets we preprocessed the response data using the following steps: lowercasing, lemmatization, stop word & special character removal. The tokenized data was then categorized using the spaCy english parts of speech tags. During the hyperparameter testing phase, the team iteratively tested implementation of unigram and bigram feature representations with TF-IDF and count vectorization. 
The record count for the supervised learning dataset was 386,712 records. The vectorized text data features that the team trained the supervised learning models on were extensive. Therefore, we are providing a list of the top fifty most important features produced by the feature_importances_ function from the XGBoost classifier library.
4.2. Methods
During this experiment we tested three machine learning models before identifying our best performing model. The supervised learning algorithms used to identify gender labels from response post data included:
k-Nearest Neighbors (KNN) classifier: The KNN classification algorithm employs instance-based learning to determine the class of a record based on the average value of its k-nearest neighbors. This simple model serves as a baseline for testing, and we implemented the KNN algorithm using the ‘neighbors’ module from the scikit learn python library.
Naive Bayes (NB classifier): The Naive Bayes classifier is a probabilistic machine learning algorithm that is based on Bayes’ theorem. This classifier is known to work well with high-dimensional data. We implemented the NB classification algorithm using the scikit learn python library.
Extreme Gradient Boosting (XGB) classifier: The XGB classification algorithm is an ensemble method that trains learners sequentially, iteratively correcting the errors of each previous learner. This model is known for being efficient and well suited to highly dimensional and sparse data. We implemented the XGB algorithm using the tree based ‘XGBClassifier’ model from the xgboost python library.
Our supervised learning hyperparameter tuning pipeline emphasized an iterative process using a dynamic function to test the efficacy of each model type (KNN, XGB, NB), vectorizer (Count & TF-IDF), and feature representation (Unigram & bigram) combination for a total of twelve model comparisons. We began pipeline creation using the annotations dataset for quick setup and transitioned to our larger dataset after hyperparameter tuning the twelve model options. Our model training and hyperparameter tuning method consisted of implementing the train, validation, test splitting technique with an allocation of 70% training, 15% validation, and 15% testing data. Hyperparameter tuning was implemented through random search using a Stratified k-Fold technique with five splits to reduce overfitting while balancing computational expenditures. We evaluated the hyperparameter grid using a combination of F-1 scoring and AUC-PR. 
The team identified a class imbalance within the annotations dataset and decided to implement F-1 cross validation scoring. Additionally, in our use case precision and recall were equally important because both the false positive and false negative scenarios would be equally harmful to model performance. We also explored using the AUC-PR metric for scoring because AUC-PR is more effective than AUC-ROC or F-1 when positive instances are the minority class, which was the case in the annotations dataset. Once again we tested all twelve models with random search, this time scoring on AUC-PR. We saw improved performance generally for all model combinations. Using random search on a subset of the larger dataset allowed our team to identify a starting place for each of the twelve models. 
After we identified the best hyperparameter combination and feature representation for each model category we began building analysis pipelines using the entire RtGender dataset. The team quickly realized that sampling the data would be necessary due to computational constraints. Another issue we identified was the composition of the RtGender source data. When viewing the feature importance dictionary from the XGBoost model, we noticed that congress person names and fitness buzzwords were overrepresented in the list of the most important features. This happened because the facebook congress and fitocracy sources were overrepresented compared to the other RtGender data sources. 
The class imbalance and source imbalance prompted our team to employ a sampling method that limited the dataset size while balancing the gender label records and data source composition. Using a process of sampling the larger data sources while maintaining all records from the smaller data sources we deployed an 80/20 train test split with stratification on the balance data source composition. Then, we randomized the order of the final balanced training dataset to mitigate potential biases. The team tested multiple sample sizes including ~400k, 700k, as well as one and two million records. The improvements in model performance were insignificant with larger samples. Whereas the decrease in computing efficiency was significant with datasets over ~400k records. We ultimately decided to conduct the final analysis using the ~400k record sample.
4.3. Evaluation
We assessed our models through stratified five-fold cross-validation, utilizing the mean and standard deviation of the F-1 score. The team decided to use F-1 scoring following the balancing of gender labels, as positive and negative misclassifications carried equal weight in our use case. Consequently, the significance of the AUC-PR metric diminished and the F-1 score became preferable due to its ability to strike a balance as the harmonic mean of precision and recall. 
Table 4.3. Model Performance Comparison
Metric
k-Nearest Neighbors
Naive Bayes
 Extreme Gradient Boosting
F-1
0.682 ± 0.004 
0.791 ± 0.001
0.794  ± 0.000

After comparing the results of the model evaluations the team opted to further analyze the Extreme Gradient Boosting model in further detail.
4.3.1. Feature Analysis
Given that vectorized text data can be difficult to analyze using ablation testing, our team employed spaCy part of speech (POS) tagging on the tokenized text data for feature categorization. We created TF-IDF vectorized matrices with both the word features and POS tags as inputs for the XGBoost model ablation testing. Our ablation testing process included training the XGB model with one part of speech removed on each iteration. The team also trained the XGB model using the word features and all POS tags and compared this model to the ablation models. The spaCy parts of speech tags include the following categories:  Noun, Proper Noun, Pronoun, Adjective, Verb, Adverb, Adposition, Conjunction, Determiner, Numeral, Particle, Interjection, Punctuation, Symbol, and Other. The team has provided a chart located in the appendix that includes the POS categories, their descriptions, as well as the F-1 mean and standard deviation of each ablation step and the impact in comparison to the hyperparameter tuned XGB model trained on all POS categories. The POS categories that had the most significant impact on the model included nouns, verbs, and adjectives. The least impactful categories were determined to be coordinating conjunctions, punctuation, and symbols. These results aligned with our expectations as we removed symbols and punctuations during the text preprocessing stage. Additionally, nouns are often the subjects of speech in a sentence while verbs and adjectives describe nouns and can be significantly related to implicit bias.

Table 4.3.1. POS Ablation Analysis
Part of Speech Tag
F-1
Impact
NOUN
0.726 ± 0.004
0.068
VERB
0.729 ± 0.003
0.065
ADJ
0.760 ± 0.002
0.034

4.3.2. Sensitivity Analysis		
The team conducted a hyperparameter sensitivity analysis to deepen our understanding of the XGB model's ability to generalize predictions on unseen data. The hyperparameters we tested included n_estimators, learning rate, max_depth, subsample, and colsample_bytree. We utilized an iterative approach, holding all but one parameter constant for each iteration. A summary chart is included in the appendix with the F-1 score for the five-fold cross-validation mean and standard deviation of each test step in our analysis. Overall, we found that our model was only marginally sensitive to hyperparameter changes in each iteration, and the F-1 score standard deviations were no larger than 0.001. The results indicated to the team that our model was relatively consistent across each iteration, and the model would likely generalize well on unseen data.
Among all the hyperparameters tested, the n_estimators and learning rate parameters created the most significant model improvement magnitudes. Starting with 100 estimators caused the model's performance to drop by 0.0017 from the hyperparameter-tuned model. Adding 100 estimators to the n_estimators parameter improved the F-1 mean of the model by 0.003. Additionally, adjusting the learning rate from 0.01 to 0.1 had the most substantial impact on the model, with the F-1 mean increasing from 0.788 to 0.796. These examples represent the most significant improvements in the model; however, they offer only marginal increases in the F-1 score.	
4.3.3 Model Enhancement Analysis
Initiating a new machine learning project often involves a series of iterative trial-and-error steps. In the context of our project, we found that a significant number of steps were either discarded or deprioritized, surpassed by the steps that ultimately made it into the final process. With machine learning there will always be a needed balance between accuracy and efficiency due to the technical limitations of computational processing.
During our hyperparameter tuning phase, we tested many variations of vectorization and feature representations using unigrams and bigrams. For this initial phase, we chose to construct the process using the subset of annotated data. This choice enabled the team to efficiently tune twelve machine learning model combinations. In the case of the Naive Bayes and XGB models, we observed that unigram feature representations consistently outperformed bigram representations across all combinations of hyperparameters and vectorization strategies. However, the K-Nearest Neighbors model consistently demonstrated improved performance with bigram feature representations. 
After completing hyperparameter tuning, we proceeded to test large samples from the full RtGender dataset. Unfortunately, we encountered issues when using bigram representations in the KNN model fitting, which failed to complete with any sample exceeding 100k records. We decided to test unigram features on the KNN model and we successfully fitted the model using sample sizes of approximately 400k, 700k, and millions of training records. The performance of the model did not significantly decrease with the use of unigram representations. Consequently, we made the decision to continue training the final KNN model using unigram feature representations rather than utilizing a different sample dataset, to align the training data between models.
4.4. Failure Analysis
4.4.1 Feature Importances
Our XGB model underwent three types of error analysis, leading to the implementation of corrections based on the identified errors in one category. The initial error analysis began as a feature importance analysis. After training our original XGB model using POS tagging and TF-IDF vectorization on approximately 400k records, we observed unusual features being prioritized in the feature importance analysis. We identified two categories of features that might have been impeding our model's accuracy in predicting gender labels.
The first category pertained to political data, specifically names of congress people and political buzzwords. The second category included health and wellness features such as 'team,' 'fitness,' 'journey,' 'exercise,' 'welcome,' and 'thanks.' To delve deeper into these issues, we investigated our source data. The team discovered two divergent issues contributing to the unusual feature importances. Firstly, the Facebook_congress data source was significantly overrepresented due to its larger record size compared to other sources. Conversely, the Fitocracy data did not represent a substantial portion of the overall dataset. However, due to the nature of the social media platform, Fitocracy exhibited a concentration of posts related to joining the community, expressing exercise goals, and welcoming others.
In response to these issues, our team took a two step approach. First we decided to balance the datasource sampling to prevent the model from overfitting on congress Facebook posts or larger datasets in general. The second step involved removing additional stopwords from the data related to politicians' names and fitness/Fitocracy buzzwords. Although this approach did not significantly impact model performance, the feature importance analysis results did improve. The results indicated that the important features were likely to contribute to enhanced generalization on unseen data. Specifically, words like 'he,' 'she,' 'beautiful,' 'victorious,' and 'introvert' became more important. However, the most important word, 'consistent,' appeared to be a remnant of the Fitocracy issue and is a candidate for removal in future model retrainings.
4.4.2 Part of Speech Tagging
In the second error analysis conducted by our team, we focused on identifying the most frequently occurring parts of speech (POS) in instances of both correctly classified and misclassified labels. The analysis process involved the following steps: Identifying the correctly and incorrectly classified records. Creating a Bag of Words (BOW) for each category; correct/incorrect classification. Calculating the frequencies of each POS in the categories and aggregating the counts for further analysis. Normalizing the POS frequencies by considering the overall distribution of POS tags in the dataset. Applying min-max scaling to the normalized results, ensuring consistency in the scale of the analyzed data. We can see both the misclassified and classified categories are similarly impacted by the ADP, SCONJ, AUX, and DET. The part of speech comparison ultimately did not provide much insight into the reasons prediction errors were occurring. This conclusion led the team into exploring the specific samples related to prediction errors.   

4.4.3 Response Subjects
The final assessment in our team's supervised pipeline examination involved a detailed review of the vectorized text data for specific misclassified records. This process entailed examining a sample of incorrectly predicted results and comparing the true label with the predicted label. During this analysis, a noteworthy pattern emerged: errors often occurred when the true label differed from the 'subject' of a response.
Below, two illustrative examples highlight this pattern. In the first instance, the subject of the response is not the original poster but rather the 'president' with the pronoun ‘he’. Conversely, in the second example, despite the original poster being male, the response references 'she' and 'Clinton,' likely referring to Hillary Clinton. Strategies for addressing these errors will be further considered in the discussion section.
True Label: W, Predicted Label: M
Vectorized Text:
“president job write program cyber security he continue working fix problem program work giving job congress make american safe president responsibility alone president working poor middle class word job moving america forward instead worrying right wing base”
True Label: M, Predicted Label: W
Vectorized Text:
“cheated perhaps even threatened presidency clinton screwed big time she known fair trump better alternative today clinton charge world would come swift harsh end american nonamericans alike she start raining nuclear bomb continent”
5. UNSUPERVISED LEARNING
In addition to assessing three supervised models, we also explored two unsupervised methods for identifying bias. The first leverages the foundational work of researchers such as Garg et al. (2018), employing a bag-of-words model informed by word embeddings. Unlike the supervised approach, this method does not rely on the gender of the original poster, focusing instead on how men and women are represented through spatial relationships in word embeddings. Our second method shifts the focus from individual words to clusters of words, using dimensionality reduction and clustering to determine whether clusters are better at capturing the nuances of implicit bias than words. Together, these unsupervised methods enrich our investigation into implicit gender bias, probing the textual landscape of social media posts to uncover bias at both the word and concept levels.

5.1. Word Embeddings
5.1.1 Features
To prepare our data for unsupervised learning, we performed several preprocessing steps. First, we added a source column and a unique identifier to each record. We then removed entries without a response and excluded quotes and double quotes from the response text to preempt encoding issues. Following this, we resampled the data with replacement to balance it across both the sources and the gender of the original posters. This resulted in three datasets: one with five million posts, another with three million posts, and the last with one million posts. These datasets were divided into training, validation, and testing sets and saved as files, each with four columns: source, source id, original poster gender, and response text. For efficiency, these preprocessing steps were executed using Git Bash.
Next, we tokenized the response text using the TweetTokenizer tool from the Natural Language Toolkit (NLTK), following the approach used by Nguyen et al. (2020) for English Tweets. TweetTokenizer was selected for its ability to preserve emojis and hashtags and to normalize words with excessive letters, reflecting the expressive nature of social media communication. During tokenization, we applied several normalization techniques: converting text to lowercase for consistency, reducing repeated characters to standardize emphasis, and stripping user handles to limit personally identifiable information. In addition, we explored the effects of lemmatization and stop word removal on our dataset. Initial analysis, based on a modified silhouette score, showed that lemmatizing responses and retaining stop words produced the best results. The silhouette score is discussed in greater detail in the method and evaluation sections.
5.1.2. Methods
Many researchers have used word embeddings to explore bias, including Garg et al. (2018), Manzini et al. (2019), and Zhao et al. (2018), among others. Where Manzini et al. (2019) use analogies to detect bias, our research follows the approach proposed by Garg et al. (2018), measuring the relative distance between word vectors and gendered representations, developed using nouns and pronouns from the HolisticBias dataset (Smith et al., 2022).
Among the algorithms available for training word embeddings, Word2Vec, introduced by Mikolov et al. (2013), stands out for its balance of simplicity and effectiveness. Word2Vec offers two architectures: Continuous Bag of Words (CBOW) and Continuous Skip-Gram. CBOW predicts a target word from a set of context words, whereas Continuous Skip-Gram predicts context words from a target word. Both versions produce a dense matrix, facilitating the analysis of semantic and syntactic word relationships.
In tuning the model, we assessed several considerations, including the source of the data, the approach to tokenization, several hyperparameters, and the number of records. A modified silhouette score, traditionally used in cluster analysis, served as our evaluation metric. This score assesses model quality by comparing the cohesion and separation of gendered noun and pronoun pairs. The higher the score, the better the model. To compute the silhouette score, we first determined the cosine similarity between pairs of gendered nouns and pronouns. We then averaged this similarity both within and across genders, converting these averages into a distance measure. Finally, we derived the score by subtracting the between-gender distance from the within-gender distance and dividing by the maximum of these distances.
Based on the silhouette score, some models trained on single sources outperform those trained on multiple sources. However, the results are inconsistent (Table 5.1.2.1). Men are represented more cohesively in the Facebook Congress and TED models than the Facebook Wiki, Fitocracy, and Reddit models.

Table 5.1.2.1. Silhouette scores for models trained on the one-million-response training dataset varying the source of the data. 
Source
Men
Women
Average
All
0.112
0.194
0.153
Facebook Congress
0.153
0.209
0.181
Facebook Wiki
0.083
0.221
0.152
Fitocracy
0.062
0.167
0.115
Reddit
0.014
0.204
0.109
TED
0.158
0.204
0.181


The influence of lemmatization and stop word removal is more subtle. There is virtually no difference between the silhouette scores for the three most successful models (Table 5.1.2.2). Thus, to facilitate interpretation, tokens were lemmatized and stopwords retained.
Table 5.1.2.2. Silhouette scores for models trained on the one-million-response training dataset with and without lemmatization and stopwords. 
Lemma
Stopwords
Men
Women
Average
False
False
0.127
0.194
0.161
False
True
0.116
0.187
0.152
True
False
0.112
0.194
0.153
True
True
0.097
0.196
0.146


Subsequent analysis focused on optimizing the number of vectors, learning rate, window size, and the choice between algorithms. Each variable was systematically varied to identify the optimal settings, balancing model performance with computational efficiency. The final model returns 100 vectors, has a learning rate of 0.025, a window size of 15, and uses the Continuous Skip-Gram algorithm with Negative Sampling. 
5.1.3. Evaluation
Mapping the similarity between gendered nouns and pronouns allows us to explore the vector space (Figure 5.1.3.1). Ideally, gender-related terms would exhibit high similarity scores within their respective quadrants, signifying clear differentiation, while maintaining low similarity scores across genders. There is a modest positive correlation within genders, suggesting gender cohesion. Crucially, the model also discerns syntactic differences (nouns and pronouns) as well as semantic differences (familial relationships). 
To evaluate words that are not inherently gendered, we use the bias score proposed by Garg et al. (2018) measuring the relative distance between word vectors and gendered representations. The bias scores across our vocabulary approximate a normal distribution, centered around a mean bias of 0.023 with a standard deviation of 0.061 (Figure 5.1.3.2). This yields 315 words with a significant male bias and 327 with a female bias at the 95% confidence level. 
The prevalence of nouns and verbs among these biased words mirrors their linguistic frequency. However, adjectives are disproportionately represented among female biased words, suggesting their potential as bias indicators (Table 5.1.3.1). The most biased female adjectives include “powerful,” “dense,” and “beautiful,” while the most biased male adjectives include “spineless,” “nuclear,” and “foolish.” Additional adjectives are available in the appendix.
Table 5.1.3.1. Part of speech frequency among male and female biased words.
Part of Speech
Share of Male Biased Words
Index to Model Vocabulary
Share of Female Biased Words
Index to Model Vocabulary
Noun
38.6%
103
39.2%
105
Verb
31.4%
103
22.6%
74
Proper Noun
18.8%
123
11.8%
77
Adjective
9.2%
80
21.3%
184
Adverb
2.0%
37
5.1%
96


It is also possible to evaluate embeddings with analogies. While our model correctly interprets the analogy “he is to boy as she is to girl,” it also produces unexpected results, such as equating “he is to husband” with “she is to aunt.” Among a set of noun and pronoun analogies developed using the HolisticBias dataset, our model predicts 36% correctly. 
Notably, volume had a measurable impact on performance. Comparing the silhouette score for the models trained on datasets of one million, three million, and five million responses reveals marginal improvement (Table 5.1.3.2). More compelling is the improvement in analogy resolution, with accuracy climbing from 25% in the one-million-response model to 36% in the five-million-response model. This suggests that increasing the number of responses—and thus the contexts in which gendered nouns and pronouns appear—is critical in improving model performance. 

Table 5.1.3.2. Silhouette scores for models trained on the one-million-response, three-million-response, and five-million-response training datasets. 
Records
Men
Women
Average
One Million
0.122
0.152
0.137
Three Million
0.074
0.227
0.151
Five Million
0.115
0.194
0.155


5.2 Clustering
5.2.1. Features
The team incorporated the findings from our prior supervised and unsupervised approaches to inform feature engineering for document clustering. The one-million-response train sample created using Git Bash (explained in section 5.1.1) was used for all clustering model exploration. This was done to prioritize speed and resources with the eventual goal of scaling our best models to a larger dataset. Links, punctuation, and stop words were all removed. We used the standard English stopwords list but retained pronouns as they can be indicative of gender. Lastly, we mirrored how we tokenized text for our word embeddings (using NLTK’s TweetTokenizer) 
Count vectors, TF-IDF vectors, and word embeddings were all used as viable feature representations during model tuning. The count and TF-IDF vectors were generated from the training data using tri-grams and a minimum term frequency of 5. To accommodate for memory constraints, the sparse count and TF-IDF matrices were reduced to 100 dimensions using Scikit-Learn’s TruncatedSVD module. We used the Word2Vec model that was trained on the one-million-response train sample where stopwords were kept and tokens were not lemmatized (see Table 5.1.2.2) as our word embeddings. Each document in the corpus was vectorized using the average of the vectorized embeddings of the words within it. 
5.2.2. Methods
The initial goal of our clustering implementation was to identify two clusters that ideally coincided with the two genders represented in our sample. For this reason, we chose clustering algorithms where the number of clusters or components could be set. KMeans was chosen as a baseline model. Hierarchical clustering was chosen for its interpretability, however we ultimately stopped pursuing it due to its tendency to surpass memory capacity. Lastly, Gaussian Mixture (GMM) and Bayesian Gaussian Mixture (BGMM) models were chosen for their ability to handle noise (common in social media corpora) by using the Gaussian distribution and, in the case of BGMM, incorporating prior distributions.
Scikit Learn’s GridSearchCV was implemented for 5-fold cross-validation hyperparameter tuning. A separate grid search was fit for each combination of model and term-matrix representation. We iterated over different combinations of the maximum iterations and tolerance parameters for KMeans and the covariance type, tolerance, maximum iterations, and regularization covariance parameters for both GMM and BGMM. Each implementation was compared over the same set of evaluation metrics.
5.2.3 Evaluation
The team was interested to see how the clusters aligned with the gender labels in our data. While these labels were not used in training the models, they were incorporated in our evaluation metrics. Our commitment to minimizing gender misrepresentation, driven by ethical considerations, led us to prioritize accuracy. Accuracy was defined as the highest percentage of either class within a cluster.. Additionally, due to some preliminary training, we were concerned our models would cluster at random. To avoid random assignment, we evaluated using the adjusted rand score where a maximal score would imply minimal randomness. While accuracy and adjusted rand score were our primary measures of performance, we included precision, recall, and adjusted mutual info score as additional measures for our reference. 
Unfortunately, the tuned models performed inline with our worst expectations. As seen in Table 5.2.3.1, our five most accurate models hovered around a 50% accuracy implying that a given class is identified in accordance with the ground-truth label for approximately 50% of documents in each cluster. This was reaffirmed by the low adjusted rand score, implying assignment of documents to clusters is almost completely random.
Table 5.2.3.1. Average accuracy and adjusted rand scores across parameters for top 5 performing clustering model and data representation combinations. 
Model
Data Representation
Avg. Accuracy
Avg. Adj. Rand Score
BGMM
Word2Vec
0.506925
0.065225
GMM
Word2Vec
0.506910
0.064955
BGMM
TF-IDF
0.501992
0.063413
GMM
TF-IDF
0.501958
0.062434
KMeans
Word2Vec
0.501710
0.009399


5.2.4 Altered Approach
The clustering evaluations rendered the binary cluster approach infeasible and elicited a new approach. Rather than forcing our models to converge for two clusters, we explored multiple clusters placing greater importance on the clusters with a dense makeup of one class. To mitigate for situations where responders discussed someone other than the original poster, we filtered our data to TED talk responses. The TED responses were typically directed to the subject within the TED talk. This limited the noise we were experiencing in our supervised learning models and word embeddings. Aside from filtering on source and vectorizing a TF-IDF matrix on the subsetted data, we copied all the other preprocessing steps from our original clustering approach. 
Considering that BGMM converged the least randomly of the clustering models in our initial approach, we opted to focus on a BGMM implementation. We conducted iterative training of a BGMM across a range of one to twenty components, maintaining default values for all other parameters. Each fitting was evaluated on the validation matrix using the average log-likelihood where a greater value means each sample is more likely to belong to its assigned cluster. Figure 5.2.4.1 illustrates our sensitivity analysis on average log-likelihood for each additional component in addition to the time to train. While the model with twenty components is the most performant it takes about thirty minutes to train. 
The trained model was used to predict the assignment of test documents into twenty components. Most of the resulting components were made up of an approximately equal number of responses to men and women. Three components contained responses to either men or women at least 60% of the time. 
Table 5.2.4.1. Components with significant TED speaker gender makeups. 
Predicted Cluster
Percent Men
Percent Women
Record Count
1
61.54%
38.46%
1191
14
60.01%
39.99%
3133
3
39.51%
60.49%
1096


Components one and fourteen were dominated by men speakers while component three was dominated by women. Using the posterior probabilities we found the documents with the largest weights for each component. We referenced our TF-IDF to take the average weights of all features within the significant documents for each component. The significant words for each component are shown below. Notice that in the context of TED talks, many of the words used in the men-dominated components are nouns and verbs while the women-dominated component has a lot of adjectives. 
Significant terms for component 1: ['ok', 'mr', 'opinion', 'cool', 'yes', 'wow', 'funny', 'oh', 'unfortunately', 'love', 'interesting', 'hello', 'idea', 'work', 'something', 'use']
Significant terms for component 14: ['im', 'comment', 'said', 'get', 'great talk', 'question', 'sorry', 'agree', 'book', 'point', 'year', 'life', 'game', 'go']
Significant terms for component 3: ['beautiful', 'inspiring', 'story', 'she', 'word', 'sharing', 'powerful', 'truly', 'absolutely', 'amazing talk', 'thanks', 'speech', 'touching', 'her', 'speaker', 'presentation', 'watched', 'ted talk', 'loved', 'thank sharing', 'inspirational', 'truly inspiring', 'best', 'thanks sharing', 'ive', 'job', 'well', 'seen', 'watching video', 'ever', 'heard', 'great ted', 'really inspiring', 'amazing speech', 'inspiring talk', 'brave']
5. DISCUSSION
While exploring our supervised learning models, the Fitocracy and Facebook Congress datasets were significant to our initial results. Words of gratitude and motivation were prevalent in our Fitocracy responses while the names of our Congress representatives had high frequency in our Facebook congress responses. The resulting important features included words like “thanks'' and “appreciate” as well as names like "Kamala" and "McCain.” We solved this in later models by removing some of these words and names that may not be generalizable in other contexts. This did not have a significant impact on model performance but did more appropriately extract the driving features of bias.
Another peculiarity of our data, that we did not have adequate time to identify a solution for, was the tendency of responses to address someone other than the original poster. This behavior was especially common in the Facebook congress data where responders may be referencing one of the original posters’ peers. This resulted in noise when the gender of the original poster and the person being mentioned in the response were different genders. 
In spite of the complexity of our data, the team refrained from using neural networks to maintain greater interpretability. We noticed that a related analysis used a Transformer model (Dinan et al., 2020)  for a similar purpose. The resulting analysis converged on a ~70% accuracy . This was encouraging as we worked towards a similar score using XGBoost, a model that can output easily interpretable artifacts like decision tree diagrams and feature importance lists.
Our team converged on multiple realizations through the analysis of both the unsupervised and supervised models. We tried to accommodate for source specific nuances by training a separate word2vec embedding for each source. To our surprise, we noticed lower silhouette scores for some sources when compared to the embedding of all sources. We infer that the diversity of sources may have added context for each term. Similarly, the added context of a larger search window seemed to be a significant contributor to silhouette scores compared to other word2vec levers like the number of vector dimensions. Our most notable discovery from using word2vec was how evenly weighted biased and unbiased terms were. This was a limitation of document-level vectorization as it made it difficult to discern between biased and unbiased responses. With more time we would have explored increasing the weights of those words that are more indicative of bias. 
Document-level unsupervised learning was an ongoing challenge when working with topic and clustering models. Our evaluation of clustering models using Adjusted Rand Score revealed that splitting documents into two gender-driven clusters is comparable to random choice. Our solution was to use more clusters and analyze the clusters that were dominated by a given class. We also attempted to use LDA to identify two gender-focused topics but experienced the same obstacle.We attempted to use Turbo Topics on top of the LDA to identify significant n-grams however we ran into several errors and ultimately decided to forego it due to time constraints. If granted more time we would pursue the successful implementation of Turbo Topics.
Despite the additional computational capacity of GLC, we noticed long runtimes and occasional kernel timeouts due to the large dimensionality of our term matrices. One solution we had for this was dimensionality reduction. Although this did improve speed and did not deteriorate performance, we were not fond of losing information at the expense of accurately representing the specific important terms. Our long term solution was to settle for a smaller sample. This ultimately worked well for us as we determined during our supervised sensitivity analysis that larger samples did not have a significant impact on our model performance.
The team sees a lot of opportunity for refinement. With more time we could have explored or manufactured a larger and more diverse dataset, tailored the POS tagger for social media text, and implemented systematic programs to flag bias. As of now our dataset is restricted to two genders. In its current state we would like to spend more time analyzing how the two genders communicate amongst each other (e.g. man to woman, woman to woman, etc.) We would also love to see what sort of language patterns transpire towards some of the more fluid genders. This may come naturally with a more recent dataset. We are also currently limited to the context of social media due to the availability of data and the ability to observe gender in this context. A more manual approach (perhaps appointed to a crowdworker on Amazon Mechanical Turks) may be necessary to label less noisy corpora. Additionally, we would have liked to experiment more with POS tagging, however we noticed that the POS tagger was another driver of noise due to its inaccuracies when parsing casual social media text – a trained POS tagger on social media may be quite an undertaking but it would produce better results for our use case. Lastly, we would love the ability to operationalize our models to dynamically flag implicit bias and make recommendations for word changes.
6. ETHICAL CONSIDERATIONS
Our goal with both our supervised and unsupervised models was to identify terms and phrases commonly used when addressing a specific gender. We approached this project with careful consideration because gender is a fundamental aspect of identity. To ensure ethical results, we prioritized privacy, interpretability, and fair representation.
We demonstrated commitment to privacy in our choice to use the RtGender datasets to train our models. While we did not create these datasets, we were confident in using them as personally identifiable information is removed –all identifiers like user ID’s and post ID’s have been stripped before our consumption of the data. The creators of the datasets used the gender that the observed subjects self-identified. 
The team hopes that our findings encourage others to more closely analyze the language they use. Conversely, we want the recipients of communication changes to understand the rationale behind them since sudden change may cause discomfort. We knew that our analysis must be easily digestible in order to get buy-in from communicators and recipients. This prompted us to prioritize the use of more interpretable models. 
Our most involved commitment to ethics was fair representation. We observed data imbalance on source and gender during our preliminary analysis. To address this, we implemented stratification and resampling in each modeling task to most evenly represent all classes. These proactive measures reduced bias in our models and enhanced their performance. We also selected our evaluation metrics based on how well they accommodate for data imbalance (i.e. F-1 score in our supervised portion).
It is exciting to witness society reevaluate gender norms. More people are recognizing the diverse experiences of gender – gender fluidity is a concept that empowers people to embrace their true identities. While our analysis focused on two genders (men and women), an expanded analysis on more genders is a potential area of opportunity – particularly in the context of fair representation and data ethics.
7. STATEMENT OF WORK
Daniel Atallah completed the data, discussion, ethical considerations sections of the report and contributed to the unsupervised learning section. He also developed the code for clustering and topic modeling approaches. 
Jayme Fisher completed the introduction and related work sections of the report and contributed to the unsupervised learning section. She also developed the code for word embeddings approach. 
Courtney Gibson completed the supervised learning sections of the report. They developed the code for preprocessing, vectorizing, tuning, and evaluating the supervised learning approach. Courtney also completed feature ablation, sensitivity, and error analysis.
8. REFERENCES
Bertrand, M., & Duflo, E. (2017). Field experiments on discrimination. In Handbook of Economic Field Experiments (pp. 309–393). https://doi.org/10.1016/bs.hefe.2016.08.004
Devlin, Jacob, et al. (2019) Bert: Pre-Training of Deep Bidirectional Transformers for Language Understanding. arXiv.Org, arxiv.org/abs/1810.04805 
Dinan, E., Fan, A., Wu, L., Weston, J., Kiela, D., & Williams, A. (2020). Multi-Dimensional Gender Bias Classification. arXiv. https://doi.org/10.18653/v1/2020.emnlp-main.23 
Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. PNAS, 115(16). https://doi.org/10.1073/pnas.1720347115 
OpenAI. (2023). ChatGPT (January 16 version 4) [Large language model]. “Imagine that you are a well-regarded copy editor at a popular scientific magazine. You are also a good friend. You have volunteered to review the introduction to a proposal I am drafting for a machine learning project in graduate school. Our objective is to use natural language processing to identify implicit bias at the document level and assess the features associated with bias. This will allow us - in subsequent research - to determine whether it is possible to mitigate bias through information nudges. Here is the text. What, if anything, would you recommend for clarity?” https://chat.openai.com 
Rivera, L. A., & Tilcsik, A. (2016). Class Advantage, Commitment Penalty: The Gendered Effect of Social Class Signals in an Elite Labor Market. American Sociological Review, 81(6), 1097-1131. https://doi.org/10.1177/0003122416668154 
Schmield, K. (2021, November 19). How natural language processing can help eliminate bias in performance Reviews. Bloomberg. Retrieved December 15, 2023, from https://sponsored.bloomberg.com/article/pwc/how-natural-language-processing-can-help-eliminate-bias-in-performance-reviews 
Smith, E. M., Hall, M. A., Kambadur, M., Presani, E., & Williams, A. (2022). “I’m sorry to hear that”: Finding New Biases in Language Models with a Holistic Descriptor Dataset. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2205.09209 
Staats, C., Capatosto, K., Tenney, L., & Mamo, S. (2017). Implicit Bias Review. State of the Science. https://wkkf.issuelab.org/resources/29868/29868.pdf 
Voigt, R., Jurgens, D., Prabhakaran, V., Jurafsky, D., & Tsvetkov, Y. (2018). RtGender: A Corpus for Studying Differential Responses to Gender. LREC, 2814–2820. https://nlp.stanford.edu/robvoigt/rtgender/rtgender.pdf 









8. APPENDIX
Appendix A: Code Repository
GitHub: https://github.com/d-atallah/implicit_gender_bias/tree/main


File Name
Description
01_Supervised_Data_ETL.ipynb


Response source data ETL.
02_Supervised_Responses_Preprocessing.ipynb
Sampling & preprocessing of the source data.
03_Supervised_Model_Creation.ipynb
Model testing of KNN, XGB, and NB models.
04_Supervised_XGB_Model.ipynb
Final XGB model with POS tagging, and unigram TF-IDF vectorization.
05_Supervised_Model_Evaluation.ipynb
Evaluation of all models (KNN, NB, XGB) using 5 fold cross validation F-1 scoring.
06_Supervised_Sensitivity_Testing.ipynb
Hyperparameter sensitivity testing on XGB model parameters: learning_rate, n_estimators, max_depth, colsample_bytree, subsample.
07_Supervised_Feature_Importance.ipynb
Feature importance ablation testing on POS tagging categories.
08_Supervised_Feature_Analysis.ipynb
Feature importance & error analysis using feature importance dictionary from XGB model, classified and misclassified BOW & POS tags, topic modeling, and response subject analysis.
bash_commands.ipynb
List of Git Bash commands used to preprocess the data for unsupervised learning.
split_stratified_data.ipynb
Train, validate, and test split for unsupervised learning.
word_embeddings.ipynb
Load files, tokenize text, train and evaluate word embeddings.
02_create_matrices_1m_sample.ipynb
Load samples and create term matrices for clustering.
03_cluster_tune_1m_sample.ipynb
Load term matrices, train and evaluate different clustering models.
04_analyze_train_results.ipynb
View the model performance and params of each model.
05_cluster_tune_BGMM.ipynb
Tune a Bayesian Gaussian Mixture Model with more components.
06_final_BGM.ipynb
Extract insights from BGMM.
07_plots.ipynb
Create sensitivity line plot for BGMMs with more components.
01_turbotopics_train.ipynb
Failed/working experimentation with turbo topics.


Appendix B: Supervised Learning Ablation Testing Results


Tag
Part of Speech Description
F-1
Impact
NOUN
Noun - a person, place, thing, or idea
0.726 ± 0.004
0.068
VERB
Verb - expresses an action, occurrence, or state of being
0.729 ± 0.003
0.065
ADJ
Adjective - describes or modifies a noun or pronoun
0.760 ± 0.002


0.034
ADV
Adverb - modifies a verb, adjective, or other adverb
0.764 ± 0.001


0.030
X
Other - uncategorized items (often used for words that are not part of the standard treebank)
0.774 ± 0.001
0.020
AUX
Auxiliary - helps form the tenses, moods, etc. of verbs
0.775 ± 0.002
0.019
PRON
Pronoun - takes the place of a noun
0.775 ± 0.002
0.019
PROPN
Proper noun - a specific name of a person, place, etc.
0.775 ± 0.001
0.019
PART
Particle - function words that have a grammatical role
0.785 ± 0.001
0.009
ADP
Adposition - shows relationships between words
0.787 ± 0.001
0.007
NUM
Numeral - represents numbers
0.787 ± 0.001
0.007
CONJ
Conjunction - connects words or groups of words
0.789 ± 0.001


0.005
INTJ
Interjection - expresses strong emotion on its own
0.798 ± 0.001
0.004
SCONJ
Subordinating conjunction - connects clauses and words in a subordinating way
0.790 ± 0.001
0.004
DET
Determiner - introduces a noun and limits its meaning
0.792 ± 0.001


0.002
CCONJ
Coordinating conjunction - connects words or groups
0.794 ± 0.001
0.000
PUNCT
Punctuation - marks used in writing, e.g., commas, etc.
0.795 ± 0.001
0.000
SYM
Symbol - includes mathematical symbols, currency signs, etc.
0.795 ± 0.001
0.000


Appendix C: Supervised Learning Hyperparameter Sensitivity Analysis


Parameter
Value
Parameter Tuning
Impact
learning_rate
0.01
0.788 ± 0.001
0.0062
learning_rate
0.1
0.796 ± 0.001
0.0017
n_estimators
100
0.793 ± 0.001
0.0017
n_estimators
200
0.796 ± 0.001
0.0012
max_depth
7
0.793 ± 0.001
0.0010
max_depth
11
0.795 ± 0.000
0.0010
subsample
0.9
0.795 ± 0.001
0.0003
colsample_bytree
0.6
0.795 ± 0.001
0.0002
colsample_bytree
0.4
0.795 ± 0.001
0.0001
subsample
0.7
0.795 ± 0.000
0.0001
max_depth
9
0.794 ± 0.000
0.0000
learning_rate
0.05
0.794 ± 0.000
0.0000
colsample_bytree
0.5
0.794 ± 0.000
0.0000
subsample
0.8
0.794 ± 0.000
0.0000
n_estimators
150
0.794 ± 0.000
0.0000


Appendix D: Most Biased Tokens in Word Embeddings


Token
Male Bias
Female Bias
Difference
Part of Speech
Analogy
spineless
0.573
0.358
0.216
Adjective
disrespectful
nuclear
0.274
0.070
0.203
Adjective
autonomous
foolish
0.466
0.276
0.189
Adjective
mock
solid
0.211
0.030
0.182
Adjective
nifty
unconstitutional
0.320
0.140
0.180
Adjective
eligible
original
0.357
0.184
0.173
Adjective
previous
legislative
0.320
0.149
0.171
Adjective
unlawful
unlawful
0.283
0.113
0.170
Adjective
judicial
considerable
0.517
0.347
0.170
Adjective
none
ruthless
0.620
0.451
0.169
Adjective
unheard
… 
…
…
…
…
…
sexy
0.330
0.483
-0.153
Adjective
handsome
painful
0.379
0.533
-0.154
Adjective
miserable
cultural
0.131
0.286
-0.155
Adjective
secular
positive
0.162
0.319
-0.157
Adjective
negative
playful
0.511
0.672
-0.161
Adjective
sinek
korean
0.248
0.411
-0.163
Adjective
native
adolescent
0.225
0.391
-0.165
Adjective
nonverbal
beautiful
0.379
0.545
-0.166
Adjective
gorgeous
dense
0.327
0.495
-0.168
Adjective
superior
powerful
0.146
0.319
-0.173
Adjective
blissful


