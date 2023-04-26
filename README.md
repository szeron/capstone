# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project - Wine Recommender

Author: Soh Sze Ron
<br>
<br>

**Executive Summary**

This capstone project aims to build a wine recommender system using machine learning and NLP techniques to leverage large wine datasets obtained from WineEnthusiast. With wine being a complex and diverse product, it is challenging for consumers to find the perfect wine that matches their tastes and preferences. Therefore, this project seeks to develop a robust and user-friendly system that can assist consumers in discovering new wines that match their preferences. Additionally, the system will aid wine retailers and distributors in effectively targeting their marketing efforts. The two datasets from Kaggle, containing wine descriptions, ratings, prices, and taster names, among other features, provide a rich source of information to develop such a system. The project's successful completion will provide a valuable tool for wine enthusiasts and industry professionals alike.
<br><br>
**Problem Statement**

As a wine lover, my objective is to develop a tailored recommendation system that addresses individual preferences and streamlines the shopping experience. Leveraging machine learning and NLP methods, I strive to provide readily available personalized wine suggestions, elevating the process of discovering new wines.

The recommender system should be capable of:

- Accurately predicting wine ratings and identifying key features that contribute to the overall quality and appeal of wines

- Analyzing wine descriptions and other relevant information (e.g., varietal, region, and price) to create meaningful clusters or categories of wines that exhibit similar characteristics

- Generating personalized recommendations for users based on their flavor profiles

The success of the wine recommender system will be evaluated by its ability to provide accurate and relevant recommendations, as well as its potential to enhance user satisfaction and engagement in the wine selection process.
<br><br>

---

### Datasets

Datasets are provided on Kaggle.

Datasets: 
* [`winemag-data-130k-v2.csv`](./data/winemag-data-130k-v2.csv)
* [`winemag-data-2017-2020.csv`](./data/winemag-data-2017-2020.csv)

<br>

#### Data dictionary of selected features only

| Feature               | Description                                                                                               |
|-----------------------|-----------------------------------------------------------------------------------------------------------|
| country               | The country that the wine is from                                                                         |
| description           | The review given to the wine                                                                              |
| points                | The number of points/ratings WineEnthusiast rated the wine on a scale of 1-100                            |
| price                 | The cost for a bottle of the wine                                                                         |
| province              | The province or state that the wine is from                                                               |
| region_1              | The wine growing area in a province or state (i.e. Napa)                                                  |
| taster_name           | Name of Wine taster                                                                                       |
| title                 | Name, Year & Vineyard of the wine. This feature is the key feature to differentiate wines from each other |
| variety               | Type of wine (i.e. Pinot Noir)                                                                            |
| winery                | The place where the wine was made                                                                         |
| vintage               | The year the wine's grapes were harvested                                                                 |

---

### Modelling

**Summary of results**

| Algorithm                         | RMSE     | Precision@k (%)| Recall@k (%)|
|-----------------------------------|----------|:-----------:|:--------:|
| Normal Predictor                  | 2.176 |   57.0  | 16.3 |
| Baseline Predictor                | 1.614 |   72.4  | 9.6 |
| KNN Basic                         | 1.632 |   59.9  | 19.8 |
| KNN with Means                    | 1.632 |   57.5  | 19.2 |
| KNN with Z-score                  | 1.632 |   58.9  | 19.9 |
| KNN Baseline                      | 1.614 |   72.1  | 9.2 |
| **SVD**                           | **1.615** | **73.8**|**6.6**|
| SVD (tuned)                    | 1.615 |   72.7  | 8.8 |
| Non-negative Matrix Factorization | 1.632 |   59.7  | 19.1 |
| Co-Clustering                     | 1.632 |   57.3  | 18.5 |



**Streamlit Deployment**

[https://wine-me-up.streamlit.app](https://wine-me-up.streamlit.app)

---

### Conclusion

In conclusion, SVD (before tuning) has the best overall performance for the wine recommender as it has the lowest RMSE and the highest Precision@k. At 74%, it is relatively high, meaning that users are more likely to receive recommendations that they will enjoy, thus increasing user satisfaction and trust in the recommendations. However, it has the lowest recall, suggesting that it may not be capturing all the relevant wines for a user within the top k recommendations, like NMF.

Overall, as our aim is to prioritize the quality of recommendations (i.e., how well the predicted ratings match the actual ratings and the relevance of recommended items), SVD would do a good job for our wine recommender.
<br><br>
**Recommendations** <br>
To improve the recall@k without sacrificing precision, additional features or metadata about the wines (e.g., grape variety, region, or tasting notes) to enhance the recommendation algorithm could be explored. 

We can consider segmenting users based on their preferences or behaviors to tailor the recommendations more effectively. Personalized recommendations can lead to improved user satisfaction and better overall performance of the recommender system.

Lastly, we can perform A/B testing with different algorithms or different configurations of the chosen algorithm to find the most effective solution for our users. This approach allows us to compare the performance of different algorithms in a controlled environment and make data-driven decisions about which algorithm to use.

---

### References

- [Wine Reviews, Kaggle. Zack Thoutt](https://www.kaggle.com/datasets/zynicide/wine-reviews)
- [Updated Wine Enthusiast Reviews, Kaggle. Valeriy Mukhtarulin](https://www.kaggle.com/datasets/manyregression/updated-wine-enthusiast-review)
