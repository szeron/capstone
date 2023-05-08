import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(layout='wide')

@st.cache_data
def load_data(file_path, num_samples):
    df = pd.read_pickle(file_path)
    df = df.sample(num_samples, random_state=42)  # Sample a smaller dataset
    return df

file_path = './data/wine_reviews_clean.pkl'
num_samples = 30000  # Reduce this value to load fewer samples
df = load_data(file_path, num_samples)
df['wine_id'] = df.index

# Instantiate the list of wine attributes
attributes = ['fruit', 'tannin', 'cherry', 'rripe', 'aroma', 'rich', 'spice', 'red', 'oak', 'blackberry', 'plum', 
              'berry', 'fresh', 'dark', 'dry', 'apple', 'lemon', 'white', 'juicy', 'balanced', 'herb', 'firm', 'raspberry', 
              'pepper', 'black-cherry', 'bright', 'full-bodied', 'citrus', 'pear', 'light', 'vanilla', 'chocolate', 'crisp', 
              'mineral', 'black-fruit', 'peach', 'currant', 'dense', 'wood', 'soft', 'savory', 'complex', 'sweet', 
              'spicy', 'smooth', 'licorice', 'orange', 'tobacco', 'strawberry', 'leather', 'fruity', 'lime', 'earthy', 
              'creamy', 'tart', 'lead', 'cassis', 'clove', 'tannic', 'grape', 'intense', 'stone',  'tight', 
              'toast', 'powerful', 'textured', 'rose', 'cranberry', 'smoky', 'grapefruit', 'cinnamon', 
              'blueberry', 'black-currant', 'coffee', 'cola', 'violet', 'floral', 'anise', 'tangy', 'bold', 'baked', 
              'cedar', 'herbal', 'lush', 'peel', 'layered', 'black-pepper', 'red-cherry', 'apricot', 'honey', 'black-plum', 
              'zest', 'rounded', 'red-fruit']
attributes.sort()

#---------------------------------------------------------------------------------------------------------

# Combine tokens into a single string for each wine
df['tokens_str'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))

# Create a TfidfVectorizer
tvec = TfidfVectorizer(min_df=5, max_df=0.95, max_features=5000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w[\w-]+\b")

# Fit the vectorizer to the 'tokens' column and transform the tokens into vectors
tmat = tvec.fit_transform(df['tokens_str'])

# Get the vocabulary
tfidf_vocab = tvec.get_feature_names_out()

#---------------------------------------------------------------------------------------------------------

def content_based_recommendations(desired_attributes, n=10):
    # Filter out attributes that are not in the vocabulary
    filtered_attributes = [attribute for attribute in desired_attributes if attribute in tfidf_vocab]

    # If all attributes are not in the vocabulary, return 0 as the similarity
    if not filtered_attributes:
        return pd.DataFrame()

    desired_attributes_vector = tvec.transform([' '.join(filtered_attributes)])

    similarities = cosine_similarity(tmat, desired_attributes_vector)

    # Get the indices of the wines with the highest similarities
    sorted_wine_indices = np.argsort(similarities[:, 0])[::-1]

    # Filter the wines that contain all filtered_attributes in their descriptions
    wine_indices_with_all_attributes = [idx for idx in sorted_wine_indices if all(attr in df.iloc[idx]['description'] for attr in filtered_attributes)]
    
    # Get the top n wines with the highest similarities that contain all filtered_attributes, without duplicates
    seen_titles = set()
    top_wine_indices = []
    for idx in wine_indices_with_all_attributes:
        title = df.iloc[idx]['title']
        if title not in seen_titles:
            seen_titles.add(title)
            top_wine_indices.append(idx)
            if len(top_wine_indices) >= n:
                break

    # Create a dataframe with the recommended wines
    recommended_wines = df.iloc[top_wine_indices][['wine_id', 'title', 'variety', 'points', 'price', 'province', 'description', 'country']]

    # Add the cosine similarity values to the dataframe
    recommended_wines['cosine_similarity'] = similarities[top_wine_indices, 0]

    # Sort the dataframe by cosine similarity in descending order
    recommended_wines = recommended_wines.sort_values(by='cosine_similarity', ascending=False)

    return recommended_wines

#---------------------------------------------------------------------------------------------------------

with open("./data/amazon_wines.json", "r") as f:
    amazon_wines = json.load(f)

def find_best_amazon_wine(recommended_wine_id, selected_attributes):
    
    # Get the recommended wine variety
    recommended_wine = df.loc[df["wine_id"] == recommended_wine_id]
    recommended_variety = recommended_wine["variety"].values[0].lower()

    # Find the best matching wine from Amazon based on user-selected attributes
    best_match = None
    max_matching_attributes = 0
    best_variety_match = None
    max_variety_matching_attributes = 0

    # Sort Amazon wines by rating
    sorted_amazon_wines = sorted(amazon_wines, key=lambda x: x["customer_rating"] if x["customer_rating"] is not None else -1, reverse=True)

    for wine in sorted_amazon_wines:
        if wine["description"] is None:
            continue

        wine_attributes = wine["description"].lower()
        matching_attributes = [attr for attr in selected_attributes if attr in wine_attributes]

        if len(matching_attributes) > max_matching_attributes:
            best_match = wine
            max_matching_attributes = len(matching_attributes)

        # Check for same variety matches only if there is no best match yet
        if best_match is None and wine["variety"] is not None and wine["variety"].lower() == recommended_variety:
            if len(matching_attributes) > max_variety_matching_attributes:
                best_variety_match = wine
                max_variety_matching_attributes = len(matching_attributes)

    return best_match if best_match is not None else best_variety_match

#---------------------------------------------------------------------------------------------------------

# Function for background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        
        [data-testid="stAppViewContainer"] {{
            font-family: 'Montserrat', sans-serif;
            background-image: url("https://images.pexels.com/photos/12584752/pexels-photo-12584752.jpeg?auto=compress&cs=tinysrgb&w=1600");
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            padding: 2rem;
        }}
        
        [data-testid="stVerticalBlock"] {{
            background-color: rgba(0,0,0,0.5);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h1, h2, p, a {{
            color: #ffffff;
        }}
        
        h1 {{
            font-weight: bold;
            font-size: 3rem;
        }}
        
        h2 {{
            font-weight: bold;
            font-size: 2rem;
        }}
        
        a {{
            text-decoration: none;
            font-weight: bold;
        }}

        a:hover {{
            color: #bbbbbb;
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

#----------------------------------------------------------------------------------------------------------

add_bg_from_url()
st.title("Sip Back and Relax: We've Got Your Wine Choices Covered")
st.subheader("By Soh Sze Ron")
st.write("[GitHub](https://github.com/szeron) | [LinkedIn](https://www.linkedin.com/in/szeron)")
st.write("Select your desired wine attributes below:")

user_attributes = st.multiselect(label = " ", options = attributes, label_visibility = "collapsed")

if st.button("Let's un-wine with the top picks!"):
    with st.spinner("Hold your wine, we're still vine-tuning the perfect recommendations..."):
        
        # Instantiate selected wine attributes
        if len(user_attributes) == 0:
            selected_attributes = attributes
        else:
            selected_attributes = user_attributes

        # Run the content-based recommender model
        model = content_based_recommendations(selected_attributes, 10)

        # Prepare the display for the top 10 recommendations
        if not model.empty:
            model_final = model[['title', 'variety', 'points', 'price', 'country', 'province', 'description']].reset_index(drop=True)
        else:
            model_final = pd.DataFrame(columns=['Name', 'Combined Rating', 'Variety', 'Sommelier Rating', 'Price (in USD)', 'Country', 'Province', 'Review'])
            st.write("No recommendations found. Please try different attributes.")
        model_final.index = model_final.index + 1
        model_final.rename(columns={'title': 'Name',
                                    'points':'Sommelier Rating',
                                    'price': 'Price (in USD)',
                                    'variety': 'Variety',
                                    'country': 'Country',
                                    'province': 'Province',
                                    'description': 'Review'}, inplace=True)
        st.balloons()

        # Display the dataframe within a scrollable container
        st.write("""
        <style>
        .dataframe {{
            white-space: nowrap;
            overflow-x: auto;
            overflow-y: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
        model_final.set_index(pd.Series(range(1, len(model_final) + 1)), inplace=True)
        st.write(model_final.style.format({"Combined Rating": "{:.2f}"}))
        st.write('</div>', unsafe_allow_html=True)

        if not model.empty:
            # Find the best wine from Amazon based on the top recommended wine's attributes
            top_recommended_wine_id = model.iloc[0]['wine_id']
            best_amazon_wine = find_best_amazon_wine(top_recommended_wine_id, selected_attributes)

            # Display the best matching wine from Amazon
            if best_amazon_wine is not None:
                st.write("Recommended Wine on Amazon:")
                st.write("Name:", best_amazon_wine["title"])
                st.write("Price:", best_amazon_wine["price"])
                st.write("Customer Rating:", best_amazon_wine["customer_rating"])
                st.write("Amazon link:", best_amazon_wine["url"])

            else:
                st.write("No matching wine found on Amazon.")
        else:
            st.write("No recommendations found. Please try different attributes.")
