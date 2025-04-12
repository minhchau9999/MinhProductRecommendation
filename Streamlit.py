import streamlit as st
import pickle
import pandas as pd
from utils import get_recommendations_gensim, get_recommendations_surprise
from PIL import Image
import requests
from io import BytesIO
import re
from bs4 import BeautifulSoup
import sys
import traceback
import os

# Set page config
st.set_page_config(
    page_title="Product Recommendation",
    page_icon="📊",
    layout="wide"
)



# Add at the start of your app
st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/dictionary.pkl', 'rb') as f1:
            dictionary = pickle.load(f1)
        with open('models/tfidf_model.pkl', 'rb') as f2:
            tfidf = pickle.load(f2)
        with open('models/lsi_model.pkl', 'rb') as f3:
            lsi_model = pickle.load(f3)
        with open('models/similarity_index.pkl', 'rb') as f4:
            similarity_index = pickle.load(f4)
        with open('models/surprise_svd_model.pkl', 'rb') as f5:
            surprise = pickle.load(f5)
        return dictionary, tfidf, lsi_model, similarity_index, surprise
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please check if all model files exist in the models/ directory")
        raise e

# Load data
@st.cache_data(ttl="1h", show_spinner="Loading data...")
def load_data():
    try:
        st.write(f"Current working directory: {os.getcwd()}")
        with open('data/processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['df']
    except Exception as e:
        st.error(f"Detailed error: {str(e)}")
        st.error(f"Error type: {type(e)}")
        st.error("Please check if processed_data.pkl exists in the data/ directory")
        raise e

# def load_user_rating_data():
#     with open('data/user_rating_df.pkl', 'rb') as f:
#         data = pickle.load(f)
#     return data

@st.cache_data
def load_sample_products():
    sample_df = pd.read_csv('sample_products.csv')
    return sample_df

def load_image_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
    except:
        return None
    return None

def extract_shopee_image_url(product_url):
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Get the product page
        response = requests.get(product_url, headers=headers)
        if response.status_code == 200:
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the picture element with class UkIsx8
            picture = soup.find('picture', {'class': 'UkIsx8'})
            if picture:
                # Find the img tag within the picture
                img_tag = picture.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    # Get the full resolution image URL by removing the resize parameters
                    image_url = img_tag['src']
                    # Remove any resize parameters from the URL
                    image_url = image_url.split('@')[0] if '@' in image_url else image_url
                    return image_url
            
            # Fallback: Look for meta tags with image information
            meta_img = soup.find('meta', {'property': 'og:image'})
            if meta_img and 'content' in meta_img.attrs:
                return meta_img['content']
    except:
        return None
    return None

def display_recommendation(row, df, search_type):
    # Get the full product details from the original dataset
    product_details = df[df['product_id'] == row['product_id']].iloc[0]
    if search_type == "User Rating":
        string = f"{row['product_name']} (Score_Prediction: {row['Score_Prediction']:.4f})"
    else:
        string = f"{row['product_name']} (Similarity: {row['similarity_score']:.4f})"
    
    with st.expander(string):
        # Create two columns for image and details
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                # Check if image exists and is a valid URL
                if 'image' in product_details and product_details['image'] and isinstance(product_details['image'], str):
                    # Try to load the image
                    img = load_image_from_url(product_details['image'])
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.write("Could not load image from URL")
                else:
                    st.write("No valid image URL available")
            except Exception as e:
                st.write(f"Error loading image: {str(e)}")
        
        with col2:
            st.write(f"**ID:** {row['product_id']}")
            st.write(f"**Sub Category:** {row['sub_category']}")
            if 'rating' in row:
                st.write(f"**Rating:** {row['rating']}")
            st.write(f"**Price:** {product_details['price']}")
            st.write(f"**Description:** {product_details['description']}")
            st.write(f"**Link:** {product_details['link']}")
            
            # Create a clickable link that looks like a button
            st.markdown(f'<a href="{product_details["link"]}" target="_blank" style="display: inline-block; padding: 0.5rem 1rem; background-color: #FF4B4B; color: white; text-decoration: none; border-radius: 0.25rem;">View Product</a>', unsafe_allow_html=True)

def show_homepage():
    st.title("Product Analytics Dashboard")
    st.markdown("""
    Welcome to the Product Analytics Dashboard! This application helps you:
    
    - Find similar products based on your selection
    - Search products using text queries
    - View detailed product information and images
    
    Use the sidebar to navigate to the Product Recommendation System.
    """)

def show_recommendations():
    st.title("🛍️ Product Recommendation System")
    st.markdown("Find similar products based on your selection or search query or on the basis of user rating")
    
    # Load models and data
    dictionary, tfidf, lsi_model, similarity_index, surprise = load_models()
    df = load_data()
    # user_rating_df = load_user_rating_data()
    sample_products = load_sample_products()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Options")
        search_type = st.radio(
            "Choose search type:",
            ["Product Selection", "Text Search", "User Rating"]
        )
        
        if search_type == "Product Selection":
            # Create a dropdown with product names and IDs
            product_options = {f"{row['product_name']} (ID: {row['product_id']})": row['product_id'] 
                             for _, row in sample_products.iterrows()}
            
            selected_product_name = st.selectbox(
                "Select a product:",
                options=list(product_options.keys())
            )
            
            product_id = product_options[selected_product_name]
            
            # Show selected product
            selected_product = df[df['product_id'] == product_id].iloc[0]
            st.markdown("**Selected Product:**")
            st.write(f"Name: {selected_product['product_name']}")
            st.write(f"Category: {selected_product['sub_category']}")
            if 'rating' in selected_product:
                st.write(f"Rating: {selected_product['rating']}")
            
            # Get recommendations automatically
            recommendations = get_recommendations_gensim(
                similarity_index=similarity_index,
                df=df,
                tfidf=tfidf,
                lsi_model=lsi_model,
                dictionary=dictionary,
                product_id=product_id,
                nums=10
            )
        elif search_type == "User Rating":
            # User rating
            user_id = st.text_input("Enter your user id (number in range 0-650636):")
            try:
                # Check for all zeros first
                if user_id and user_id.strip('0') == '':
                    st.error("Invalid user ID - cannot be all zeros")
                    return
                user_id = int(user_id)
            except ValueError:
                st.error("Please enter a valid numeric user ID")
                return
            if user_id:
                if user_id not in range(0,650636):
                    st.error("User ID not found in the dataset")
                else:
                    recommendations = get_recommendations_surprise(
                        df_productid=df[['product_id']],  # Pass DataFrame with product_ids
                        full_product_df=df,
                        surprise=surprise,
                        user_id=user_id,
                        nums=10
                )
                # st.write(recommendations)
        else:
            # Text search
            query = st.text_input("Enter your search query:")
            if query:
                recommendations = get_recommendations_gensim(
                    similarity_index=similarity_index,
                    df=df,
                    tfidf=tfidf,
                    lsi_model=lsi_model,
                    dictionary=dictionary,
                    query=query,
                    nums=10
                )
                # st.write(recommendations)
    
    with col2:
        st.subheader("Recommendations")
        if search_type == "Product Selection" and 'recommendations' in locals():
            # Display recommendations as a list
            for _, row in recommendations.iterrows():
                display_recommendation(row, df, search_type)
        elif search_type == "Text Search" and 'recommendations' in locals():
            # Display recommendations as a list
            for _, row in recommendations.iterrows():
                display_recommendation(row, df, search_type)
        elif search_type == "User Rating" and 'recommendations' in locals():
            # Display recommendations as a list
            for _, row in recommendations.iterrows():
                display_recommendation(row, df, search_type)

def main():
    try:
        # Add navigation in sidebar
        st.sidebar.title("Navigation")
        
        # Create buttons for navigation
        if st.sidebar.button("🏠 Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("🔍 Product Recommendations"):
            st.session_state.page = "Product Recommendations"
        
        # Initialize session state if not exists
        if 'page' not in st.session_state:
            st.session_state.page = "Home"
        
        # Show the selected page
        if st.session_state.page == "Home":
            show_homepage()
        else:
            show_recommendations()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

#    # Cosine in Streamlit
#    @st.cache_data
#    def load_cosine_models():
#        with open('models/vectorizer.pkl', 'rb') as f:
#            vectorizer = pickle.load(f)
#        with open('models/tfidf_matrix.pkl', 'rb') as f:
#            tfidf_matrix = pickle.load(f)
#        return vectorizer, tfidf_matrix


