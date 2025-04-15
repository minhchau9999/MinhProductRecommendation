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
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Make the sidebar wider with custom CSS
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 300px;
            max-width: 300px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add at the start of your app
# st.write(f"Python version: {sys.version}")
# st.write(f"Streamlit version: {st.__version__}")

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
        # st.write(f"Current working directory: {os.getcwd()}")
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
    """Display a single recommendation in a card format"""
    st.markdown("""
        <div style="
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 400px;  /* Fixed height for the container */
            display: flex;
            flex-direction: column;
        ">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Get the image URL directly from the row
        if 'image' in row and isinstance(row['image'], str) and row['image'].strip():
            try:
                image = load_image_from_url(row['image'])
                if image:
                    st.markdown("""
                        <div style="
                            height: 200px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            overflow: hidden;
                        ">
                    """, unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                else:
                    st.markdown("""
                        <div style="
                            height: 200px;
                            background-color: #2E2E2E;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            border-radius: 5px;
                        ">
                            <span style="color: #666;">No image available</span>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown("""
                    <div style="
                        height: 200px;
                        background-color: #2E2E2E;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 5px;
                    ">
                        <span style="color: #666;">No image available</span>
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="color: white; height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h3 style="color: #FF4B4B;">{row['product_name']}</h3>
                    <p style="color: #FFD700;">{'Predicted Rating' if search_type == 'collaborative' else 'Similarity Score'}: {row.get('predicted_rating', row.get('similarity_score', 0)):.2f}</p>
                    <p>Category: {row.get('category', row.get('sub_category', ''))}</p>
                    <p>Category: {row.get('category', row.get('sub_category', ''))}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Get the product link from the original DataFrame
        try:
            if 'link' in df.columns:
                product_link = df[df['product_id'] == row['product_id']]['link'].iloc[0]
                st.markdown(f"""
                    <a href="{product_link}" target="_blank" style="
                        display: inline-block;
                        padding: 8px 16px;
                        background-color: #FF4B4B;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                        margin-top: 10px;
                    ">View Product</a>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("Product link not available")

def show_homepage():
    # Add a header with a nice icon
    st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #FF4B4B;'>🛍️ Product Analytics Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='margin-top: 20px;'>
            <h3>✨ Project Overview</h3>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🤖</span>
                <span>This product is the demonstration of using using NLP methods to build the recommendation system for an online shopping service.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features with icons
        st.markdown("""
        <div style='margin-top: 20px;'>
            <h3>🎯 Key Features</h3>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🔄</span>
                <span>Find similar products based on your selection</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🔍</span>
                <span>Search products using natural language queries</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>👥</span>
                <span>Get personalized recommendations based on user ratings</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>📊</span>
                <span>View detailed product information with images</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical Description with better formatting
    st.markdown("""
        <div style='margin-top: 30px;'>
            <h3 style='color:rgb(255, 255, 255);'>⚙️ Technical Description</h3>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>📝</span>
                <span><strong>Content-based Filtering</strong> - Analyzes product descriptions</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>💫</span>
                <span><strong>Collaborative Filtering</strong> - Learns from user ratings</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🧠</span>
                <span><strong>Advanced Models</strong>:</span>
            </div>
            <div style='margin-left: 40px;'>
                <div style='display: flex; align-items: center; margin: 10px 0;'>
                    <span style='font-size: 24px; margin-right: 10px;'>🤖</span>
                    <span>Gensim for semantic analysis</span>
                </div>
                <div style='display: flex; align-items: center; margin: 10px 0;'>
                    <span style='font-size: 24px; margin-right: 10px;'>📈</span>
                    <span>Surprise for rating predictions</span>
                </div>
                <div style='display: flex; align-items: center; margin: 10px 0;'>
                    <span style='font-size: 24px; margin-right: 10px;'>📉</span>
                    <span>LSI for dimensionality reduction</span>
                </div>
                <div style='display: flex; align-items: center; margin: 10px 0;'>
                    <span style='font-size: 24px; margin-right: 10px;'>🔤</span>
                    <span>TF-IDF for text processing</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Add a decorative image or illustration
        st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <img src='data:image/jpeg;base64,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' style='width: 300px; height: 200px;'>
        </div>
        """, unsafe_allow_html=True)
        
        # Group Members section
        st.markdown("""
        <div style='margin-top: 20px; padding: 20px; border-radius: 1px;'>
            <h3 style='color:rgb(255, 255, 255);'>👨‍💻👩‍💻 Group Members</h3>
            <p style='font-size: 28px;'><br>Châu Nhật Minh - Phạm Đình Anh Duy</p>
        </div>
        """, unsafe_allow_html=True)

def show_recommendations():
    st.title("🛍️ Product Recommendation System")
    
    # Load models and data
    dictionary, tfidf, lsi_model, similarity_index, surprise = load_models()
    df = load_data()
    sample_products = load_sample_products()
    
    # Create two columns for search options
    st.subheader("Search Options")
    search_col1, search_col2 = st.columns([1, 1])
    
    with search_col1:
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
            st.write(f"Price: {selected_product['price']}")
            st.markdown(f'<a href="{selected_product["link"]}" target="_blank" style="display: inline-block; padding: 0.5rem 1rem; background-color: #FF4B4B; color: white; text-decoration: none; border-radius: 0.25rem;">View Product</a>', unsafe_allow_html=True)
            
            # Get recommendations automatically
            recommendations = get_recommendations_gensim(
                similarity_index=similarity_index,
                df=df,
                tfidf=tfidf,
                lsi_model=lsi_model,
                dictionary=dictionary,
                product_id=product_id,
                nums=4  # Increased to show more recommendations
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
                        nums=4  # Increased to show more recommendations
                )
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
                    nums=4  # Increased to show more recommendations
                )
    
    with search_col2:
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px;'>
            <h3 style='color: #FF4B4B;'>Search Tips</h3>
            <ul style='color: white;'>
                <li>Product Selection: Choose from our curated list of products</li>
                <li>Text Search: Use natural language to describe what you're looking for</li>
                <li>User Rating: Get personalized recommendations based on your preferences</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Show recommendations below the search options
    if 'recommendations' in locals():
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; font-size: 1.5em;'>Recommended Products</h3>", unsafe_allow_html=True)
        
        # Create a grid of 4 columns for recommendations
        cols = st.columns(4)
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            with cols[idx % 4]:
                
                # Get the product image from the original DataFrame
                product_image = df[df['product_id'] == row['product_id']]['image'].iloc[0]
                
                # Display product image
                try:
                    if product_image:
                        img = load_image_from_url(product_image)
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown("""
                        <div style="
                            height: 200px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background-color: #2E2E2E;
                            border-radius: 8px;
                            margin-bottom: 10px;
                        ">
                            <span style="color: #666;">No image available</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="
                            height: 200px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background-color: #2E2E2E;
                            border-radius: 8px;
                            margin-bottom: 10px;
                        ">
                            <span style="color: #666;">No image available</span>
                        </div>
                        """, unsafe_allow_html=True)
                except:
                    st.markdown("""
                        <div style="
                            height: 200px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background-color: #2E2E2E;
                            border-radius: 8px;
                            margin-bottom: 10px;
                        ">
                            <span style="color: #666;">No image available</span>
                        </div>
                        """, unsafe_allow_html=True)
                                
                # Display product details
                price = df[df['product_id'] == row.product_id]['price'].iloc[0]
                des = df[df['product_id'] == row.product_id]['description'].iloc[0]

                st.markdown(f"""
                    <div style='
                        color: white; 
                        flex-grow: 1;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    '>
                        <div>
                            <h4 style='color: #FF4B4B; margin-top: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{row['product_name']}</h4>
                            <p style='color: #FFD700;'>{'Predicted Rating' if search_type == 'User Rating' else 'Similarity Score'}: {row.get('Score_Prediction', row.get('similarity_score', 0)):.2f}</p>
                            <p style='color: white; font-size: 0.9em;'><strong>Category:</strong> {row.get('sub_category', '')}</p>
                            <p style='color: white; font-size: 0.9em;'><strong>Price:</strong> {price}</p>
                            <p style='color: white; font-size: 0.9em;'><strong>Description:</strong> {des}</p>
                        </div>
                        
                """, unsafe_allow_html=True)
                
                # Get the product link from the original DataFrame
                try:
                    product_link = df[df['product_id'] == row['product_id']]['link'].iloc[0]
                    st.markdown(f"""
                        <a href="{product_link}" target="_blank" style="
                            display: inline-block;
                            width: 100%;
                            padding: 8px 0;
                            background-color: #FF4B4B;
                            color: white;
                            text-decoration: none;
                            border-radius: 4px;
                            margin-top: 10px;
                            text-align: center;
                        ">View Product</a>
                    """, unsafe_allow_html=True)
                except:
                    st.markdown("<p style='color: #666;'>Product link not available</p>", unsafe_allow_html=True)

def main():
    try:
        # Add navigation in sidebar
        st.markdown(
        """
        <style>
            [data-testid="stSidebar"][aria-expanded="true"]{
                min-width: 300px;
                max-width: 300px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
        
            # Add empty space to push footer to bottom
        st.sidebar.markdown("<br>" * 13, unsafe_allow_html=True)

        # Footer for author and source code
        st.sidebar.markdown("---")
        st.sidebar.markdown("Streamlit UI made with ☕︎ by [Minh] and Cursor-AI")
        st.sidebar.markdown("Source code: [GitHub](https://github.com/minhchau9999/MinhProductRecommendation.git)")
        st.sidebar.markdown("Gensim and Surprise models by Minh + Duy")
        st.sidebar.markdown("---")

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


