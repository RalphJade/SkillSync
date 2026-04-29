import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load and Merge O*NET Relational Data
@st.cache_data
def load_data():
    # Load the 3 raw files
    occ_df = pd.read_excel("Occupation Data.xlsx")
    skills_df = pd.read_excel("Skills.xlsx")
    int_df = pd.read_excel("Interests.xlsx")
    
    # 1A. Map O*NET-SOC Codes to Broad Categories based on the first 2 digits
    # (e.g., Code '27' represents Arts, Entertainment, and Sports!)
    soc_map = {
        '11': 'Management', '13': 'Business & Financial', '15': 'Computer & Math',
        '17': 'Architecture & Engineering', '19': 'Science', '21': 'Community & Social',
        '23': 'Legal', '25': 'Education', '27': 'Arts, Entertainment & Sports',
        '29': 'Healthcare Practitioners', '31': 'Healthcare Support', '33': 'Protective Service',
        '35': 'Food Preparation', '37': 'Building & Grounds Cleaning', '39': 'Personal Care',
        '41': 'Sales', '43': 'Office & Administrative', '45': 'Farming, Fishing & Forestry',
        '47': 'Construction & Extraction', '49': 'Installation, Maintenance & Repair',
        '51': 'Production', '53': 'Transportation & Material Moving', '55': 'Military'
    }
    occ_df['soc_group'] = occ_df['O*NET-SOC Code'].str[:2]
    occ_df['category'] = occ_df['soc_group'].map(soc_map).fillna('Other')
    
    # 1B. Aggregate Top Skills per Occupation
    # We filter by 'Importance', sort by highest score, and merge them into a single string
    skills_imp = skills_df[skills_df['Scale Name'] == 'Importance']
    skills_imp = skills_imp.sort_values(by=['O*NET-SOC Code', 'Data Value'], ascending=[True, False])
    skills_grouped = skills_imp.groupby('O*NET-SOC Code')['Element Name'].apply(lambda x: ', '.join(x)).reset_index()
    skills_grouped.rename(columns={'Element Name': 'skills_list'}, inplace=True)
    
    # 1C. Aggregate Top Interests per Occupation
    # We do the same for 'Occupational Interests' (Realistic, Investigative, etc.)
    interests_oi = int_df[int_df['Scale Name'] == 'Occupational Interests']
    interests_oi = interests_oi.sort_values(by=['O*NET-SOC Code', 'Data Value'], ascending=[True, False])
    interests_grouped = interests_oi.groupby('O*NET-SOC Code')['Element Name'].apply(lambda x: ', '.join(x)).reset_index()
    interests_grouped.rename(columns={'Element Name': 'interests_list'}, inplace=True)
    
    # 1D. MERGE Everything Together!
    df = occ_df.merge(skills_grouped, on='O*NET-SOC Code', how='left')
    df = df.merge(interests_grouped, on='O*NET-SOC Code', how='left')
    
    # 1E. Prepare columns to match your existing app logic
    df['job_title'] = df['Title']
    df['job_description'] = df['Description']
    # Stitch skills and interests together
    df['clean_skills'] = df['skills_list'].fillna('') + ", " + df['interests_list'].fillna('')
    # Stitch everything into one massive context window for the AI
    df['combined_features'] = df['clean_skills'] + " " + df['job_description'].fillna('')
    
    df = df.dropna(subset=['category', 'clean_skills'])
    return df

# 2. Load Knowledge Base
@st.cache_data
def load_knowledge_base():
    try:
        with open('rules.json', 'r') as file:
            return json.load(file)['rules']
    except FileNotFoundError:
        return []

# 3. Train Supervised Classifier (Random Forest)
@st.cache_resource
def train_classifier(df):
    X = df['combined_features']
    y = df['category']
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    model_pipeline.fit(X, y)
    return model_pipeline

# Initialize Components
df = load_data()
rules = load_knowledge_base()
with st.spinner('Training AI Classifier on O*NET Data... Please wait.'):
    classifier_model = train_classifier(df)

st.title("SkillSync AI: An AI-Based Job Recommendation System")
st.write("University of Southern Mindanao - Project Implementation")

# 4. User Input
user_skills = st.text_input("Enter your skills (e.g., programming, negotiation, drawing):").lower()
selected_interest = st.text_input("Enter your primary interest (e.g., realistic, investigative, sports):").lower()

# 5. The REASONING ENGINE
if st.button("Get Recommendations"):
    if not user_skills and not selected_interest:
        st.warning("Please enter some skills or interests.")
    else:
        user_input = user_skills + " " + selected_interest
        
        # --- STEP 1: Supervised ML Predicts Category ---
        predicted_category = classifier_model.predict([user_input])[0]
        st.success(f"🤖 **AI Classifier:** Predicted your ideal career sector is **{predicted_category}**")
        
        # Filter dataset to only jobs in the predicted category
        category_jobs = df[df['category'] == predicted_category].copy()
        
        # --- STEP 2: Unsupervised ML (Cosine Similarity) ---
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(category_jobs['combined_features'])
        user_vector = vectorizer.transform([user_input])
        
        base_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
        category_jobs['final_score'] = base_scores
        
        # --- STEP 3: Knowledge Base (Rule-Based Boosts) ---
        applied_rules = []
        for rule in rules:
            if rule['if_keyword'] in user_input:
                applied_rules.append(rule['if_keyword'])
                category_jobs.loc[category_jobs['category'].str.contains(rule['then_boost_category'], case=False, na=False), 'final_score'] += rule['boost_amount']
        
        # --- STEP 4: Rank and Display ---
        top_matches = category_jobs.sort_values(by='final_score', ascending=False).head(5)
        top_matches = top_matches[top_matches['final_score'] > 0]
        
        st.write("---")
        if applied_rules:
            st.info(f"🧠 **Knowledge Base Active:** Rules applied based on keywords: {', '.join(applied_rules)}")
            
        st.subheader(f"🎯 Top Careers in {predicted_category}:")
        
        if not top_matches.empty:
            for index, row in top_matches.iterrows():
                st.markdown(f"### 💼 {row['job_title']}")
                
                display_score = min(float(row['final_score']), 1.0)
                st.progress(display_score, text=f"Match Score: {round(display_score * 100, 1)}%")
                
                st.write(f"**Description:** {row['job_description']}")
                skills_snippet = row['clean_skills'][:150] + "..." if len(row['clean_skills']) > 150 else row['clean_skills']
                st.write(f"**Top Skills & Interests:** {skills_snippet}")
                st.write("") 
        else:
            st.error("No exact matches found in this sector. Try adding more skills or interests!")