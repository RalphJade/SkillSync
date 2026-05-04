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
    
    # --- NEW: Extract unique skills and interests for the dropdown menus ---
    unique_skills = sorted(skills_df[skills_df['Scale Name'] == 'Importance']['Element Name'].dropna().unique().tolist())
    unique_interests = sorted(int_df[int_df['Scale Name'] == 'Occupational Interests']['Element Name'].dropna().unique().tolist())

    # 1A. Map O*NET-SOC Codes to Broad Categories based on the first 2 digits
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
    skills_imp = skills_df[skills_df['Scale Name'] == 'Importance']
    skills_imp = skills_imp.sort_values(by=['O*NET-SOC Code', 'Data Value'], ascending=[True, False])
    skills_grouped = skills_imp.groupby('O*NET-SOC Code')['Element Name'].apply(lambda x: ', '.join(x)).reset_index()
    skills_grouped.rename(columns={'Element Name': 'skills_list'}, inplace=True)
    
    # 1C. Aggregate Top Interests per Occupation
    interests_oi = int_df[int_df['Scale Name'] == 'Occupational Interests']
    interests_oi = interests_oi.sort_values(by=['O*NET-SOC Code', 'Data Value'], ascending=[True, False])
    interests_grouped = interests_oi.groupby('O*NET-SOC Code')['Element Name'].apply(lambda x: ', '.join(x)).reset_index()
    interests_grouped.rename(columns={'Element Name': 'interests_list'}, inplace=True)
    
    # 1D. MERGE Everything Together
    df = occ_df.merge(skills_grouped, on='O*NET-SOC Code', how='left')
    df = df.merge(interests_grouped, on='O*NET-SOC Code', how='left')
    
    # 1E. Prepare columns
    df['job_title'] = df['Title']
    df['job_description'] = df['Description']
    df['clean_skills'] = df['skills_list'].fillna('') + ", " + df['interests_list'].fillna('')
    df['combined_features'] = df['clean_skills'] + " " + df['job_description'].fillna('')
    
    df = df.dropna(subset=['category', 'clean_skills'])
    
    # Return the dataframe AND the lists for our dropdowns
    return df, unique_skills, unique_interests

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
df, unique_skills, unique_interests = load_data()
rules = load_knowledge_base()

with st.spinner('Training AI Classifier on O*NET Data... Please wait.'):
    classifier_model = train_classifier(df)

st.title("SkillSync AI: An AI-Based Job Recommendation System")
st.write("University of Southern Mindanao - Project Implementation")

# 4. User Input (HYBRID APPROACH)
st.markdown("### Tell us about yourself")

# Free text for general vibes/keywords
general_input = st.text_input("1. Enter general keywords (e.g., chemistry, drawing, managing people):").lower()

# Structured multi-selects for perfect data alignment
selected_skills = st.multiselect("2. Select specific skills you possess:", unique_skills)
selected_interests = st.multiselect("3. Select your core interests:", unique_interests)


# 5. The REASONING ENGINE
if st.button("Get Recommendations"):
    if not general_input and not selected_skills and not selected_interests:
        st.warning("Please enter some keywords or select at least one skill/interest.")
    else:
        # Combine all inputs into one rich string for the AI
        user_input_parts = [general_input] + selected_skills + selected_interests
        user_input = " ".join([p for p in user_input_parts if p]).strip()
        
        # --- STEP 1: Supervised ML Predicts Primary Category ---
        predicted_category = classifier_model.predict([user_input])[0]
        st.success(f"🤖 **AI Classifier:** Predicted your primary ideal sector is **{predicted_category}**")
        
        # Create a copy of the dataframe so we don't mutate the cached version
        results_df = df.copy()
        
        # --- STEP 2: Unsupervised ML (Cosine Similarity on ALL Jobs) ---
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(results_df['combined_features'])
        user_vector = vectorizer.transform([user_input])
        
        base_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
        results_df['base_score'] = base_scores
        results_df['final_score'] = results_df['base_score']
        
        # --- STEP 3: Knowledge Base & ML Boosts ---
        # 3A. Give a score boost to jobs inside the ML's predicted category
        results_df.loc[results_df['category'] == predicted_category, 'final_score'] += 0.15
        
        # 3B. Apply JSON Rules
        applied_rules = []
        for rule in rules:
            if rule['if_keyword'].lower() in user_input.lower():
                applied_rules.append(rule['if_keyword'])
                results_df.loc[results_df['category'].str.contains(rule['then_boost_category'], case=False, na=False), 'final_score'] += rule['boost_amount']
        
        # --- STEP 4: Rank and Display ---
        # Sort by the final calculated score
        top_matches = results_df.sort_values(by='final_score', ascending=False).head(5)
        top_matches = top_matches[top_matches['final_score'] > 0]
        
        st.write("---")
        if applied_rules:
            st.info(f"🧠 **Knowledge Base Active:** Rules applied based on keywords: {', '.join(applied_rules)}")
            
        st.subheader("🎯 Top Career Matches for You:")
        
        if not top_matches.empty:
            for index, row in top_matches.iterrows():
                st.markdown(f"### 💼 {row['job_title']} ({row['category']})")
                
                # Cap the display score at 1.0 (100%) in case boosts push it over
                display_score = min(float(row['final_score']), 1.0)
                st.progress(display_score, text=f"Match Score: {round(display_score * 100, 1)}%")
                
                st.write(f"**Description:** {row['job_description']}")
                skills_snippet = row['clean_skills'][:150] + "..." if len(row['clean_skills']) > 150 else row['clean_skills']
                st.write(f"**Top Skills & Interests:** {skills_snippet}")
                st.write("") 
        else:
            st.error("No exact matches found. Try adding more skills or interests!")