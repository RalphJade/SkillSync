import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SkillSync AI - Career Discovery",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# 1. LOAD AND MERGE O*NET RELATIONAL DATA
# ============================================================================
@st.cache_data(show_spinner="Loading O*NET occupational database...")
def load_data():
    occ_df = pd.read_excel("Occupation Data.xlsx")
    skills_df = pd.read_excel("Skills.xlsx")
    int_df = pd.read_excel("Interests.xlsx")

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

    occ_df['soc_group'] = occ_df['O*NET-SOC Code'].astype(str).str[:2]
    occ_df['category'] = occ_df['soc_group'].map(soc_map).fillna('Other')

    # Aggregate Top Skills per Occupation (Importance scale)
    skills_imp = skills_df[skills_df['Scale Name'] == 'Importance'].copy()
    skills_imp = skills_imp.sort_values(by=['O*NET-SOC Code', 'Data Value'], ascending=[True, False])
    skills_grouped = skills_imp.groupby('O*NET-SOC Code')['Element Name'].apply(lambda x: ', '.join(x.head(10))).reset_index()
    skills_grouped.rename(columns={'Element Name': 'skills_list'}, inplace=True)

    # Aggregate Top Interests per Occupation
    interests_oi = int_df[int_df['Scale Name'] == 'Occupational Interests'].copy()
    interests_oi = interests_oi.sort_values(by=['O*NET-SOC Code', 'Data Value'], ascending=[True, False])
    interests_grouped = interests_oi.groupby('O*NET-SOC Code')['Element Name'].apply(lambda x: ', '.join(x.head(5))).reset_index()
    interests_grouped.rename(columns={'Element Name': 'interests_list'}, inplace=True)

    # Merge everything
    df = occ_df.merge(skills_grouped, on='O*NET-SOC Code', how='left')
    df = df.merge(interests_grouped, on='O*NET-SOC Code', how='left')

    df['job_title'] = df['Title']
    df['job_description'] = df['Description'].fillna('')
    df['skills_text'] = df['skills_list'].fillna('')
    df['interests_text'] = df['interests_list'].fillna('')
    df['clean_skills'] = df['skills_text'] + ", " + df['interests_text']
    df['combined_features'] = (
        df['job_title'] + " " + 
        df['job_description'] + " " + 
        df['clean_skills']
    ).fillna('')

    df = df.dropna(subset=['category', 'clean_skills'])
    df = df.reset_index(drop=True)

    return df

# ============================================================================
# 2. KNOWLEDGE BASE & REASONING ENGINE
# ============================================================================
def load_knowledge_base():
    """Enhanced knowledge base with positive, negative, conditional, and chaining rules.
    NOTE: Not cached because chaining rule conditions use runtime evaluation."""
    kb = {
        "boost_rules": [
            {"if_keyword": "math", "then_boost_category": "Computer & Math", "boost_amount": 0.15, "priority": 2},
            {"if_keyword": "math", "then_boost_category": "Architecture & Engineering", "boost_amount": 0.10, "priority": 2},
            {"if_keyword": "communication", "then_boost_category": "Sales", "boost_amount": 0.15, "priority": 2},
            {"if_keyword": "communication", "then_boost_category": "Management", "boost_amount": 0.10, "priority": 2},
            {"if_keyword": "creative", "then_boost_category": "Arts, Entertainment & Sports", "boost_amount": 0.20, "priority": 1},
            {"if_keyword": "creative", "then_boost_category": "Architecture & Engineering", "boost_amount": 0.10, "priority": 2},
            {"if_keyword": "technology", "then_boost_category": "Computer & Math", "boost_amount": 0.20, "priority": 1},
            {"if_keyword": "technology", "then_boost_category": "Installation, Maintenance & Repair", "boost_amount": 0.10, "priority": 2},
            {"if_keyword": "data", "then_boost_category": "Computer & Math", "boost_amount": 0.15, "priority": 1},
            {"if_keyword": "data", "then_boost_category": "Business & Financial", "boost_amount": 0.10, "priority": 2},
            {"if_keyword": "help", "then_boost_category": "Healthcare Practitioners", "boost_amount": 0.15, "priority": 1},
            {"if_keyword": "help", "then_boost_category": "Community & Social", "boost_amount": 0.15, "priority": 1},
            {"if_keyword": "help", "then_boost_category": "Education", "boost_amount": 0.10, "priority": 2},
            {"if_keyword": "physical", "then_boost_category": "Construction & Extraction", "boost_amount": 0.20, "priority": 1},
            {"if_keyword": "physical", "then_boost_category": "Transportation & Material Moving", "boost_amount": 0.15, "priority": 2},
            {"if_keyword": "science", "then_boost_category": "Science", "boost_amount": 0.20, "priority": 1},
            {"if_keyword": "science", "then_boost_category": "Healthcare Practitioners", "boost_amount": 0.10, "priority": 2},
            {"if_keyword": "design", "then_boost_category": "Architecture & Engineering", "boost_amount": 0.15, "priority": 1},
            {"if_keyword": "design", "then_boost_category": "Arts, Entertainment & Sports", "boost_amount": 0.15, "priority": 1},
            {"if_keyword": "legal", "then_boost_category": "Legal", "boost_amount": 0.25, "priority": 1},
            {"if_keyword": "teach", "then_boost_category": "Education", "boost_amount": 0.20, "priority": 1},
        ],
        "suppress_rules": [
            {"if_keyword": "minimal physical", "suppress_category": "Construction & Extraction", "suppress_amount": 0.30},
            {"if_keyword": "minimal physical", "suppress_category": "Transportation & Material Moving", "suppress_amount": 0.25},
            {"if_keyword": "minimal physical", "suppress_category": "Building & Grounds Cleaning", "suppress_amount": 0.25},
            {"if_keyword": "minimal physical", "suppress_category": "Installation, Maintenance & Repair", "suppress_amount": 0.20},
            {"if_keyword": "fast-paced", "suppress_category": "Education", "suppress_amount": 0.15},
            {"if_keyword": "fast-paced", "suppress_category": "Office & Administrative", "suppress_amount": 0.10},
        ],
        "chaining_rules": [
            {
                "name": "Tech-Finance Bridge",
                "trigger_profile": "Computer & Math",
                "trigger_threshold": 0.25,
                "trigger_keywords": ["data", "business"],
                "action": {"Business & Financial": 0.08, "Management": 0.05},
                "explanation": "Strong tech interest + business keywords → Finance/Management boost"
            },
            {
                "name": "Healthcare-Education Bridge",
                "trigger_profile": "Healthcare Practitioners",
                "trigger_threshold": 0.20,
                "trigger_keywords": ["teach", "education"],
                "action": {"Education": 0.10, "Community & Social": 0.05},
                "explanation": "Healthcare + teaching interest → Education/Community boost"
            },
            {
                "name": "Creative-Tech Bridge",
                "trigger_profile": "Arts, Entertainment & Sports",
                "trigger_threshold": 0.20,
                "trigger_keywords": ["technology", "design", "software"],
                "action": {"Computer & Math": 0.08, "Architecture & Engineering": 0.05},
                "explanation": "Creative + tech interest → Design/Engineering boost"
            },
            {
                "name": "Leadership-Sales Bridge",
                "trigger_profile": "Management",
                "trigger_threshold": 0.25,
                "trigger_keywords": ["communication", "people"],
                "action": {"Sales": 0.10, "Community & Social": 0.05},
                "explanation": "Management + people skills → Sales/Community boost"
            }
        ]
    }
    return kb

# ============================================================================
# 3. QUESTIONNAIRE STRUCTURE
# ============================================================================
QUESTIONNAIRE = {
    1: {
        "question": "🎯 How do you prefer to spend your time?",
        "options": {
            "Solving problems with data and technology": {
                "Computer & Math": 0.50, "Business & Financial": 0.25, "Science": 0.15, "Architecture & Engineering": 0.10
            },
            "Creating and designing things": {
                "Arts, Entertainment & Sports": 0.40, "Architecture & Engineering": 0.35, "Education": 0.15, "Computer & Math": 0.10
            },
            "Helping and supporting others directly": {
                "Healthcare Practitioners": 0.35, "Community & Social": 0.30, "Education": 0.20, "Healthcare Support": 0.15
            },
            "Managing projects, teams, or organizations": {
                "Management": 0.50, "Business & Financial": 0.25, "Sales": 0.15, "Legal": 0.10
            },
            "Building, fixing, or working with my hands": {
                "Construction & Extraction": 0.30, "Installation, Maintenance & Repair": 0.30, "Production": 0.25, "Transportation & Material Moving": 0.15
            },
            "Studying science, nature, or how things work": {
                "Science": 0.45, "Healthcare Practitioners": 0.25, "Architecture & Engineering": 0.20, "Education": 0.10
            }
        }
    },
    2: {
        "question": "🏢 What work environment do you thrive in?",
        "options": {
            "Fast-paced, competitive, results-driven": {
                "Sales": 0.40, "Management": 0.30, "Computer & Math": 0.15, "Business & Financial": 0.15
            },
            "Stable, structured, with clear routines": {
                "Office & Administrative": 0.35, "Legal": 0.25, "Military": 0.20, "Protective Service": 0.20
            },
            "Outdoors, hands-on, physically active": {
                "Construction & Extraction": 0.30, "Transportation & Material Moving": 0.25, 
                "Farming, Fishing & Forestry": 0.25, "Installation, Maintenance & Repair": 0.20
            },
            "Collaborative, team-oriented, people-focused": {
                "Community & Social": 0.30, "Education": 0.30, "Healthcare Practitioners": 0.20, "Management": 0.20
            },
            "Creative, flexible, independent work": {
                "Arts, Entertainment & Sports": 0.40, "Architecture & Engineering": 0.25, 
                "Education": 0.20, "Computer & Math": 0.15
            },
            "Patient-focused, caring, service-oriented": {
                "Healthcare Practitioners": 0.40, "Healthcare Support": 0.30, 
                "Community & Social": 0.20, "Personal Care": 0.10
            }
        }
    },
    3: {
        "question": "💡 What motivates you most in a career?",
        "options": {
            "High income and financial growth": {
                "Computer & Math": 0.30, "Management": 0.25, "Business & Financial": 0.25, "Architecture & Engineering": 0.20
            },
            "Making a positive impact on people": {
                "Healthcare Practitioners": 0.30, "Community & Social": 0.30, "Education": 0.25, "Personal Care": 0.15
            },
            "Recognition, status, and achievement": {
                "Management": 0.30, "Sales": 0.25, "Arts, Entertainment & Sports": 0.25, "Legal": 0.20
            },
            "Continuous learning and intellectual growth": {
                "Education": 0.30, "Computer & Math": 0.25, "Science": 0.25, "Architecture & Engineering": 0.20
            },
            "Job security, benefits, and stability": {
                "Government": 0.30, "Legal": 0.25, "Protective Service": 0.25, "Military": 0.20
            },
            "Work-life balance and personal well-being": {
                "Education": 0.30, "Office & Administrative": 0.25, "Healthcare Support": 0.25, "Personal Care": 0.20
            }
        }
    },
    4: {
        "question": "💪 How do you feel about physical or labor-intensive work?",
        "options": {
            "Very comfortable — I enjoy physical activity": {
                "Construction & Extraction": 0.30, "Installation, Maintenance & Repair": 0.25,
                "Transportation & Material Moving": 0.25, "Production": 0.20
            },
            "Somewhat comfortable — I can handle it when needed": {
                "Production": 0.25, "Food Preparation": 0.25, 
                "Building & Grounds Cleaning": 0.25, "Farming, Fishing & Forestry": 0.25
            },
            "Prefer minimal physical work — I like desk-based roles": {
                "Computer & Math": 0.35, "Management": 0.25, "Business & Financial": 0.20, "Legal": 0.20
            },
            "Depends on the role — mixed is fine": {
                "Healthcare Practitioners": 0.25, "Architecture & Engineering": 0.25,
                "Office & Administrative": 0.25, "Personal Care": 0.25
            }
        }
    },
    5: {
        "question": "⭐ What are your top 3 strengths? (Select up to 3)",
        "options": {
            "Technical & analytical problem-solving": {
                "Computer & Math": 0.40, "Science": 0.25, "Architecture & Engineering": 0.20, "Business & Financial": 0.15
            },
            "Communication, leadership, and persuasion": {
                "Management": 0.35, "Education": 0.25, "Sales": 0.25, "Community & Social": 0.15
            },
            "Creativity, design, and artistic vision": {
                "Arts, Entertainment & Sports": 0.40, "Architecture & Engineering": 0.30, "Education": 0.20, "Computer & Math": 0.10
            },
            "Attention to detail, organization, and precision": {
                "Legal": 0.30, "Office & Administrative": 0.30, "Business & Financial": 0.25, "Science": 0.15
            },
            "Empathy, emotional intelligence, and people skills": {
                "Healthcare Practitioners": 0.30, "Community & Social": 0.30, "Personal Care": 0.25, "Education": 0.15
            },
            "Adaptability, resilience, and hands-on problem-solving": {
                "Management": 0.25, "Computer & Math": 0.25, "Installation, Maintenance & Repair": 0.25, "Construction & Extraction": 0.25
            }
        },
        "allow_multiple": True
    }
}

# ============================================================================
# 4. SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    defaults = {
        'current_question': 1,
        'answers': {},
        'user_profile': None,
        'show_results': False,
        'ml_model_ready': False,
        'vectorizer': None,
        'job_vectors': None,
        'fired_rules': [],
        'explanations': {}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ============================================================================
# 5. PROFILE CALCULATION (Knowledge-Based Component)
# ============================================================================
def calculate_user_profile(answers, df):
    """Calculate category scores from questionnaire answers."""
    all_categories = df['category'].unique()
    category_scores = {cat: 0.0 for cat in all_categories}

    for question_id, answer in answers.items():
        if isinstance(answer, list):
            for single_answer in answer:
                if single_answer in QUESTIONNAIRE[question_id]["options"]:
                    weights = QUESTIONNAIRE[question_id]["options"][single_answer]
                    for category, weight in weights.items():
                        if category in category_scores:
                            category_scores[category] += weight
        else:
            if answer in QUESTIONNAIRE[question_id]["options"]:
                weights = QUESTIONNAIRE[question_id]["options"][answer]
                for category, weight in weights.items():
                    if category in category_scores:
                        category_scores[category] += weight

    total = sum(category_scores.values())
    if total > 0:
        category_scores = {k: v / total for k, v in category_scores.items()}

    return category_scores

# ============================================================================
# 6. ML MODEL: TF-IDF CONTENT-BASED FILTERING
# ============================================================================
def train_ml_model(df):
    """Train TF-IDF vectorizer on job descriptions for content-based filtering."""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85
    )
    job_vectors = vectorizer.fit_transform(df['combined_features'])
    return vectorizer, job_vectors

def get_ml_scores(user_text, vectorizer, job_vectors):
    """Get cosine similarity scores between user profile text and all jobs."""
    if vectorizer is None or job_vectors is None:
        return np.zeros(job_vectors.shape[0] if job_vectors is not None else 1)
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, job_vectors).flatten()
    return similarities

# ============================================================================
# 7. REASONING ENGINE
# ============================================================================
def apply_reasoning_engine(profile_scores, answers, kb):
    """
    Forward-chaining reasoning engine with:
    - Keyword-based positive boosts
    - Keyword-based negative suppressions  
    - Conditional chaining rules (using declarative data, not lambdas)
    - Explanation generation
    """
    scores = dict(profile_scores)
    fired = []
    explanations = []

    user_text = " ".join([str(v) for v in answers.values()]).lower()

    # 7A. Apply boost rules (forward chaining, priority-ordered)
    boost_rules = sorted(kb["boost_rules"], key=lambda x: x["priority"])
    for rule in boost_rules:
        if rule["if_keyword"].lower() in user_text:
            cat = rule["then_boost_category"]
            if cat in scores:
                scores[cat] += rule["boost_amount"]
                fired.append({
                    "type": "boost",
                    "rule": f"'{rule['if_keyword']}' → +{rule['boost_amount']} to {cat}",
                    "priority": rule["priority"]
                })

    # 7B. Apply suppression rules (negative evidence)
    for rule in kb["suppress_rules"]:
        if rule["if_keyword"].lower() in user_text:
            cat = rule["suppress_category"]
            if cat in scores:
                scores[cat] = max(0, scores[cat] - rule["suppress_amount"])
                fired.append({
                    "type": "suppress",
                    "rule": f"'{rule['if_keyword']}' → -{rule['suppress_amount']} from {cat}"
                })

    # 7C. Apply chaining rules (higher-order inference using declarative conditions)
    for chain_rule in kb["chaining_rules"]:
        trigger_cat = chain_rule["trigger_profile"]
        threshold = chain_rule["trigger_threshold"]
        keywords = chain_rule["trigger_keywords"]

        # Check profile condition
        profile_met = scores.get(trigger_cat, 0) > threshold
        # Check keyword condition
        keyword_met = any(kw.lower() in user_text for kw in keywords)

        if profile_met and keyword_met:
            for cat, boost in chain_rule["action"].items():
                if cat in scores:
                    scores[cat] += boost
            fired.append({
                "type": "chain",
                "rule": chain_rule["name"],
                "explanation": chain_rule["explanation"]
            })
            explanations.append(chain_rule["explanation"])

    # Re-normalize
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return scores, fired, explanations

def generate_job_explanation(job_row, user_profile, answers, fired_rules, ml_score, kb_score):
    """Generate human-readable explanation for a specific job recommendation."""
    reasons = []
    category = job_row['category']
    cat_score = user_profile.get(category, 0)

    # Profile match reason
    if cat_score >= 0.20:
        reasons.append(f"Your profile strongly aligns with **{category}** careers ({cat_score*100:.0f}% preference match)")
    elif cat_score >= 0.10:
        reasons.append(f"You show moderate interest in **{category}** careers")

    # Content similarity reason
    if ml_score > 0.15:
        reasons.append(f"High content similarity ({ml_score*100:.0f}%) between your interests and this role's requirements")
    elif ml_score > 0.08:
        reasons.append(f"Good content match with this role's skills and description")

    # Specific fired rules
    for rule in fired_rules:
        if category in rule.get("rule", ""):
            if rule["type"] == "boost":
                reasons.append(f"Knowledge base rule fired: {rule['rule']}")
            elif rule["type"] == "chain":
                reasons.append(f"Inferred connection: {rule.get('explanation', rule['rule'])}")

    # Skills overlap
    user_keywords = set(" ".join([str(v) for v in answers.values()]).lower().split())
    job_skills = set(job_row['clean_skills'].lower().split(", "))
    overlap = user_keywords & job_skills
    if overlap:
        reasons.append(f"Shared interests: {', '.join(list(overlap)[:3])}")

    if not reasons:
        reasons.append("Recommended based on ensemble scoring of multiple factors")

    return reasons

# ============================================================================
# 8. HYBRID ENSEMBLE RECOMMENDATION
# ============================================================================
def hybrid_recommendation(df, user_profile, answers, kb, vectorizer, job_vectors, top_n=5):
    """
    Ensemble recommendation combining:
    1. Knowledge-based profile scoring
    2. ML content-based similarity (TF-IDF + cosine)
    3. Reasoning engine rules
    """
    df_scored = df.copy()

    # 8A. KB Profile Score
    df_scored['kb_score'] = df_scored['category'].map(user_profile).fillna(0)

    # 8B. ML Content Score
    user_text = " ".join([str(v) for v in answers.values()])
    ml_scores = get_ml_scores(user_text, vectorizer, job_vectors)
    df_scored['ml_score'] = ml_scores

    # 8C. Apply Reasoning Engine
    kb_profile, fired_rules, chain_explanations = apply_reasoning_engine(user_profile, answers, kb)
    df_scored['kb_score'] = df_scored['category'].map(kb_profile).fillna(0)

    # 8D. Normalize scores
    for col in ['kb_score', 'ml_score']:
        min_v, max_v = df_scored[col].min(), df_scored[col].max()
        if max_v > min_v:
            df_scored[col] = (df_scored[col] - min_v) / (max_v - min_v)
        else:
            df_scored[col] = 0

    # 8E. Ensemble (weighted combination)
    # KB gets higher weight when rules fire strongly, ML gets higher weight when content is rich
    rule_strength = len(fired_rules) / 10  # normalize
    kb_weight = 0.55 + (0.15 * min(rule_strength, 1.0))
    ml_weight = 1.0 - kb_weight

    df_scored['ensemble_score'] = (kb_weight * df_scored['kb_score']) + (ml_weight * df_scored['ml_score'])

    # 8F. Generate explanations for top jobs
    top_jobs = df_scored.nlargest(top_n, 'ensemble_score')
    explanations = {}
    for idx, row in top_jobs.iterrows():
        explanations[idx] = generate_job_explanation(
            row, kb_profile, answers, fired_rules, 
            row['ml_score'], row['kb_score']
        )

    return top_jobs, fired_rules, explanations, kb_weight, ml_weight

# ============================================================================
# 9. EVALUATION METRICS
# ============================================================================
def calculate_evaluation_metrics(df, top_jobs, user_profile):
    """Calculate recommendation quality metrics."""
    metrics = {}

    # Category concentration (diversity check)
    top_cats = top_jobs['category'].value_counts()
    metrics['category_diversity'] = len(top_cats) / len(top_jobs)

    # Average profile alignment of top jobs
    metrics['avg_profile_alignment'] = top_jobs['kb_score'].mean()

    # Average ML confidence
    metrics['avg_ml_confidence'] = top_jobs['ml_score'].mean()

    # Score spread (confidence in ranking)
    scores = top_jobs['ensemble_score'].values
    if len(scores) > 1:
        metrics['score_spread'] = scores[0] - scores[-1]
    else:
        metrics['score_spread'] = 0

    # Coverage: % of categories represented in top N
    all_cats = df['category'].nunique()
    metrics['coverage'] = len(top_cats) / all_cats

    return metrics

# ============================================================================
# 10. UI COMPONENTS
# ============================================================================
def render_questionnaire():
    """Render the multi-step questionnaire."""
    progress = st.session_state.current_question / 5
    st.progress(progress, text=f"Step {st.session_state.current_question} of 5")

    current_q = st.session_state.current_question
    q_data = QUESTIONNAIRE[current_q]

    st.subheader(q_data["question"])

    if q_data.get("allow_multiple", False):
        answer = st.multiselect(
            "Select up to 3 options:",
            options=list(q_data["options"].keys()),
            max_selections=3,
            key=f"q{current_q}",
            help="Choose the strengths that best describe you"
        )
    else:
        answer = st.radio(
            "Choose the option that best fits you:",
            options=list(q_data["options"].keys()),
            key=f"q{current_q}",
            label_visibility="visible"
        )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Back", disabled=(current_q == 1), use_container_width=True):
            st.session_state.current_question -= 1
            st.rerun()

    with col3:
        is_last = (current_q == 5)
        btn_label = "🎯 See My Career Matches" if is_last else "Next ➡️"

        if st.button(btn_label, disabled=(not answer), use_container_width=True):
            st.session_state.answers[current_q] = answer

            if is_last:
                # Compute everything
                st.session_state.user_profile = calculate_user_profile(st.session_state.answers, df)
                st.session_state.show_results = True
                st.rerun()
            else:
                st.session_state.current_question += 1
                st.rerun()

def render_results(df, kb):
    """Render the results dashboard."""
    st.success("✅ Analysis Complete! Here are your personalized career recommendations.")
    st.write("---")

    # Ensure ML model is ready
    if not st.session_state.ml_model_ready:
        with st.spinner("Training ML content model on O*NET data..."):
            vectorizer, job_vectors = train_ml_model(df)
            st.session_state.vectorizer = vectorizer
            st.session_state.job_vectors = job_vectors
            st.session_state.ml_model_ready = True

    # Get recommendations
    top_jobs, fired_rules, explanations, kb_w, ml_w = hybrid_recommendation(
        df, st.session_state.user_profile, st.session_state.answers, 
        kb, st.session_state.vectorizer, st.session_state.job_vectors, top_n=5
    )

    # Calculate metrics
    metrics = calculate_evaluation_metrics(df, top_jobs, st.session_state.user_profile)

    # ---- PROFILE SUMMARY ----
    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.subheader("📊 Your Career Profile")

        profile_data = pd.DataFrame(
            list(st.session_state.user_profile.items()),
            columns=['Category', 'Score']
        ).sort_values('Score', ascending=False)
        profile_data['Score'] = (profile_data['Score'] * 100).round(1)

        # Bar chart
        fig = px.bar(
            profile_data.head(8), x='Score', y='Category', orientation='h',
            color='Score', color_continuous_scale='Blues',
            title="Top Career Category Preferences"
        )
        fig.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🧠 System Architecture & Reasoning")

        # Show ensemble weights
        st.write(f"**Ensemble Weights:** Knowledge Base {kb_w*100:.0f}% | ML Model {ml_w*100:.0f}%")

        # Show fired rules
        with st.expander(f"🔍 View Fired Rules ({len(fired_rules)} rules activated)"):
            if fired_rules:
                for rule in fired_rules:
                    emoji = {"boost": "⬆️", "suppress": "⬇️", "chain": "🔗"}.get(rule["type"], "⚡")
                    st.write(f"{emoji} **{rule['type'].upper()}**: {rule['rule']}")
            else:
                st.write("No specific rules fired — recommendation based on questionnaire alignment.")

        # Show metrics
        with st.expander("📈 Recommendation Quality Metrics"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Category Diversity", f"{metrics['category_diversity']*100:.0f}%")
            c2.metric("KB Alignment", f"{metrics['avg_profile_alignment']*100:.0f}%")
            c3.metric("ML Confidence", f"{metrics['avg_ml_confidence']*100:.0f}%")
            c4.metric("Score Spread", f"{metrics['score_spread']*100:.1f}")

    st.write("---")

    # ---- JOB RECOMMENDATIONS ----
    st.subheader("💼 Top 5 Recommended Careers")

    for rank, (idx, row) in enumerate(top_jobs.iterrows(), 1):
        match_pct = min(row['ensemble_score'] * 100, 100)

        with st.container():
            col_main, col_score = st.columns([4, 1])

            with col_main:
                st.markdown(f"### {rank}. {row['job_title']}")
                st.write(f"**Category:** {row['category']} | **O*NET Code:** {row['O*NET-SOC Code']}")

                # Description
                desc = row['job_description'][:250]
                if len(row['job_description']) > 250:
                    desc += "..."
                st.write(f"**Description:** {desc}")

                # Skills
                skills = row['clean_skills'][:200]
                if len(row['clean_skills']) > 200:
                    skills += "..."
                st.write(f"**Key Skills & Interests:** {skills}")

                # Explanation
                with st.expander("🧩 Why was this recommended?"):
                    for reason in explanations.get(idx, ["No specific explanation available."]):
                        st.write(f"• {reason}")

                    st.write("---")
                    st.write(f"**Score Breakdown:**")
                    st.write(f"- Knowledge Base Profile Score: {row['kb_score']*100:.1f}%")
                    st.write(f"- ML Content Similarity Score: {row['ml_score']*100:.1f}%")
                    st.write(f"- Final Ensemble Score: {row['ensemble_score']*100:.1f}%")

            with col_score:
                st.metric("Match Score", f"{match_pct:.0f}%")

                # Mini gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=match_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 40], 'color': "lightgray"},
                               {'range': [40, 70], 'color': "yellow"},
                               {'range': [70, 100], 'color': "lightgreen"}]
                          }
                ))
                fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()

    # ---- COMPARISON VIEW ----
    with st.expander("🔬 Compare ML-Only vs KB-Only vs Hybrid"):
        comp_data = top_jobs[['job_title', 'category', 'kb_score', 'ml_score', 'ensemble_score']].copy()
        comp_data.columns = ['Job Title', 'Category', 'KB-Only', 'ML-Only', 'Hybrid Ensemble']
        comp_data = comp_data.round(3)
        st.dataframe(comp_data, use_container_width=True)

        fig_comp = px.bar(
            comp_data.melt(id_vars=['Job Title'], value_vars=['KB-Only', 'ML-Only', 'Hybrid Ensemble'],
                          var_name='Method', value_name='Score'),
            x='Job Title', y='Score', color='Method', barmode='group',
            title="Score Comparison Across Methods"
        )
        fig_comp.update_layout(height=400)
        st.plotly_chart(fig_comp, use_container_width=True)

    # ---- RESET ----
    st.write("---")
    if st.button("🔄 Start New Discovery", use_container_width=True):
        defaults = {
            'current_question': 1,
            'answers': {},
            'user_profile': None,
            'show_results': False,
            'ml_model_ready': False,
            'vectorizer': None,
            'job_vectors': None,
            'fired_rules': [],
            'explanations': {}
        }
        for key in ['current_question', 'answers', 'user_profile', 'show_results', 
                    'ml_model_ready', 'vectorizer', 'job_vectors', 'fired_rules', 'explanations']:
            st.session_state[key] = defaults.get(key, None) if key in defaults else False
        st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================
init_session_state()

try:
    df = load_data()
    kb = load_knowledge_base()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.title("🎯 SkillSync AI: Career Discovery Journey")
st.caption("University of Southern Mindanao — AI-Based Job Recommendation System | Hybrid ML + Knowledge-Based Reasoning")
st.write("---")

if not st.session_state.show_results:
    render_questionnaire()
else:
    render_results(df, kb)