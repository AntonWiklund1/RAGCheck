import streamlit as st
import pandas as pd
import glob
import plotly.express as px
import os
import sys
# Add path to src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

folder = "results" # Folder where results are stored


st.set_page_config(
    page_title="RAG Evaluation Results",
    page_icon="📊",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Add this after your imports and before the title
def clear_results_folder(folder):
    try:
        files = glob.glob(f"{folder}/*.csv")
        for file in files:
            os.remove(file)
        return True
    except Exception as e:
        st.error(f"Error clearing results: {str(e)}")
        return False

def get_sorted_csv_files(folder):
    csv_files = glob.glob(f"{folder}/*.csv")
    # Sort files by timestamp (newest first)
    return sorted(csv_files, key=lambda x: os.path.getmtime(x), reverse=True)

st.title("RAG Evaluation Results Visualization")

# Add this after the title
with st.sidebar:
    st.subheader("Maintenance")
    # Use session state to track confirmation state
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False

    if not st.session_state.confirm_delete:
        if st.button("Clear Results Folder"):
            st.session_state.confirm_delete = True
            st.rerun()
    
    else:
        st.warning("⚠️ Are you sure you want to delete all results?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete all"):
                if clear_results_folder(folder):
                    st.success("Results folder cleared!")
                    st.session_state.confirm_delete = False
                    st.rerun()
        with col2:
            if st.button("No, cancel"):
                st.session_state.confirm_delete = False
                st.rerun()

# Get all CSV files in the test folder
csv_files = get_sorted_csv_files(folder)

# Add refresh button and file selector in columns
if st.button("🔄 Refresh files"):
    st.rerun()
selected_file = st.selectbox("Select a results file:", csv_files)

if selected_file:
    # Read the selected CSV file
    df = pd.read_csv(selected_file)
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    
    # Add error handling and data cleaning for scores
    try:
        # Convert score column to numeric, forcing errors to NaN
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        
        # Show any problematic rows
        invalid_scores = df[df['score'].isna()]
        if not invalid_scores.empty:
            st.error("⚠️ Found invalid scores in the following rows:")
            st.dataframe(invalid_scores[['question_num', 'question', 'score']])
        
        # Calculate average excluding NaN values
        avg_score = df['score'].mean()
        if pd.isna(avg_score):
            st.warning("Could not calculate average score - all values are invalid")
        else:
            st.metric("Average Score", f"{avg_score:.2f}/10")
            
    except Exception as e:
        st.error(f"Error processing scores: {str(e)}")
        st.write("Raw score values:", df['score'].unique())
    
    # Score Distribution
    st.subheader("Score Distribution")
    fig_hist = px.histogram(
        df, 
        x="score",
        nbins=2,
        title="Distribution of Scores",
        labels={'score': 'Score', 'count': 'Frequency'}
    )
    st.plotly_chart(fig_hist)
    
    # Detailed Results Table
    st.subheader("Detailed Results")
    st.dataframe(
        df[['question_num', 'score', 'question', 'explanation', 'rag_response', 'expected_answer']]
        .sort_values('score', ascending=False)
    )
    
    # Download Results
    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False),
        file_name=os.path.basename(selected_file),
        mime='text/csv'
    )
