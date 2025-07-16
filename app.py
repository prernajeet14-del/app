__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
from datetime import datetime
import time
import os
import json
from pathlib import Path

# Import the core logic from your app.py file
from logic import (
    MarketingAnalyticsOrchestrator,
    ReportGenerator,
    load_analysis_mappings_from_excel,
    create_rag_chain,
    llm,
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Marketing Analytics AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
def initialize_session_state():
    """Initialize all necessary session state variables."""
    if "current_step" not in st.session_state:
        st.session_state.current_step = "start"
    if "analysis_mappings" not in st.session_state:
        st.session_state.analysis_mappings = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "db_schema" not in st.session_state:
        st.session_state.db_schema = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = None
    if "analysis_info" not in st.session_state:
        st.session_state.analysis_info = None
    if "clarification_question" not in st.session_state:
        st.session_state.clarification_question = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "report_generator" not in st.session_state:
        st.session_state.report_generator = ReportGenerator()
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []
    # New state variables
    if "selected_churn_models" not in st.session_state:
        st.session_state.selected_churn_models = None
    if "clarification_complete" not in st.session_state:
        st.session_state.clarification_complete = False


# --- UI Rendering Functions ---

def render_start_page():
    """Renders the initial welcome page."""
    st.title("🚀 Welcome to the Marketing Analytics AI Assistant!")
    st.markdown("""
    This intelligent assistant, powered by CrewAI, LangChain, and Google's Gemini, streamlines complex marketing data analysis.
    
    **Follow the steps to get started:**
    1.  **Initiate Analysis:** Connect to the database and view available data.
    2.  **Select a Query:** Choose a business question and any model-specific options.
    3.  **Clarify Details:** Answer questions to refine the analysis and provide feedback.
    4.  **Review Report:** Get a comprehensive technical report with insights and recommendations.
    5.  **Ask Follow-ups:** Chat with the AI to dive deeper into the results.
    """)
    if st.button("Begin Analysis", type="primary", use_container_width=True):
        with st.spinner("Loading analysis options and connecting to database..."):
            try:
                mappings_file = 'question_mappings.xlsx'
                st.session_state.analysis_mappings = load_analysis_mappings_from_excel(mappings_file)
                if not st.session_state.analysis_mappings:
                    st.error("Failed to load `question_mappings.xlsx`. Please ensure the file is present and correctly formatted.")
                    return

                st.session_state.orchestrator = MarketingAnalyticsOrchestrator(st.session_state.analysis_mappings)
                
                schema_info = st.session_state.orchestrator.present_database_schema()
                if "error" in schema_info:
                    st.error(f"Database Connection Failed: {schema_info['error']}")
                    return
                
                st.session_state.db_schema = schema_info.get("schema_data", {})
                st.session_state.current_step = "select_query"
                st.rerun()

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

def render_query_selection_page():
    """Renders the page for selecting the business query and model options."""
    st.title("Step 1: Select a Business Query")
    
    with st.expander("View Database Schema", expanded=False):
        if st.session_state.db_schema:
            for table, columns in st.session_state.db_schema.items():
                st.markdown(f"**Table:** `{table}`")
                st.text(", ".join(columns))
        else:
            st.warning("Could not retrieve database schema.")

    st.markdown("### Choose the question you want to analyze:")
    analysis_options = list(st.session_state.analysis_mappings.keys())
    
    user_query = st.selectbox(
        "Select an analysis:",
        options=analysis_options,
        index=None,
        placeholder="Choose a business question..."
    )

    if user_query:
        st.session_state.user_query = user_query
        analysis_info = st.session_state.analysis_mappings[user_query]
        st.session_state.analysis_info = analysis_info
        st.info(f"You selected: **{st.session_state.user_query}**\n\nThis will perform a **{analysis_info['internal_name']}** analysis.")
        
        # --- Churn Model Selection ---
        if analysis_info['internal_name'] == 'Churn':
            st.markdown("#### Churn Model Options")
            selected_models = st.multiselect(
                "Select which prediction models to run (or leave blank to run all and compare):",
                options=["Logistic Regression", "Random Forest", "Gradient Boosting"],
                default=[],
                key="churn_model_selector"
            )
            # Store the selection in the session state immediately
            st.session_state.selected_churn_models = selected_models
            if selected_models:
                st.write("You have selected the following models:", selected_models)
            else:
                st.write("No specific models selected. The system will run all and find the best one.")


    # Only show the 'Next' button if a query has been chosen
    if st.session_state.user_query:
        if st.button("Next: Answer Clarifying Questions", type="primary", use_container_width=True):
            # The model selection is already stored, so we can proceed
            st.session_state.current_step = "clarification"
            st.rerun()

def render_clarification_page():
    """Renders the interactive clarification Q&A page."""
    st.title("Step 2: Clarify Analysis Details")
    st.markdown("To ensure the analysis is accurate, please answer the following questions.")
    
    # Display conversation history
    if st.session_state.orchestrator.conversation_history:
        for item in st.session_state.orchestrator.conversation_history:
            if item['question'] != "USER_REQUEST_MORE_QUESTIONS":
                st.chat_message("assistant").write(item['question'])
                st.chat_message("user").write(item['answer'])

    # Satisfaction check
    if st.session_state.clarification_complete:
        st.info("The initial set of questions is complete.")
        st.markdown("Are you satisfied with the questions asked so far?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, proceed to analysis", type="primary", use_container_width=True):
                st.session_state.current_step = "run_analysis"
                st.rerun()
        with col2:
            if st.button("❌ No, ask me more questions", use_container_width=True):
                st.session_state.orchestrator.conversation_history.append({
                    "question": "USER_REQUEST_MORE_QUESTIONS",
                    "answer": "The user was not satisfied and requested more clarifying questions."
                })
                st.session_state.clarification_question = None
                st.session_state.clarification_complete = False
                st.rerun()
        return

    # Get the next question if we don't have one
    if not st.session_state.clarification_question:
        with st.spinner("Thinking of the next question..."):
            st.session_state.clarification_question = st.session_state.orchestrator.get_next_clarifying_question(
                st.session_state.user_query,
                st.session_state.analysis_info
            )

    question = st.session_state.clarification_question
    
    if "I don't have any more questions" in question:
        st.session_state.clarification_complete = True
        st.rerun()

    # Display the current question
    st.chat_message("assistant").write(question)
    
    # Get user answer
    answer = st.text_input("Your Answer", key=f"answer_for_{question}")
    
    if st.button("Submit Answer", use_container_width=True):
        if answer:
            # Save history and reset for the next question
            st.session_state.orchestrator.conversation_history.append({"question": question, "answer": answer})
            st.session_state.clarification_question = None
            st.rerun()
        else:
            st.warning("Please provide an answer.")

def render_analysis_page():
    """Renders the analysis progress page and then the results."""
    st.title("Step 3: Running the Analysis")
    st.markdown("The AI-powered crew is now performing the analysis. This may take several minutes.")
    
    with st.spinner("🤖 The agent crew is working... Please wait."):
        progress_placeholder = st.empty()
        # This is a simplified progress indicator. The actual crewAI logs will appear in the console.
        progress_steps = [
            "Step 1: Data Analyst is loading and preprocessing data...",
            "Step 2: Data Analyst is performing EDA...",
            "Step 3: Feature Engineer is creating new features...",
            "Step 4: Analytics Expert is building models...",
            "Step 5: Business Strategist is synthesizing insights..."
        ]
        for i, step in enumerate(progress_steps):
            progress_placeholder.info(f"**Current Status:** {step}")
            time.sleep(3) # Simulate time passing for each step

        analysis_params = {
            "selected_churn_models": st.session_state.get('selected_churn_models')
        }

        results = st.session_state.orchestrator.run_analysis(
            st.session_state.user_query,
            st.session_state.analysis_info,
            st.session_state.db_schema,
            analysis_params
        )
        st.session_state.analysis_results = results

    if st.session_state.analysis_results and st.session_state.analysis_results.get("success"):
        st.success("🎉 Analysis complete! Generating report...")
        st.session_state.current_step = "show_report"
        st.rerun()
    else:
        st.error("❌ Analysis Failed!")
        error_details = st.session_state.analysis_results.get("error", "An unknown error occurred.")
        st.code(error_details)
        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

def render_report_page():
    """Renders the final analysis report and Q&A section."""
    st.title("Step 4: Analysis Report & Recommendations")

    results = st.session_state.analysis_results
    if not results:
        st.error("No analysis results found.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # This now correctly receives the full markdown content string from the generator
    md_content = st.session_state.report_generator.generate_markdown_report(results, timestamp)

    # --- Sidebar for Downloads ---
    st.sidebar.subheader("Download Artifacts")

    # Download button for ydata-profiling report
    eda_results = results.get('execution_steps', {}).get('exploratory_analysis', {})
    if eda_results and isinstance(eda_results, dict):
        profile_report_path = eda_results.get("profile_report_path")
        if profile_report_path and os.path.exists(profile_report_path):
            with open(profile_report_path, "rb") as fp:
                st.sidebar.download_button(
                    label="📥 Download Data Profile Report (.html)",
                    data=fp,
                    file_name=os.path.basename(profile_report_path),
                    mime="text/html"
                )

    # Intermediate Data Downloads (Cleaned and Engineered files)
    intermediate_files = results.get("intermediate_files", {})
    if intermediate_files:
        cleaned_path = intermediate_files.get("cleaned_data")
        engineered_path = intermediate_files.get("engineered_data")
        if cleaned_path and os.path.exists(cleaned_path):
            with open(cleaned_path, "rb") as fp:
                st.sidebar.download_button(
                    label="📥 Download Cleaned Data (.pkl)",
                    data=fp,
                    file_name="cleaned_data.pkl"
                )
        if engineered_path and os.path.exists(engineered_path):
            with open(engineered_path, "rb") as fp:
                st.sidebar.download_button(
                    label="📥 Download Engineered Data (.pkl)",
                    data=fp,
                    file_name="engineered_data.pkl"
                )

    # Final Report Downloads
    st.sidebar.download_button(
        label="📥 Download Full Report (.md)",
        data=md_content,
        file_name=f"{results['analysis_type']}_report_{timestamp}.md",
        mime="text/markdown",
    )
    json_report_path = st.session_state.report_generator.generate_json_report(results, timestamp)
    with open(json_report_path, "r", encoding="utf-8") as f:
        st.sidebar.download_button(
            label="📥 Download Raw Data (.json)",
            data=f.read(),
            file_name=os.path.basename(json_report_path),
            mime="application/json",
        )

    # Detailed RFM Report Download
    if results['analysis_type'] == 'RFM':
        detailed_rfm_path = st.session_state.report_generator.generate_detailed_rfm_report(results, timestamp)
        with open(detailed_rfm_path, "r", encoding="utf-8") as f:
            st.sidebar.download_button(
                label="📥 Download Detailed RFM Report (.md)",
                data=f.read(),
                file_name=os.path.basename(detailed_rfm_path),
                mime="text/markdown",
            )

    # Display the main text-based report on the page
    st.markdown(md_content, unsafe_allow_html=True)

    # --- Add a clear divider and header for the visuals section ---
    st.subheader("🖼️ Generated Visualizations", divider='rainbow')
    st.markdown("This section contains all charts generated during the analysis, such as data distributions, correlation heatmaps, and model-specific plots.")

    # Display generated visuals
    visual_folders = ["eda_visuals", "rfm_visuals", "segmentation_visuals"]
    found_visuals = False
    for folder in visual_folders:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                found_visuals = True
                # Use a more prominent header for each folder
                st.markdown(f"#### {folder.replace('_', ' ').title()}")
                for img in images:
                    image_path = os.path.join(folder, img)
                    if os.path.exists(image_path):
                        st.image(image_path, caption=img.replace('_', ' ').title(), use_column_width=True)

    if not found_visuals:
        st.info("No visualization files were generated for this analysis.")


    # --- RAG Q&A Section ---
    st.subheader("Step 5: Ask Follow-up Questions", divider='rainbow')
    st.markdown("Chat with the AI to get more details about the report.")

    if not st.session_state.rag_chain:
        with st.spinner("Building knowledge base from the report..."):
            st.session_state.rag_chain = create_rag_chain(st.session_state.analysis_results)

    if not st.session_state.rag_chain:
        st.error("Could not initialize the Q&A assistant.")
        return

    # Display chat history
    for msg in st.session_state.qa_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get new question
    if prompt := st.chat_input("Ask a question about the report..."):
        st.session_state.qa_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "I couldn't find an answer.")
                st.write(answer)
                st.session_state.qa_messages.append({"role": "assistant", "content": answer})
                
# --- Main App Controller ---

def main():
    initialize_session_state()

    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Current Step:** `{st.session_state.current_step.replace('_', ' ').title()}`")

    if st.sidebar.button("Restart Analysis", use_container_width=True):
        # Clear visuals and temp data folders on restart
        folders_to_clear = [
            "analysis_reports", "analysis_temp_data", "eda_visuals",
            "rfm_visuals", "segmentation_visuals", "profiling_reports"
        ]
        for folder in folders_to_clear:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    try:
                        os.remove(os.path.join(folder, file))
                    except OSError as e:
                        st.error(f"Error removing file {file}: {e}")

        st.session_state.clear()
        st.rerun()

    st.sidebar.info("Logs from the AI agents will be printed to your terminal console.")

    if st.session_state.current_step == "start":
        render_start_page()
    elif st.session_state.current_step == "select_query":
        render_query_selection_page()
    elif st.session_state.current_step == "clarification":
        render_clarification_page()
    elif st.session_state.current_step == "run_analysis":
        render_analysis_page()
    elif st.session_state.current_step == "show_report":
        render_report_page()

if __name__ == "__main__":
    main()
