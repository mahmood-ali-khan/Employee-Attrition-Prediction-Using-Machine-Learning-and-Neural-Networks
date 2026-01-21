import streamlit as st

# Page config
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="🤖",
    layout="centered"
)

# Sidebar
st.sidebar.title("⚙ Model Settings")
st.sidebar.info("Select a machine learning model and run predictions.")

# Main title
st.markdown("<h1 style='text-align: center;'>🤖 ML Model Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Interactive Streamlit Application</h4>", unsafe_allow_html=True)
st.divider()

# Welcome section
st.markdown("### 👋 Welcome")
st.write("This application allows you to select and run different machine learning models interactively.")

# Model selection section
st.markdown("### 🔍 Choose Your Model")

models = st.selectbox(
    "Select a Machine Learning Model:",
    ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Networks"]
)

# Show selected model
st.success(f"✅ You selected: **{models}**")

# Run button
if st.button("🚀 Run Model"):
    st.info(f"Running **{models}** model...")
    st.success("🎉 Model executed successfully!")

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by Mahmood Ali Khan | Streamlit ML App</p>",
    unsafe_allow_html=True
)