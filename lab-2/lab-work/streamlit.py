import streamlit as st
import joblib
model=joblib.load("model.joblib")

st.title("News Categorization Using BBC Dataset🗞️")
st.markdown("## Enter the news below:")
user_input = st.text_area( label=' ',max_chars=1000,height=300)
if st.button('Predict Category'):
    pred = model.predict([user_input])[0]
    st.success(f'News Category: {pred}')