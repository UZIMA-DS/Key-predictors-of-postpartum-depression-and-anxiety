import streamlit as st
import pandas as pd
import joblib

model = model = joblib.load("logistic_regression_trained_with_11_model.pkl")

st.title("Maternal & Child Risk Screening")

# ---- Depression items (3 questions) ----
st.subheader("Depression symptoms")
dep_options = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}
dep1 = st.selectbox("Little interest or pleasure in doing things", list(dep_options.keys()))
dep2 = st.selectbox("Feeling down, depressed, or hopeless", list(dep_options.keys()))
dep3 = st.selectbox("Thought that you would be better off dead or of hurting yourself in some way", list(dep_options.keys()))

depression_total = dep_options[dep1]+  dep_options[dep2] +  dep_options[dep3]

# ---- Anxiety items (2 questions) ----
st.subheader("Anxiety symptoms")
anx_options = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}
anx1 = st.selectbox("In the last 2 weeks, have you felt nervous, anxious or on edge?", list(anx_options.keys()))
anx2 = st.selectbox("In the last 2 weeks, have you not been able to stop or control worrying?", list(anx_options.keys()))

anxiety_total = anx_options[anx1] + anx_options[anx2]

# ---- home screening items (2 questions) ----
st.subheader("home environment screening")
env_options = {
    "No": 0,
    "Yes": 1,
    
}
env1 = st.selectbox("Una mnyama yeyote mnayemueka kwa nyumba kuwafurahisha?", list(env_options.keys()))
env2 = st.selectbox("Je, una marafiki wowote wenye watoto wenye umri sawa na mtoto wako?", list(env_options.keys()))



# ----Insufficient food intake and its physical consequences items (3 questions) ----
st.subheader("Insufficient food intake from food insecurity scale")
food_options = {
  
   "Rarely (once or twice in the past four weeks)": 1,
    "Sometimes (three to ten times in the past four weeks)": 2,
    "Often (more than ten times in the past four weeks)": 3
}
food1 = st.selectbox("Did you or any household member have to eat a smaller meal than you felt you needed because there was not enough food?", list(food_options.keys()))
food2 = st.selectbox("Did you or any household member have to eat fewer meals in a day because there was not enough food?" , list(food_options.keys()))
food3 = st.selectbox("Was there ever no food to eat of any kind in your household because of a lack of resources to get food?", list(food_options.keys()))
food4 = st.selectbox("Did you or any household member go to sleep at night hungry because there was not enough food?", list(food_options.keys()))
food5 = st.selectbox("Did you or any household member go a whole day and night without eating anything because there was not enough food?", list(food_options.keys()))

food_total = food_options[food1] + food_options[food2] + food_options[food3]+food_options[food4]+food_options[food5]

# Other inputs
Muac = st.number_input("Maternal Muac", 10.0, 60.0, 20.0)
Muac_child = st.number_input("Child Muac", 5.0, 20.0, 20.0)
child_age = st.number_input("Child age (months)", 0, 6, 6)
maternal_age = st.number_input("Maternal age", 11, 70, 30)
Num_child = st.number_input("Number of children", 0, 30, 30)
head_cirm = st.number_input("Child head circunference", 10, 100, 30)
if st.button("Predict"):

    X = pd.DataFrame([{
        'anxiety_with2_items': anxiety_total,
        'hs9_1':  env_options[env1],
      'depression_with3_items': depression_total,
        "Maternal.age..years.": maternal_age,
         'Maternal.muac': Muac,
          "Child.age..months.": child_age,
          'Household_food_insecurity_score_new5': food_total,
          'Child.head.circumference.in.cm':head_cirm,
        'Total.number.of.children':Num_child,
        'Child.muac':Muac_child ,
        'hs21_1':  env_options[env2]
       
    }])

    child_label = model.predict(X)[0]

    #st.write(f"Predicted probability: {prob:.3f}")

    if  child_label == 1:
        st.warning("Child is off-track")
    else:
        st.success("Child is on-track")