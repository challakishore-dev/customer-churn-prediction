
import streamlit as st, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction ")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")
model=load_model()

tab1,tab2,tab3=st.tabs(["Prediction","Analytics","About"])

with tab1:
    st.subheader("Predict Churn Risk")
    c1,c2,c3=st.columns(3)
    tenure=c1.slider("Tenure (months)",1,72,12)
    monthly=c2.slider("Monthly Charges",20,120,75)
    support=c3.slider("Support Tickets",0,10,1)
    contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
    internet=st.selectbox("Internet Service",["DSL","Fiber","None"])
    paperless=st.selectbox("Paperless Billing",["Yes","No"])
    if st.button("Predict"):
        row=pd.DataFrame([{
            "tenure":tenure,"MonthlyCharges":monthly,"SupportTickets":support,
            "Contract":contract,"InternetService":internet,"PaperlessBilling":paperless
        }])
        p=float(model.predict_proba(row)[0,1])
        st.metric("Churn Probability", f"{p*100:.1f}%")
        if p>0.65: st.error("High Risk Customer")
        elif p>0.35: st.warning("Medium Risk Customer")
        else: st.success("Low Risk Customer")

with tab2:
    df=pd.read_csv("data/churn.csv")
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots()
        sns.countplot(x="Churn",data=df,ax=ax)
        ax.set_title("Churn Distribution")
        st.pyplot(fig)
    with c2:
        fig,ax=plt.subplots()
        sns.boxplot(x="Churn",y="MonthlyCharges",data=df,ax=ax)
        ax.set_title("Charges vs Churn")
        st.pyplot(fig)
    fig,ax=plt.subplots()
    sns.histplot(data=df,x="tenure",hue="Churn",multiple="stack",ax=ax)
    ax.set_title("Tenure Impact")
    st.pyplot(fig)

with tab3:
    st.markdown("""
    ### Business Use Case
    Predict customers likely to leave and take retention actions.

    **Actions**
    - Discount offers
    - Support callback
    - Plan upgrade/downgrade
    - Loyalty rewards
    """)
