import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf


model = tf.keras.models.load_model('BreastCancer_DL.h5')

def predict_survival(age_at_diagnosis, overall_survival_months, lymph_nodes_examined_positive, tumor_size, tumor_stage, brca1, brca2, tp53, pten, egfr):
    df = pd.read_csv('METABRIC_RNA_Mutation_Signature_Preprocessed.csv', delimiter=',')
    #Convert Categorical values to Numerical values
    #features_to_drop = df.columns[52:]
    #df = df.drop(features_to_drop, axis=1)
    #all_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    #unwanted_columns = ['patient_id', 'death_from_cancer']
    #all_categorical_columns = [ele for ele in all_categorical_columns if ele not in unwanted_columns]
    #dummies_df = pd.get_dummies(df.drop('patient_id',axis=1),columns = all_categorical_columns,dummy_na=True)
    #dummies_df.dropna(inplace = True)
    X = df.drop(['death_from_cancer', 'overall_survival'], axis = 1)

    TestData = X.iloc[[9],:]
    TestData [['age_at_diagnosis', 'overall_survival_months','lymph_nodes_examined_positive', 'tumor_size', 'tumor_stage', 'brca1', 'brca2', 'tp53', 'pten', 'egfr']] = [age_at_diagnosis, overall_survival_months, lymph_nodes_examined_positive,tumor_size,tumor_stage,brca1,brca2,tp53,pten,egfr]
    TestData = np.asarray(TestData).astype(np.float32)
    prediction = model.predict(TestData)
    pred='{0:.{1}f}'.format(prediction[0][0],2)
    return float(pred)

def main():
    st.title("Streamlit Tutorial")

    with st.sidebar:
        st.write("This tool employs CNN.")

    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">AI Breast Cancer Prognosis Web Tool </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    age_at_diagnosis = st.text_input("Age At Diagnosis", "")
    overall_survival_months = st.text_input("Overall Survival Months","")
    lymph_nodes_examined_positive = st.text_input("Positive Lymph Nodes","")
    tumor_size = st.text_input("Tumor Size", "")
    tumor_stage = st.text_input("Tumor Stage", "")
    brca1 = st.text_input("brca1", "")
    brca2 = st.text_input("brca2", "")
    tp53 = st.text_input("tp53","")
    pten = st.text_input("pten","")
    egfr = st.text_input("egfr", "")

    living_html = """
    <div style="background-color:#F4D03F ;padding:10px">
    <h2 style="color:white;text-align:center;">High Risk</h2>
    </div>
    """

    death_html = """
    <div style="background-color:#F08080 ;padding:10px">
    <h2 style="color:white;text-align:center;">Low Risk</h2>
    </div>
    """

    if st.button("Predict"):
        output=predict_survival(age_at_diagnosis, overall_survival_months,lymph_nodes_examined_positive, tumor_size, tumor_stage, brca1, brca2, tp53,pten, egfr)
        st.success('The probability of survival is {}'.format(output))

        if output > .5:
            st.markdown(death_html, unsafe_allow_html=True)
        else:
            st.markdown(living_html, unsafe_allow_html=True)
    


if __name__ == '__main__':
    main()