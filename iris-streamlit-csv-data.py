import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset from CSV
@st.cache_data
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Streamlit App
st.title("Iris Dataset SVM Classification")
st.write("SVM Classification algorithm is used to classify data points into different classes based on labeled training data.")
st.write("The SVM algorithm is well-suited for the Iris dataset because of its ability to handle clear class separations effectively. It minimizes overfitting risks, offers kernel options for nonlinear relationships, and retains model interpretability through support vectors. These qualities make SVMs a reliable choice for this classic classification problem.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("#### Sample data of Iris dataset:")
    st.write(df.head())

    # Display summary statistics
    st.write("#### Summary statistics of the dataset:")
    st.write(df.describe())

    # Check the distribution of the target variable
    @st.cache_resource
    def plot_distribution(df):
        fig, ax = plt.subplots()
        sns.countplot(x='species', data=df, ax=ax)
        return fig

    # st.write("#### Distribution of the target variable:")
    # fig = plot_distribution(df)
    # st.pyplot(fig)

    # Pairplot to visualize relationships between features
    @st.cache_resource
    def plot_pairplot(df):
        fig = sns.pairplot(df, hue='species')
        return fig

    @st.cache_resource
    def visualize_data(dataframe, input_df=None):
        if input_df is not None:
            dataframe = pd.concat([dataframe, input_df.assign(species='User Input')])
        st.write("#### Visualize relations between features")
        fig = plot_pairplot(dataframe)
        st.pyplot(fig)

    visualize_data(df)

    # Split the data
    @st.cache_data
    def split_data(df):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = split_data(df)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define and train the model
    @st.cache_resource
    def train_model(X_train, y_train):
        model = SVC(kernel='linear', random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and display accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'<h4>Accuracy: {accuracy:.2%}</h4>', unsafe_allow_html=True)

    # User input for prediction
    st.sidebar.header('Input Data for Prediction')

    def user_input_features():
        inputs = {}
        for col in df.columns[:-1]:
            inputs[col] = st.sidebar.slider(f'{col}', float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        data = pd.DataFrame([inputs])
        return data

    input_df = user_input_features()

    # Predict button
    if st.sidebar.button("Predict"):

        # Standardize the user input
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)
        st.write(f'<h3>>>>Prediction: {prediction[0]}</h3>', unsafe_allow_html=True)

        # Visualize user input in the pairplot
        visualize_data(df, input_df)
else:
    st.write("Upload a CSV file to start analyzing the Iris dataset.")
