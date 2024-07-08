import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y
    df['species'] = df['species'].apply(lambda x: iris.target_names[x])
    return df, iris

df, iris = load_data()

# Streamlit App
st.title("Iris Dataset SVM Classification")
st.write("SVM Classification algorithm is used to classify data points into different classes based on labeled training data.")
st.write("The SVM algorithm is well-suited for the Iris dataset because of its ability to handle clear class separations effectively. It minimizes overfitting risks, offers kernel options for nonlinear relationships, and retains model interpretability through support vectors. These qualities make SVMs a reliable choice for this classic classification problem.")

# Display the first few rows of the dataset
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
def visualize_data(dataframe):
    st.write("#### Visualize relations between features")
    fig = plot_pairplot(dataframe)
    st.pyplot(fig)

visualize_data(df)

# Split the data
@st.cache_data
def split_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].apply(lambda x: list(iris.target_names).index(x)).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)

# Standardize the features
@st.cache_resource
def get_scaler():
    scaler = StandardScaler()
    return scaler

scaler = get_scaler()
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
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Predict button
if st.sidebar.button("Predict"):

    # Standardize the user input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    prediction_species = iris.target_names[prediction][0]

    st.write(f'<h3>>>>Prediction: {prediction_species} </h3>', unsafe_allow_html=True)

    # Visualize user input in the pairplot
    st.write("Vizualisation of prediction:")
    df_with_input = pd.concat([df, input_df.assign(species='User Input')])
    palette = {name: color for name, color in zip(iris.target_names, sns.color_palette())}
    palette['User Input'] = 'red'
    
    @st.cache_resource
    def plot_input_pairplot(df_with_input, palette):
        fig = sns.pairplot(df_with_input, hue='species', palette=palette, markers=['o', 's', 'D', 'X'])
        return fig
    
    fig = plot_input_pairplot(df_with_input, palette)
    st.pyplot(fig)
