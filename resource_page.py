import streamlit as st
import pandas as pd


st.write("""### There are some resources about this survey""")
@st.cache_resource
def load_data():
    df = pd.read_csv("survey.csv")
    return df
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
my_df = load_data()
csv = convert_df(my_df)

def download_file():
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='servey.csv',
        mime='text/csv',
    )
def resource_page():
    st.subheader("Stackoverflow Developer Survey 2022")
    download_file()
    code = '''
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv("survey.csv")

        df.head()

        df.info()

        df["ConvertedCompYearly"]

        df =df[["Country", "EdLevel", "YearsCodePro", "ConvertedCompYearly"]]
        df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
        df.head()

        df = df[df["Salary"].notnull()]
        df.head()

        df.info()

        df = df.dropna()
        df.isnull().sum()


        df.info()

        def shorten_categories(categories, cutoff):
            categorical_map = {}
            for i in range(len(categories)):
                if categories.values[i] >= cutoff:
                    categorical_map[categories.index[i]] = categories.index[i]
                else:
                    categorical_map[categories.index[i]] = "Other"
            return categorical_map

        country_map = shorten_categories(df.Country.value_counts(), 400)
        df["Country"] = df["Country"].map(country_map)
        df.Country.value_counts()

        fig, ax = plt.subplots(1,1, figsize = (12, 7))
        df.boxplot("Salary", "Country", ax=ax)
        plt.suptitle("Salary (US$) vs Country", fontsize=18)
        plt.title("")
        plt.ylabel("Salary (US$)", fontsize=14)
        plt.xticks(rotation=90)
        plt.show()

        df = df[df["Salary"] <= 250000]
        df = df[df["Salary"] >= 10000]
        df = df[df["Country"] != "Other"]

        fig, ax = plt.subplots(1,1, figsize = (12, 7))
        df.boxplot("Salary", "Country", ax=ax)
        plt.suptitle("Salary (US$) vs Country", fontsize=18)
        plt.title("")
        plt.ylabel("Salary (US$)", fontsize=14)
        plt.xticks(rotation=90)
        plt.show()

        df["YearsCodePro"].unique()

        def clean_experience(x):
            if x == "More than 50 years":
                return 51
            elif x == "Less than 1 year":
                return 0.5
            else:
                return float(x)
        df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)

        df["YearsCodePro"].unique()

        df["EdLevel"].unique()

        def clean_education(x):
            if "Bachelor’s degree" in x:
                return "Bachelor’s degree"
            elif "Master’s degree" in x:
                return "Master’s degree"
            elif "Professional degree" in x or "Other doctoral degree" in x:
                return "Post grad"
            else:
                return "Less than a Bachelors"

        df["EdLevel"] = df["EdLevel"].apply(clean_education)

        df["EdLevel"].unique()

        from sklearn.preprocessing import LabelEncoder
        le_education = LabelEncoder()
        df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
        df["EdLevel"].unique()
        #le.classes_

        le_country = LabelEncoder()
        df['Country'] = le_country.fit_transform(df['Country'])
        df["Country"].unique()

        X = df.drop("Salary", axis=1)
        y = df["Salary"]

        from sklearn.linear_model import LinearRegression
        linear_reg = LinearRegression()
        linear_reg.fit(X, y.values)

        y_pred = linear_reg.predict(X)

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        error = np.sqrt(mean_squared_error(y, y_pred))

        error

        from sklearn.tree import DecisionTreeRegressor
        dec_tree_reg = DecisionTreeRegressor(random_state=0)
        dec_tree_reg.fit(X, y.values)

        y_pred = dec_tree_reg.predict(X)

        error = np.sqrt(mean_squared_error(y, y_pred))
        print("${:,.02f}".format(error))

        from sklearn.ensemble import RandomForestRegressor
        random_forest_reg = RandomForestRegressor(random_state=0)
        random_forest_reg.fit(X, y.values)

        y_pred = random_forest_reg.predict(X)

        error = np.sqrt(mean_squared_error(y, y_pred))
        print("${:,.02f}".format(error))

        from sklearn.model_selection import GridSearchCV

        max_depth = [None, 2,4,6,8,10,12]
        parameters = {"max_depth": max_depth}

        regressor = DecisionTreeRegressor(random_state=0)
        gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
        gs.fit(X, y.values)

        regressor = gs.best_estimator_

        regressor.fit(X, y.values)
        y_pred = regressor.predict(X)
        error = np.sqrt(mean_squared_error(y, y_pred))
        print("${:,.02f}".format(error))

        X

        # country, edlevel, yearscode
        X = np.array([["United States of America",'Master’s degree', 16 ]])
        X

        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
        X

        y_pred = regressor.predict(X)
        y_pred

        import pickle

        data = {"model": regressor, "le_country": le_country, "le_education": le_education}
        with open('saved_steps.pkl', 'wb') as file:
            pickle.dump(data, file)

        with open('saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)

        regressor_loaded = data["model"]
        le_country = data["le_country"]
        le_education = data["le_education"]

        y_pred = regressor_loaded.predict(X)
        y_pred
'''
    st.code(code, language='python')
    