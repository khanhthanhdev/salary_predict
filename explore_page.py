import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def education(x):

    if 'Bachelor’s degree' in x:
        return "Bachelor's degree"
    elif 'Master’s degree' in x:
        return "Master's degree"
    elif 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    else:
        return 'Less than a Bachelors'


@st.cache_resource
def load_data():
    df = pd.read_csv("survey.csv")
    df = df[df["ConvertedCompYearly"].notnull()]
    df =df.dropna()
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 800)
    df["Country"] = df["Country"].map(country_map)

    df = df[df["ConvertedCompYearly"] <= 250000]
    df = df[df["ConvertedCompYearly"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(experience)
    df["EdLevel"] = df["EdLevel"].apply(education)

    return df

df = load_data()
def show_explore_page():


    st.write(
        """
    ### Stack Overflow Developer Survey 2022
    """
    )

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal") 
    st.write("""#### Number of Data from countries""")
    st.pyplot(fig1)
    
    st.write(
        """
    #### Mean Salary In Other Country
    """
    )

    data = df.groupby(["Country"])["ConvertedCompYearly"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["ConvertedCompYearly"].mean().sort_values(ascending=True)
    st.line_chart(data)


if __name__ == "__main__":
    show_explore_page()
