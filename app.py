# ==========================================================
# WORKPLACE MENTAL-HEALTH STREAMLIT APPLICATION
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="Workplace Mental Health Analytics",
    page_icon="🧠",
    layout="wide"
)


# ----------------------------------------------------------
# CUSTOM FORMATTING
# ----------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: #F5F8FC;
        border: 1px solid #DDE5EE;
        border-radius: 10px;
        padding: 14px;
    }

    .insight-box {
        background-color: #EEF6FF;
        border-left: 5px solid #1F77B4;
        border-radius: 5px;
        padding: 12px;
        margin: 10px 0 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------------------------
# DATA CLEANING
# ----------------------------------------------------------

def standardize_gender(value):
    """Convert inconsistent gender responses into reporting groups."""

    if pd.isna(value):
        return "Not reported"

    gender = str(value).strip().lower()

    if gender in {"a little about you", "p", ""}:
        return "Not reported"

    other_markers = [
        "trans", "non-binary", "nonbinary", "queer",
        "genderqueer", "agender", "androgyne",
        "enby", "fluid", "neuter", "unsure"
    ]

    if any(marker in gender for marker in other_markers):
        return "Other / Non-binary"

    female_values = {
        "f", "female", "woman", "femake", "femail",
        "cis female", "female (cis)", "cis-female/femme"
    }

    if (
        gender in female_values
        or "female" in gender
        or "woman" in gender
    ):
        return "Female"

    male_values = {
        "m", "male", "man", "make", "mal", "maile",
        "msle", "mail", "malr", "cis male",
        "male (cis)", "cis man", "male-ish",
        "guy (-ish) ^_^", "something kinda male?"
    }

    if gender in male_values or "male" in gender:
        return "Male"

    return "Other / Non-binary"


@st.cache_data
def load_and_prepare_data(file_source):
    """Load, validate and clean the survey dataset."""

    try:
        # Load the supplied CSV file.
        data = pd.read_csv(file_source, low_memory=False)

        # Confirm that the file contains records.
        if data.empty:
            raise ValueError("The uploaded dataset is empty.")

        required_columns = {
            "Timestamp", "Age", "Gender", "Country",
            "state", "self_employed", "family_history",
            "treatment", "work_interfere", "no_employees",
            "remote_work", "tech_company", "benefits",
            "care_options", "wellness_program", "seek_help",
            "anonymity", "leave",
            "mental_health_consequence",
            "phys_health_consequence", "coworkers",
            "supervisor", "mental_health_interview",
            "phys_health_interview", "mental_vs_physical",
            "obs_consequence", "comments"
        }

        missing_columns = required_columns.difference(data.columns)

        if missing_columns:
            raise ValueError(
                f"Required columns are missing: "
                f"{sorted(missing_columns)}"
            )

        # Preserve the source and remove exact duplicates.
        cleaned = data.copy(deep=True)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        # Remove unnecessary spaces from text values.
        object_columns = cleaned.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in object_columns:
            cleaned[column] = cleaned[column].apply(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )

        # Convert timestamp into datetime.
        cleaned["Timestamp"] = pd.to_datetime(
            cleaned["Timestamp"],
            errors="coerce"
        )

        # Convert unrealistic ages into missing values.
        cleaned["Age"] = pd.to_numeric(
            cleaned["Age"],
            errors="coerce"
        )

        cleaned.loc[
            ~cleaned["Age"].between(18, 80),
            "Age"
        ] = np.nan

        # Create age groups.
        cleaned["Age_Group"] = pd.cut(
            cleaned["Age"],
            bins=[17, 24, 34, 44, 54, 64, 80],
            labels=[
                "18–24", "25–34", "35–44",
                "45–54", "55–64", "65–80"
            ]
        )

        # Standardize gender responses.
        cleaned["Gender_Group"] = cleaned["Gender"].apply(
            standardize_gender
        )

        # Handle state values according to applicability.
        cleaned.loc[
            cleaned["Country"].ne("United States")
            & cleaned["state"].isna(),
            "state"
        ] = "Not applicable"

        cleaned.loc[
            cleaned["Country"].eq("United States")
            & cleaned["state"].isna(),
            "state"
        ] = "Not reported"

        # Retain unknown categorical responses explicitly.
        cleaned["self_employed"] = (
            cleaned["self_employed"]
            .fillna("Not reported")
        )

        cleaned["work_interfere"] = (
            cleaned["work_interfere"]
            .fillna("Not reported")
        )

        # Create a numerical treatment indicator.
        cleaned["Treatment_Flag"] = (
            cleaned["treatment"]
            .map({"Yes": 1, "No": 0})
        )

        if cleaned["Treatment_Flag"].isna().any():
            raise ValueError(
                "Unexpected values were found in treatment."
            )

        # Calculate a workplace-support score from 0 to 5.
        support_columns = [
            "benefits",
            "care_options",
            "wellness_program",
            "seek_help",
            "anonymity"
        ]

        cleaned["Support_Score"] = sum(
            cleaned[column].eq("Yes").astype(int)
            for column in support_columns
        )

        return cleaned

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        raise RuntimeError(
            f"Dataset preparation failed: {error}"
        ) from error


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------

data_path = Path(__file__).parent / "survey.csv"

try:
    if not data_path.exists():
        st.error(
            "survey.csv was not found. Place it in the same "
            "folder as app.py."
        )
        st.stop()

    df = load_and_prepare_data(data_path)

except RuntimeError as error:
    st.error(str(error))
    st.stop()


# ----------------------------------------------------------
# APPLICATION HEADER
# ----------------------------------------------------------

st.title("🧠 Workplace Mental Health Analytics")

st.caption(
    "Analysis of employee treatment, workplace support, "
    "care awareness and mental-health disclosure concerns."
)


# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------

st.sidebar.header("Dashboard Filters")

country_options = sorted(df["Country"].dropna().unique())

selected_countries = st.sidebar.multiselect(
    "Select country",
    options=country_options,
    default=country_options
)

gender_options = sorted(
    df["Gender_Group"].dropna().unique()
)

selected_genders = st.sidebar.multiselect(
    "Select gender group",
    options=gender_options,
    default=gender_options
)

company_options = sorted(
    df["no_employees"].dropna().unique()
)

selected_company_sizes = st.sidebar.multiselect(
    "Select organization size",
    options=company_options,
    default=company_options
)

# Apply all selected filters.
filtered_df = df[
    df["Country"].isin(selected_countries)
    & df["Gender_Group"].isin(selected_genders)
    & df["no_employees"].isin(selected_company_sizes)
].copy()

if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Please change the filter selection."
    )
    st.stop()

st.sidebar.success(
    f"{len(filtered_df):,} respondents selected"
)


# ----------------------------------------------------------
# KPI CALCULATIONS
# ----------------------------------------------------------

total_respondents = len(filtered_df)

treatment_rate = (
    filtered_df["Treatment_Flag"].mean() * 100
)

interference_rate = (
    filtered_df["work_interfere"]
    .isin(["Sometimes", "Often"])
    .mean() * 100
)

benefit_uncertainty = (
    filtered_df["benefits"]
    .eq("Don't know")
    .mean() * 100
)


# ----------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Respondents",
    f"{total_respondents:,}"
)

kpi2.metric(
    "Treatment Rate",
    f"{treatment_rate:.1f}%"
)

kpi3.metric(
    "Frequent Work Interference",
    f"{interference_rate:.1f}%"
)

kpi4.metric(
    "Unaware of Benefits",
    f"{benefit_uncertainty:.1f}%"
)


# ----------------------------------------------------------
# DASHBOARD TABS
# ----------------------------------------------------------

overview_tab, support_tab, drivers_tab, action_tab = st.tabs(
    [
        "Overview",
        "Workplace Support",
        "Treatment Drivers",
        "Recommendations"
    ]
)


# ----------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------

with overview_tab:

    st.subheader("Respondent Overview")

    chart1, chart2 = st.columns(2)

    with chart1:

        treatment_counts = (
            filtered_df["treatment"]
            .value_counts()
            .rename_axis("Treatment")
            .reset_index(name="Respondents")
        )

        treatment_chart = px.bar(
            treatment_counts,
            x="Treatment",
            y="Respondents",
            color="Treatment",
            color_discrete_map={
                "Yes": "#2A9D8F",
                "No": "#E76F51"
            },
            text="Respondents",
            title="Mental-Health Treatment Distribution"
        )

        treatment_chart.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            treatment_chart,
            use_container_width=True
        )

        st.info(
            f"{treatment_rate:.1f}% of the selected respondents "
            "reported seeking mental-health treatment."
        )

    with chart2:

        # Plotly automatically ignores missing ages.
        age_chart = px.histogram(
            filtered_df.dropna(subset=["Age"]),
            x="Age",
            nbins=18,
            color_discrete_sequence=["#457B9D"],
            title="Respondent Age Distribution"
        )

        st.plotly_chart(
            age_chart,
            use_container_width=True
        )

        median_age = filtered_df["Age"].median()

        st.info(
            f"The median valid respondent age is "
            f"{median_age:.0f} years."
        )

    work_order = [
        "Never",
        "Rarely",
        "Sometimes",
        "Often",
        "Not reported"
    ]

    work_counts = (
        filtered_df["work_interfere"]
        .value_counts()
        .reindex(work_order, fill_value=0)
        .rename_axis("Work Interference")
        .reset_index(name="Respondents")
    )

    work_chart = px.bar(
        work_counts,
        x="Work Interference",
        y="Respondents",
        text="Respondents",
        color="Respondents",
        color_continuous_scale="Purples",
        title="Mental-Health Interference With Work"
    )

    st.plotly_chart(
        work_chart,
        use_container_width=True
    )


# ----------------------------------------------------------
# TAB 2: WORKPLACE SUPPORT
# ----------------------------------------------------------

with support_tab:

    st.subheader("Workplace Support and Awareness")

    support1, support2 = st.columns(2)

    with support1:

        benefits_counts = (
            filtered_df["benefits"]
            .value_counts()
            .rename_axis("Benefits")
            .reset_index(name="Respondents")
        )

        benefits_chart = px.bar(
            benefits_counts,
            x="Benefits",
            y="Respondents",
            color="Benefits",
            text="Respondents",
            color_discrete_map={
                "Yes": "#2A9D8F",
                "No": "#E76F51",
                "Don't know": "#E9C46A"
            },
            title="Mental-Health Benefit Availability"
        )

        benefits_chart.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            benefits_chart,
            use_container_width=True
        )

    with support2:

        care_counts = (
            filtered_df["care_options"]
            .value_counts()
            .rename_axis("Care Options")
            .reset_index(name="Respondents")
        )

        care_chart = px.bar(
            care_counts,
            x="Care Options",
            y="Respondents",
            color="Care Options",
            text="Respondents",
            color_discrete_map={
                "Yes": "#2A9D8F",
                "No": "#E76F51",
                "Not sure": "#E9C46A"
            },
            title="Awareness of Care Options"
        )

        care_chart.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            care_chart,
            use_container_width=True
        )

    leave_order = [
        "Very easy",
        "Somewhat easy",
        "Don't know",
        "Somewhat difficult",
        "Very difficult"
    ]

    leave_counts = (
        filtered_df["leave"]
        .value_counts()
        .reindex(leave_order, fill_value=0)
        .rename_axis("Leave")
        .reset_index(name="Respondents")
    )

    leave_chart = px.bar(
        leave_counts,
        x="Leave",
        y="Respondents",
        text="Respondents",
        color="Respondents",
        color_continuous_scale="RdYlGn_r",
        title="Ease of Taking Mental-Health Leave"
    )

    st.plotly_chart(
        leave_chart,
        use_container_width=True
    )


# ----------------------------------------------------------
# TAB 3: TREATMENT DRIVERS
# ----------------------------------------------------------

with drivers_tab:

    st.subheader("Factors Associated With Treatment")

    driver1, driver2 = st.columns(2)

    with driver1:

        family_rate = (
            filtered_df.groupby(
                "family_history"
            )["Treatment_Flag"]
            .mean()
            .mul(100)
            .reset_index(name="Treatment Rate")
        )

        family_chart = px.bar(
            family_rate,
            x="family_history",
            y="Treatment Rate",
            color="family_history",
            text_auto=".1f",
            color_discrete_map={
                "Yes": "#2A9D8F",
                "No": "#E76F51"
            },
            title="Treatment Rate by Family History"
        )

        family_chart.update_layout(
            showlegend=False,
            yaxis_title="Treatment rate (%)"
        )

        st.plotly_chart(
            family_chart,
            use_container_width=True
        )

    with driver2:

        interference_rate_df = (
            filtered_df.groupby(
                "work_interfere",
                observed=True
            )["Treatment_Flag"]
            .mean()
            .mul(100)
            .reindex(work_order)
            .dropna()
            .reset_index(name="Treatment Rate")
        )

        interference_chart = px.line(
            interference_rate_df,
            x="work_interfere",
            y="Treatment Rate",
            markers=True,
            text="Treatment Rate",
            title="Treatment Rate by Work Interference"
        )

        interference_chart.update_traces(
            line_color="#8064A2",
            texttemplate="%{text:.1f}%"
        )

        interference_chart.update_layout(
            xaxis_title="Work interference",
            yaxis_title="Treatment rate (%)"
        )

        st.plotly_chart(
            interference_chart,
            use_container_width=True
        )

    support_chart = px.box(
        filtered_df,
        x="treatment",
        y="Support_Score",
        color="family_history",
        category_orders={
            "treatment": ["No", "Yes"],
            "family_history": ["No", "Yes"]
        },
        color_discrete_map={
            "Yes": "#2A9D8F",
            "No": "#E76F51"
        },
        title=(
            "Support Score by Treatment and Family History"
        )
    )

    st.plotly_chart(
        support_chart,
        use_container_width=True
    )

    st.warning(
        "These relationships are associations from self-reported "
        "survey data. They do not establish causation."
    )


# ----------------------------------------------------------
# TAB 4: RECOMMENDATIONS
# ----------------------------------------------------------

with action_tab:

    st.subheader("Recommended Business Actions")

    st.markdown(
        """
        1. **Improve benefit awareness:** Publish a simple guide explaining
           available benefits, eligibility and access procedures.

        2. **Create a centralized care portal:** Combine counselling,
           insurance, helpline and Employee Assistance Program information.

        3. **Clarify mental-health leave:** Explain eligibility, approval,
           documentation and confidentiality procedures.

        4. **Train managers:** Provide training on supportive communication,
           referral boundaries, confidentiality and non-retaliation.

        5. **Provide early intervention:** Offer confidential counselling,
           flexible work arrangements and workload support.

        6. **Monitor progress:** Conduct anonymous pulse surveys and track
           awareness, work interference, psychological safety, absence
           and employee retention.
        """
    )

    st.success(
        "Expected outcome: stronger employee wellbeing, better awareness, "
        "lower work interference and improved employee retention."
    )


# ----------------------------------------------------------
# OPTIONAL DATA PREVIEW
# ----------------------------------------------------------

with st.expander("View analysis-ready data"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )