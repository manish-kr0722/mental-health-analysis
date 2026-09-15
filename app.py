"""Professional Streamlit application using only the charts in Mental_health.ipynb."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Workplace Mental Health Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The professional corporate-healthcare palette is used consistently.
NAVY = "#102A43"
BLUE = "#2563EB"
TEAL = "#0F9D8A"
ORANGE = "#F59E0B"
RED = "#E05252"
INK = "#243B53"
MUTED = "#627D98"
PALE = "#F5F8FC"
GRID = "#E6ECF2"
YES_NO = {"Yes": TEAL, "No": RED}

# Custom CSS improves spacing, hierarchy and card presentation.
st.markdown(
    f"""
    <style>
    .stApp {{ background: {PALE}; }}
    .block-container {{ max-width: 1500px; padding-top: 1.25rem; padding-bottom: 3rem; }}
    [data-testid="stSidebar"] {{ background: {NAVY}; }}
    [data-testid="stSidebar"] * {{ color: #E6EEF5; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        background: rgba(255,255,255,.045); border-radius: 10px;
        padding: .42rem .55rem; margin-bottom: .20rem;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: rgba(15,157,138,.28); border: 1px solid rgba(110,231,183,.45);
    }}
    h1, h2, h3 {{ color: {NAVY}; letter-spacing: -.02em; }}
    .hero {{
        background: linear-gradient(125deg, {NAVY} 0%, #174B66 65%, #126E70 100%);
        border-radius: 20px; padding: 1.85rem 2.1rem; color: white;
        margin-bottom: 1.2rem; box-shadow: 0 12px 30px rgba(16,42,67,.14);
    }}
    .hero .eyebrow {{ color: #8FE3D2; font-size: .78rem; font-weight: 800; letter-spacing: .15em; }}
    .hero h1 {{ color: white; font-size: 2.2rem; margin: .25rem 0 .4rem 0; }}
    .hero p {{ color: #D9E5EF; max-width: 980px; font-size: 1rem; line-height: 1.55; margin: 0; }}
    .metric-card {{
        background: white; border: 1px solid #DDE5EE; border-radius: 16px;
        padding: 1.05rem 1.12rem; min-height: 128px;
        box-shadow: 0 7px 18px rgba(16,42,67,.055);
    }}
    .metric-label {{ color: {MUTED}; font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; }}
    .metric-value {{ color: {NAVY}; font-size: 1.95rem; line-height: 1.2; font-weight: 800; margin: .3rem 0; }}
    .metric-detail {{ color: {MUTED}; font-size: .81rem; line-height: 1.35; }}
    .chart-note {{
        background: #EAF7F4; border-left: 4px solid {TEAL}; border-radius: 10px;
        padding: .8rem 1rem; color: {INK}; margin: .3rem 0 1.05rem 0; line-height: 1.48;
    }}
    .chart-note-risk {{
        background: #FFF7E6; border-left: 4px solid {ORANGE}; border-radius: 10px;
        padding: .8rem 1rem; color: {INK}; margin: .3rem 0 1.05rem 0; line-height: 1.48;
    }}
    .action-card {{
        background: white; border: 1px solid #DDE5EE; border-radius: 16px;
        padding: 1.1rem 1.2rem; min-height: 185px;
        box-shadow: 0 7px 18px rgba(16,42,67,.05);
    }}
    .action-number {{ color: {TEAL}; font-size: 1.55rem; font-weight: 900; }}
    .action-title {{ color: {NAVY}; font-size: 1.02rem; font-weight: 800; margin: .2rem 0 .42rem 0; }}
    .action-text {{ color: {MUTED}; font-size: .87rem; line-height: 1.5; }}
    .footer {{ text-align:center; color:{MUTED}; font-size:.78rem; padding-top:2rem; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #DDE5EE; border-radius: 12px; overflow: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA PREPARATION — MATCHES THE NOTEBOOK LOGIC
# =============================================================================

REQUIRED_COLUMNS = {
    "Timestamp", "Age", "Gender", "Country", "state", "self_employed",
    "family_history", "treatment", "work_interfere", "no_employees",
    "remote_work", "tech_company", "benefits", "care_options",
    "wellness_program", "seek_help", "anonymity", "leave",
    "mental_health_consequence", "phys_health_consequence", "coworkers",
    "supervisor", "mental_health_interview", "phys_health_interview",
    "mental_vs_physical", "obs_consequence", "comments",
}


def standardize_gender(value: object) -> str:
    """Convert inconsistent free-text gender values into reporting groups."""
    if pd.isna(value) or not str(value).strip():
        return "Not reported"

    text = str(value).strip().lower()
    if text in {"a little about you", "p"}:
        return "Not reported"

    diverse_markers = (
        "trans", "non-binary", "nonbinary", "queer", "genderqueer",
        "agender", "androgyne", "enby", "fluid", "neuter", "unsure",
        "all", "nah",
    )
    if any(marker in text for marker in diverse_markers):
        return "Other / Non-binary"

    female_values = {
        "f", "female", "woman", "femake", "femail", "cis female",
        "female (cis)", "cis-female/femme",
    }
    if text in female_values or "female" in text or "woman" in text:
        return "Female"

    male_values = {
        "m", "male", "man", "make", "mal", "maile", "msle", "mail",
        "malr", "cis male", "male (cis)", "cis man", "male-ish",
        "guy (-ish) ^_^", "something kinda male?",
    }
    if text in male_values or "male" in text:
        return "Male"

    return "Other / Non-binary"


@st.cache_data(show_spinner=False)
def load_and_prepare(source: str | io.BytesIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the CSV while preserving a missing-value audit."""
    try:
        raw = pd.read_csv(source, low_memory=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"The CSV could not be loaded: {exc}") from exc

    if raw.empty:
        raise ValueError("The CSV contains no survey responses.")

    missing_columns = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    null_audit = pd.DataFrame({
        "Variable": raw.columns,
        "Missing values": raw.isna().sum().values,
        "Missing (%)": (raw.isna().mean().values * 100).round(2),
    })

    df = raw.copy(deep=True)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove extra spaces and convert blank strings into missing values.
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].apply(lambda value: value.strip() if isinstance(value, str) else value)
        df[column] = df[column].replace(r"^\s*$", np.nan, regex=True)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Preserve raw age and keep only plausible ages for age-based analysis.
    df["Age_Original"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age_Original"].where(df["Age_Original"].between(18, 80))
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[17, 24, 34, 44, 54, 64, 80],
        labels=["18–24", "25–34", "35–44", "45–54", "55–64", "65–80"],
    )

    df["Gender_Group"] = df["Gender"].map(standardize_gender)

    # Preserve the meaning of missing categorical responses.
    df.loc[df["Country"].ne("United States") & df["state"].isna(), "state"] = "Not applicable"
    df.loc[df["Country"].eq("United States") & df["state"].isna(), "state"] = "Not reported"
    df["self_employed"] = df["self_employed"].fillna("Not reported")
    df["work_interfere"] = df["work_interfere"].fillna("Not reported")
    df["Has_Comment"] = df["comments"].notna().astype("int8")

    unexpected_treatment = set(df["treatment"].dropna().unique()).difference({"Yes", "No"})
    if unexpected_treatment:
        raise ValueError(f"Unexpected treatment categories: {sorted(unexpected_treatment)}")

    df["Treatment_Flag"] = df["treatment"].map({"Yes": 1, "No": 0}).astype("int8")

    support_columns = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]
    df["Support_Score"] = sum(df[column].eq("Yes").astype("int8") for column in support_columns)

    return df, null_audit


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def treatment_rate(data: pd.DataFrame, category: str, order: list[str] | None = None) -> pd.DataFrame:
    """Calculate sample size and treatment rate by one categorical variable."""
    result = (
        data.groupby(category, observed=True, dropna=False)["Treatment_Flag"]
        .agg(Responses="size", Treatment_Rate="mean")
        .reset_index()
    )
    result["Treatment_Rate"] *= 100
    if order:
        result[category] = pd.Categorical(result[category], categories=order, ordered=True)
        result = result.sort_values(category)
    return result


def style_figure(fig: go.Figure, height: int = 420, percentage_axis: bool = False) -> go.Figure:
    """Apply the shared professional chart theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=22, r=22, t=66, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color=INK, size=13),
        title=dict(font=dict(size=18, color=NAVY), x=0.01, xanchor="left"),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    if percentage_axis:
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def render_plot(fig: go.Figure, key: str) -> None:
    """Render one responsive interactive chart."""
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={"displaylogo": False, "scrollZoom": False, "responsive": True},
    )


def hero(eyebrow: str, title: str, description: str) -> None:
    """Render the page header."""
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str) -> None:
    """Render one KPI card."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def chart_note(why: str, finding: str, impact: str, risk: str) -> None:
    """Show the four required chart explanations directly below the graph."""
    st.markdown(
        f'<div class="chart-note"><strong>Why this chart?</strong> {why}<br>'
        f'<strong>Insight:</strong> {finding}<br>'
        f'<strong>Business impact:</strong> {impact}<br>'
        f'<strong>Risk / negative signal:</strong> {risk}</div>',
        unsafe_allow_html=True,
    )


def action_card(number: str, title: str, text: str) -> None:
    """Render one business recommendation."""
    st.markdown(
        f'<div class="action-card"><div class="action-number">{number}</div><div class="action-title">{title}</div><div class="action-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA LOADING AND GLOBAL FILTERS
# =============================================================================

csv_path = Path(__file__).resolve().parent / "survey.csv"

try:
    if not csv_path.is_file():
        st.error("survey.csv was not found beside app.py.")
        st.stop()
    full_df, null_audit = load_and_prepare(str(csv_path))
except ValueError as exc:
    st.error(f"Data preparation failed: {exc}")
    st.stop()


with st.sidebar:
    st.markdown("## 🧠 Mental Health Analytics")
    st.caption("20-chart EDA application")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "Overview & Demographics",
            "Workplace Support",
            "Treatment Factors",
            "Multivariate Analysis",
            "Recommendations & Quality",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Global filters")

    countries = sorted(full_df["Country"].dropna().unique().tolist())
    genders = [value for value in ["Male", "Female", "Other / Non-binary", "Not reported"] if value in full_df["Gender_Group"].unique()]
    company_order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
    valid_ages = full_df["Age"].dropna()

    selected_countries = st.multiselect("Country", countries, default=countries, key="country_filter")
    selected_genders = st.multiselect("Gender group", genders, default=genders, key="gender_filter")
    selected_sizes = st.multiselect("Organization size", company_order, default=company_order, key="company_filter")
    age_range = st.slider(
        "Valid age range",
        int(valid_ages.min()),
        int(valid_ages.max()),
        (int(valid_ages.min()), int(valid_ages.max())),
        key="age_filter",
    )

    def reset_filters() -> None:
        """Restore every global filter to its original full-sample value."""
        st.session_state["country_filter"] = countries
        st.session_state["gender_filter"] = genders
        st.session_state["company_filter"] = company_order
        st.session_state["age_filter"] = (int(valid_ages.min()), int(valid_ages.max()))

    st.button("Reset filters", width="stretch", on_click=reset_filters)
    st.markdown("---")
    st.caption("Built by Manish Kumar")


# Rows with invalid ages remain when the full valid-age range is selected.
df = full_df[
    full_df["Country"].isin(selected_countries)
    & full_df["Gender_Group"].isin(selected_genders)
    & full_df["no_employees"].isin(selected_sizes)
    & (full_df["Age"].isna() | full_df["Age"].between(age_range[0], age_range[1]))
].copy()

if df.empty:
    st.warning("No records match the selected filters. Expand the sidebar selections.")
    st.stop()

respondents = len(df)
treatment_pct = df["Treatment_Flag"].mean() * 100
interference_pct = df["work_interfere"].isin(["Sometimes", "Often"]).mean() * 100
benefit_unknown_pct = df["benefits"].eq("Don't know").mean() * 100
care_gap_pct = df["care_options"].isin(["No", "Not sure"]).mean() * 100


# =============================================================================
# PAGE 1 — CHARTS 1 TO 5
# =============================================================================

if page == "Overview & Demographics":
    hero(
        "UNIVARIATE ANALYSIS · CHARTS 1–5",
        "Workplace Mental Health & Employee Wellbeing",
        "Understand the respondent profile, treatment prevalence and reported effect of mental health on work. These are the first five charts from the notebook.",
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Respondents", f"{respondents:,}", f"{respondents / len(full_df) * 100:.1f}% of the full dataset")
    with k2:
        metric_card("Treatment rate", f"{treatment_pct:.1f}%", "Respondents answering Yes")
    with k3:
        metric_card("Work interference", f"{interference_pct:.1f}%", "Sometimes or Often")
    with k4:
        metric_card("Median valid age", f"{df['Age'].median():.0f}", "Eight invalid ages excluded")

    # Chart 1 — Age Distribution
    age_data = df.dropna(subset=["Age"])
    fig = px.histogram(
        age_data,
        x="Age",
        nbins=18,
        color_discrete_sequence=[BLUE],
        title="Chart 1: Distribution of Respondent Age",
        labels={"count": "Respondents"},
    )
    fig.add_vline(x=age_data["Age"].median(), line_dash="dash", line_color=RED, annotation_text=f"Median {age_data['Age'].median():.0f}")
    render_plot(style_figure(fig), "chart_01")
    chart_note(
        "A histogram shows the shape, concentration and spread of a continuous variable.",
        f"The median valid age is {age_data['Age'].median():.0f} years.",
        "Support communication can reflect the life and career stage of the main respondent population.",
        "Smaller young and older groups limit generalization across all ages.",
    )

    c1, c2 = st.columns(2)
    with c1:
        # Chart 2 — Age Group
        age_counts = df["Age_Group"].value_counts(sort=False).reset_index()
        age_counts.columns = ["Age group", "Respondents"]
        fig = px.bar(age_counts, x="Age group", y="Respondents", color_discrete_sequence=[TEAL], text_auto=True, title="Chart 2: Respondents by Age Group")
        render_plot(style_figure(fig), "chart_02")
        largest_age = age_counts.loc[age_counts["Respondents"].idxmax()]
        chart_note(
            "A bar chart accurately compares counts across discrete age bands.",
            f"{largest_age['Age group']} is the largest age group with {largest_age['Respondents']:,} respondents.",
            "Age segments can inform accessible, relevant awareness campaigns.",
            "Unequal representation may hide the needs of smaller groups.",
        )

    with c2:
        # Chart 3 — Gender Distribution
        gender_counts = df["Gender_Group"].value_counts().reset_index()
        gender_counts.columns = ["Gender group", "Respondents"]
        fig = px.bar(
            gender_counts,
            x="Gender group",
            y="Respondents",
            color="Gender group",
            color_discrete_map={"Male": BLUE, "Female": "#D65C9E", "Other / Non-binary": "#805AD5", "Not reported": MUTED},
            text_auto=True,
            title="Chart 3: Respondents by Standardized Gender Group",
        )
        fig.update_layout(showlegend=False)
        render_plot(style_figure(fig), "chart_03")
        largest_gender = gender_counts.loc[gender_counts["Respondents"].idxmax()]
        chart_note(
            "A bar chart clearly compares the sizes of categorical gender groups.",
            f"{largest_gender['Gender group']} is the largest group with {largest_gender['Respondents'] / respondents * 100:.1f}% of selected respondents.",
            "Representation checks help improve future survey outreach.",
            "An imbalanced sample can understate the experiences of smaller groups.",
        )

    c3, c4 = st.columns(2)
    with c3:
        # Chart 4 — Treatment Distribution
        treatment_counts = df["treatment"].value_counts().reindex(["Yes", "No"], fill_value=0).reset_index()
        treatment_counts.columns = ["Treatment", "Respondents"]
        fig = px.bar(
            treatment_counts,
            x="Treatment",
            y="Respondents",
            color="Treatment",
            color_discrete_map=YES_NO,
            text_auto=True,
            title="Chart 4: Mental-Health Treatment Distribution",
        )
        fig.update_layout(showlegend=False)
        render_plot(style_figure(fig), "chart_04")
        chart_note(
            "A two-bar comparison communicates a binary outcome more precisely than a multi-slice chart.",
            f"{treatment_pct:.1f}% report having sought mental-health treatment.",
            "The baseline supports capacity planning for benefits and confidential services.",
            "Treatment is a positive help-seeking action; non-treatment does not prove absence of need.",
        )

    with c4:
        # Chart 5 — Work Interference Frequency
        work_order = ["Never", "Rarely", "Sometimes", "Often", "Not reported"]
        work_counts = df["work_interfere"].value_counts().reindex(work_order, fill_value=0).reset_index()
        work_counts.columns = ["Work interference", "Respondents"]
        fig = px.bar(
            work_counts,
            x="Work interference",
            y="Respondents",
            color="Respondents",
            color_continuous_scale=[[0, "#D8D4EB"], [1, "#6D4E9B"]],
            text_auto=True,
            title="Chart 5: Work-Interference Frequency",
        )
        fig.update_layout(coloraxis_showscale=False)
        render_plot(style_figure(fig), "chart_05")
        chart_note(
            "An ordered bar chart preserves the progression from Never to Often.",
            f"{interference_pct:.1f}% report interference Sometimes or Often.",
            "Early assistance and reasonable accommodations may prevent escalation.",
            "Frequent interference can affect productivity, absence and retention.",
        )


# =============================================================================
# PAGE 2 — CHARTS 6 TO 8
# =============================================================================

elif page == "Workplace Support":
    hero(
        "UNIVARIATE ANALYSIS · CHARTS 6–8",
        "Workplace Support, Awareness & Leave",
        "Evaluate whether employees know about mental-health benefits and care options, and whether they expect mental-health leave to be accessible.",
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Benefits available", f"{df['benefits'].eq('Yes').mean() * 100:.1f}%", "Respondents answering Yes")
    with k2:
        metric_card("Benefit uncertainty", f"{benefit_unknown_pct:.1f}%", "Respondents answering Don't know")
    with k3:
        metric_card("Care-navigation gap", f"{care_gap_pct:.1f}%", "No or Not sure")
    with k4:
        leave_risk = df["leave"].isin(["Don't know", "Somewhat difficult", "Very difficult"]).mean() * 100
        metric_card("Leave barrier", f"{leave_risk:.1f}%", "Uncertain or expects difficulty")

    c1, c2 = st.columns(2)
    with c1:
        # Chart 6 — Availability of Mental-Health Benefits
        order = ["Yes", "No", "Don't know"]
        counts = df["benefits"].value_counts().reindex(order, fill_value=0).reset_index()
        counts.columns = ["Response", "Respondents"]
        fig = px.bar(counts, x="Response", y="Respondents", color="Response", color_discrete_map={"Yes": TEAL, "No": RED, "Don't know": ORANGE}, text_auto=True, title="Chart 6: Availability of Mental-Health Benefits")
        fig.update_layout(showlegend=False)
        render_plot(style_figure(fig), "chart_06")
        chart_note(
            "A bar chart exposes both benefit availability and uncertainty.",
            f"{benefit_unknown_pct:.1f}% do not know whether benefits are available.",
            "Clear communication can improve utilization without changing the underlying plan.",
            "Uncertainty can make existing benefit investment underperform.",
        )

    with c2:
        # Chart 7 — Awareness of Care Options
        order = ["Yes", "No", "Not sure"]
        counts = df["care_options"].value_counts().reindex(order, fill_value=0).reset_index()
        counts.columns = ["Response", "Respondents"]
        fig = px.bar(counts, x="Response", y="Respondents", color="Response", color_discrete_map={"Yes": TEAL, "No": RED, "Not sure": ORANGE}, text_auto=True, title="Chart 7: Awareness of Mental-Health Care Options")
        fig.update_layout(showlegend=False)
        render_plot(style_figure(fig), "chart_07")
        chart_note(
            "A bar chart distinguishes clear awareness, uncertainty and absence.",
            f"{care_gap_pct:.1f}% answer No or Not sure about available care options.",
            "A centralized care directory can reduce navigation friction.",
            "Low awareness may delay care and weaken program effectiveness.",
        )

    # Chart 8 — Ease of Taking Mental-Health Leave
    leave_order = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
    counts = df["leave"].value_counts().reindex(leave_order, fill_value=0).reset_index()
    counts.columns = ["Leave difficulty", "Respondents"]
    fig = px.bar(
        counts,
        x="Leave difficulty",
        y="Respondents",
        color="Respondents",
        color_continuous_scale=[[0, TEAL], [.5, ORANGE], [1, RED]],
        text_auto=True,
        title="Chart 8: Ease of Taking Mental-Health Leave",
    )
    fig.update_layout(coloraxis_showscale=False)
    render_plot(style_figure(fig), "chart_08")
    chart_note(
        "An ordered bar chart communicates the direction and intensity of leave accessibility.",
        f"{leave_risk:.1f}% are uncertain about leave or expect difficulty.",
        "Clear standards and manager guidance can improve confidence and consistency.",
        "Uncertainty or difficulty can encourage presenteeism and delayed treatment.",
    )


# =============================================================================
# PAGE 3 — CHARTS 9 TO 17
# =============================================================================

elif page == "Treatment Factors":
    hero(
        "BIVARIATE ANALYSIS · CHARTS 9–17",
        "Factors Associated With Treatment-Seeking",
        "Compare treatment experience across age, gender, family history, work interference and workplace-support perceptions. These relationships are descriptive, not causal.",
    )

    # Chart 9 — Age by Treatment Status
    age_treatment = df.dropna(subset=["Age", "treatment"])
    fig = px.box(
        age_treatment,
        x="treatment",
        y="Age",
        color="treatment",
        category_orders={"treatment": ["No", "Yes"]},
        color_discrete_map=YES_NO,
        title="Chart 9: Age Distribution by Treatment Status",
        labels={"treatment": "Sought treatment"},
    )
    fig.update_layout(showlegend=False)
    render_plot(style_figure(fig), "chart_09")
    age_medians = age_treatment.groupby("treatment")["Age"].median()
    chart_note(
        "A box plot compares the median, spread and overlap of age across treatment groups.",
        f"Median age is {age_medians.get('Yes', np.nan):.0f} for Yes and {age_medians.get('No', np.nan):.0f} for No.",
        "Age-sensitive communication can be tested while keeping support universal.",
        "Strong overlap means age alone is a poor targeting rule.",
    )

    # Chart definitions exactly follow notebook Charts 10–17.
    gender_rate = treatment_rate(df, "Gender_Group")
    family_rate = treatment_rate(df, "family_history")
    work_order = ["Never", "Rarely", "Sometimes", "Often", "Not reported"]
    work_rate = treatment_rate(df, "work_interfere", work_order)
    benefits_order = ["Yes", "No", "Don't know"]
    benefits_rate = treatment_rate(df, "benefits", benefits_order)
    care_order = ["Yes", "No", "Not sure"]
    care_rate = treatment_rate(df, "care_options", care_order)
    remote_rate = treatment_rate(df, "remote_work")
    consequence_order = ["No", "Maybe", "Yes"]
    consequence_rate = treatment_rate(df, "mental_health_consequence", consequence_order)
    supervisor_order = ["Yes", "Some of them", "No"]
    supervisor_rate = treatment_rate(df, "supervisor", supervisor_order)

    chart_specs = [
        (10, gender_rate, "Gender_Group", "Treatment Rate by Gender Group", {"Male": BLUE, "Female": "#D65C9E", "Other / Non-binary": "#805AD5", "Not reported": MUTED}, "A percentage bar chart enables comparison across differently sized gender groups.", "Differences may reflect access, stigma or sample composition.", "Segment gaps can guide inclusive benefit communication.", "Small groups can create unstable percentages; avoid individual conclusions."),
        (11, family_rate, "family_history", "Treatment Rate by Family History", YES_NO, "A rate chart measures association between two binary categorical variables.", "Treatment is more common among respondents reporting family history.", "Privacy-safe education may support earlier help-seeking.", "Family history must never inform employment decisions."),
        (13, benefits_rate, "benefits", "Treatment Rate by Mental-Health Benefits", {"Yes": TEAL, "No": RED, "Don't know": ORANGE}, "A rate chart compares treatment experience by benefit access and awareness.", "Treatment rates vary across benefit-response categories.", "Benefits and communication can reduce access barriers.", "Treatment may itself increase benefit awareness; causality is not established."),
        (14, care_rate, "care_options", "Treatment Rate by Care-Option Awareness", {"Yes": TEAL, "No": RED, "Not sure": ORANGE}, "A rate chart compares treatment across care-awareness categories.", "Treatment experience is associated with care-option awareness.", "Improved navigation may encourage timely care.", "Awareness may follow treatment rather than cause it."),
        (15, remote_rate, "remote_work", "Treatment Rate by Remote-Work Status", YES_NO, "A rate chart fairly compares two groups with unequal sample sizes.", "Remote and non-remote treatment rates can be compared directly.", "Support delivery can reflect different work arrangements.", "Remote work should not be treated as a cause of treatment differences."),
        (16, consequence_rate, "mental_health_consequence", "Treatment Rate by Expected Consequences", {"No": TEAL, "Maybe": ORANGE, "Yes": RED}, "An ordered rate chart compares escalating levels of perceived workplace risk.", "Treatment rates vary with expected disclosure consequences.", "Confidentiality and non-retaliation policies can reduce fear.", "Fear of consequences may suppress disclosure and early support."),
        (17, supervisor_rate, "supervisor", "Treatment Rate by Supervisor Openness", {"Yes": TEAL, "Some of them": ORANGE, "No": RED}, "A rate chart compares treatment across supervisor-discussion willingness.", "Treatment experience differs across openness categories.", "Manager training can create safer referral routes.", "Employees must never be pressured to disclose."),
    ]

    for row_index in range(0, len(chart_specs), 2):
        containers = st.columns(2)
        for container, spec in zip(containers, chart_specs[row_index:row_index + 2]):
            number, data, category, title, palette, why, finding, impact, risk = spec
            with container:
                fig = px.bar(
                    data,
                    x=category,
                    y="Treatment_Rate",
                    color=category,
                    color_discrete_map=palette,
                    text_auto=".1f",
                    hover_data=["Responses"],
                    title=f"Chart {number}: {title}",
                    labels={category: category.replace("_", " ").title(), "Treatment_Rate": "Treatment rate (%)"},
                )
                fig.update_layout(showlegend=False)
                render_plot(style_figure(fig, percentage_axis=True), f"chart_{number:02d}")
                chart_note(why, finding, impact, risk)

    # Chart 12 — line chart retained separately to match the notebook.
    fig = px.line(
        work_rate,
        x="work_interfere",
        y="Treatment_Rate",
        markers=True,
        text="Treatment_Rate",
        hover_data=["Responses"],
        title="Chart 12: Treatment Rate by Work-Interference Frequency",
        labels={"work_interfere": "Work interference", "Treatment_Rate": "Treatment rate (%)"},
    )
    fig.update_traces(line=dict(color="#6D4E9B", width=4), marker=dict(size=10), texttemplate="%{text:.1f}%", textposition="top center")
    render_plot(style_figure(fig, percentage_axis=True), "chart_12")
    known_rates = work_rate.set_index("work_interfere")["Treatment_Rate"]
    often_never_gap = known_rates.get("Often", np.nan) - known_rates.get("Never", np.nan)
    chart_note(
        "An ordered line chart emphasizes progression across work-interference frequency.",
        f"The Often-versus-Never treatment-rate difference is {often_never_gap:+.1f} percentage points.",
        "Confidential support at early interference stages may reduce escalation.",
        "The association does not prove that workplace conditions or treatment caused interference.",
    )


# =============================================================================
# PAGE 4 — CHARTS 18 TO 20
# =============================================================================

elif page == "Multivariate Analysis":
    hero(
        "MULTIVARIATE ANALYSIS · CHARTS 18–20",
        "Priority Segments and Workplace Support",
        "Combine treatment, family history, work interference and workplace-support scores to understand interacting patterns without creating individual risk labels.",
    )

    known_interference = df[df["work_interfere"].ne("Not reported")].copy()
    work_order = ["Never", "Rarely", "Sometimes", "Often"]

    # Chart 18 — Work Interference, Family History and Treatment
    multi_rate = (
        known_interference.groupby(["work_interfere", "family_history"], observed=True)["Treatment_Flag"]
        .agg(Responses="size", Treatment_Rate="mean")
        .reset_index()
    )
    multi_rate["Treatment_Rate"] *= 100
    multi_rate["work_interfere"] = pd.Categorical(multi_rate["work_interfere"], categories=work_order, ordered=True)
    multi_rate = multi_rate.sort_values("work_interfere")

    fig = px.bar(
        multi_rate,
        x="work_interfere",
        y="Treatment_Rate",
        color="family_history",
        barmode="group",
        category_orders={"work_interfere": work_order, "family_history": ["No", "Yes"]},
        color_discrete_map=YES_NO,
        text_auto=".1f",
        hover_data=["Responses"],
        title="Chart 18: Treatment by Work Interference and Family History",
        labels={"work_interfere": "Work interference", "family_history": "Family history", "Treatment_Rate": "Treatment rate (%)"},
    )
    render_plot(style_figure(fig, percentage_axis=True), "chart_18")
    priority_row = multi_rate.loc[multi_rate["Treatment_Rate"].idxmax()]
    chart_note(
        "Grouped bars compare family-history groups within every known interference level.",
        f"The highest observed segment is {priority_row['work_interfere']} interference / family history {priority_row['family_history']} at {priority_row['Treatment_Rate']:.1f}% (n={int(priority_row['Responses'])}).",
        "The interaction can prioritize voluntary education and support.",
        "Family history is sensitive, and small subgroups can produce volatile rates.",
    )

    # Chart 19 — Treatment Priority Heatmap
    priority = known_interference.pivot_table(
        index="family_history",
        columns="work_interfere",
        values="Treatment_Flag",
        aggfunc="mean",
        observed=True,
    ) * 100
    priority = priority.reindex(index=["No", "Yes"], columns=work_order)
    fig = px.imshow(
        priority,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=100,
        title="Chart 19: Treatment Priority Heatmap",
        labels={"x": "Work interference", "y": "Family history", "color": "Treatment rate (%)"},
    )
    render_plot(style_figure(fig, height=475), "chart_19")
    max_cell = priority.stack().idxmax()
    max_value = priority.stack().max()
    chart_note(
        "A heatmap makes high- and low-rate intersections visible through colour intensity.",
        f"The highest cell is family history {max_cell[0]} / {max_cell[1]} interference at {max_value:.1f}%.",
        "The grid supports aggregated prioritization of manager resources and outreach.",
        "High treatment is not negative; underlying interference and unmet need are the risks.",
    )

    # Chart 20 — Support Score by Treatment and Family History
    fig = px.box(
        df,
        x="treatment",
        y="Support_Score",
        color="family_history",
        category_orders={"treatment": ["No", "Yes"], "family_history": ["No", "Yes"]},
        color_discrete_map=YES_NO,
        title="Chart 20: Support Score by Treatment and Family History",
        labels={"treatment": "Sought treatment", "family_history": "Family history", "Support_Score": "Support score (0–5)"},
    )
    render_plot(style_figure(fig), "chart_20")
    support_summary = df.groupby(["treatment", "family_history"])["Support_Score"].mean()
    low_segment = support_summary.idxmin()
    high_segment = support_summary.idxmax()
    chart_note(
        "A grouped box plot compares the distribution and overlap of a numeric score across two categories.",
        f"Mean support ranges from {support_summary.loc[low_segment]:.2f} for {low_segment} to {support_summary.loc[high_segment]:.2f} for {high_segment}.",
        "Support gaps can guide policy communication while keeping access universal.",
        "The score is perception-based and must not be used to assess individuals.",
    )


# =============================================================================
# PAGE 5 — RECOMMENDATIONS AND NOTEBOOK DATA-QUALITY CHART
# =============================================================================

else:
    hero(
        "BUSINESS ACTION · DATA QUALITY",
        "Recommendations Based on the Notebook Analysis",
        "The application contains the same 20 numbered analytical charts as the notebook. The missing-values graph below is an existing data-quality visualization and is not counted as an additional analytical chart.",
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        action_card("01", "Make support discoverable", "Publish one confidential guide covering benefits, eligibility, care options and access routes. Repeat it during onboarding and regular employee communication.")
    with r2:
        action_card("02", "Clarify mental-health leave", "Explain approval steps, documentation, confidentiality and escalation. Apply the process consistently across managers and teams.")
    with r3:
        action_card("03", "Train managers safely", "Teach supportive listening, professional referral, confidentiality and non-retaliation without asking managers to diagnose employees.")

    st.write("")
    r4, r5, r6 = st.columns(3)
    with r4:
        action_card("04", "Intervene before escalation", "Offer voluntary counselling, flexible arrangements and reasonable workload support when recurring work interference is reported.")
    with r5:
        action_card("05", "Strengthen psychological safety", "Provide confidential channels that do not require disclosure to a direct supervisor and communicate non-retaliation clearly.")
    with r6:
        action_card("06", "Measure change", "Track benefit awareness, leave clarity, work interference, psychological safety, absence, engagement and retention through anonymous pulse surveys.")

    st.markdown("### Existing notebook data-quality visualization")

    # This null-values graph exists in the notebook and is not an extra analytical chart.
    missing = null_audit[null_audit["Missing values"] > 0].sort_values("Missing (%)", ascending=True)
    fig = px.bar(
        missing,
        x="Missing (%)",
        y="Variable",
        orientation="h",
        color="Missing (%)",
        color_continuous_scale="Oranges",
        text="Missing (%)",
        title="Missing Values by Variable",
        labels={"Missing (%)": "Missing values (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(coloraxis_showscale=False)
    render_plot(style_figure(fig), "null_values_chart")

    st.markdown(
        """
        <div class="chart-note-risk">
        <strong>Handling applied:</strong> State was separated into Not applicable and Not reported; missing self-employment and work-interference answers were retained as Not reported; optional comments were kept separately; and eight implausible ages were excluded only from age analysis. No valid respondent rows were unnecessarily removed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Manipulations", "Responsible use", "Filtered data"])
    with tab1:
        st.markdown(
            """
            - Standardized 49 raw gender labels into inclusive reporting groups.
            - Converted eight implausible ages into missing values for age-based analysis.
            - Handled missing categorical values without inventing Yes or No answers.
            - Created `Age_Group`, `Treatment_Flag`, `Has_Comment` and `Support_Score`.
            - Preserved original values for audit and applied schema/category validation.
            """
        )
    with tab2:
        st.markdown(
            """
            - Report only aggregated patterns and avoid very small segments.
            - Never use treatment or family history for hiring, promotion, insurance or performance decisions.
            - Treat treatment-seeking as a potentially positive act of getting help.
            - The survey is observational and self-reported; associations do not prove causation.
            """
        )
    with tab3:
        visible_columns = [
            "Age", "Age_Group", "Gender_Group", "Country", "state",
            "family_history", "treatment", "work_interfere", "no_employees",
            "remote_work", "benefits", "care_options", "leave", "Support_Score",
        ]
        st.dataframe(df[visible_columns], width="stretch", hide_index=True, height=470)
        st.download_button(
            "Download filtered data",
            data=df[visible_columns].to_csv(index=False).encode("utf-8"),
            file_name="mental_health_filtered_data.csv",
            mime="text/csv",
            width="stretch",
        )


st.markdown(
    '<div class="footer">Workplace Mental Health & Employee Wellbeing Analysis · 20 notebook charts · Built by Manish Kumar</div>',
    unsafe_allow_html=True,
)
