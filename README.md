# Workplace Mental Health Survey Analysis

**Python • Pandas • Plotly • Streamlit | Survey analytics project**

A survey analysis examining treatment history, reported work interference and awareness of workplace support.

## Business question

What gaps in benefit awareness and workplace support appear in the survey, and what could organisations investigate to improve access to resources?

## Dataset

The included survey contains **1,259 responses and 27 original variables**, covering demographics, employment conditions, treatment history and workplace attitudes.

## Data preparation

- Standardised text and created grouped gender categories.
- Identified eight invalid ages and excluded them from age-based analysis rather than removing the respondents entirely.
- Distinguished missing states from states that are not applicable.
- Preserved unknown categorical responses rather than guessing Yes or No.
- Created treatment, support and age-group variables.
- Retained optional comments separately from structured analysis.

The final preparation output preserves 1,259 responses and identifies zero exact duplicate rows.

## Verified findings

Percentages below use all 1,259 responses as the denominator.

| Measure | Result |
|---|---:|
| Reported having sought treatment | 50.60% |
| Reported work interference sometimes or often | 48.37% |
| Did not know whether benefits were available | 32.41% |
| Reported no awareness or uncertainty about care options | 64.73% |

Treatment history is not a diagnosis or a direct measure of current need.

## Application

The Streamlit app includes demographic, workplace-support, treatment-factor and multivariate analysis views. Interactive filters and Plotly charts support exploration of the survey.

[Open the published app](https://mental-health-in-tech-survey-eda-analysis-erntqjzaftszeeuhsqvn.streamlit.app/)

## Recommendations

Improve communication of benefits and care options, clarify leave processes and investigate concerns about confidentiality. These are proposed actions, not measured improvements in wellbeing or productivity.

## Run locally

~~~bash
git clone https://github.com/manish-kr0722/mental-health-analysis.git
cd mental-health-analysis
python -m venv .venv
~~~

Activate the environment:

~~~bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
~~~

~~~bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
~~~

To run Mental_health.ipynb, also install Jupyter, matplotlib and seaborn. Keep survey.csv alongside the app and notebook.

## Limitations

The survey is self-reported and its demographic composition limits generalisation. Relationships are observational. The custom support score is an exploratory summary, not a validated clinical measure.

Report sensitive information in aggregate. Do not use this project for medical diagnosis or individual employment decisions.

## Repository files

- [Mental_health.ipynb](Mental_health.ipynb)
- [app.py](app.py)
- [requirements.txt](requirements.txt)
- [survey.csv](survey.csv)

## Author

**Manish Kumar** — banking professional transitioning into Data Analytics.

[LinkedIn](https://www.linkedin.com/in/manish071096/) · [GitHub](https://github.com/manish-kr0722)
