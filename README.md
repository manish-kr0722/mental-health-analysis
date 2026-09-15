# Workplace Mental Health and Employee Wellbeing Analysis

## Project Overview

This project analyses workplace mental-health survey data to understand how employee demographics, organizational support, workplace attitudes and employment conditions are associated with mental-health treatment and work interference.

The project includes data cleaning, exploratory data analysis using the UBM framework, more than 20 meaningful visualizations, business recommendations and an interactive Streamlit dashboard.

## Business Objective

The objective is to identify factors associated with mental-health treatment and work interference, evaluate employees’ awareness of workplace support, and recommend actions that can:

* Improve access to mental-health care
* Increase awareness of available benefits
* Reduce workplace stigma
* Strengthen psychological safety
* Reduce absenteeism and productivity loss
* Improve employee engagement and retention

## Dataset Overview

The dataset contains:

* **1,259 survey responses**
* **27 original variables**
* Demographic, employment and workplace-support information
* Mental-health treatment and disclosure-related responses

Important variables include:

* `Age`
* `Gender`
* `Country`
* `family_history`
* `treatment`
* `work_interfere`
* `benefits`
* `care_options`
* `wellness_program`
* `seek_help`
* `leave`
* `supervisor`
* `mental_health_consequence`

## Data Preparation

The following data-cleaning and preparation steps were performed:

* Removed unnecessary spaces from column names and text values
* Checked for duplicate rows and duplicate column names
* Converted `Timestamp` into date-time format
* Identified eight unrealistic age values
* Excluded invalid ages only from age-based analysis
* Created meaningful age groups
* Standardized 49 inconsistent gender responses
* Classified missing states as `Not applicable` or `Not reported`
* Replaced missing `self_employed` and `work_interfere` responses with `Not reported`
* Excluded the highly incomplete `comments` field from structured analysis
* Created a numerical `Treatment_Flag`
* Created a workplace `Support_Score`
* Applied logical ordering to ordinal variables
* Added validation and exception handling

No valid respondent records were unnecessarily removed.

## Exploratory Data Analysis

The visualizations were created using the UBM framework.

### Univariate Analysis

Univariate analysis was used to understand individual variables:

* Age distribution
* Age-group distribution
* Gender distribution
* Country distribution
* Treatment distribution
* Work-interference frequency
* Mental-health benefit availability
* Care-option awareness
* Ease of taking mental-health leave
* Organization-size distribution

### Bivariate Analysis

Bivariate analysis examined relationships between two variables:

* Age and treatment
* Gender and treatment
* Family history and treatment
* Work interference and treatment
* Benefits and treatment
* Care-option awareness and treatment
* Remote work and treatment
* Technology-company status and treatment
* Workplace consequences and treatment
* Supervisor openness and treatment

### Multivariate Analysis

Multivariate analysis examined interacting workplace factors:

* Work interference, family history and treatment
* Family history and treatment-priority heatmap
* Support score by treatment and family history
* Country, treatment rate, respondent volume and median age

Each visualization includes:

* Reason for selecting the chart
* Important insights
* Potential positive business impact
* Negative-growth or risk indicators

## Key Findings

* Approximately **50.6%** of respondents had sought mental-health treatment.
* Around **48.4%** reported that mental health interfered with work sometimes or often.
* Approximately **32.4%** did not know whether mental-health benefits were available.
* Around **64.7%** reported no awareness or uncertainty about available care options.
* Many respondents were uncertain about the process for taking mental-health leave.
* Treatment rates increased with the frequency of reported work interference.
* Respondents with a family history of mental illness had a higher treatment rate.
* Fear of workplace consequences may discourage employees from seeking support.
* Providing benefits alone may not eliminate workplace stigma.
* The sample is dominated by US and male respondents, limiting wider generalization.

## Business Recommendations

Organizations should:

1. Clearly communicate available mental-health benefits.
2. Create a centralized and confidential care-resource portal.
3. Explain mental-health leave policies and approval procedures.
4. Train supervisors in supportive communication and referral practices.
5. Strengthen confidentiality and non-retaliation policies.
6. Provide confidential early-intervention support.
7. Offer accessible support for remote and office-based employees.
8. Conduct regular anonymous employee pulse surveys.
9. Monitor benefit awareness, psychological safety and work interference.
10. Never use sensitive mental-health data for individual employment decisions.

## Streamlit Dashboard

The interactive Streamlit application provides:

* Country, gender and organization-size filters
* Total respondent KPI
* Treatment-rate KPI
* Work-interference KPI
* Benefit-awareness KPI
* Respondent overview
* Workplace-support analysis
* Treatment-driver analysis
* Interactive Plotly charts
* Business recommendations
* Analysis-ready data preview

## Project Structure

```text
Mental_Health_Streamlit_App/
│
├── app.py
├── survey.csv
├── requirements.txt
├── Workplace_Mental_Health_Analysis.ipynb
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Jupyter Notebook
* VS Code

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the displayed local address, normally:

```text
http://localhost:8501
```

For more information, refer to the [official Streamlit installation guide](https://docs.streamlit.io/get-started/installation/command-line).

## Requirements

```text
streamlit
pandas
numpy
plotly
matplotlib
seaborn
```

## Deployment

The application can be deployed using Streamlit Community Cloud:

1. Upload the project files to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Select **Create app**.
4. Choose the GitHub repository.
5. Set `app.py` as the main application file.
6. Click **Deploy**.

See the [official Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

## Limitations

* The survey data is self-reported.
* The analysis identifies associations, not causal relationships.
* The survey sample is not evenly distributed across countries and gender groups.
* Treatment history does not necessarily measure current mental-health need.
* Sensitive information should only be reported in aggregated form.

## Conclusion

The analysis demonstrates that workplace mental health is closely associated with employee wellbeing and work performance. Important concerns include limited awareness of benefits, uncertainty about care options and leave procedures, frequent work interference, and fear of negative workplace consequences. Clear communication, confidential care access, trained managers and stigma-reduction policies can improve employee wellbeing, productivity and retention.

## Author

**Manish Kumar**

Data Analytics Project — Python, Exploratory Data Analysis and Streamlit
