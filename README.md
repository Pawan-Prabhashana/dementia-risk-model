# TEAM 404 - Dementia Risk Prediction from Non-Medical Data

This project was built for a hackathon focused on **estimating dementia risk** using information that normal people can realistically know about themselves (age, education, lifestyle, living situation, and simple diagnoses) – **without** using detailed medical tests, scans, or specialist cognitive scores.

The end goal is a model that could sit behind a simple website or app where a user answers questions like:

- How old are you?
- What is your highest level of education?
- Who do you live with?
- Do you smoke or drink alcohol?
- Have you ever been told you have diabetes, high blood pressure, or a stroke?

…and the system returns:

- an **estimated probability** of having dementia
- a label like **“Low”, “Medium”, or “High (at risk)”**

> ⚠️ **Important:** This model is **not a diagnosis tool** and must **not** be used as a substitute for medical advice. It is a research prototype built on a research dataset.

---

## 1. Dataset & Task

- Source: curated dementia dataset based on the **NACC UDS** (National Alzheimer’s Coordinating Center Uniform Data Set).
- Each row (after preprocessing) corresponds to **one participant at baseline**.
- Target variable: `DEMENTED`
  - `1` = participant met criteria for dementia
  - `0` = no dementia

After filtering to baseline visits (`NACCVNUM == 1`):

- **Participants:** 52,537
- **Train / Validation / Test (by subject):**
  - Train: 36,775
  - Validation: 7,881
  - Test: 7,881
- Dementia prevalence (baseline): ~**32.5%** dementia, **67.5%** no dementia.

The task is **binary classification**:

> Given a person’s non-medical information, estimate the probability that they currently have dementia (`DEMENTED = 1`).

---

## 2. Feature Selection – Non-Medical Only

We strictly followed the hackathon requirement to **exclude medical-only features**:

- ❌ No detailed neuropsychological test scores (MMSE/MoCA, CDR, FAQ, GDS, etc.)
- ❌ No imaging, lab results, or clinician-only scales
- ❌ No genetic/mutation markers
- ❌ No medication classes

We **allowed** only information a typical person can reasonably know or self-report:

### 2.1 Demographics & Social context

From forms like A1 (subject demographics):

- `NACCAGE`, `NACCAGEB` – age at visit / baseline
- `SEX` – sex
- `HISPANIC`, `RACE` – ethnicity & race
- `EDUC` – years of education
- `MARISTAT` – marital status
- `NACCLIVS` – living situation (alone / with spouse / others)
- `RESIDENC` – type of residence (private home, retirement community, etc.)
- `PRIMLANG` – primary language
- `HANDED` – handedness

These are all standard self-reported or basic demographic fields.

### 2.2 Lifestyle & substance use

From subject health history:

- `TOBAC30`, `TOBAC100`, `SMOKYRS`, `PACKSPER`, `QUITSMOK` – smoking status & intensity
- `ALCOCCAS`, `ALCFREQ` – alcohol use and frequency

These are everyday-language questions (“Do you smoke?”, “How often do you drink?”).

### 2.3 Simple diagnoses (patient-known)

We allowed **binary diagnoses** that people are typically told explicitly by clinicians:

Cardio-metabolic and vascular:

- `CVHATT` (heart attack), `HATTMULT` (multiple), `CBSTROKE` (stroke), `STROKMUL` (multiple strokes), `CBTIA` (TIA)
- `DIABETES` – diabetes
- `HYPERTEN` – high blood pressure
- `HYPERCHO` – high cholesterol
- `CVCHF` – congestive heart failure
- `CVANGINA` – angina
- `CVHVALVE` – valve repair / replacement

Other chronic conditions:

- `THYROID` – thyroid disease
- `B12DEF` – B12 deficiency
- `ARTHRIT` – arthritis

Sleep:

- `APNEA` – sleep apnea
- `INSOMN` – insomnia / hyposomnia
- `OTHSLEEP` – other sleep disorders

Psychiatric:

- `DEP2YRS`, `DEPOTHR` – depression (recent / past)
- `ANXIETY`, `BIPOLAR`, `SCHIZ`, `PTSD`
- `ALCOHOL`, `ABUSOTHR` – alcohol or other substance abuse

Neurological:

- `PD` – Parkinson’s disease
- `SEIZURES` – seizure disorder
- `TBI`, `NACCTBI` – history of traumatic brain injury

### 2.4 Family history

- `NACCFAM` – any first-degree relative with significant memory problems
- `NACCMOM` – mother with cognitive impairment
- `NACCDAD` – father with cognitive impairment

Most users can answer “Did your mother/father have dementia or serious memory problems?” without needing specialist tests.

### 2.5 Physical characteristics & sensory function

- `HEIGHT`, `WEIGHT`, `NACCBMI` – height, weight, BMI
- `VISION`, `VISCORR`, `VISWCORR` – vision with/without correction
- `HEARING`, `HEARAID`, `HEARWAID` – hearing function & hearing aids

Users typically know their approximate height/weight and whether they use glasses or hearing aids.

### 2.6 Engineered features

To give the model more structure and interpretability, we created:

- `pack_years` = `SMOKYRS * PACKSPER` – lifetime smoking burden
- `cardio_burden` – count of major vascular conditions (heart attack, stroke, TIA, diabetes, hypertension, cholesterol, CHF, angina, valve disease)
- `psych_burden` – count of psychiatric conditions (depression, anxiety, bipolar, schizophrenia, PTSD, alcohol/substance abuse)
- `sleep_burden` – number of sleep problems (apnea, insomnia, other sleep disorders)
- `parent_dementia_any` – 1 if mother or father had cognitive impairment
- `BMI_CAT` – BMI category: `underweight`, `normal`, `overweight`, `obese`

These engineered features map well to how clinicians and lay people talk about risk (“many vascular problems”, “recent depression”, “obesity”, “family history”).

---

## 3. Modeling Approach

We used only **baseline** visits (`NACCVNUM == 1`), so each person appears once.

### 3.1 Data splitting

- `NACCID` used as a subject identifier
- Splits at **subject level** (no leakage across visits):
  - Train: 70%
  - Validation: 15%
  - Test: 15%
- Stratified by `DEMENTED` to keep class balance consistent.

### 3.2 Preprocessing

Implemented with `sklearn` Pipelines and `ColumnTransformer`:

- **Numeric features:**
  - `SimpleImputer(strategy="median")`
  - (Scaled for logistic regression; unscaled for tree models)
- **Categorical features:**
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

All imputation and encoding steps are inside the pipelines, so there is **no leakage** from validation/test into training statistics.

### 3.3 Models trained

We trained and compared:

1. **Logistic Regression (with feature engineering)**

   - `class_weight="balanced"`
   - Good baseline + interpretable coefficients

2. **Random Forest**

   - `class_weight="balanced"`
   - Tuned via `RandomizedSearchCV` (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)

3. **XGBoost (XGBClassifier)** – **final model**
   - Objective: `binary:logistic`
   - Tuned via `RandomizedSearchCV` over:
     - `n_estimators`, `max_depth`, `learning_rate`
     - `subsample`, `colsample_bytree`, `min_child_weight`
   - Best configuration (example from this run):
     - `n_estimators = 400`
     - `max_depth = 5`
     - `learning_rate = 0.05`
     - `subsample = 0.9`
     - `colsample_bytree = 1.0`
     - `min_child_weight = 1`

Evaluation used ROC-AUC as the primary metric, with accuracy, precision, recall, and F1 as secondary metrics.

---

## 4. Results

### 4.1 Validation performance (ROC-AUC)

| Model         | Val ROC-AUC | Test ROC-AUC |
| ------------- | ----------: | -----------: |
| LogReg + FE   |      ~0.731 |       ~0.733 |
| Random Forest |      ~0.767 |       ~0.767 |
| **XGBoost**   |  **~0.783** |   **~0.784** |

XGBoost consistently performed best on both validation and test sets.

### 4.2 Final XGBoost performance on test set (threshold = 0.30)

We selected a probability threshold of **0.30** based on validation performance to prioritize **higher recall** for dementia cases.

On the **test set**:

- **ROC-AUC:** ~**0.784**
- **Accuracy:** ~**0.69**
- For dementia class (`DEMENTED = 1`):
  - **Precision:** ~**0.52**
  - **Recall (sensitivity):** ~**0.75**
  - F1 ≈ 0.61

This operating point means:

> The model correctly identifies about **75%** of people with dementia (high sensitivity), while about **52%** of people flagged as “at risk” truly have dementia in the dataset.

This is a reasonable trade-off for a **screening** scenario where missing at-risk individuals is more costly than having some false alarms.

---

## 5. Explainability (SHAP)

To understand what the XGBoost model learned, we used **SHAP (TreeExplainer)** on a sample of 5,000 training participants.

### 5.1 Global feature importance

Top features by mean absolute SHAP value included:

- **Depression in last 2 years (`DEP2YRS`)** – strong positive association with dementia risk
- **Living situation (`NACCLIVS`)** – certain living arrangements (e.g. not living with a partner) associated with higher risk
- **Years of education (`EDUC`)** – lower education increased risk; higher education was protective
- **Age (`NACCAGE`)** – older age increased risk
- **Sex (`SEX`)** – one sex showed slightly higher modeled risk depending on coding
- **Weight / BMI (`WEIGHT`, `NACCBMI`)** – obesity / underweight patterns affected risk
- **Arthritis (`ARTHRIT`)**, **thyroid disease (`THYROID`)**, **alcohol use (`ALCOCCAS`, `ALCFREQ`)**
- Various **residence** and **race** categories

These patterns are broadly consistent with known epidemiological risk factors (age, depression, cardiometabolic health, education, and social context).

### 5.2 Individual-level explanations

Using SHAP force plots and per-person explanations, we can show:

- For a given individual, **which features push their risk up** (e.g. older age, recent depression, multiple vascular conditions)
- And **which features push it down** (e.g. higher education, absence of depression, no vascular issues)

This is ideal for a real app where, in addition to a risk score, users could see **“Your risk is mainly higher because of X, Y, Z”** in simple language.

---

## 6. Simple API-like interface

We wrapped the final XGBoost pipeline into a function:

```python
def predict_dementia_risk(person_features: dict):
    """
    person_features: dict mapping feature name -> value
    (missing features are allowed and will be imputed)
    Returns:
        prob: predicted probability of dementia
        category: "Low risk", "Medium risk", or "High risk (at risk)"
    """
```
