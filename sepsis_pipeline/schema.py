"""PhysioNet 2019 column order and the project's fixed evaluation scope."""

VARIABLES = (
    'HR O2Sat Temp SBP MAP DBP Resp EtCO2 BaseExcess HCO3 FiO2 pH PaCO2 '
    'SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct '
    'Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total '
    'TroponinI Hct Hgb PTT WBC Fibrinogen Platelets'
).split()
CONTEXT = ['Age', 'Gender', 'HospAdmTime', 'ICULOS']
HEADER = VARIABLES + ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
STANDARD_ID = 'SEPSIS_FULL_RECORD_V2'
SPLIT_SEED = 42
CV_SEED = 2026
