# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import sklearn
from sklearn.model_selection import train_test_split
import dash_daq as daq
import pandas as pd

# Imports from this application
from app import app


#---------------------------------------------------
# Load wrangling function for data

# Create function to clean data
# The data dictionary provided with the article does not match the dataset well. Check readme for explanation as to why I removed some columns. I think the question #'s got mislabeled somehow.

def clean_data(X):
    X = X.copy()

    
    #Filter out 300 rows that chose to not answer target question (those that preferred not to answer)
    X = X[((X['Q28'] == 'Yes') | (X['Q28'] == 'No'))]
    
    #Drop columns containing unique metadata for each test
    X = X.drop(columns = ['ID.format', 'ID.endDate', 'ID.completed', 'ID.site', 'ID.date', 'ID.start', 'ID.end', 'ID.time'])
    
    #Drop columns with extra / specific data. Most are a step down from 'Other' answers in survey
    X = X.drop(columns = ['Q1.a', 'Q2', 'Q3.a', 'Q6', 'Q6.a', 'Q7', 'Q7.a', 'Q8', 'Q8.a', 'Q9', 'Q9.a', 'Q10', 'Q10.a', 'Q11', 'Q11.a',
                         'Q12.a', 'Q14', 'Q14.a', 'Q15.a', 'Q15.b', 'Q15.c', 'Q15.d', 'Q15.e', 'Q15.f', 'Q15.g', 'Q15.h', 'Q15.i', 'Q15.j',
                          'Q15.k', 'Q15.l', 'Q15.m', 'Q15.n', 'Q16','Q17.a', 'Q25.a', 'Q26', 'Q29', 'Q29.a', 'Q32:1', 'Q32:2', 'Q32:3', 'Q32:4', 
                          'Q32:5', 'Q32:6', 'Q32:7', 'Q32.a', 'Q33', 'Q35:1', 'Q35:2', 'Q35:3', 'Q35:4', 'Q35:5', 'Q35:6', 'Q35:7', 'Q35:8', 'Q35:9', 
                          'Q35.a', 'Q36.a','Q37.a', 'Q37.b', 'Q37.c', 'Q37.d', 'Q37.e','Q39:1', 'Q39:2', 'Q39:3', 'Q39:4', 'Q39:5', 'Q39:6', 'Q39:7', 'Q39.a',
                          'Q39:8', 'Q39:9', 'Q39:1', 'Q39:2', 'Q39:3', 'Q39:4', 'Q39:5', 'Q39:6', 'Q39:7', 'Q39:8', 'Q39:9','Q39.a', 'Q40.a', 'Q41', 'Q44', 'Q44.a', 
                          'Q45:1', 'Q45:2', 'Q45:3', 'Q45:4', 'Q45:5', 'Q45:6', 'Q45:7', 'Q45:8', 'Q45:9', 'Q45:10', 'Q45:11', 'Q45.a', 'Q46:1', 'Q46:2', 'Q46:3', 'Q46:4', 'Q46:5', 'Q46:6', 'Q46:7', 'Q46:8', 'Q46:9', 'Q46:10', 'Q46:11', 'Q46:12', 
                          'Q46:13', 'Q46:14', 'Q46.a', 'Q48:1', 'Q48:2', 'Q48:3', 'Q59:5', 'Q59.a', 'Q60', 'Q61', 'Q62', 'Q63', 'Q64', 'Q65.a', 'Q65.b',
                          'Q48:4', 'Q48:5', 'Q48:6', 'Q48:7', 'Q48.a', 'Q49:8', 'Q49.a', 'Q52.a', 'Q53:1', 'Q53:2', 'Q53:3', 'Q53:4', 'Q53:5', 'Q53:6', 'Q53:7', 
                          'Q53.a', 'Q54:5', 'Q54.a', 'Q55', 'Q58.a', 'Q59:1', 'Q59:2', 'Q59:3', 'Q47.a'])
    
    #Q3 Replace 'Other, please specify with other'
    X['Q3'] = X['Q3'].replace('Other, please specify', 'Other')
    
    #Q4: Are you studying where you grew up?
    X['Q4'] = X['Q4'].replace('yes', 1)
    X['Q4'] = X['Q4'].replace('no', 0)
    
    #Q12:1-:12 1 or 0 if they went to graduate school for this reason
    Q12_ans = ['Q12:1', 'Q12:2', 'Q12:3', "Q12:4", 'Q12:5', 'Q12:6', 'Q12:7', 'Q12:8', 'Q12:9', 'Q12:10', 'Q12:11']
    for i in Q12_ans:
        X[i].fillna(0, inplace= True)
    X['Q12:1'] = (X['Q12:1'].replace('To study at a specific university', 1))
    X['Q12:2'] = (X['Q12:2'].replace('Lack of quality PhD programmes in my home country', 1))
    X['Q12:3'] = (X['Q12:3'].replace('Lack of funding opportunities in my home country', 1))
    X['Q12:4'] = (X['Q12:4'].replace('Lack of PhD programmes in my subject of choice', 1))
    X['Q12:5'] = (X['Q12:5'].replace('Chance to pursue a specific research question', 1))
    X['Q12:6'] = (X['Q12:6'].replace('Higher salaries post-study', 1))
    X['Q12:7'] = (X['Q12:7'].replace('More job opportunities post-study', 1))
    X['Q12:8'] = (X['Q12:8'].replace('Family reasons', 1))
    X['Q12:9'] = (X['Q12:9'].replace('To experience another culture', 1))
    X['Q12:10'] = (X['Q12:10'].replace('Political reasons', 1))
    X['Q12:11'] = (X['Q12:11'].replace('Other, please specify', 1))
    
    #Q18a - How satisfied are you with your decision to pursue a phd?
    X['Q18.a'] = (X['Q18.a'].replace('Neither satisfied nor dissatisfied', 'Neither'))
    
    #Q19a - How satisfied are you with your phd experience?
    X['Q19.a'] = (X['Q19.a'].replace('4 = Neither satisfied nor dissatisfied', 4))
    X['Q19.a'] = (X['Q19.a'].replace('1 = Not at all satisfied', 1))
    X['Q19.a'] = (X['Q19.a'].replace('7 = Extremely satisfied', 7))
    
    #Q21a:i & Q22a:i - How satisfied are you with x (Ordinal)
    q_21s = ['Q21.a', 'Q21.b', 'Q21.c', 'Q21.d', 'Q21.e', 'Q21.f', 'Q21.g', 'Q21.h', 'Q21.i', 'Q22.a', 'Q22.b', 'Q22.c', 'Q22.d', 'Q22.e', 'Q22.f', 'Q22.g', 'Q22.h', 'Q22.i']
    for every in q_21s:
        X[q_21s] = (X[q_21s].replace('4 = Neither satisfied nor dissatisfied', 4))
        X[q_21s] = (X[q_21s].replace('1 = Not at all satisfied', 1))
        X[q_21s] = (X[q_21s].replace('7 = Extremely satisfied', 7))
        
    #Q24 Workload on average (Ordinal)
    X['Q24'] = (X['Q24'].replace('41-50 hours','41-50'))
    X['Q24'] = (X['Q24'].replace('51-60 hours','51-60'))
    X['Q24'] = (X['Q24'].replace('61-70 hours','61-70'))
    X['Q24'] = (X['Q24'].replace('71-80 hours','71-80'))
    X['Q24'] = (X['Q24'].replace('21-30 hours','21-30'))
    X['Q24'] = (X['Q24'].replace('31-40 hours','31-40'))
    X['Q24'] = (X['Q24'].replace('Less than 11 hours','0-10'))
    X['Q24'] = (X['Q24'].replace('More than 80 hours','81 and up'))
    
    #Q25.a - extra for other
    X['Q25'] = (X['Q25'].replace('Other, please specify', 'Other'))
    
    #Q30.a - Q30.f - strongly agree or disagree
    X['Q30.a'] = (X['Q30.a'].replace('Neither satisfied nor dissatisfied', 'Neither'))
    X['Q30.b'] = (X['Q30.b'].replace('Neither satisfied nor dissatisfied', 'Neither'))
    X['Q30.c'] = (X['Q30.c'].replace('Neither satisfied nor dissatisfied', 'Neither'))
    X['Q30.d'] = (X['Q30.d'].replace('Neither satisfied nor dissatisfied', 'Neither'))
    X['Q30.e'] = (X['Q30.e'].replace('Neither satisfied nor dissatisfied', 'Neither'))
    X['Q30.f'] = (X['Q30.f'].replace('Neither satisfied nor dissatisfied', 'Neither'))

    #Q34 Did you feel discriminated against or harassed? No issue
    X['Q34'].fillna('Prefer not to say', inplace=True)
    
    #Q47 : Which of these are most difficult based on your discipline
    X['Q47:1'] = (X['Q47:1'].replace('Learning what career possibilities exist', 1))
    X['Q47:2'] = (X['Q47:2'].replace('Finding research careers within academia', 1))
    X['Q47:3'] = (X['Q47:3'].replace('Finding research careers within industry', 1))
    X['Q47:4'] = (X['Q47:4'].replace('Finding research careers within government', 1))
    X['Q47:5'] = (X['Q47:5'].replace('Finding research careers within charity/non-profit', 1))
    X['Q47:6'] = (X['Q47:6'].replace('Obtaining skills for careers in industry', 1))
    X['Q47:7'] = (X['Q47:7'].replace('Obtaining skills for careers in non-profits \xa0', 1))
    X['Q47:8'] = (X['Q47:8'].replace('Finding non-research careers that use my skills', 1))
    X['Q47:1'].fillna(0, inplace= True)
    X['Q47:2'].fillna(0, inplace= True)
    X['Q47:3'].fillna(0, inplace= True)
    X['Q47:4'].fillna(0, inplace= True)
    X['Q47:5'].fillna(0, inplace= True)
    X['Q47:6'].fillna(0, inplace= True)
    X['Q47:7'].fillna(0, inplace= True)
    X['Q47:8'].fillna(0, inplace= True)
    
    #Q49 - Which of the following do you think is needed
    X['Q49:1'].fillna(0, inplace= True)
    X['Q49:2'].fillna(0, inplace= True)
    X['Q49:3'].fillna(0, inplace= True)
    X['Q49:4'].fillna(0, inplace= True)
    X['Q49:5'].fillna(0, inplace= True)
    X['Q49:6'].fillna(0, inplace= True)
    X['Q49:1'] = (X['Q49:1'].replace('Lower competition for grants', 1))
    X['Q49:2'] = (X['Q49:2'].replace('Mentorship with individuals in my field/department/institution', 1))
    X['Q49:3'] = (X['Q49:3'].replace('Gender-specific mentorship with individuals in my field/department/institution', 1))
    X['Q49:4'] = (X['Q49:4'].replace('Better data/information about available career opportunities', 1))
    X['Q49:5'] = (X['Q49:5'].replace('A record of, or data on, career paths of previous graduates from my programme', 1))
    X['Q49:6'] = (X['Q49:6'].replace('More jobs in academia', 1))
    X['Q49:7'] = (X['Q49:7'].replace('Grants to help PhD holders transition to permanent positions', 1))
    
    #Q52:1 - 52:8 Which, if any, of the following activities have you done to advance your career? 
    X['Q52:1'].fillna(0, inplace= True)
    X['Q52:2'].fillna(0, inplace= True)
    X['Q52:3'].fillna(0, inplace= True)
    X['Q52:4'].fillna(0, inplace= True)
    X['Q52:5'].fillna(0, inplace= True)
    X['Q52:6'].fillna(0, inplace= True)
    X['Q52:7'].fillna(0, inplace= True)
    X['Q52:8'].fillna(0, inplace= True)
    X['Q52:1'] = (X['Q52:1'].replace('Attended career seminars and/or workshops',1)) 
    X['Q52:2'] = (X['Q52:2'].replace('Attended networking events',1))
    X['Q52:3'] = (X['Q52:3'].replace('Developed my social media profile',1))
    X['Q52:4'] = (X['Q52:4'].replace('Worked out an individualized development plan',1))
    X['Q52:5'] = (X['Q52:5'].replace('Discussed my career future with a supervisor/PI',1))
    X['Q52:6'] = (X['Q52:6'].replace('Discussed my career future with a mentor',1))
    X['Q52:7'] = (X['Q52:7'].replace('Discussed my career future with a careers counsellor at my institution',1))
    X['Q52:8'] = (X['Q52:8'].replace('Other, please specify',1))
    
    #Q54:1	Q54:2 Q54:3 Q54:4 Q54:5 Q54.a what would you do differently
    X['Q54:1'].fillna(0, inplace= True)
    X['Q54:2'].fillna(0, inplace= True)
    X['Q54:3'].fillna(0, inplace= True)
    X['Q54:4'].fillna(0, inplace= True)
    X['Q54:1'] = (X['Q54:1'].replace('Change area of study',1))
    X['Q54:2'] = (X['Q54:2'].replace('Change supervisor/PI ',1))
    X['Q54:3'] = (X['Q54:3'].replace('Not pursue a PhD at all',1))
    X['Q54:4'] = (X['Q54:4'].replace('Nothing',1))

    #Q58
    X['Q58:1'].fillna(0, inplace= True)
    X['Q58:2'].fillna(0, inplace= True)
    X['Q58:3'].fillna(0, inplace= True)
    X['Q58:4'].fillna(0, inplace= True)
    X['Q58:5'].fillna(0, inplace= True)
    X['Q58:6'].fillna(0, inplace= True)
    X['Q58:7'].fillna(0, inplace= True)
    X['Q58:8'].fillna(0, inplace= True)
    X['Q58:9'].fillna(0, inplace= True)
    X['Q58:10'].fillna(0, inplace= True)
    X['Q58:11'].fillna(0, inplace= True)
    X['Q58:12'].fillna(0, inplace= True)
    X['Q58:1'] = (X['Q58:1'].replace('Caucasian',1))
    X['Q58:2'] = (X['Q58:2'].replace('Latino/Hispanic',1))
    X['Q58:3'] = (X['Q58:3'].replace('Middle Eastern',1))
    X['Q58:4'] = (X['Q58:4'].replace('African',1))
    X['Q58:5'] = (X['Q58:5'].replace('Caribbean',1))
    X['Q58:6'] = (X['Q58:6'].replace('South Asian',1))
    X['Q58:7'] = (X['Q58:7'].replace('East Asian',1))
    X['Q58:8'] = (X['Q58:8'].replace('Pacific Islander',1))
    X['Q58:9'] = (X['Q58:9'].replace('American Indian',1))
    X['Q58:10'] = (X['Q58:10'].replace('Mixed ethnicity',1))
    X['Q58:11'] = (X['Q58:11'].replace('Other, please specify',1))
    X['Q58:12'] = (X['Q58:12'].replace('Prefer not to say',1))

    #Q59
    X['Q59:4'] = X['Q59:4'].replace('No', 0)
    X['Q59:4'].fillna(1, inplace= True)
    

    return X


## Load in data
#URL from Nature article (https://www.nature.com/articles/d41586-019-03459-7)
url = 'https://ndownloader.figshare.com/files/18543320?private_link=74a5ea79d76ad66a8af8'

#Extract Columns while importing data
df = pd.read_excel(url)
raw_cols = df.columns

#Create df with questions as column names
df = pd.read_excel(url, header=None, skiprows = 2, names = raw_cols)

#Wrangle dataset
survey = clean_data(df)

##Create train, test data
train, test = train_test_split(df,train_size = 0.8, random_state = 42)

#Creating X_test, y_test from features, target
X_train = train.drop(columns='Q28')
X_test = test.drop(columns='Q28')
features = X_train.columns.tolist()
target = 'Q28'
y_train = train[target]
y_test = test[target]


#Make pipeline and prediction

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Predictions

            Your instructions: How to use your app to get new predictions.

            """
        ),
    ],
    md=4,
)

column2 = dbc.Col(
    [
        daq.Gauge(
        id='my-daq-gauge',
        max=1,
        min=0,
        value=0.1
        )

    ]
)

layout = dbc.Row([column1, column2])