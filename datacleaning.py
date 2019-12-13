import pandas as pd


def data_clean(xls_name):
    
    df = pd.read_excel(xls_name, skiprows = 1)
    new_column_names = ['ID', 'limit_balance', 'sex', 'education', 'marriage', 'age', 'pay_status_sep', 
                   'pay_status_aug', 'pay_status_jul', 'pay_status_jun', 'pay_status_may',
                   'pay_status_apr', 'bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun',
                   'bill_amt_may', 'bill_amt_apr', 'pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul',
                    'pay_amt_jun', 'pay_amt_may', 'pay_amt_apr', 'default_next_month']

    df.columns = new_column_names
    
    df.drop(columns = 'ID', inplace = True)
    
    df.sex.replace({2: 'female', 1: 'male'}, inplace = True)
    
    median_edu = int(df.education.median())
    df.education.replace({5: median_edu, 6:median_edu, 0:median_edu}, inplace = True)
    
    df.marriage.replace({0:3}, inplace = True)
    
    return df
    
def get_dummies(df):
    
    df = pd.get_dummies(df, columns = ['sex']) #age dummy
    
    df = pd.get_dummies(df, columns = ['education']) #education dummy
    
    df = pd.get_dummies(df, columns = ['marriage']) #marriage dummy
    
    features = ['pay_status_sep', 'pay_status_aug', 'pay_status_jul', 'pay_status_jun', 'pay_status_may',
            'pay_status_apr']
    df = pd.get_dummies(df, columns = features) #pay status to dummy
    
    return df

def del_months(df):

    for col in df.columns:
        if 'jul' in col:
            del df[col]
        elif 'jun' in col:
            del df[col]
        elif 'may' in col:
            del df[col]
        elif 'apr' in col:
            del df[col]
        
    return df

    

