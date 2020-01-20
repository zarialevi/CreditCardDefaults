def FixEducation(df):
    
    to_drop = (df.loc[(df.Education > 4 )|(df.Education == 0) ]).index
    to_drop_again = (df.loc[df.MaritalStatus == 0]).index
    df = df.drop(to_drop)
    df = df.drop(to_drop_again)
    
    return df
 

def AddAverages(df):
    
    df['average_bill'] =  (df['BillApr'] + df['BillMay'] + df['BillJun'] + df['BillJul'] + df['BillAug'] + df['BillSep'])/6                                                                           
    df['average_payment'] = (df['PrevPaymentSep']  + df['PrevPaymentAug'] + df['PrevPaymentJul'] + df['PrevPaymentJun'] + df['PrevPaymentMay'] + df['PrevPaymentApr'])/6
    df['total_payment'] = (df['PrevPaymentSep'] + df['PrevPaymentAug'] + df['PrevPaymentJul'] + df['PrevPaymentJun'] + df['PrevPaymentMay'] + df['PrevPaymentApr'])                                                                                  
    df['average_percentage_of_bill_paid'] = (df['average_payment']/df['average_bill'])*100
    df['bill_paid/credit_limit'] = (df['total_payment']/df['CreditLimit'])*100
    df['average_bill_paid/credit_limit'] = (df['average_payment']/df['CreditLimit'])*100
    
    return df

def AddStrings(df):
    gender_dict ={1:'male', 2:'female'}
    education_dict = {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'}
    marriage_dict = {1 : 'married', 2 : 'single', 3 : 'others'}
    df['Gender'] = df['Gender'].map(gender_dict)
    df['Education'] = df['Education'].map(education_dict)
    df['MaritalStatus'] = df['MaritalStatus'].map(marriage_dict)
    
    return df

def DropNonUsers(df):
    drop_non_users =( df.loc[(df.average_bill == 0) & (df.total_payment == 0)]).index
    df = df.drop(drop_non_users)
    df_test = df.loc[df.average_percentage_of_bill_paid == np.inf].index
    df = df.drop(df_test)
    return df

def FixNegativestats(df):
    fil = (df.RepayStatApr == -2) | (df.RepayStatApr == -1) | (df.RepayStatApr == 0)
    df.loc[fil, 'RepayStatApr'] = 0
    fil = (df.RepayStatMay == -2) | (df.RepayStatMay == -1) | (df.RepayStatMay == 0)
    df.loc[fil, 'RepayStatMay'] = 0
    fil = (df.RepayStatJun == -2) | (df.RepayStatJun == -1) | (df.RepayStatJun == 0)
    df.loc[fil, 'RepayStatJun'] = 0
    fil = (df.RepayStatJul == -2) | (df.RepayStatJul == -1) | (df.RepayStatJul == 0)
    df.loc[fil, 'RepayStatJul'] = 0
    fil = (df.RepayStatAug == -2) | (df.RepayStatAug == -1) | (df.RepayStatAug == 0)
    df.loc[fil, 'RepayStatAug'] = 0
    fil = (df.RepayStatSep == -2) | (df.RepayStatSep == -1) | (df.RepayStatSep == 0)
    df.loc[fil, 'RepayStatSep'] = 0
    
    return df

def TheUltimateCleaner(df):
    df = FixEducation(df)
