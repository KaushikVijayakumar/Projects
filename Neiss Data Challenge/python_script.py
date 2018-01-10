import os
import numpy as np
import pandas as pd
import matplotlib as mlib
import sys
import gc

os.getcwd()
os.chdir('D:\\neiss_datachallenge')

class DataChallange_questions(object):
    
     
     def __init__(self):
        # set the current working directory
        try:
          os.chdir('D:\\neiss_datachallenge')
        except:
            print("Error: Connection Failed")
      
     def question1(self):
        
        # Create a subset dataframe which is only body_part
        df_neiss_body_part = pd.DataFrame(df_raw_neiss, columns  = ['body_part'])
        # Validate the data
        df_neiss_body_part.head()
        # Join the body part code with the look up table for the body part to retrive the meaningful string
        df_q1 = pd.merge(df_neiss_body_part,df_raw_bodypart, left_on = 'body_part', right_on = 'Code')
        
        #Print the restult for the question 1
        print("\n\n1.a) The top 3 body parts MOST frequently represented in the dataset are below:")
        print(df_q1['BodyPart'].value_counts().head(3))
        print("\n\n1.b) The top 3 body parts LEAST frequently represented in the dataset are below:")      
        print(df_q1['BodyPart'].value_counts().tail(3))
        print('--------------------------------------------------------------------------------------------------------------------')
        
        #Release the memory to python
        del df_neiss_body_part
        del df_q1
        return
    
     def question2(self):     
        #Create the subset of the dataset with only required fields for this questions
        df_q2 =pd.DataFrame(df_raw_neiss, columns  = ['age','sex','prod1','prod2'])
        
        #Check if either of prod1 or prod2 involves skateboard
        #refer to NEISS coding manual to get the code for skateboard as 1333
        df1 = df_q2[(df_q2['prod1'] == 1333) | (df_q2['prod2'] == 1333)]
        
        #Print the results
        print("\n\n2.a)Total injuries that involves skateboard:")
        # Get the first array value which is the number of rows    
        print(df1.shape[0])
            
        print("\n\n2.b)Percentage of male and female in the accidents involving skateboards")
        df2 = df1['sex'].value_counts()
        # Use the labmda function to calculate the percentage of male and female who are injured by using skateboards
        print(df2.apply(lambda x: str(100 * x /df2.sum())+'%'))
        
        print("\n\n2.c)Mean age of the persons with accidents involving skateboards")
        #Get the mean age
        print(round(df1['age'].astype(int).mean(),2))
        print('--------------------------------------------------------------------------------------------------------------------')
        
        #calculate the correct age for the toddlers, less than 1 year 
        #df1['age_converted'] = df1['age_str'].apply(lambda x:  int(str(x)[-2:])/12 if(len(str(x)) ==3) else str(x))

        #Release the memory to python
        del df_q2
        del df1
        
        return
    
     def question3(self):
        # Get the subset with the fields relevant to only this questions
        df_q3 = pd.DataFrame(df_raw_neiss , columns = ['diag','disposition', 'diag_other'])
        # merge the neiss data with the disposition look-up table to get the disposition description 
        df_q3_1 = pd.merge(df_q3, df_raw_disposition, left_on = 'disposition', right_on = 'Code')
        # merge the neiss data with the diagnosis look-up table to get the diagnosis description 
        df_q3_2 = pd.merge(df_q3_1, df_raw_diagnosiscodes, left_on = 'diag', right_on = 'Code')
        #df_q3_2.head()
        #df_q3_2.columns
        df_q3_3 = pd.DataFrame(df_q3_2, columns = ['Code_x','Disposition','Diagnosis','diag_other'])
        #df_q3_3.columns
        
        print("\n\n3.a) The diagnosis with the highest hospitalization rate:")
        print(df_q3_3[df_q3_3['Code_x'] == '4']['Diagnosis'].value_counts().head(1))
        
        df_q3_3.head()
        df_q3_3.columns
        
        #pd.DataFrame(df_q3_3[df_q3_3['Code_x'] == '4']['Diagnosis']).T.plot.bar(stacked=True)
        #pd.DataFrame(df_q3_3[df_q3_3['Code_x'] == '4']['Diagnosis']).T.plot.bar(stacked=True)
     
        # Print the insights
        print("\n\n3.b) The diagnosis which more people leaving without being seen:")       
        print(df_q3_3[df_q3_3['Code_x'] == '6']['Diagnosis'].value_counts().head(1))
        
        print('\n\n3.c) Most of the people leave due to "Other/Non-stated" reasons.')
        print('So we need to look at the "diag_other" field to analyse this scenario.')
        print('Since the frequency of "PAIN" is high, we would like to understand how many patients who reported PAIN are leaving')
        
        df_q3_4 = df_q3_3[(df_q3_3['Code_x'] == '6') & (df_q3_3['Diagnosis'] == 'Other/Not Stated')]['diag_other'].str.contains('PAIN').value_counts()
        print("\nBelow is the % of people with pain and without pain:")
        print(df_q3_4.apply(lambda x: str(round(100.00 * x /df_q3_4.sum(),0))+'%'))
        
        print('\nConclusion: Most people who are leaving the hospital without being seen are the ones with diagnosis categorized as "other/not stated" . \nAmong them, 70% have reported some from of PAIN')
        print('--------------------------------------------------------------------------------------------------------------------')
        
        #Release memory to python
        del df_q3
        del df_q3_1
        del df_q3_2
        del df_q3_3      
        
        return
       
     def question4(self):
        
         # Fetch only the required fields for the questions into a dataframe
        df_raw_neiss_bin = pd.DataFrame(df_raw_neiss, columns = ['age','diag'])
        # create an array to set the boundaries for the age. This is used for age binning
        age_bins = [1, 10, 20, 30, 40, 50, 60, 110, 299]
        # Create an array for the age binned group
        age_group = ['1-10','11-20','21-30','31-40','41-50','51-60','61+','<1']
        
        # Create a new column with the age binned column
        df_raw_neiss_bin['age_groups'] = pd.cut(df_raw_neiss_bin['age'], age_bins, labels=age_group) 
        #convert the code column to integer. This is for the merge command and need to ensure the datatype is the same for the columns
        df_raw_diagnosiscodes['Code'] = df_raw_diagnosiscodes['Code'].astype(int)
        #convert the diag column to integer. This is for the merge command and need to ensure the datatype is the same for the columns
        df_raw_neiss_bin['diag'] = df_raw_neiss_bin['diag'].astype(int)
        # Fetch only the required columns for analysis
        df_raw_neiss_bin = pd.DataFrame(df_raw_neiss_bin , columns = ['age_groups','diag'])
     
        # Join the tables to derive the diagnosis description from the code
        df_raw_neiss_bin = pd.merge(df_raw_neiss_bin, df_raw_diagnosiscodes, left_on = 'diag' , right_on = 'Code' )
        # Fetch only the required columns for analysis. In this case the diagnosis is the description and not the code
        df_raw_neiss_bin = pd.DataFrame(df_raw_neiss_bin , columns = ['age_groups','Diagnosis'])
        
        # Fet the count of the diagnosis the data
        df_raw_neiss_bin['Diagnosis'].value_counts()
        
        # NOTE: The plots needs to be executed individually for analysis
        
        #Get the top 5 diagnosis for the total data and plot in graph
        df_raw_neiss_bin['Diagnosis'].value_counts()[:5].plot(kind='barh')
        #Get the top 5 diagnosis for patients with age less than 1 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '<1']['Diagnosis'].value_counts()[:5].plot(kind='barh')
        #Get the top 5 diagnosis for patients with age between 1 to 10 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '1-10']['Diagnosis'].value_counts()[:5].plot(kind='barh')
       #Get the top 5 diagnosis for patients with age between 11 to 20 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '11-20']['Diagnosis'].value_counts()[:5].plot(kind='barh')
        #Get the top 5 diagnosis for patients with age between 21 to 30 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '21-30']['Diagnosis'].value_counts()[:5].plot(kind='barh')
        #Get the top 5 diagnosis for patients with age between 31 to 40 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '31-40']['Diagnosis'].value_counts()[:5].plot(kind='barh')
        #Get the top 5 diagnosis for patients with age between 41 to 50 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '41-50']['Diagnosis'].value_counts()[:5].plot(kind='barh')
        #Get the top 5 diagnosis for patients with age between 51 to 60 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '51-60']['Diagnosis'].value_counts()[:5].plot(kind='barh')
        #Get the top 5 diagnosis for patients with age above 61 and plot in graph
        df_raw_neiss_bin[df_raw_neiss_bin['age_groups'] == '61+']['Diagnosis'].value_counts()[:5].plot(kind='barh')
        
        #Below are the insights
        print('\n\n4. Insights from the visualization:')
        print('Overalll trend: Laceration, Contusions, Abrasions and Strain or Sprain seems to be the top 3 diagnosis among all patients')
        print('1. Besides the general trend, below are the observations:')
        print('2. Internal organ injury seems to be substantially more among toddlers (<1 years) with 25% of them reported')
        print('3. Teen aged children tend to visit the hospital more due to "Strain and Sprain" compared to other ages')
        print('4. There are more cases of "Fracture" among the patients over 60 years of age (22%)')
        print('5. Substantial number of children ages less than 10 years are admitted due to poisoning, compared to other ages')
        print('--------------------------------------------------------------------------------------------------------------------')
        
        #Release memory to python
        del df_raw_neiss_bin
    
        return
        
     def main(self):
        
        global df_raw_bodypart
        global df_raw_diagnosiscodes
        global df_raw_disposition
        global df_raw_neiss
       
           
        df_raw_bodypart         = pd.read_csv('BodyParts.csv')
        df_raw_diagnosiscodes   = pd.read_csv('DiagnosisCodes.csv')
        df_raw_disposition      = pd.read_csv('Disposition.csv')
        df_raw_neiss            =  pd.read_csv('NEISS2014.csv')
        
        df_raw_neiss['disposition']=df_raw_neiss['disposition'].astype(str)
        df_raw_disposition['Code']=df_raw_disposition['Code'].astype(str)
        df_raw_neiss['diag']=df_raw_neiss['diag'].astype(str)
        df_raw_diagnosiscodes['Code']=df_raw_diagnosiscodes['Code'].astype(str)
        
        self.question1();
        self.question2();
        self.question3();
        self.question4();
        
        del df_raw_bodypart
        del df_raw_diagnosiscodes
        del df_raw_disposition
        del df_raw_neiss
        
        return


        
if __name__ == "__main__":
    # calling main function
    sys.stdout = open('NEISS_DataChallenge_Insights.txt', 'w')
    DataChallange_questions = DataChallange_questions()
    DataChallange_questions.main()
    sys.stdout.close()
    gc.collect
