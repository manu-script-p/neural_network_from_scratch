# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:55:37 2020

@author: manu, sathwik, goutam
"""

import pandas as pd

def clean(df):
    #filling the null values with suitable parameters.
    df['Age']=df['Age'].fillna(df['Age'].mean())
    df['Weight']=df['Weight'].fillna(df['Weight'].mean())
    df['Delivery phase']=df['Delivery phase'].fillna(df['Delivery phase'].mode()[0])
    
        
    df['HB']=df['HB'].fillna(df['HB'].median())
    #1.5 is the normal bp ratio of a person.
    df['BP']=df['BP'].fillna(1.5)
    df['Education']=df['Education'].fillna(df['Education'].mode()[0])
    df['Residence']=df['Residence'].fillna(df['Residence'].mode()[0])
    
    #normalizing values
    df1=df['Age']
    df.Age=(df1-df1.min())/(df1.max()-df1.min())*10+0.01
    df2=df['Weight']
    df.Weight=(df2-df2.min())/(df2.max()-df2.min())*10+0.01
    
    df1=df['Community']
    df.Community=(df1-df1.min())/(df1.max()-df1.min())*10+0.01
    df2=df['IFA']
    df.IFA=(df2-df2.min())/(df2.max()-df2.min())*10+0.01
    df1=df['Delivery phase']
    df['Delivery phase']=(df1-df1.min())/(df1.max()-df1.min())*10+0.01
    
    df1=df['BP']
    df['BP']=(df1-df1.min())/(df1.max()-df1.min())*10+0.01

    return df

df=pd.read_csv('LBW_Dataset.csv')
#cleaning
df=clean(df)
#writing the dataframe to a csv file
df.to_csv('processed.csv',index=False)