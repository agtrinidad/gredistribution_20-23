# %%
import pandas as pd

#img2table is a really awesome open-source package dedicated to converting images to manipulable tabular data
from img2table.document import PDF #Specifying PDF usage
from img2table.ocr import PaddleOCR #Specifying OCR system

from statsmodels.stats.weightstats import DescrStatsW 
#We're going to be doing some statistics...

# %%
gre_pdf = PDF("gre-guide-table-3a.pdf") #Pathing to particular PDF

# %%
padd_ocr = PaddleOCR(lang="en") #Loading PaddleOCR, an open-source OCR toolkit based on PaddlePaddle
pdf_tables = gre_pdf.extract_tables(ocr=padd_ocr)

# %%
greAvg = pd.concat([pdf_tables[0][0].df,
                    pdf_tables[1][0].df, 
                    pdf_tables[2][0].df, 
                    pdf_tables[3][0].df], 
                    axis = 'index')

#There are four pages, each with one table, so we have to specify a dataframe from each page
#Numeration above reflects Python's native zero-indexing

greAvg = greAvg.drop_duplicates(subset=0) #There's a few duplicate rows that ETS had for viewer clarity
greAvg.columns = greAvg.iloc[0]

#Column name cleaning
greAvg.columns = greAvg.columns.str.replace(r'\n',' ', regex=True)
greAvg.columns = greAvg.columns.str.replace('- ',' - ')
greAvg.columns = greAvg.columns.str.replace(r'^([a-zA-Z]{2}) ', r'\1: ', regex=True)
greAvg.columns = greAvg.columns.str.replace(r' M$', r' Mean', regex=True)

greAvg['Intended Graduate Major'] = greAvg['Intended Graduate Major'].str.replace(r' ─ ',r': ').str.replace(r'\n',' ', regex=True)

greAvg = greAvg.drop(greAvg.index[0]).reset_index().drop('index', axis = 'columns')



# %%
#Casting this field as string
greAvg['Intended Graduate Major'] = greAvg['Intended Graduate Major'].astype('string')

# %%
#Filling N/A values
greAvg = greAvg.fillna(0)

#Removing commas from the N values, caused them to have been read as string
greAvg['VR: N'] = greAvg['VR: N'].str.replace(',','').astype('float')
greAvg['QR: N'] = greAvg['QR: N'].str.replace(',','').astype('float')
greAvg['AW: N'] = greAvg['AW: N'].str.replace(',','').astype('float')


# %%
#Casting numerical values as float
#String major name is stored as string, so will remain unaffected

for column in greAvg.columns:
    if greAvg[column].dtype == 'object':
        greAvg[column] = greAvg[column].astype(float)

# %%
greAvg['Intended Graduate Major'] = pd.Series(range(1,61)).astype('string').str.pad(2, side='left', fillchar='0') + ': ' + greAvg['Intended Graduate Major']
#Enumerating beginning of each string to maintain ETS-intended order

# %%
#Adding suffixes to entries
greAvg.iloc[12:18,0] = greAvg.iloc[12:18,0].astype('string') + ' Engineering' #Varieties of engineering

greAvg.iloc[37:39,0] = greAvg.iloc[37:39,0].astype('string') + ' Education' #Levels of pupil education
greAvg.iloc[40:43,0] = greAvg.iloc[40:43,0].astype('string') + ' Education' #Levels of pupil education II

# %%
#Making new entry for "all" test takers
#This is going to avoid "other fields" as ETS didn't specify values for them
#This can be accomplished by multiplying each row's values by the corresponding "N" value
#To avoid unnecessary errors, a copy of the dataframe will be made here

greOperational = greAvg.copy()

#Dropping the empty categories
greOperational = greOperational.drop(greOperational[greOperational['Intended Graduate Major'] == '52: OTHER FIELDS'].index)
greOperational = greOperational.drop(greOperational[greOperational['Intended Graduate Major'] == '60: Other Fields, Other*'].index)

#Dropping the supercategories
supercatlist = ['01: LIFE SCIENCES', 
        '05: PHYSICAL SCIENCES', 
        '12: ENGINEERING', 
        '20: SOC. & BEHAVIORAL SCI.', 
        '27: HUMANITIES & ARTS', 
        '35: EDUCATION', '46: BUSINESS',]

for cmajor in supercatlist:
    greOperational = greOperational.drop(greOperational[greOperational['Intended Graduate Major'] == cmajor].index)

# %%
#Take each percentage, divide by 100 to convert to proportion, multiply by N to approximate reversion to raw frequency
greOperational.iloc[:,1:10] = greOperational.iloc[:,1:10].div(100).mul(greOperational['VR: N'], axis='index').round(0)
greOperational.iloc[:,13:22] = greOperational.iloc[:,13:22].div(100).mul(greOperational['QR: N'], axis='index').round(0)
greOperational.iloc[:,24:32] = greOperational.iloc[:,24:32].div(100).mul(greOperational['AW: N'], axis='index').round(0)


# %%
#And then we sum all these items...

greGrandtotal = greOperational.sum(axis = 'index', numeric_only = True).to_frame().transpose()
greGrandtotal.insert(0, column='placeholder', value='')


# %%
#Setting column names...
greGrandtotal.columns = greOperational.columns

# %%
#Defining weighted means formula using DescrStatsW...
def wmean(observations, weightlist):
    result = DescrStatsW(observations, weights = weightlist, ddof = 1).mean
    return result

# %%
#Using greOperational because we dropped the empty "Other Fields" values there

greGrandtotal['VR: Mean'] = wmean(greOperational['VR: Mean'],greOperational['VR: N'])
greGrandtotal['QR: Mean'] = wmean(greOperational['QR: Mean'],greOperational['QR: N'])
greGrandtotal['AW: Mean'] = wmean(greOperational['AW: Mean'],greOperational['AW: N'])

# %%
#Calculating proportions then converting to percentages
greGrandtotal.iloc[:,1:10] = greGrandtotal.iloc[:,1:10].div(greGrandtotal['VR: N'], axis = 'index').round(4) * 100
greGrandtotal.iloc[:,13:22] = greGrandtotal.iloc[:,13:22].div(greGrandtotal['QR: N'], axis = 'index').round(4) * 100
greGrandtotal.iloc[:,24:32] = greGrandtotal.iloc[:,24:32].div(greGrandtotal['AW: N'], axis = 'index').round(4) * 100

#Resetting calculated means
greGrandtotal['VR: Mean'] = greGrandtotal['VR: Mean'].round(1)
greGrandtotal['QR: Mean'] = greGrandtotal['QR: Mean'].round(1)
greGrandtotal['AW: Mean'] = greGrandtotal['AW: Mean'].round(2)

# %%
#Calcuating a weighted standard deviation is a little more complicated!
#The good thing is that greAvg is left unaltered...
#So we're still able to use that set as our observations to calculate the weighted SD.

#Using DescrStatsW that we called above...

def wstd(observations, weightlist):
    result = DescrStatsW(observations, weights = weightlist, ddof = 1).std 
    return result

# %%
#Calculating standard deviation from here...
greGrandtotal['QR: SD'] = wstd(greOperational['QR: Mean'], greOperational['QR: N'])
greGrandtotal['VR: SD'] = wstd(greOperational['VR: Mean'], greOperational['VR: N'])
greGrandtotal['AW: SD'] = wstd(greOperational['AW: Mean'], greOperational['AW: N'])

print(greGrandtotal['QR: SD']) #Checking that things make sense!

# %%
greGrandtotal.iloc[:,1:] = greGrandtotal.iloc[:,1:].round(2)

# %%
#Prepping for integration into main sheet
greGrandtotal['Intended Graduate Major'] = '61: GRAND TOTAL'

#It should be noted here that the data only concerns those who reported particular intended majors
#If ETS data from July 2020 to June 2023 is to be believed, only half of test-takers repoted majors
#That is, roughly 591,000 out of 1,041,330 at the highest estimate

# %%
#Integrating into main
greGrandtotal.rename(index={0:(len(greAvg))},inplace=True)
greAvg = pd.concat([greAvg,greGrandtotal])

# %%
#We'll be tacking the GRE Population data from the same period on for an improved user experience.
greETS = {'Intended Graduate Major':    '62: ETS-Provided Population',
          'VR: Mean':   float(151.29),
          'QR: Mean':   float(156.93),
          'AW: Mean':   float(3.49),
          
          'VR: SD':   float(8.27),
          'QR: SD':   float(9.89),
          'AW: SD':   float(0.88),
          
          'VR: N':   float(1039310),
          'QR: N':   float(1041330),
          'AW: N':   float(1037639)}

greETS = pd.Series(greETS).to_frame().transpose()
greETS.rename(index={0:(len(greAvg))},inplace=True)

# %%
greAvg = pd.concat([greAvg,greETS])
greAvg = greAvg.fillna(0)

# %%
#Exporting to CSV for further manipulation
greAvg.to_csv("gre-table.csv", sep=',', encoding='utf-8', index=False)


