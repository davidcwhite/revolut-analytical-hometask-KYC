import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import numpy as np
from functools import reduce
import matplotlib.style as style
from matplotlib import rcParams
import ast
sns.set_theme(style="darkgrid")
rcParams.update({'figure.autolayout': True})
style.use('ggplot')

#read in data
docs = pd.read_csv("doc_reports_sample.csv")
df_docs = docs
df_docs.sort_values('user_id', inplace=True)
faces = pd.read_csv("face_reports_sample.csv")
df_faces = faces
df_faces.sort_values('user_id', inplace=True)

# set suffixes for new df
doc_report = "_doc_report"
face_report = "_face_report"

# merge on user_id and attempt_id
result = pd.merge(df_docs, df_faces, how='left', on=[
                  'user_id', 'attempt_id'], suffixes=("_doc_report", "_face_report"))

# convert timestamps to pd datetime
result["created_at" +
       doc_report] = pd.to_datetime(result["created_at" + doc_report])
result["created_at" +
       face_report] = pd.to_datetime(result["created_at" + face_report])
# sort
result.sort_values("created_at" + doc_report, inplace=True)

# measure difference in times
creation_time_difference = result["created_at" +
                                  doc_report] - result["created_at" + face_report]
creation_time_difference = pd.Series(
    [i.total_seconds() for i in creation_time_difference])
result['creation_time_difference'] = creation_time_difference
creation_time_difference.value_counts()

# logical test for passing
result['pass'] = (result["result" + doc_report] ==
                  "clear") & (result["result" + face_report] == "clear")
result['pass'] = result['pass'].map({True: 'clear', False: 'consider'})

# quick sanity check of the data
# result.to_csv(r'output.csv', index=False, header=True)


def plot_applicants_by_day():
    days_in_period = pd.date_range('2017-05-23', '2017-10-31', normalize=False)
    day_bins = []
    for k in range(len(days_in_period)-1):
        upper = result["created_at" + doc_report] < days_in_period[k+1]
        lower = result["created_at" + doc_report] > days_in_period[k]
        day_bins.append(len(result[upper & lower]))
    # inputs
    num = np.array(days_in_period[:-1])
    applications = np.array(day_bins)
    # convert to pandas dataframe
    d = {'Days': num, 'Applications': applications}
    data = pd.DataFrame(d)
    data['Applications SMA (20d)'] = data.Applications.rolling(
        20).mean().shift(-3)
    sns.set_context("talk")
    plt.figure(figsize=(9, 6))
    # Time series plot with Seaborn lineplot() with label
    sns.lineplot(x="Days", y="Applications",
                 label="Daily Applications", data=data,
                 ci=None)
    # 7-day rolling average Time series plot with Seaborn lineplot() with label
    sns.lineplot(x="Days", y="Applications SMA (20d)",
                 label="Applications SMA (20d)",
                 data=data,
                 ci=None)
    plt.xlabel("Days", size=14)
    plt.ylabel("Applications", size=14)
    plt.title('Growth in Applications')
    plt.show()

# -------plot applications per day#-------
# plot_applicants_by_day()

def SMA(column, period):
    return column.rolling(period).mean()

def corr(x, y, period):
    return x.rolling(period).corr(y)

def plotter(x, y, title):
    plt.legend(loc='best', prop={'size': 10})
    plt.legend(loc='best')
    plt.ylabel(y, fontsize=10)
    plt.xlabel(x, fontsize=10)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.title(title)
    plt.show()

def sns_plotter(x, y, title):
    sns.set(style='darkgrid')
    sns.set_context("talk")
    plt.legend(loc='best', prop={'size': 10})
    plt.title(title, size=14)
    plt.xlabel(x, size=12)
    plt.ylabel(y, size=12)
    plt.show()

def plot_compare_avg(variables, labels, title=''):
    df = pd.DataFrame()
    for i in range(len(variables)):
        label = labels[i]
        result[variables[i] + "_int"] = result[variables[i]].astype(int)
        result[variables[i] + "_SMA"] = SMA(result[variables[i] + "_int"], 500)
        df[variables[i] + "_SMA"] = result[variables[i] + "_SMA"]
        if i == 0:
            sns.lineplot(x=result["created_at" + doc_report],
                         y=result[variables[i]+"_SMA"], label=label.format(i=i), linewidth=4)
        else:
            sns.lineplot(x=result["created_at" + doc_report],
                         y=result[variables[i]+"_SMA"], label=label.format(i=i))
        plt.xticks(rotation=20)
    sns_plotter("Time", "Pass Rate", title)
    plt.show()

# core correlation matrix method
def corr_matrix(variables, labels, title=''):
    df = pd.DataFrame()
    for i in range(len(variables)):
        label = labels[i]
        result[variables[i] + "_int"] = result[variables[i]].astype(int)
        result[variables[i] + "_SMA"] = SMA(result[variables[i] + "_int"], 500)
        df[label.format(i=i)] = result[variables[i] + "_SMA"]
    sns.set(rc={'figure.figsize': (6, 5)})
    sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1,
                center=0,  cbar=False, fmt='.1g')
    sns.set(style='darkgrid')
    sns.set_context("talk")
    plt.title(title, size=14)
    plt.xticks(rotation=25)
    plt.yticks(rotation=25)

# for rolling correlation coefficient time series
def plot_compare_corr(variables, labels, title=''):
    result['count'] = 1
    for i in range(len(variables)):
        label = labels[i]
        result[variables[i] + "_int"] = result[variables[i]].astype(int)
        result[variables[i] +
               "_SMA"] = SMA(result[variables[i] + "_int"], 1500)
        result[variables[i]+"_corr_to_pass_rate"] = result['pass_SMA'].rolling(
            1500).corr(result[variables[i]+"_SMA"])
        if i == 0:
            sns.lineplot(x=result["created_at" + doc_report], y=result[variables[i] +
                                                                       "_corr_to_pass_rate"], label=label.format(i=i), linewidth=4)
        else:
            sns.lineplot(x=result["created_at" + doc_report],
                         y=result[variables[i]+"_corr_to_pass_rate"], label=label.format(i=i))
        plt.xticks(rotation=20)
    sns_plotter("Time", "Correlation (1500d, rolling)", "A closer look")
    plt.figure(figsize=(9, 6))


def plot_compare_corr_timeseries(variables, labels, title=''):
    df = pd.DataFrame()
    for i in range(len(variables)):
        label = labels[i]
        result[variables[i] + "_int"] = result[variables[i]].astype(int)
        result[variables[i] +
               "_SMA"] = SMA(result[variables[i] + "_int"], 1200)
        df[variables[i]+"_corr_to_pass_rate"] = result[variables[0] +
                                                       "_SMA"].rolling(1200).corr(result[variables[i]+"_SMA"])
        if i == 0:
            sns.lineplot(x=result["created_at" + doc_report], y=df[variables[i] +
                                                                   "_corr_to_pass_rate"], label=label.format(i=i), linewidth=4)
        else:
            sns.lineplot(x=result["created_at" + doc_report],
                         y=df[variables[i]+"_corr_to_pass_rate"], label=label.format(i=i))
        plt.xticks(rotation=20)
    sns_plotter("Time", "Correlation (1200d, rolling)", title)
    plt.show()

# main handler for SMA output
def results_avg(variables, labels, title=''):
    for i in range(len(variables)):
        result[variables[i]] = result[variables[i]] == "clear"
    plot_compare_avg(variables, labels, title)
    plt.show()

# -------plot declining pass rate with two conditional checks' pass rate-------
# results_avg(['pass',
#                 'result' + doc_report,
#                 'result' + face_report],
#                 ['Overall Pass Rate','Document Report Check Pass Rate','Facial Similarity Check Pass Rate'],
#                 'A declining pass rate - one cause')

# -------plot "clear" values over time for doc report conditional checks for IIR-------
# results_avg(["result" + doc_report,
#                 'conclusive_document_quality_result',
#                 'image_integrity_result',
#                 'colour_picture_result',
#                 'supported_document_result',
#                 'image_quality_result'],
#                 ['Doc pass rate', 'Image integrity result','Conclusive document quality result','Colour picture result','Supported document result','Image quality result'],
#                 'Document report check breakdown')

# -------plot clear values over time for doc report breakdown checks 1-------
# results_avg(["result" + doc_report,
#                 'visual_authenticity_result' + doc_report,
#                 'image_integrity_result',
#                 'data_validation_result',
#                 'data_consistency_result',
#                 'data_comparison_result',
#                 'police_record_result',
#                 'compromised_document_result'],
#                 ['DPR','VAR','IIR','DVR','DConsisR','DComparR','PRR','CDR'],
#                 'Document report check breakdown')

# -------plot "clear" values over time for doc report conditional checks for other doc report check factors-------
# results_avg(["result" + doc_report,
#                 'data_validation_result',
#                 'data_consistency_result',
#                 'data_comparison_result',
#                 'police_record_result',
#                 'compromised_document_result'],
#                 ['Doc pass rate', 'Data Validation Result','Data Consistency Result','Data Comparison Result','Police Record Result','Compromised Document Result'],
#                 'Document report check breakdown')

# -------plot "clear" valuese for doc report sub-breakdown -> IRR conditional checks-------
# results_avg([
#                         'image_integrity_result',
#                         'supported_document_result',
#                         'image_quality_result',
#                         'colour_picture_result',
#                         'conclusive_document_quality_result'],
#                         ['IIR','SDR','IQR','CPR','CDQR'],
#                         'IIR failure rate cause')

# main handler for correlation matrix output
def results_corr_matrix(variables, labels, title=''):
    for i in range(len(variables)):
        result[variables[i]] = result[variables[i]] == "clear"
    corr_matrix(variables, labels, title)
    plt.show()

# -------show correlation matrix for doc report breakdown checks 1-------
# results_corr_matrix(["result" + doc_report,
#                 'visual_authenticity_result' + doc_report,
#                 'image_integrity_result',
#                 'data_validation_result',
#                 'data_consistency_result',
#                 'data_comparison_result',
#                 'police_record_result',
#                 'compromised_document_result'],
#                 ['DPR','VAR','IIR','DVR','DConsisR','DComparR','PRR','CDR'],
#                 'Document report check breakdown analysis')

# -------show correlation matrix for doc report conditional checks 1 -> IRR breakdown-------
# results_corr_matrix(["result" + doc_report,
#                 'data_validation_result',
#                 'data_consistency_result',
#                 'data_comparison_result',
#                 'police_record_result',
#                 'image_integrity_result',
#                 'conclusive_document_quality_result',
#                 'colour_picture_result',
#                 'supported_document_result',
#                 'image_quality_result'],
#                 ['DPR','DVR','DConsistR','DComparR','PRR','IIR','CDQR','CPR','SDR','IQR'],
#                 'Document report check analysis')

# -------show correlation matrix for doc report conditional checks-------
# results_corr_matrix(["result" + doc_report,
#                 'image_integrity_result',
#                 'conclusive_document_quality_result',
#                 'colour_picture_result',
#                 'supported_document_result',
#                 'image_quality_result'],
#                 ['DPR','IIR','CDQR','CPR','SDR','IQR'],
#                 'Document report check analysis')

# main handler for correlation time series output
def results_corr_timeseries(variables, labels, title=''):
    for i in range(len(variables)):
        result[variables[i]] = result[variables[i]] == "clear"
    plot_compare_corr_timeseries(variables, labels, title)
    plt.show()

# -------plot correlation time series for doc report first (2) sub-checks-------
# results_corr_timeseries(['pass',
#                         'result' + doc_report,
#                         'result' + face_report,],
#                         ['Overall Pass Rate','Document Check Pass Rate','Facial Similarity Check Pass Rate'],
#                         'A closer look')

# -------plot correlation time series for doc report conditional checks-------
# results_corr_timeseries([
#                         "result" + doc_report,
#                         'visual_authenticity_result' + doc_report,
#                         'image_integrity_result',
#                         'data_validation_result',
#                         'data_consistency_result',
#                         'data_comparison_result',
#                         'police_record_result',
#                         'compromised_document_result'],
#                         ['DPR','VAR','IRR','DVR','DConsisR','DComparR','PRR','CDR'],
#                         'Closer still...')

# -------plot correlation time series for doc report -> IRR conditional checks-------
# results_corr_timeseries([
#                         'image_integrity_result',
#                         'supported_document_result',
#                         'image_quality_result',
#                         'colour_picture_result',
#                         'conclusive_document_quality_result'],
#                         ['IIR','SDR','IQR','CPR','CDQR'],
#                         'CDQR - a sharp change from inverse to perfect correlation')

# doc_prop = pd.json_normalize(result['properties_doc_report'])

# result.sort_values("created_at" + doc_report, inplace=True)
properties_doc_report = pd.DataFrame(
    result['properties_doc_report'].apply(ast.literal_eval).values.tolist())
properties_doc_report.columns = 'properties_doc_report.' + \
    properties_doc_report.columns

df1 = pd.DataFrame()
df1['created_at_doc_report'] = result["created_at" + doc_report]
df1['result_doc_report'] = result['result_doc_report']
df1['conclusive_document_quality_result'] = result['conclusive_document_quality_result']
df1 = df1.reset_index(drop=True)

properties_doc_report['created_at_doc_report'] = df1['created_at_doc_report']
properties_doc_report['result_doc_report'] = df1['result_doc_report']
properties_doc_report['conclusive_document_quality_result'] = df1['conclusive_document_quality_result']

# properties_doc_report.to_csv(r'properties_output_1.csv', index = False, header=True)

def results_avg_2(variables, labels, title=''):
    for i in range(len(variables)):
        if i > 0:
            properties_doc_report[variables[i]
                                  ] = properties_doc_report[variables[i]].isnull()
        else:
            properties_doc_report[variables[i]
                                  ] = properties_doc_report[variables[i]] != "clear"
    plot_compare_avg_2(variables, labels, title)
    plt.show()

def plot_compare_avg_2(variables, labels, title=''):
    for i in range(len(variables)):
        label = labels[i]
        properties_doc_report[variables[i] +
                              "_int"] = properties_doc_report[variables[i]].astype(int)
        properties_doc_report[variables[i] + "_SMA"] = SMA(
            properties_doc_report[variables[i] + "_int"], 500)
        if i == 0:
            sns.lineplot(x=properties_doc_report["created_at" + doc_report],
                         y=properties_doc_report[variables[i]+"_SMA"], label=label.format(i=i), linewidth=4)
        else:
            sns.lineplot(x=properties_doc_report["created_at" + doc_report],
                         y=properties_doc_report[variables[i]+"_SMA"], label=label.format(i=i))
        plt.xticks(rotation=20)
    sns_plotter("Time", "Rate", title)
    plt.show()

# -------plot "clear" values over time for doc report conditional checks for other doc report check factors-------
# results_avg_2(['result_doc_report',
#               'conclusive_document_quality_result'],
#                 ['Doc Report Failure Rate', 'CDQR Null Rate'],
#                 'CDQR Null Values')

properties_doc_report['pass'] = properties_doc_report['result_doc_report'] != "clear"
properties_doc_report['GBR'] = properties_doc_report['properties_doc_report.issuing_country'] == 'GBR'
properties_doc_report['FRA'] = properties_doc_report['properties_doc_report.issuing_country'] == 'FRA'
properties_doc_report['IRL'] = properties_doc_report['properties_doc_report.issuing_country'] == 'IRL'
properties_doc_report['POL'] = properties_doc_report['properties_doc_report.issuing_country'] == 'POL'
properties_doc_report['PRT'] = properties_doc_report['properties_doc_report.issuing_country'] == 'PRT'
properties_doc_report['LTU'] = properties_doc_report['properties_doc_report.issuing_country'] == 'LTU'
properties_doc_report['ESP'] = properties_doc_report['properties_doc_report.issuing_country'] == 'ESP'

# Compare issuing countries
# plot_compare_avg_2(['pass',
#                     'GBR',
#                     'FRA',
#                     'IRL',
#                     'POL',
#                     'PRT',
#                     'LTU',
#                     'ESP', ],
#                    ['Document Report Failure rate', 'GBR',
#                        'FRA', 'IRL', 'POL', 'PRT', 'LTU', 'ESP'],
#                    'Document issuing country breakdown')

properties_doc_report['Male'] = properties_doc_report['properties_doc_report.gender'] == 'Male'
properties_doc_report['Female'] = properties_doc_report['properties_doc_report.gender'] == 'Female'

# plot_compare_avg_2(['pass',
#                     'Male',
#                     'Female'],
#                    ['Document Report Failure rate', 'Male',
#                        'Female'],
#                    'Gender breakdown')

properties_doc_report['passport'] = properties_doc_report['properties_doc_report.document_type'] == 'passport'
properties_doc_report['driving_licence'] = properties_doc_report['properties_doc_report.document_type'] == 'driving_licence'
properties_doc_report['national_identity_card'] = properties_doc_report['properties_doc_report.document_type'] == 'national_identity_card'

# plot_compare_avg_2(['pass',
#                     'passport',
#                     'driving_licence',
#                     'national_identity_card'],
#                    ['Document Report Failure rate', 'Passport', 'Driving Licence',
#                        'National Identity Card'],
#                    'Document type breakdown')

# result.to_csv(r'output', index = False, header=True)
