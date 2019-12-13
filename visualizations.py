import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

def distplot(variable_name, data):
    plt.figure(figsize = (16,9))
    sns.set(font_scale = 1.5)
    plt.title("Probability of Default by {}".format(variable_name))
    sns.distplot(data.loc[data['default_next_month']==0]["{}".format(variable_name)],
                       kde=True, bins=20, color = 'r',
                       label = "{} of Non-Defaulters".format(variable_name))
    sns.distplot(data.loc[data['default_next_month']==1]["{}".format(variable_name)],
                       kde=True, bins=20, color = 'g',
                       label = "{} of Defaulters".format(variable_name))
    plt.legend(loc = 'best')
    plt.ylabel('Probability')
    plt.xlabel(variable_name)

    
def bar_plot(variable_name, data):
    groupby = pd.DataFrame(data.groupby([variable_name, 'default_next_month'])
                           .size().unstack())  
    groupby.plot(kind = 'bar', color = 'rg', figsize = (16,9), legend = True)
    
def correlation_matrix(data, columns, type_of_variables):
    var = columns
    plt.figure(figsize = (16,9))
    plt.title("Correlation Matrix (Pearson) for {}".format(type_of_variables))
    corr = data[var].corr()
    cmap = sns.color_palette("YlGnBu")
    sns.heatmap(corr,xticklabels = corr.columns,yticklabels=corr.columns,
                linewidths=.1, cmap=cmap)
    

    