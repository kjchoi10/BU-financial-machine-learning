%matplotlib inline
         import sys
         import numpy as np
         import matplotlib as plt
         from pylab import *
         from scipy import *
         from math import sqrt, log
         from sklearn import svm, metrics
         from sklearn.ensemble import RandomForestRegressor
         import csv, random
         read_data(’test_mr.csv’, ’sp_g.csv’)
         x_start,x_pred,x_test = initialize("2012Jan","2014Jan","2014Feb")
         # call initialize with training period and test month

mom_range = [3,6,9,12]
1
# number of months in momentum
predicted,sp500,random25,test_months = machine(mom_range,x_start,x_pred,x_test) # call machine which returns predicted & sp arrays
         plot_results(predicted,sp500,random25,test_months)

#### Function to Open Data Files ###################### def read_data(mr, sp):
             global data, data_sp, nrows, ncols
             with open(mr, ’rU’) as f:
                 reader = csv.reader(f)
                 data = list(reader)
             nrows = len(data)
             ncols = len(data[0])
             with open(sp, ’rU’) as g:
                 reader = csv.reader(g)
                 data_sp = list(reader)
            return

### Function to return xy coordinates of a cell in a 2D list
         def index_2d(myList, v):
             for i, x in enumerate(myList):
                 if v in x:
                     return (i, x.index(v))

### Initializing useful arrays and variables
def initialize(m_start,m_pred,m_sp):
             x_start,y = index_2d(data,m_start)
            print "start training:", data[x_start][y] # start training from here
             x_pred,y = index_2d(data,m_pred)
              print "predict:", data[x_pred][y] # predict this month
             x_test,y_sp = index_2d(data_sp,m_sp)
              print "start testing:", data_sp[x_test][y_sp] # start testing from here, till 2015Apr
              return x_start, x_pred, x_test

### Populating training and testing data sets
def machine(mom_range,x_start,x_pred,x_test):
             ## defining useful arrays & variables:
            predicted = [1]
            # array storing predicted cumulative returns for top 25 companies sp500 = [1]
            # array storing S&P500 cumulative returns
            random25 = [1]
            test_months = [data[x_pred][0]]
            # stores months over which test cumulative returns are calculated
             prod_pred = 1
             prod_sp = 1
             prod_rand = 1
            for m in range(x_test,nrows): # testing period
                test = []
              # testing data for month m and company i
                result = []
              # predicted result of model, changes dynamically train = []
              # to store training set of feature vectors target = []
              # to store current returns
              for i in range(1,ncols): # loop for each company
                for j in range(x_start,x_pred): # dynamic training on 2 years data
                    temp = []
                    for k in mom_range:
                        fv, stdev = mom(k,j,i)

            # geting feature vector for month j, company i
            if(stdev!=0): temp.append(fv) #/stdev)
                else:
                    temp.append(0)
            # filling zeros for 0 stdevs
                    train.append((temp)) target.append(data[j][i]) # current return
                    model = RandomForestRegressor(n_estimators=20) #, min_samples_split=1) model.fit(train,target)
                    for i in range(1,ncols):
                    # loop for each company
                    # testing data of technical feature vectors
                    temp = []
                        for k in mom_range:
                            fv, stdev = mom(k,m,i)
                            # getting feature vector for testing month m, company i if(stdev!=0):
                            temp.append(fv) #/stdev) else:
                            temp.append(0)
                            # filling zeros for 0 stdevs
                            test.append((temp))
                            result.append(model.predict(test))
                            # dynamic shifting of training data for next run
                            x_start = x_start + 1
                            x_pred = x_pred + 1
                            pred25 = np.argsort(result)
                            # pred25[0] contains sorted list of indices of predicted returns, starting from 0
                            sum25 = 0
                            sum25_rand = 0
                                for j in range(ncols-26,ncols-1):
                                    sum25 = sum25 + (1+float(data[m][pred25[0][j]+1]))
                                    # cumulative return for top 25 companies for mth test month
                                    sum25_rand = sum25_rand + (1+float(data[m][random.randint(1,ncols-1)])) # cumulative return for random 25 companies for mth test month
                                    prod_pred = prod_pred*sum25/25
                                    # average cumulative return for mth test month
                                    prod_sp = prod_sp*float(data_sp[m][1]) # S&P cumulative return
                                    prod_rand = prod_rand*sum25_rand/25
                                    # random 25 companies’, average cumulative return for mth test month
                                                     random25.append(prod_rand)
                                    predicted.append(prod_pred)
                                    sp500.append(prod_sp)
                                    test_months.append(data[m][0])
                                    print "now testing:", data[m][0]
                                 return predicted, sp500, random25, test_months

### Plot predicted results against S&P ####
def plot_results(predicted, sp500, random25, test_months):
             technical = ’Cumulative Returns Comparison\n' + str(mom_range) + '-momentum'
             plt.title(technical,fontsize=18)
             plt.xlabel(’Test Months’,fontsize=18)
             plt.ylabel(’$\prod(1+r)$’,fontsize=20)
             xval = range(0,len(test_months))
             plt.xticks(xval,test_months,rotation=45)
             plt.plot(xval, predicted, ’rD-’,label=’Predicted’)
             plt.plot(xval, sp500, ’bo-’,label=’S&P500’)
             plt.plot(xval, random25, ’gv-’,label=’Random 25 (Reality Check)’)
             plt.tight_layout()
             plt.legend(loc=’best’, fontsize=’small’)
             plt.savefig(’rf.png’)
             plt.show()

# Momentum Technical
def mom(m_range,month,company):
    mom = []
    for l in range(1,m_range+1):
        mom.append(1.0+float(data[month-l][company]))
    return(prod(mom),std(mom))
