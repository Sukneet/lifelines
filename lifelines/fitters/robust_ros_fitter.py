from __future__ import print_function, division
  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from lifelines.fitters import BaseFitter
from sklearn import linear_model

import code

class RobustROSFitter(BaseFitter):
    '''
    Class to implement the Robust regression-on-order statistics (ROS)
    method outlined in Nondetects and Data Analysis by Dennis R. Helsel
    (2005) to estimate the left-censored (non-detect) values of a
    dataset.
    Parameters
    ----------
    data : pandas DataFrame
        The censored dataset for which the non-detect values need to be
        estimated.
    result_col : optional string (default='res')
        The name of the column containing the numerical values of the
        dataset. Left-censored values should be set to the detection
        limit.
    censorship_col : optional string (default='cen')
        The name of the column containing indicating which observations
        are censored.
        `True` implies Left-censorship. `False` -> uncensored.
    Attributes
    ----------
    N_obs : int
        Total number of results in the dataset
    N_cen : int
        Total number of non-detect results in the dataset.
    cohn : pandas DataFrame
        A DataFrame of the unique detection limits found in `data` along
        with the `A`, `B`, `C`, and `PE` quantities computed by the
        estimation.
    data : pandas DataFrame
        An expanded version of the original dataset `data` passed the
        constructor included in the `modeled` column.
    debug : pandas DataFrame
        A full version of the `data` DataFrame that includes preliminary
        quantities.
    Example
    -------
    >>> from lifelines.estimation import RobustROS
    >>> ros = RobustROSFitter(myDataFrame, result_col='conc',
                              censorship_col='censored')
    >>> ros.fit()
    >>> print(ros.data)
    Notes
    -----
    It is inappropriate to replace specific left-censored values with
    the estimated values from this method. The estimated values
    (self.data['modeled']) should instead be used to refine descriptive
    statistics of the dataset as a whole.
    '''
    def __init__(self, data, result_col='res', censorship_col='cen'):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input `data` must be a pandas DataFrame")

        if not data.index.is_unique:
            raise ValueError("Index of input DataFrame `data` must be unique")

        if data[result_col].min() <= 0:
            raise ValueError('All result values of `data` must be positive')

        # rename the dataframe columns to the standard names
        # these will be used throughout ros.py when convenient
        newdata = data.rename(columns={
            result_col: 'res', censorship_col: 'cen'
        })

        # confirm a datatype real quick
        try:
            newdata.res = np.float64(newdata.res)
        except ValueError:
            raise ValueError('Result data is not uniformly numeric')

        # and get the basic info
        self.N_obs = newdata.shape[0]
        self.N_cen = newdata[newdata.cen].shape[0]

        # sort the data
        self.data = _ros_sort(newdata, result_col='res', censorship_col='cen')

        # create a dataframe of detection limits and their parameters
        # used in the ROS estimation
        self.cohn = self._get_cohn_numbers()

    def _get_cohn_numbers(self):
        '''
        Computes the Cohn numbers for the delection limits in the dataset.
        '''

        def _A(row):
            '''
            Helper function to compute the `A` quantity.
            '''
            # index of results above the lower DL
            above = self.data.res >= row['lower']

            # index of results below the upper DL
            below = self.data.res < row['upper']

            # index of non-detect results
            detect = self.data.cen == False

            # return the number of results where all condictions are True
            return self.data[above & below & detect].shape[0]

        def _B(row):
            '''
            Helper function to compute the `B` quantity
            '''
            # index of data less than the lower DL
            less_than = self.data.res < row['lower']

            # index of data less than or equal to the lower DL
            less_thanequal = self.data.res <= row['lower']

            # index of detects, non-detects
            detect = self.data.cen == False
            nondet = self.data.cen == True

            # number results less than or equal to lower DL and non-detect
            LTE_nondets = self.data[less_thanequal & nondet].shape[0]

            # number of results less than lower DL and detected
            LT_detects = self.data[less_than & detect].shape[0]

            # return the sum
            return LTE_nondets + LT_detects

        def _C(row):
            '''
            Helper function to compute the `C` quantity
            '''
            censored_below = self.data.res[self.data.cen] == row['lower']
            return censored_below.sum()

        # unique values
        cohn = pd.unique(self.data.res[self.data.cen])

        # if there is a results smaller than the minimum detection limit,
        # add that value to the array
        if cohn.shape[0] > 0:
            if self.data.res.min() < cohn.min():
                cohn = np.hstack([self.data.res.min(), cohn])

            # create a dataframe
            cohn = pd.DataFrame(cohn, columns=['DL'])

            # copy the cohn in two columns. offset the 2nd (upper) column
            cohn['lower'] = cohn['DL']
            if cohn.shape[0] > 1:
                cohn['upper'] = cohn.DL.shift(-1).fillna(value=np.inf)
            else:
                cohn['upper'] = np.inf

            # compute A, B, and C
            cohn['A'] = cohn.apply(_A, axis=1)
            cohn['B'] = cohn.apply(_B, axis=1)
            cohn['C'] = cohn.apply(_C, axis=1)

            # add an extra row
            cohn = cohn.reindex(range(cohn.shape[0]+1))

            # add the 'PE' column, initialize with zeros
            cohn['PE'] = 0.0

        else:
            dl_cols = ['DL', 'lower', 'upper', 'A', 'B', 'C', 'PE']
            cohn = pd.DataFrame(np.empty((0,7)), columns=dl_cols)

        return cohn

    def fit(self, weighted=False):
        '''
        Estimates the values of the censored data
        '''
        def _ros_DL_index(row):
            '''
            Helper function to create an array of indices for the
            detection  limits (self.cohn) corresponding to each
            data point
            '''
            DLIndex = np.zeros(len(self.data.res))
            if self.cohn.shape[0] > 0:
                index, = np.where(self.cohn['DL'] <= row['res'])
                DLIndex = index[-1]
            else:
                DLIndex = 0

            return DLIndex

        def _ros_plotting_pos(row):
            '''
            Helper function to compute the ROS'd plotting position
            '''
            dl_1 = self.cohn.iloc[row['DLIndex']]
            dl_2 = self.cohn.iloc[row['DLIndex']+1]
            if row['cen']:
                return (1 - dl_1['PE']) * row['Rank']/(dl_1['C']+1)
            else:
                return (1 - dl_1['PE']) + (dl_1['PE'] - dl_2['PE']) * \
                        row['Rank'] / (dl_1['A']+1)

        def _select_modeled(row):
            '''
            Helper fucntion to select "final" data from original detects
            and estimated non-detects
            '''
            if row['cen']:
                return row['modeled_data']
            else:
                return row['res']

        def _select_half_DL(row):
            '''
            Helper function to select half cohn when there are
            too few detections
            '''
            if row['cen']:
                return 0.5 * row['res']
            else:
                return row['res']

        # create a DLIndex column that references self.cohn
        self.data['DLIndex'] = self.data.apply(_ros_DL_index, axis=1)

        # compute the ranks of the data
        self.data['Rank'] = 1
        groupcols = ['DLIndex', 'cen', 'Rank']
        rankgroups = self.data[groupcols].groupby(by=['DLIndex', 'cen'])
        self.data['Rank'] = rankgroups.transform(lambda x: x.cumsum())

        # detect/non-detect selectors
        detect_selector = self.data.cen == False
        nondet_selector = self.data.cen == True

        # if there are no non-detects, just spit everything back out
        if self.N_cen == 0:
            if not weighted:
                (self.data['Zprelim'],self.data['modeled']), (self.slope,self.intercept,self.r_squared) = stats.probplot(self.data['res'], fit=True)
            else:
                self.data['Zprelim'],self.data['modeled'] = stats.probplot(self.data['res'], fit=False)
                
                w = 1 - self.data['Zprelim']/(self.data['Zprelim'].max()+0.01)
                fit = linear_model.LinearRegression().fit(self.data['Zprelim'].to_frame(),np.log(self.data['res']).to_frame(),sample_weight=w)
                self.fit = fit
                self.slope = fit.coef_[0][0]
                self.intercept = fit.intercept_[0]
                self.r_squared = fit.score(self.data['Zprelim'].to_frame(),np.log(self.data['res']).to_frame())#,sample_weight=w)
                
            self.data['modeled_data'] = ""

        # if there are too few detects, use half DL
        #elif self.N_obs - self.N_cen < 2 or self.N_cen/self.N_obs > 0.8:
        #    self.data['modeled'] = self.data.apply(_select_half_DL, axis=1)

        # in most cases, actually use the MR method to estimate NDs
        else:
            # compute the PE values
            for j in self.cohn.index[:-1][::-1]:
                self.cohn.iloc[j]['PE'] = self.cohn.iloc[j+1]['PE'] + \
                   self.cohn.iloc[j]['A'] / \
                   (self.cohn.iloc[j]['A'] + self.cohn.iloc[j]['B']) * \
                   (1 - self.cohn.loc[j+1]['PE'])

            # compute the plotting position of the data (uses the PE stuff)
            self.data['plot_pos'] = self.data.apply(_ros_plotting_pos, axis=1)

            # correctly sort the plotting positions of the ND data:
            ND_plotpos = self.data['plot_pos'][self.data.cen]
            ND_plotpos.values.sort()
            self.data['plot_pos'][self.data.cen] = ND_plotpos.copy()

            # estimate a preliminary value of the Z-scores
            self.data['Zprelim'] = stats.norm.ppf(self.data['plot_pos'])

            # fit a line to the logs of the detected data
            if not weighted or len(self.data['Zprelim'][detect_selector]) < 3:
                fit = stats.linregress(self.data['Zprelim'][detect_selector],
                                   np.log(self.data['res'][detect_selector]))

                # save the fit params to an attribute
                self.fit = fit

                # pull out the slope and intercept for use later
                self.slope, self.intercept = fit[:2]
                self.r_squared = fit[2]**2
            else:
                w = stats.norm.pdf(self.data['Zprelim'][detect_selector])
                fit = linear_model.LinearRegression().fit(self.data['Zprelim'][detect_selector].to_frame(),np.log(self.data['res'][detect_selector]).to_frame(),sample_weight=w)
                self.fit = fit
                self.slope = fit.coef_[0][0]
                self.intercept = fit.intercept_[0]
                self.r_squared = fit.score(self.data['Zprelim'][detect_selector].to_frame(),np.log(self.data['res'][detect_selector]).to_frame(),sample_weight=w)
                
            #code.interact(local=dict(globals(), **locals()))
                       
            # model the data based on the best-fit curve
            self.data['modeled_data'] = np.exp(
                self.slope*self.data['Zprelim'][nondet_selector] + self.intercept
            )

            # select out the final data
            self.data['modeled'] = self.data.apply(_select_modeled, axis=1)

        # create the debug attribute as a copy of the self.data attribute
        self.debug = self.data.copy(deep=True)

        # select out only the necessary columns for data
        # Drop censored values that exceed max of uncensored values
        self.data = self.data[self.data.modeled <= self.data[self.data.cen == False]["modeled"].max()]#[['modeled', 'res', 'cen']]
        
        return self

    def plot(self, ax=None, show_raw=False, raw_kwds={}, model_kwds={},
             leg_kwds={}, xlog=True, show_cens=False, show_lof=True, marker = None, color =None):
        '''
        Generate a QQ plot of the raw (censored) and modeled data.
        Parameters
        ----------
        ax : optional matplotlib Axes
            The axis on which the figure will be drawn. If no specified
            a new one is created.
        show_raw : optional boolean (default = True)
            Toggles on (True) or off (False) the drawing of the censored
            quantiles.
        raw_kwds : optional dict
            Plotting parameters for the censored data. Passed directly
            to `ax.plot`.
        model_kwds : optional dict
            Plotting parameters for the modeled data. Passed directly to
            `ax.plot`.
        leg_kwds : optional dict
            Optional kwargs for the legend, which is only drawn if
            `show_raw` is True. Passed directly to `ax.legend`.
        ylog : optional boolean (default = True)
            Toggles the logarthmic scale of the y-axis.
        Returns
        -------
        ax : matplotlib Axes
        '''
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # legend options
        leg_params = {
            'loc': 'upper left',
            'fontsize': 8
        }
        leg_params.update(leg_kwds)

        # modeled data
        mod_symbols = {
            'marker': ('o' if marker is None else marker),
            'color': ('CornflowerBlue' if color is None else color),
            'label': 'Modeled data',
            'alpha': 0.87
        }
        mod_symbols.update(model_kwds)
        #osm_mod, osr_mod = stats.probplot(self.data['modeled'], fit=False)
        self.data.sort_values('modeled',inplace=True)
        
        if show_cens:
            ax.plot(self.data['modeled'], self.data['Zprelim'], linestyle ='none', **mod_symbols)
        else:
            ax.plot(self.data[self.data.cen == False]['modeled'], self.data[self.data.cen == False]['Zprelim'], linestyle= 'none', **mod_symbols)
            #code.interact(local=dict(globals(), **locals()))
        
        if show_lof:
            x = np.array([self.data['Zprelim'].min(),self.data['Zprelim'].max()])
            ax.plot(np.exp(self.slope*x+ self.intercept), x, linestyle='-', marker=None, color=color)

        # raw data
        if show_raw:
            raw_symbols = {
                'marker': 's',
                'markersize': 6,
                'markeredgewidth': 1.0,
                'markeredgecolor': '0.35',
                'markerfacecolor': 'none',
                'linestyle': 'none',
                'label': 'Censored data',
                'alpha': 0.70
            }
            raw_symbols.update(raw_kwds)
            osm_raw, osr_raw = stats.probplot(self.data['res'], fit=False)
            ax.plot(osr_raw, osm_raw, **raw_symbols)
            ax.legend(**leg_params)

        #ax.set_ylabel('Theoretical Quantiles')
        #ax.set_xlabel('Observations')
        if xlog:
            ax.set_xscale('log')

        return ax

def _ros_sort(dataframe, result_col='res', censorship_col='cen'):
    '''
    This function prepares a dataframe for ROS. It sorts ascending with
    left-censored observations on top.

    Parameters
    ----------
    dataframe : a pandas dataframe with results and qualifiers.
        The qualifiers of the dataframe must have two states:
        detect and non-detect.
    result_col (default = 'res') : name of the column in the dataframe
        that contains result values.
    censorship_col(default = 'cen') : name of the column in the dataframe
        that indicates that a result is left-censored.
        (i.e., True -> censored, False -> uncensored)

    Output
    ------
    Sorted pandas DataFrame.
    '''
    # separate detects from non-detects
    nondetects = dataframe[dataframe[censorship_col]].sort_values(result_col)
    detects = dataframe[~dataframe[censorship_col]].sort_values(result_col)

    return nondetects.append(detects)