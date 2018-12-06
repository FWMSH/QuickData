import pandas as pd
import datetime
import numpy as np
import seaborn as sb
from statsmodels import robust
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import glob

def set_plot_size(w,h, *args, **kwargs):
    
    # Function to set the size of a plot
    
    sb.set_context('poster', font_scale=0.5)
    sb.set_palette('colorblind')
    sb.set_style('darkgrid')
    plt.rcParams["figure.figsize"] = [w,h]


def load_data(path):
    # This function loads the Tessitura data from stored files. path specifies
    # the location of the csv files containing recent data, grouped_data.pkl,
    # which contains past Tessitura  data, and hist_combined.pkl, which contains
    # the pre-Tessitura data.
    
    # Read and process the data

    files = glob.glob(path + '/*.csv')
    raw_dfs = list()
    if len(files) > 0:
        for file in files:
            df = pd.read_csv(file,
                           encoding='latin1',
                           skiprows=1,
                           names=['Description', 'Code', 'Perf date', 'Email#', 'Email$', 'Person#', 'Person$', 'Mail#', 'Mail$', 'Phone#', 'Phone$', 'Web#', 'Web$'])

            df['Order date'] = pd.to_datetime(file[13:23])
            raw_dfs.append(df)
        full_data = pd.concat(raw_dfs)

        # Add new columns
        full_data['Perf date'] = pd.to_datetime(full_data['Perf date'])
        full_data = full_data.fillna(0)
        full_data['Tickets'] = full_data['Email#'] + full_data['Person#'] + full_data['Mail#'] + full_data['Phone#'] + full_data['Web#']
        full_data['Revenue'] = full_data['Email$'] + full_data['Person$'] + full_data['Mail$'] + full_data['Phone$'] + full_data['Web$']
        #full_data['Day of week'] = full_data['Perf date'].dt.weekday
        #full_data['Day name'] = full_data['Perf date'].dt.strftime('%A')
        #full_data['Shows'] = 1

        # Drop rows with no useful data before we do expensive operations
        full_data = full_data[full_data['Tickets'] != 0]

        # Fix names
        full_data = full_data.tt.fix_names()

        # Add venue column
        full_data = full_data.tt.add_venue()

        # Add Audience column
        full_data = full_data.tt.add_audience()
        
        new_data = full_data[['Description', 'Perf date', 'Tickets', 'Revenue', 'Venue', 'Audience', 'Order date']]

    # Unpack the stored data for past months (pickled for performance)
    old_data = pd.read_pickle(path + '/grouped_data.pkl')
    old_data = old_data[old_data['Perf date'] > '2018-01-02'] # To avoid overlaps with hist_data
    # Unpack the stored data from the historical file (Pre-2018)
    hist_data = pd.read_pickle(path + '/hist_combined.pkl')
    if len(files) > 0:
        data = pd.concat([new_data,old_data])
    else:
        data = old_data.copy()
    data = pd.concat([data, hist_data])
    
    return(data)

def add_venue(df):
    # This function takes an input dateframe df and adds a
    # 'Venue' column.

    df = df.reset_index(drop=True)
    
    df['Venue'] = 'Other'
    
    venues = df['Venue'].values
    
    noble_list = ['Texas Sky Tonight', 'Our Solar System',
                   'Sun, Earth, Moon: Advance', 'Sun, Earth, and Moon: Bas',
                   'One World, One Sky: Big B', 'Planetarium Experience',
                   'Earth, Sun, Moon: Beginni', 'Earth, Sun, Moon: Connect',
                   'Earth, Sun, Moon: Explora', 'This Is Your Captain',
                   'Fragile Planet', 'Design A Mission', 'One World, One Sky',
                   'Take Me To The Stars', 'Noble Admission',
                   'Black Holes', 'Stars Of The Pharaohs']
    noble = df.loc[df['Description'].isin(noble_list)]
    venues[noble.index] = 'Noble'
    
    omni_list = ['A Beautiful Planet', "America's Musical Journey",
                 'Tornado Alley', 'Coral Reef Adventure', 'Coco',
                 'Pandas', 'Star Wars VIII: The Last', 'Jaws',
                 'Flight of the Butterflies', 'Lewis & Clark: Great Jour',
                 'Jerusalem', 'Dream Big', 'Dolphins', 'Night at the Museum',
                 'Rolling Stones at the Max', 'Journey to the South Paci',
                 'Born to be Wild', 'Dinosaurs Alive', 'D-Day: Normandy 1944',
                 'Backyard Wilderness', 'National Parks Adventure',
                 'The Polar Express', 'Omni Admission']
    omni = df.loc[df['Description'].isin(omni_list)]
    venues[omni.index] = 'Omni'
    studios_list = ['Science on Tap', "FAMapalooza", 'Birthday Parties',
                                                  'Reel Adventures']
    studios = df.loc[df['Description'].isin(studios_list)]
    venues[studios.index] = 'Studios'
    
    df['Venue'] = venues
    
    return(df)

def add_audience(df):
    
    # Function to differentiate between a school and public
    # audience based on the day of the week and the time of day
    
    df = df.reset_index(drop=True)
    
    if len(df) == 0:
        return(df)
    
    # Add Audience column
    df['Audience'] = 'Public'
    values = df['Audience'].values
    # no_school days are weekdays when we're on a public schedule
    no_school = pd.to_datetime(['2018-03-12', '2018-03-13', '2018-03-14', '2018-03-15', '2018-03-16',
                 '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', '2018-01-15',
                 '2018-05-28', '2018-09-03', '2018-09-10', '2018-01-01', '2018-10-08', '2018-11-23',
                 '2018-11-19', '2018-11-20', '2018-11-21', '2018-11-23'])

    # This is all the weekdays within the general range of the school year
    school_dates = pd.bdate_range('2018-01-02', '2018-05-31').union(pd.bdate_range('2018-09-04', '2018-12-21'))
    
    # This is all those weekdays which also have shows before 1:30 PM
    match = df['Perf date'].dt.date.isin(pd.Series(school_dates).dt.date) & (df['Perf date'].dt.time < pd.to_datetime('13:30:00').time())
    values[match] = 'School'
    
    # Now set the days in no_school back to Public
    match2 = df['Perf date'].dt.date.isin(pd.Series(no_school).dt.date)
    values[match2] = 'Public'
                 
    df['Audience'] = values
    
    return(df)
    
def fix_names(df):

    # This function takes a dataframe df and fixes some of the naming
    # inconsistancies in the database.

    # Remove 'FYXX ' from names
    if 'Description' in df.keys():
        df['Description'] = df['Description'].str.slice_replace(0,5,'')

    
    # Fix changed names
    latn_oss = ('LaN: Our Solar Sy', 'Our Solar System', 'LaN: Our Solar Syste',
                'LaN:Our Solar Syste', 'LaN: Our Solar Sys', 'LaN:Our Solar System')
    df = df.replace(to_replace=latn_oss, value="Our Solar System")
        
    latn_tst = ('LaN: Texas Sky Tonight', 'Texas Sky Tonight', 'LaN: Texas Sky',
                'TX Sky Tonight', 'LaN: Texas Sky Tonig', 'FY18 LaN: Texas Sky',
                'FY18 Texas Sky Tonig')
    df = df.replace(to_replace=latn_tst, value="Texas Sky Tonight")
        
    omni_amj = ("America's Musical J", "America's Musical Jo", )
    df = df.replace(to_replace=omni_amj, value="America's Musical Journey")
        
    omni_cor = ('Coral Reef Adventure', 'Coral Reef', 'Coral Reef Adv', 'Coral Reef Adve', 'FY18 Coral Reef Adve')
    df = df.replace(to_replace=omni_cor, value="Coral Reef Adventure")
        
    omni_rs = ('FY18 Rolling Stones', 'Rolling Stones')
    df = df.replace(to_replace=omni_rs, value="Rolling Stones")
        
    nob_owos = ('Big Bird', 'One World, One Sky', 'One World, One Sky: Big B')
    df = df.replace(to_replace=nob_owos, value="One World, One Sky")
        
    bday_extra = ('Birthday Party Extra', 'Birthday Party Ex')
    df = df.replace(to_replace=bday_extra, value="Birthday Party Extra")
        
    bday = ('FY18 Birthday Partie', 'Birthday Party')
    df = df.replace(to_replace=bday, value="Birthday Party")
        
    mus_park = ('Parking', 'Museum Parking')
    df = df.replace(to_replace=mus_park, value="Parking")
        
    omni_jaws = ('FY18 Jaws')
    df = df.replace(to_replace=omni_jaws, value="Jaws")
        
    omni_dinos = ('Dinosaurs Alive', 'Dinos Alive')
    df = df.replace(to_replace=omni_dinos, value="Dinosaurs Alive")
        
    omni_dday = ('FY18 D-Day: Normandy')
    df = df.replace(to_replace=omni_dday, value="D-Day: Normandy 1944")

    omni_butterfly = ('Flight of the Butter')
    df = df.replace(to_replace=omni_butterfly, value="Flight of the Butterflies")
        
    reel_adv = ('RA Harry Potter 4', 'Reel Adventures', 'RA Night at the Museum')
    df = df.replace(to_replace=reel_adv, value="Reel Adventures")
    
    return(df)
    
def add_attach(to_add, all_data):

    # This function adds the attach rate to the DataFrame to_add by
    # searching the DataFrame all_data for the correct value. It 
    # caches those values in admis_dict to improve performance.
    
    if len(to_add) > 0:
    
        admis_dict = {}
        
        str_dates = to_add['Perf date'].dt.strftime("%Y-%m-%d")

        # Pre-compute the attendance for each day in question
        for str_date in str_dates.unique():
            admis = get_performance(all_data,'General Admission', str_date)['Tickets'].sum()
            admis_dict[str_date] = admis
        
        attach = list()
        for i in range(len(to_add)):    
            row = to_add.iloc[i]
            str_date = str_dates.values[i]
            admis = admis_dict[str_date]
            try:
                attach.append(np.round((row['Tickets'] / admis), 3))
            except:
                attach.append(np.nan)
                
        ret = to_add.copy()
        ret['Attach rate'] = attach
        
        return(ret)
        
    else:
        return(to_add)
    
def add_weekday(df):

    # Function that adds a column to df that gives the name of the day_index
    # of the week for that show
    if len(df) > 0:
        df['Day of week'] = df['Perf date'].dt.strftime('%A')
    
    return(df)

 
def search(data, name='', date='', time='', venue='', audience='',
            tickets='', revenue='', day_of_week=-1, weekday=False,
            weekend=False, group='', attach=False):

    # This function returns a dateframe that filters the overall
    # dataset based on the specified parameters. If you know the
    # show you're looking for, use get_show
    
    all_data = data.copy()
    
    if len(name) > 0:
        if isinstance(name, list):
            match_list = list()
            for item in name:
                match_list.append(data[data['Description'].str.lower().str.contains(item.lower())])
            data = pd.concat(match_list)
        else:
            data = data[data['Description'].str.lower().str.contains(name.lower())]
   
    if len(date) > 0: # We were passed a date string
        if isinstance(date, str):
            date = date.lower()
        
            # Shortcuts -- can onnly have one of these
            if 'today' in date:
                today = datetime.datetime.today().strftime('%Y-%m-%d')
                date.replace('today', today)
            elif 'cy' in date: # Calendar year
                if len(date) == 4: # CYXX
                    year = int(date[2:4])
                else: # CY20XX
                    year = int(date[4:6])
                date = '>20'+str(year-1)+'-12-31<20'+str(year+1)+'-01-01'
            elif 'fy' in date: # Fiscal year
                if len(date) == 4: # FYXX
                    year = int(date[2:4])
                else: # FY20XX
                    year = int(date[4:6])
                date = '>20'+str(year-1)+'-09-30<20'+str(year)+'-10-01'
                
            single = True # We're searching just one date
            if '>' in date:
                index = date.find('>')
                d1 = date[index+1:index+11]
                d1 = pd.to_datetime(d1)
                data = data[data['Perf date'].dt.date > d1.date()]     
                single = False
            if '<' in date:
                index = date.find('<')
                d2 = date[index+1:index+11]
                d2 = pd.to_datetime(d2)
                data = data[data['Perf date'].dt.date < d2.date()] 
                single = False
            if single:
                date = pd.to_datetime(date)
                data = data[data['Perf date'].dt.date == date.date()]    
                
        elif isinstance(date, list): # We were passed datetimes directly
            if date[0] is not None:
                if len(date) == 1: # Specific date
                    data = data[data['Perf date'].dt.date == date[0]]
                elif len(date) == 2: # [start_date, end_date]
                    if (date[0] == date[1]) or (date[1] == None):
                        data = data[data['Perf date'].dt.date == date[0]]
                    else:   
                        data = data[(data['Perf date'] >= min(date)) & (data['Perf date'] <= max(date))]          

    if weekday:
        data = data[data['Perf date'].dt.weekday < 5]
    elif weekend:
        data = data[data['Perf date'].dt.weekday > 4]

    # Need to group by name and date before checking against summed values
    data = data.groupby(['Description', 'Perf date']).sum().reset_index()
    # String columns are lost, so let's re-add them.
    data = add_venue(data)
    data = add_audience(data) 
    data = add_weekday(data)
            
    if len(time) > 0:
    
        if time[0] == '>':
            time_param = pd.to_datetime(time[1:])
            data = data[data['Perf date'].dt.time > time_param.time()]
        elif time[0] == '<':
            time_param = pd.to_datetime(time[1:])
            data = data[data['Perf date'].dt.time < time_param.time()]
        else:
            time_param = pd.to_datetime(time)
            data = data[data['Perf date'].dt.time == time_param.time()]    
    
    if len(tickets) > 0:    
        if tickets[0] == '>':
            data = data[data['Tickets'] > int(tickets[1:])]
        elif tickets[0] == '<':
            data = data[data['Tickets'] < int(tickets[1:])]

    if len(revenue) > 0:    
        if revenue[0] == '>':
            data = data[data['Revenue'] > float(revenue[1:])]
        elif revenue[0] == '<':
            data = data[data['Revenue'] < float(revenue[1:])]

    if day_of_week >= 0:
        data = data[data['Perf date'].dt.weekday == int(day_of_week)]
            
    if len(venue) > 0:
        data = data[data['Venue'] == venue]
        
    if len(audience) > 0:
        data = data[data['Audience'] == audience]
    
    #if len(data['Description'].unique()) > 1:
    #    print('Table contains:')
    #    print(data['Description'].unique())
        
    if group != '':
        if group.lower() in ['d', 'm', 'y']:
            data = data.set_index('Perf date').groupby(pd.Grouper(freq=group)).sum().reset_index()
        elif group.lower() == 'show':
            data = data.groupby(['Description']).sum().reset_index()
        if len(venue) > 0:
            data['Venue'] = venue
        if len(audience) > 0:
            data['Audience'] = audience
    
    # This adds a column that is Tickets/General Admission
    if ((attach == True) or (len(data) < 25)) and (group == ''):
        data = add_attach(data, all_data)
    
    if 'Perf date' in data:
        data = data.sort_values('Perf date', ascending=False)
    
    return(data)
    
def get_show(data, name, summarize=False, all=False):
    
    # Function to return all instances of a given show across  
    # order and perf dates. Set summarize=True to group by perf
    # date. By default, only events in the future are returned.
    # Set all=True to also retrieve past events.
    
    if all:
        result = data[data['Description'] == name]   
    else:
        result = data[(data['Description'] == name) & (data['Perf date'] > datetime.datetime.now())]      
    if summarize:
        temp = result.groupby('Perf date').sum().reset_index()
        temp['Description'] = name
        temp = add_venue(temp)
        temp = add_audience(temp)
        result = temp[['Description','Perf date', 'Tickets', 'Revenue', 'Venue',
                        'Audience']]
        
    result = result.sort_values('Perf date')
        
    return(result)

def get_performance(data, name, perf_date, fast=False, summarize=False):
    
    # Function to return all instances of a given performance
    # across all order dates. Summarize=True sums the results for 
    # each perf_date. Fast=True doesn't add the venue and audience
    # back in if summarize=True is set.
    
    if isinstance(perf_date, str):
        perf_date = pd.to_datetime(perf_date)
    
    if perf_date.time() == datetime.time(0,0):
        # We haven't supplied a show time, so it won't be an exact match
        result = data[(data['Description'] == name) & (data['Perf date'].dt.date == perf_date.date())]
    else:
        # Looking for an exact match
        result = data[(data['Description'] == name) & (data['Perf date'] == perf_date)]
        
    if len(result['Perf date'].unique()) > 1:
        print('Warning: Contains multiple performances: ' + name + ' ' + str(perf_date))
    
    if summarize:
        temp = result.groupby('Perf date').sum().reset_index()
        temp['Description'] = name
        if not fast:
            temp = add_venue(temp)
            temp = add_audience(temp)
            result = temp[['Description','Perf date', 'Tickets', 'Revenue', 'Venue','Audience']]
        else:
            result = temp[['Description','Perf date', 'Tickets', 'Revenue']]            
    
    return(result)
    
def summarize_day(data, date, verbose=True):
    # Function to list all performances on a given day
    
    if isinstance(date, str):
        date = pd.to_datetime(date)
        
    result = data[data['Perf date'].dt.date == date.date()]
    
    result = result.groupby(['Perf date', 'Description']).sum().reset_index()
    result = result.sort_values('Tickets', ascending=False)
    # String columns are lost, so let's re-add them.
    result = add_venue(result)
    result = add_audience(result)
    result = add_attach(result, data)
    
    # Print some useful info about this day
    if verbose:
        print('Day of week: ' + date.strftime('%A'))
    
    return(result.sort_values(['Venue', 'Perf date']))
       
        
def get_sales_curve(data, name, perf_date, max_tickets=0, end_on_event=False):
    
    # Function to return a cumulative distribution of sales for a show
    # set end_on_event=True to truncate the sales curve on the day of the 
    # event.
    
    if isinstance(perf_date, str):
        perf_date = pd.to_datetime(perf_date)
    
    orders = get_performance(data, name, perf_date)
    if len(orders) == 0:
        # No data found for this performance
        print('Error: performance not found: ' + name + ' ' + str(perf_date))
        return(pd.DataFrame())
        
    # This useless-seeming line adds rows where we have a missing date
    orders = orders.set_index('Order date').groupby(pd.Grouper(freq='d')).sum()
    
    cumsum = orders.Tickets.cumsum()
    if max_tickets == 0:
        frac_sum = cumsum/max(cumsum)
    else:
        frac_sum = cumsum/max_tickets

    #diff = orders.reset_index()
    diff = orders.reset_index()['Order date'].sub(pd.to_datetime(perf_date))/np.timedelta64(1, 'D')

    result = pd.DataFrame()
    result['Days before'] = diff.values
    result['Tickets'] = orders.Tickets.values
    result['Total tickets'] = cumsum.values
    result['Frac sold'] = frac_sum.values
    
    if end_on_event:
        result = result[result['Days before'] <= 0]
    
    return(result)

def create_presale_model(data, curve_list):
    
    # Function to create a model of how future presales might
    # look based on the sales curves of past events. curve_list 
    # is an array of lists of the format [(name, date), (name2, date2), ...]
    
    # Fetch the sales curve for each event
    curves = list()
    for curve in curve_list:
        temp = get_sales_curve(data, curve[0], curve[1], end_on_event=True)
        curves.append(temp)
        
    # Re-index all curves to have the length of the longest
    fixed_curves = list()
    max_len = max([len(x) for x in curves])
    for i in range(len(curves)):
        if len(curves[i]) == max_len:
            max_index = curves[i]['Days before']
    for i in range(len(curves)):
        fixed_curves.append(curves[i].set_index('Days before').reindex(max_index, fill_value=0).reset_index())
    
    # Combine the fixed curves together
    combo = pd.concat(fixed_curves)
    collapsed = combo.groupby('Days before').sum().reset_index()
    
    # Compute robust statistics 
    mad_by_day = np.zeros(len(collapsed))
    med_by_day = np.zeros(len(collapsed))
    for i in range(len(collapsed)):
        day = collapsed['Days before'].iloc[i]
        dslice = combo[combo['Days before'] == day]
        med_by_day[i] = dslice['Frac sold'].median()
        mad_by_day[i] = robust.mad(dslice['Frac sold'].values)
 
    # Finalize columns
    collapsed['Frac sold'] = med_by_day
    collapsed['Uncertainty'] = mad_by_day # median absolute deviation estimating the stdev
    
    # New error estimation
    
    err = np.zeros((len(fixed_curves),len(max_index)))
    for i in range(len(fixed_curves)):
        for j in np.arange(1,len(max_index)):
            err[i,j-1] = abs((project_sales(data, collapsed, (fixed_curves[i])['Total tickets'].values[-j], (fixed_curves[i])['Days before'].values[-j], verbose=False)[0] - (fixed_curves[i])['Total tickets']).values[-1])/(fixed_curves[i])['Total tickets'].values[-1]

    err_90 = np.percentile(err, 80, axis=0)
    collapsed['New error'] = np.flip(err_90,axis=0)
    collapsed = collapsed.drop(['Tickets', 'Total tickets'], axis=1)

    return(collapsed)

def project_sales(data, model, input1, input2, max_tickets=0, verbose=True, new_err=False):
    
    # Function to project how many tickets will ultimately be sold given
    # the number currently sold, the time to the event, and a presale
    # model.
    if isinstance(input1, str):
        # We were passed a ('name', 'date') combo
        curve = get_sales_curve(data, input1, input2)
        sold = curve['Total tickets'].values[-1]
        days_out = curve['Days before'].values[-1]
    else:
        # We were passed a (sold, days_out) combo
        sold = input1
        days_out = input2

    
    # We index with negative numbers
    if days_out > 0:
        days_out = -days_out
        
    if (min(model['Days before']) > days_out): # Model doesn't extend far enough back
        if verbose:
            print('Error: model does not extend to this date')
            
        if max_tickets > 0:
            return(max_tickets/2., sold, max_tickets)
        else:
            return(np.nan, sold, np.nan)
    else:
        day = model[model['Days before'] == days_out]
        if day['Frac sold'].values[0] == 0:
            if verbose:
                print('Error: model does not extend to this date')
            
            if max_tickets > 0:
                return(max_tickets/2., sold, max_tickets)
            else:
                return(np.nan, sold, np.nan)
        proj = round(sold/day['Frac sold'].values[0]) # The best-guess value
        
        if new_err:
            high = round(proj + proj*day['New error'].values[0])
        else:
            high = round(sold/(day['Frac sold'].values[0] - day['Uncertainty'].values[0]))      
        if (high < 0) and (max_tickets == 0): 
            high = np.nan
        elif (high < 0) or (not np.isfinite(high)):
            high = max_tickets
        
        if new_err:
            low = round(proj - proj*day['New error'].values[0])
        else:
            low = round(sold/(day['Frac sold'].values[0] + day['Uncertainty'].values[0]))
        if not np.isfinite(low):
            low = sold

        if max_tickets > 0:
            low = min(low, max_tickets)
            proj = min(proj, max_tickets)
            high = min(high, max_tickets)

        return(proj, low, high)
    
def project_sales_path(data, model, sold, days_out, max_tickets=0, full=False):
    # Function to project what the sales path should look like
    # going forward. This takes into account the possibility
    # of a sellout. 
    # model is the sales model to use
    # sold is the number of tickets sold on day "days_out"
    # max=0 means the number of sales is unbounded
    # full=True means the path will include every day in the model,
    # rather than just from "days_out"
    
    proj, low, high = project_sales(data, model, sold, days_out, max_tickets=0)
    
    path = proj*model['Frac sold']
    # correct for a sellout
    if max_tickets > 0:
        path = path.clip(upper=max_tickets)
        
    if full == True:
        return(model['Days before'].values, path.values)
    else:
        day_index = np.where(model['Days before'].values == days_out)[0]
        return(model['Days before'].values[day_index[0]:], path.values[day_index[0]:])

def create_model_chart(data, curve_list, filename='', active=[], simple=False, title=''):
    
    # Function to plot the model for a given curve list and
    # all of the data that went into it. Curves passed to
    # active do not contribute to the model and are plotted
    # in color. The format is [(name, date, max_tickets)]
    
    # Clear the current figure
    plt.clf()
    
    model = create_presale_model(data, curve_list)
    
    sb.set_context('poster')
    sb.set_palette('colorblind')
    
    # Recursively plot the curves from curve_list
    if not simple:
        for curve in curve_list:
            temp = get_sales_curve(data, curve[0], curve[1], end_on_event=True)
            plt.plot(temp['Days before'], 100*temp['Frac sold'], color='gray', label='_nolegend_')
        
    # Overplot the model
    plt.plot(model['Days before'], 100*model['Frac sold'], linewidth=7, label='Model')
    
    # Recursively plot the curves from active
    for curve in active:
        temp = get_sales_curve(data, curve[0], curve[1], max_tickets=int(curve[2]))
        plt.plot(temp['Days before'], 100*temp['Frac sold'], label=curve[0] + ' ' + curve[1])

    if len(active) > 0:
        plt.legend()
    plt.xlabel('Days until event')
    plt.ylabel('Percent of tickets sold')
    plt.title(title)
    
    if filename != '':
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    #plt.show()
    return(plt.gcf())
    
def create_revenue_chart(df, *args, 
                         filename='',
                         title='Revenue Over Time'):
    
    # Function to take a DataFrame and make a time vs revenue chart
    
    # Clear the current figure
    plt.clf()
    
    sb.set_context('poster')
    sb.set_palette('colorblind')
    
    if ('Perf date' in df) and ('Revenue' in df):
        plt.plot(df['Perf date'], df['Revenue'])
        plt.xlabel('Date')
        plt.ylabel('Revenue ($)')
        plt.title(title)
        
        if filename != '':
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        fig = plt.gcf()
        plt.close()
        
        return(fig)

def create_tickets_chart(df, *args, 
                         filename='',
                         title='Tickets Over Time'):
    
    # Function to take a DataFrame and make a time vs tickets chart
    
    # Clear the current figure
    plt.clf()
    
    sb.set_context('poster')
    sb.set_palette('colorblind')
    
    if ('Perf date' in df) and ('Tickets' in df):
        plt.plot(df['Perf date'], df['Tickets'])
        plt.xlabel('Date')
        plt.ylabel('Tickets sold')
        plt.title(title)    
        
        if filename != '':
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        fig = plt.gcf()
        plt.close()
        return(fig)

def create_sales_chart(*args,
                       title='',
                       frac=False,
                       end_on_event=False,
                       filename='',
                       max_tickets=0):
    
    # Function to plot the sales curves for given events/
    # curve list should be a list of tuples in the format
    # (perf_name, perf_date, max_tickets) with max_tickets
    # being an optional parameter
    
    # Clear the figure
    plt.clf()
    
    sb.set_context('poster')
    sb.set_palette('colorblind')
    
    data = args[0]
    if len(args) == 2:
        # We were passed the DataFrame and either a list of (name, date) tuples or 
        # another DataFrame which Description and Perf date fields specifying the
        # event. Useful for feeding in the results of tt.search()
        if isinstance(args[1], list):
            curve_list = args[1]
        elif isinstance(args[1], pd.DataFrame):
            curve_list = list()
            df = args[1].groupby(['Description', 'Perf date']).sum().reset_index()
            names = df['Description']
            dates = df['Perf date']
            for i in range(len(names)):
                curve_list.append((names[i], dates[i]))            
            
    elif len(args) == 3:
        # We were passed the DataFrame plus either a name and a date or a list of 
        # names and a list of dates or a single show name and a list of dates
        if isinstance(args[1], str):
            if isinstance(args[2], list):
                curve_list = list()
                for item in args[2]:
                    curve_list.append((args[1], item))
            else:                    
                curve_list = [(args[1], args[2])]
        else:
            curve_list = list()
            name = args[1]
            date = args[2]
            for i in range(len(args[1])):
                curve_list.append((name[i], date[i]))
                
    # Recursively plot the curves
    if not frac: # We're plotting the total number of tickets
        for curve in curve_list:
            temp = get_sales_curve(data, curve[0], curve[1])
            plt.plot(temp['Days before'], temp['Total tickets'], label=curve[0] + ' ' + str(curve[1]))
            plt.ylabel('Tickets sold')
        
    else:
        for curve in curve_list:
            if len(curve) == 3:
                temp = get_sales_curve(data, curve[0], curve[1], max_tickets=int(curve[2]))
            else: # No max tickets supplied
                temp = get_sales_curve(data, curve[0], curve[1])            
            plt.plot(temp['Days before'], 100*temp['Frac sold'], label=curve[0] + ' ' + str(curve[1]))
            plt.ylabel('Percent of tickets sold')
    
    plt.legend()
    plt.xlabel('Days until event')
    plt.title(title)
    
    if end_on_event:
        left, right = plt.xlim()
        plt.xlim(left,0)
        
    if max_tickets > 0:
        plt.ylim(0, 1.05*max_tickets)
    
    if len(filename) > 0:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
     
    fig = plt.gcf()
    plt.close()
    return(fig)
    
def create_projection_chart(data, model, name, perf_date,
                            filename='',
                            max_tickets=0, 
                            simple=False,
                            title=''):
    
    # Function to create a chart that plots the past and predicted sales
    # for a given event.
    
    # Clear the current figure
    plt.clf()
    
    if isinstance(model, list):
        # We were passed a list of curves to use as the model
        model = create_presale_model(data, model)

    # Set basic plot parameters
    sb.set_context('poster')
    pal = sb.color_palette('colorblind')
    
    perf_curve = get_sales_curve(data, name, perf_date, max_tickets=max_tickets)
    
    if len(perf_curve) == 0:
        # Performance not found
        return
    
    if simple:
        proj, low, high = project_sales(data, model, 
                                            perf_curve['Total tickets'].iloc[-1], 
                                            perf_curve['Days before'].iloc[-1], 
                                            max_tickets=max_tickets, 
                                            verbose=False)
    else:
        for i in range(len(perf_curve)):
            proj, low, high = project_sales(data, model, 
                                            perf_curve['Total tickets'].iloc[i], 
                                            perf_curve['Days before'].iloc[i], 
                                            max_tickets=max_tickets, 
                                            verbose=False)
            # proj2, low2, high2 = project_sales(data, model, 
                                            # perf_curve['Total tickets'].iloc[i], 
                                            # perf_curve['Days before'].iloc[i], 
                                            # max_tickets=max_tickets, 
                                            # new_err=True,
                                            # verbose=False)
                                            
            plt.errorbar(perf_curve['Days before'].iloc[i], 
                         proj, yerr=np.array([[proj-low],[high-proj]]), 
                         fmt='o', color=pal[0])            
            # plt.errorbar(perf_curve['Days before'].iloc[i]+0.25, 
                         # proj2, yerr=np.array([[proj2-low2],[high2-proj2]]), 
                         # fmt='o', color=pal[2])
        
    if simple:
        full_bool = False
    else:
        full_bool = True
    path_days, path_proj = project_sales_path(data, model, 
                                              perf_curve['Total tickets'].iloc[-1], 
                                              perf_curve['Days before'].iloc[-1], 
                                              max_tickets=max_tickets, 
                                             full=full_bool)
    plt.plot(path_days, path_proj, 
        color=pal[2], label='Model', linewidth=6)
    plt.errorbar(0, proj, yerr=np.array([[proj-low],[high-proj]]), fmt='o', color=pal[0],
        label='Projected attendence')

    plt.scatter(np.array(perf_curve['Days before']),
                perf_curve['Total tickets'],
                color=pal[1], label='Sold to date')
    
    plt.annotate(str(int(perf_curve['Total tickets'].iloc[-1])),
                 xy=(perf_curve['Days before'].iloc[-1], perf_curve['Total tickets'].iloc[-1]),
                 xytext=(0,-20), textcoords='offset points')
    plt.annotate(str(int(proj)),xy=(0, proj), xytext=(8,-8), textcoords='offset points')
    plt.annotate(str(int(low)),xy=(0, low), xytext=(8,-8), textcoords='offset points')
    if np.isfinite(high):
        plt.annotate(str(int(high)),xy=(0, high), xytext=(8,-8), textcoords='offset points')

    plt.xlim((min(perf_curve['Days before'])-2,1))
    plt.ylim((0,1.05*max_tickets))
    plt.xlabel('Days until event')
    plt.ylabel('Projected attendence')
    if title == '':
        plt.title(str(name) + ' ' + str(perf_date))
    else:
        plt.title(title)
    plt.legend()    
    
    if simple:
        # Add an explainer to the bottom of the chart
        plt.figtext(0.1, -0.0375,
                    'The green dots mark the number of actual tickets sold so far.\nThe blue dot is the predicted final sales and the blue line is the range of reasonable possibilities.\nThe orange line shows the expected path of ticket sales from today to the final projection.',
                    fontsize='small')
                    
    if len(filename) > 0:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    #plt.show()
    return(plt.gcf())
    
# This binds the methods to the DataFrame objects
@pd.api.extensions.register_dataframe_accessor("tt")
class ttAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
    def add_venue(self):
        self._obj = add_venue(self._obj)
        return(self._obj)
    
    def add_audience(self):
        self._obj = add_audience(self._obj)
        return(self._obj)
        
    def add_weekday(self):
        self._obj = add_weekday(self._obj)
        return(self._obj)
        
    def fix_names(self):
        self._obj = fix_names(self._obj)
        return(self._obj)
        
    def search(self, **kwargs):
        return(search(self._obj, **kwargs))
        
    def get_show(self, *args, **kwargs):
        return(get_show(self._obj, *args, **kwargs))
    
    def get_performance(self, *args, **kwargs):
        return(get_performance(self._obj, *args, **kwargs))
    
    def summarize_day(self, *args, **kwargs):
        return(summarize_day(self._obj, *args, **kwargs))
    
    def get_sales_curve(self, *args, **kwargs):
        return(get_sales_curve(self._obj, *args, **kwargs))
    
    def create_presale_model(self, *args, **kwargs):
        return(create_presale_model(self._obj, *args, **kwargs))
    
    def project_sales(self, *args, **kwargs):
        return(project_sales(self._obj, *args, **kwargs))
    
    def project_sales_path(self, *args, **kwargs):
        return(project_sales_path(self._obj, *args, **kwargs))    
        
    def create_model_chart(self, *args, **kwargs):
        return(create_model_chart(self._obj, *args, **kwargs))
        
    def create_sales_chart(self, *args, **kwargs):
        return(create_sales_chart(self._obj, *args, **kwargs))
        
    def create_projection_chart(self, *args, **kwargs):
        return(create_projection_chart(self._obj, *args, **kwargs))
    
    def create_revenue_chart(self, *args, **kwargs):
        return(create_revenue_chart(self._obj, *args, **kwargs))
    
    def create_tickets_chart(self, *args, **kwargs):
        return(create_tickets_chart(self._obj, *args, **kwargs))
    
    
    