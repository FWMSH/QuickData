import pandas as pd
import datetime
import numpy as np
import seaborn as sb
from statsmodels import robust
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import glob
#from numba import jit

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
    
    # Fix names
    data = data.tt.fix_names()

    # Add venue column
    data = data.tt.add_venue()

    # Add Audience column
    data = data.tt.add_audience()
    
    return(data)

def add_venue(df):
    # This function takes an input dateframe df and adds a
    # 'Venue' column.

    df = df.reset_index(drop=True)
    
    df['Venue'] = 'Other'
    
    venues = df['Venue'].values
    
    noble_list = ['Texas Sky Tonight', 'Our Solar System','Planetarium Experience', 'This Is Your Captain Speaking','Fragile Planet', 'Design A Mission', 'One World, One Sky','Take Me To The Stars', 'Noble Admission','Black Holes', 'Stars Of The Pharaohs', 'Sun, Earth, Moon: Beginnings','Sun, Earth, Moon: Explorations','Sun, Earth, Moon: Connections','Sun, Earth, Moon: Advanced', 'Imagine the Moon']
    noble = df.loc[df['Description'].isin(noble_list)]
    venues[noble.index] = 'Noble'
    
    omni_list = ['A Beautiful Planet', "America's Musical Journey",
                 'Tornado Alley', 'Coral Reef Adventure', 'Coco',
                 'Pandas', 'Star Wars VIII: The Last Jedi', 'Jaws',
                 'Flight of the Butterflies', 'Lewis & Clark: Great Journey',
                 'Jerusalem', 'Dream Big', 'Dolphins', 'Night at the Museum',
                 'Rolling Stones at the Max', 'Journey to the South Paci',
                 'Born to be Wild', 'Dinosaurs Alive', 'D-Day: Normandy 1944',
                 'Backyard Wilderness', 'National Parks Adventure', 'Frozen', 'The Incredibles',
                 'The Polar Express', 'Omni Admission', 'Moana',
                 'Superpower Dogs', 'Under the Sea', 'Apollo 11', 'Hidden Figures',
                 'Independence Day', 'Serenity','Fifth Element',
                 'How To Train Your Dragon', 'The Secret Life of Pets', 'Fantastic Beasts']
    omni = df.loc[df['Description'].isin(omni_list)]
    venues[omni.index] = 'Omni'
    studios_list = ['Science on Tap', "FAMapalooza", 'Birthday Parties',
                    'Reel Adventures','Polar Express PJ Party']
    studios = df.loc[df['Description'].isin(studios_list)]
    venues[studios.index] = 'Studios'
    
    df['Venue'] = venues
    
    return(df)

def _compute_audience(ser, all_price):
    
    # Helper function called by add_audience() to apply a School/Public label to a dataframe.
    # Assumes ttCode has already been added to ser
    # all_price is a pricetyple DataFrame covering ser and any other rows being computed.

    df = pd.DataFrame(ser).transpose()
    
    desc = df['Description'].values[0]
    code = df['ttCode'].values
    extracted = _slice_pricing_data(all_price, code)
    if  (desc == 'General Admission') or (desc == 'Parking'):
        return(['Public'])
    frac = get_school_data(extracted, silent=True)['School fraction']
    if np.isfinite(frac): # Matching pricetype data found
        if (frac > 0.1):
            return(['School'])
        else:
            return(['Public'])
    else: # No pricetype data match found.
        return(['Unknown'])
        
def add_audience(data):
    
    # Function to differentiate between a school and public
    # audience based on the pricetype data available.
    
    df = data.copy()
    
    df = df.reset_index(drop=True)
    
    if len(df) == 0:
        return(df)
            
    df = add_unique_code(df)
    pricing_data = get_pricing_data(df, silent=True, return_ttCode=True)
    df['Audience'] = df.apply(_compute_audience,axis=1, result_type='expand', args=[pricing_data])
    df = df.drop('ttCode', axis=1)
    
    # # Add Audience column
    # df['Audience'] = 'Public'
    # values = df['Audience'].values
    # # no_school days are weekdays when we're on a public schedule
    # no_school = pd.to_datetime(['2018-03-12', '2018-03-13', '2018-03-14', '2018-03-15', '2018-03-16',
                 # '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', '2018-01-15',
                 # '2018-05-28', '2018-09-03', '2018-09-10', '2018-01-01', '2018-10-08', '2018-11-23', '2018-11-19', '2018-11-20', '2018-11-21', '2018-11-23', '2019-03-11', '2019-03-12', '2019-03-13', '2019-03-14', '2019-03-15'])

    # # This is all the weekdays within the general range of the school year
    # school_dates = pd.bdate_range('2018-01-02', '2018-05-31').union(pd.bdate_range('2018-09-04', '2018-12-21')).union(pd.bdate_range('2019-02-12', '2019-05-31'))
    
    # # This is all those weekdays which also have shows before 1:30 PM
    # match = df['Perf date'].dt.date.isin(pd.Series(school_dates).dt.date) & (df['Perf date'].dt.time < pd.to_datetime('13:30:00').time())
    # values[match] = 'School'
    
    # # Now set the days in no_school back to Public
    # match2 = df['Perf date'].dt.date.isin(pd.Series(no_school).dt.date)
    # values[match2] = 'Public'
                 
    # df['Audience'] = values
    
    return(df)
    
def fix_names(df):

    # This function takes a dataframe df and fixes some of the naming
    # inconsistancies in the database.

    # Remove 'FYXX ' from names
    if 'Description' in df.keys():
        if (df.iloc[0])['Description'][0:2] == 'FY':
            df['Description'] = df['Description'].str.slice_replace(0,5,'')

    
    # Fix changed names
    latn_oss = ('LaN: Our Solar Sy', 'Our Solar System', 'LaN: Our Solar Syste',
                'LaN:Our Solar Syste', 'LaN: Our Solar Sys', 'LaN:Our Solar System',
                'LaN: Our Solar System', 'LaN: Solar System', 'LaN Solar System',
                'Live at the Noble Solar System', 'LaN: Our Solar System Grp',
                'LaN: Our Amazing Solar System', 'LAN: Our Solar System',
                'LATN: Our Solar System', 'Live at the Noble:Solar System')
    df = df.replace(to_replace=latn_oss, value="Our Solar System")
        
    latn_tst = ('LaN: Texas Sky Tonight', 'Texas Sky Tonight', 'LaN: Texas Sky',
                'TX Sky Tonight', 'LaN: Texas Sky Tonig', 'FY18 LaN: Texas Sky',
                'FY18 Texas Sky Tonig', 'LaN Texas Sky Tonight',
                'LaN Texas sky Tonight', 'Live at the Noble: Texas Sky',
                'LaN: Texas Sky Tonight Grp', 'LaN: Texas sky Tonight',
                'LAN: Texas Sky Tonight', 'LATN: Texas Sky Tonight')
    df = df.replace(to_replace=latn_tst, value="Texas Sky Tonight")
        
    omni_amj = ("America's Musical J", "America's Musical Jo", )
    df = df.replace(to_replace=omni_amj, value="America's Musical Journey")
        
    omni_superpower = ("Superpower Dogs Grp")
    df = df.replace(to_replace=omni_superpower, value="Superpower Dogs")
        
    omni_tornado = ("Tornado Alley Grp")
    df = df.replace(to_replace=omni_tornado, value="Tornado Alley")
        
    omni_cor = ('Coral Reef Adventure', 'Coral Reef', 'Coral Reef Adv', 'Coral Reef Adve', 'FY18 Coral Reef Adve')
    df = df.replace(to_replace=omni_cor, value="Coral Reef Adventure")
        
    omni_rs = ('FY18 Rolling Stones', 'Rolling Stones')
    df = df.replace(to_replace=omni_rs, value="Rolling Stones")
        
    nob_owos = ('Big Bird', 'One World, One Sky', 'One World, One Sky: Big B',
                'One World, One Sky: Big Birds', 'One World One Sky',
                'One World, One Sky Grp', 'One World, One Sky: Big Bird',
                "One World, One Sky: Big Bird's", "One World One Sky: Big Bird's")
    df = df.replace(to_replace=nob_owos, value="One World, One Sky")
        
    nob_SEM_adv = ('Sun, Earth, Moon: Advance', 'Sun, Earth, Moon: Adv', 'Sun, Earth, and Moon: Ad', 'Sun, Earth, Moon: Ad Grp')
    df = df.replace(to_replace=nob_SEM_adv, value="Sun, Earth, Moon: Advanced")
    
    nob_SEM_con = ('Earth, Sun, Moon: C', 'Earth, Sun, Moon: Connect')
    df = df.replace(to_replace=nob_SEM_con, value="Sun, Earth, Moon: Connections")
    
    nob_SEM_exp = ('Earth, Sun, Moon: E', 'Earth, Sun, Moon: Explora')
    df = df.replace(to_replace=nob_SEM_exp, value="Sun, Earth, Moon: Explorations")
    
    nob_SEM_beg = ('Earth, Sun, Moon: B', 'Sun, Earth, and Moon: Basic', 'Sun, Earth, and Moon: Bas', 'Earth, Sun, Moon: Beginni', 'Sun, Earth, Moon:B Grp')
    df = df.replace(to_replace=nob_SEM_beg, value="Sun, Earth, Moon: Beginnings")
        
    nob_TIYCS = ('This Is Your Captain')
    df = df.replace(to_replace=nob_TIYCS, value="This Is Your Captain Speaking")
        
    nob_fragile_planet = ('FY18 Fragile Planet')
    df = df.replace(to_replace=nob_fragile_planet, value="Fragile Planet")
        
    nob_black_holes = ('FY18 Black Holes')
    df = df.replace(to_replace=nob_black_holes, value="Black Holes")
        
    bday_extra = ('Birthday Party Extra', 'Birthday Party Ex', 'Birthday Party Extras')
    df = df.replace(to_replace=bday_extra, value="Birthday Party Extra")
        
    bday = ('FY18 Birthday Partie', 'Birthday Party','Birthday Parties')
    df = df.replace(to_replace=bday, value="Birthday Party")
        
    mus_park = ('Parking', 'Museum Parking')
    df = df.replace(to_replace=mus_park, value="Parking")
        
    omni_jaws = ('FY18 Jaws')
    df = df.replace(to_replace=omni_jaws, value="Jaws")
        
    omni_tlj = ('Star Wars VIII: The Last Jedi', 'Star Wars VIII: The Last')
    df = df.replace(to_replace=omni_tlj, value="Star Wars VIII: The Last Jedi")
        
    omni_born = ('Born to be Wild Grp', 'Born to Be Wild')
    df = df.replace(to_replace=omni_born, value="Born to be Wild")
        
    omni_dinos = ('Dinosaurs Alive', 'Dinos Alive', 'Dinosaurs Alive Grp')
    df = df.replace(to_replace=omni_dinos, value="Dinosaurs Alive")
        
    omni_coco = ('Coco (en Español)')
    df = df.replace(to_replace=omni_coco, value="Coco")
        
    omni_dolphins = ('Dolphins Grp')
    df = df.replace(to_replace=omni_dolphins, value="Dolphins")
        
    omni_dday = ('FY18 D-Day: Normandy', 'D-Day: Normandy Grp',
                 'D-Day: Normandy')
    df = df.replace(to_replace=omni_dday, value="D-Day: Normandy 1944")

    omni_butterfly = ('Flight of the Butter')
    df = df.replace(to_replace=omni_butterfly, value="Flight of the Butterflies")
    
    omni_backyard = ('Backyard Wilderness Grp')
    df = df.replace(to_replace=omni_backyard, value="Backyard Wilderness")
        
    omni_lewisclark = ('Lewis & Clark: Great Jour')
    df = df.replace(to_replace=omni_lewisclark, value="Lewis & Clark: Great Journey")
        
    omni_dream = ('Dream Big Grp', 'FY18 Dream Big')
    df = df.replace(to_replace=omni_dream, value="Dream Big")
        
    reel_adv = ('RA Harry Potter 4', 'Reel Adventures', 'RA Night at the Museum',
                "Reel Adventures: Harry Potter",
                'RA: Night at the Museum', 'Reel Adventures Harry Potter 4', 'RA Harry Potter 1')
    df = df.replace(to_replace=reel_adv, value="Reel Adventures")
    
    celeb_lecture = ('H. P. Newquist Lecture', 'David Zinn Celebrity Lecture', 'Adrian Garza',
                     'Matt Tedder',
                     'Brantley Hargrove Lecture', 'Paul Sutter Lecture', 'Michael Webber Lecture')
    df = df.replace(to_replace=celeb_lecture, value="Lecture Series")
    
    live_at_the_museum = ('The Vince Lujan Project','Buffalo Ruckus','Danni and Kris', 'Center 313','Light Crust Doughboys')
    df = df.replace(to_replace=live_at_the_museum, value="Live at the Museum")
    
    famipalooza = ('FAMapalooza: Bubble Festival', 'FAMapalooza: Beat the Heat',
                   'FAMapalooza: Pawesome Animals')
    df = df.replace(to_replace=famipalooza, value="FAMapalooza")
    
    polar_pj = ('Polar Express Pajama Party')
    df = df.replace(to_replace=polar_pj, value="Polar Express PJ Party")
    
    science_on_tap = ('Science on Tap: Sweet & Savory')
    df = df.replace(to_replace=science_on_tap, value="Science on Tap")
    
    return(df)
    
def add_attach(to_add, all_data):

    # This function adds the attach rate to the DataFrame to_add by
    # searching the DataFrame all_data for the correct value. It 
    # caches those values in admis_dict to improve performance.
    
    if len(to_add) > 0:
    
        admis_dict = {}
        
        str_dates = to_add['Perf date'].dt.strftime("%Y-%m-%d")

        # Pre-compute the attendance for each day in question
        #t1 = time.time()
        for str_date in str_dates.unique():
            admis = get_performance(all_data,'General Admission', str_date)['Tickets'].sum()
            admis_dict[str_date] = admis
        #print(time.time() - t1)
        
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

def get_yoy(data, **kwargs):

    # Function to compute the year over year change for a search. Accepts all 
    # arguments supported by search() and performs two searches, one for the
    # given year and one for the prior year. Then computes the YoY change and
    # returns it as a DataFrame
    
    if 'date' not in kwargs:
        print('get_yoy: error: must specifiy a date range to get a year-over-year result')
        return(pd.DataFrame())
    
    # These are the inputed arguments, assumed to the present reference frame
    cur_args = kwargs.copy()
    date = cur_args['date']
    if isinstance(date, str):
        date = resolve_string_date(date)
    elif isinstance(date, list):
        date = pd.to_DatetimeIndex(date)
    cur_args['date'] = date
    cur_result = search(data, **cur_args)
    
    # Copy the args to modify them for the past date one year ago
    past_args = cur_args.copy()
    date = past_args['date']
    shifted_date = date - pd.DateOffset(years=1)
    past_args['date'] = shifted_date
    past_result = search(data, **past_args)
    
    result = pd.DataFrame()
    #result['Period'] = 
    result['Period'] = [cur_args['date'][0].strftime('%Y-%m-%d')+' - '+cur_args['date'][1].strftime('%Y-%m-%d'), past_args['date'][0].strftime('%Y-%m-%d')+' - '+past_args['date'][1].strftime('%Y-%m-%d'), 'Year-over-year']
    
    cur_tick = cur_result['Tickets'].sum()
    past_tick = past_result['Tickets'].sum()
    result['Tickets'] = [cur_tick, past_tick, -1*(past_tick-cur_tick)/past_tick]
    
    cur_rev = cur_result['Revenue'].sum()
    past_rev = past_result['Revenue'].sum()
    result['Revenue'] = [cur_rev, past_rev, -1*(past_rev-cur_rev)/past_rev]
    
    return(result)
    

def get_age_data(data, get='', debug=False):
    
    # Function to return the number of tickets sold to adults and kids.
    # NOTE: some price types (e.g., comp)cannot be matched with adults
    # or children and are omitted. 
    # set get='' to get a summary DataFrame
    # set get='adult' to get a DataFrame with only adult tickets
    # set get='child' to get a DataFrame with only child tickets
    #'Adult fraction' is the fraction of *tickets* sold to adults
 
           
    df = data.copy()
           
    # Check if our dataframe includes price types
    if 'Price type' in df:
        
        adult_types = ['Adult', 'Combo 2 Adult', 'Member Adult','Wonder Free Adult','Combo 3 Adult','Service','ASTC Adult','Perot Adult','Staff Adult','Special Event Member Adult','School Combo Adult','School Combo Free Adult','School Free Adult','FWISD Free Adult','FWISD Adult','FMN Adults','FMN Chaperones','School Adult','Adult Upcharge','Group Adult','Wonder Discount Adult','Group C2 Adult','Stock Show Adult','Group C3 Adult', 'Stock Show Member Adult', 'Add-on Planetarium Adult','Group Free Adult','Group C2 Free Adult','Group C3 Free Adult']
        
        child_types = ['Combo 2 Junior', 'Junior','Member Junior', 'Wonder Free Junior',  'Combo 3 Junior','ASTC Junior',  'Perot Junior','Special Event Member Junior','School Combo Student',  'School Student','FWISD Student','FMN Students','School Free Student','Wonder Discount Junior','Stock Show Junior','Stock Show Member Junior', 'Stock Show Member Under 6','Stock Show Under 6','Add-on Planetarium Jr', 'Junior Upcharge','Group Junior','Staff Junior', 'Group C2 Junior','Group C3 Junior']
        
        unknown_types = ['$5 Add-on',  'Comp', 'Omni Staff Guest','Birthday Guest', 'Birthday Paid Guest', 'Member Parking', 'Parking', 'Museum Parking Comp', 'Planetarium Member','DMR Discover Member', 'DMR Wonder Member', 'Museum Parking - CR/NCM', 'Kitchen Chemistry','Scribblebots','Special Event','2 Day Pass', '3 Day Pass','Party Animals','Dino Discovery']
        
        # Make sure the categories above contain all the
        # price types. Over time, as new ones are added, the lists will need to be
        # updated.
        check_data = df[~df['Price type'].isin(adult_types+child_types+unknown_types)]
        if len(check_data) > 0:
            print('get_age_data: Warning: New price types detected. New types:' + str(check_data['Price type'].unique()))
            
        # Reduce to the data that we know as adult or child_types and
        # then split it.
        known_data = df[~df['Price type'].isin(unknown_types)]
        adult_data = known_data[known_data['Price type'].isin(adult_types)]
        child_data = known_data[known_data['Price type'].isin(child_types)]
        
        if get == 'adult':
            return(adult_data)
        elif get == 'child':
            return(child_data)
        else:    
            results = {}
            results['Adult tickets'] = adult_data.sum()['Tickets']
            results['Adult revenue'] = adult_data.sum()['Revenue']
            results['Child tickets'] = child_data.sum()['Tickets']
            results['Child revenue'] = child_data.sum()['Revenue']
            results['Adult fraction'] = np.round(results['Adult tickets']/(results['Adult tickets']+results['Child tickets']),3)
            return(results)
        
    else: # We need to match (description, perf date) tuples to the price type data
        df = get_pricing_data(df, debug=debug)
        return(get_age_data(df, get=get))
        
def get_group_data(data, get='', debug=False):

    # Function to return data related to groups.
    # Set get='group' to return a DataFrame containing all the data for groups
    # Set get='regular' to return a DataFrame containing all the non-group data
    # set get='' to return a summary DataFrame
    # 'Group fraction' is the fraction of *tickets* sold to  groups.
           
    df = data.copy()
           
    # Check if our dataframe includes price types
    if 'Price type' in df:
        
        group_types = ['School Combo Adult','School Combo Free Adult','School Free Adult','FWISD Free Adult','FWISD Adult','FMN Adults','FMN Chaperones','School Adult','Group Adult','Group C2 Adult','Group C3 Adult','Group Free Adult','Group C2 Free Adult','Group C3 Free Adult','School Combo Student',  'School Student','FWISD Student','FMN Students','School Free Student','Group Junior','Group C2 Junior','Group C3 Junior','Birthday Guest', 'Birthday Paid Guest']
        
        regular_types = ['Combo 2 Junior', 'Junior','Member Junior', 'Wonder Free Junior',  'Combo 3 Junior','ASTC Junior',  'Perot Junior','Special Event Member Junior','Wonder Discount Junior','Stock Show Junior','Stock Show Member Junior', 'Stock Show Member Under 6','Stock Show Under 6','Add-on Planetarium Jr', 'Junior Upcharge','Staff Junior','Adult', 'Combo 2 Adult', 'Member Adult','Wonder Free Adult','Combo 3 Adult','Service','ASTC Adult','Perot Adult','Staff Adult','Special Event Member Adult','Adult Upcharge','Wonder Discount Adult','Stock Show Adult', 'Stock Show Member Adult', 'Add-on Planetarium Adult','$5 Add-on',  'Comp', 'Omni Staff Guest', 'Member Parking', 'Parking', 'Museum Parking Comp', 'Planetarium Member','DMR Discover Member', 'DMR Wonder Member', 'Museum Parking - CR/NCM','Special Event','2 Day Pass', '3 Day Pass']
        
        unknown_types = ['Kitchen Chemistry','Scribblebots','Party Animals','Dino Discovery']
        
        # Make sure the categories above contain all the
        # price types. Over time, as new ones are added, the lists will need to be
        # updated.
        check_data = df[~df['Price type'].isin(group_types+regular_types+unknown_types)]
        if len(check_data) > 0:
            print('get_group_data: Warning: New price types detected. New types:' + str(check_data['Price type'].unique()))
            
        # Reduce to the data that we know as a type
        known_data = df[~df['Price type'].isin(unknown_types)]
        group_data = known_data[known_data['Price type'].isin(group_types)]
        regular_data = known_data[known_data['Price type'].isin(regular_types)]
        
        if get.lower() == 'group':
            return(group_data)
        elif get.lower() == 'regular':
            return(regular_data)
        else:       
            results = {}
            results['Group tickets'] = group_data.sum()['Tickets']
            results['Group revenue'] = group_data.sum()['Revenue']
            results['Regular tickets'] = regular_data.sum()['Tickets']
            results['Regular revenue'] = regular_data.sum()['Revenue']
            results['Group fraction'] = np.round(results['Group tickets']/(results['Group tickets']+results['Regular tickets']),3)
            return(results)
        
    else: # We need to match (description, perf date) tuples to the price type data
        df = get_pricing_data(df, debug=debug)
        return(get_group_data(df, get=get))
        
def get_school_data(data, get='', debug=False, silent=False):

    # Function to return data related to school groups.
    # Set get='school' to return a DataFrame containing all the data for school groups
    # Set get='regular' to return a DataFrame containing all the data that isn't schools
    # set get='' to return a summary DataFrame
    # 'School fraction' is the fraction of *tickets* sold to schools.
           
    df = data.copy()
           
    # Check if our dataframe includes price types
    if 'Price type' in df:
        
        school_types = ['School Combo Adult','School Combo Free Adult','School Free Adult','FWISD Free Adult','FWISD Adult','FMN Adults','FMN Chaperones','School Adult','School Combo Student',  'School Student','FWISD Student','FMN Students','School Free Student']
        
        regular_types = ['Group Junior','Group C2 Junior','Group C3 Junior','Birthday Guest', 'Birthday Paid Guest','Group C2 Adult','Group C3 Adult','Group Free Adult','Group C2 Free Adult','Group C3 Free Adult','Group Adult','Combo 2 Junior', 'Junior','Member Junior', 'Wonder Free Junior',  'Combo 3 Junior','ASTC Junior',  'Perot Junior','Special Event Member Junior','Wonder Discount Junior','Stock Show Junior','Stock Show Member Junior', 'Stock Show Member Under 6','Stock Show Under 6','Add-on Planetarium Jr', 'Junior Upcharge','Staff Junior','Adult', 'Combo 2 Adult', 'Member Adult','Wonder Free Adult','Combo 3 Adult','Service','ASTC Adult','Perot Adult','Staff Adult','Special Event Member Adult','Adult Upcharge','Wonder Discount Adult','Stock Show Adult', 'Stock Show Member Adult', 'Add-on Planetarium Adult','$5 Add-on',  'Comp', 'Omni Staff Guest', 'Member Parking', 'Parking', 'Museum Parking Comp', 'Planetarium Member','DMR Discover Member', 'DMR Wonder Member', 'Museum Parking - CR/NCM','Special Event','2 Day Pass', '3 Day Pass']
        
        unknown_types = ['Kitchen Chemistry','Scribblebots','Party Animals','Dino Discovery']
        
        # Make sure the categories above contain all the
        # price types. Over time, as new ones are added, the lists will need to be
        # updated.
        check_data = df[~df['Price type'].isin(school_types+regular_types+unknown_types)]
        if len(check_data) > 0:
            if not silent:
                print('get_school_data: Warning: New price types detected. New types:' + str(check_data['Price type'].unique()))
            
        # Reduce to the data that we know as a type
        known_data = df[~df['Price type'].isin(unknown_types)]
        school_data = known_data[known_data['Price type'].isin(school_types)]
        regular_data = known_data[known_data['Price type'].isin(regular_types)]
        
        if get.lower() == 'school':
            return(school_data)
        elif get.lower() == 'regular':
            return(regular_data)
        else:       
            results = {}
            results['School tickets'] = school_data.sum()['Tickets']
            results['School revenue'] = school_data.sum()['Revenue']
            results['Regular tickets'] = regular_data.sum()['Tickets']
            results['Regular revenue'] = regular_data.sum()['Revenue']
            results['School fraction'] = np.round(results['School tickets']/(results['School tickets']+results['Regular tickets']),3)
            return(results)
        
    else: # We need to match (description, perf date) tuples to the price type data
        df = get_pricing_data(df, debug=debug, silent=silent)
        return(get_school_data(df, get=get))
        
def get_member_data(data, get='', debug=False):
    
    # Function to return the number and fraction of tickets sold to members.
    # Set get='' to get a summary DataFrame
    # Set get='member' to get only tickets sold to members
    # Set get='guest' to get only tickets sold to members
    
    df = data.copy()
    if 'Price type' in data: # Need the DataFrame to include princing info
        member_types = ['Member Adult', 'Member Junior', 'Wonder Free Adult','Wonder Free Junior','Special Event Member Adult', 'Special Event Member Junior','Wonder Discount Adult', 'Wonder Discount Junior','Stock Show Member Adult',
       'Stock Show Member Junior', 'Stock Show Member Under 6','Member Parking','Planetarium Member', 'DMR Discover Member', 'DMR Wonder Member',]
       
        guest_types = ['Adult', 'Combo 2 Adult', 'Combo 2 Junior', 'Junior','Combo 3 Adult', 'Combo 3 Junior','$5 Add-on', 'Service', 'Comp', 'Omni Staff Guest', 'ASTC Adult','ASTC Junior', 'Perot Adult', 'Perot Junior', 'Staff Adult','Birthday Guest', 'Birthday Paid Guest','School Combo Adult', 'School Combo Free Adult','School Combo Student', 'School Free Adult', 'School Student','FWISD Free Adult', 'FWISD Student', 'FWISD Adult', 'FMN Adults','FMN Chaperones', 'FMN Students', 'School Adult','School Free Student', 'Adult Upcharge', 'Group Adult','Group C2 Adult', 'Group C3 Adult', 'Stock Show Adult','Stock Show Junior', 'Stock Show Under 6','Parking', 'Museum Parking Comp', 'Museum Parking - CR/NCM', 'Kitchen Chemistry', 'Birthday Gift', 'Birthday Shirt', 'Add-on Planetarium Adult', 'Add-on Planetarium Jr', 'Junior Upcharge', 'Scribblebots', 'Additional Demo P Animals', 'Group Junior', 'Liquid Nitrogen Ice Cream', 'Group Free Adult', 'Special Event', 'Group C2 Free Adult', 'Group C2 Junior', 'Group C3 Junior', 'Dino Discovery', 'Party Animals', 'Popcorn Pack', 'Punch', 'Water', 'Hot Dog Pack', 'Group C3 Free Adult', 'Staff Junior', '2 Day Pass', '3 Day Pass', 'Apple Juice', '24 Cookie Tray']
       
        # Make sure the categories above contain all the
        # price types. Over time, as new ones are added, the lists will need to be
        # updated.
        check_data = df[~df['Price type'].isin(member_types+guest_types)]
        if len(check_data) > 0:
            print('get_member_data: Warning: New price types detected. New types:' + str(check_data['Price type'].unique()))
        
        member_data = df[df['Price type'].isin(member_types)]
        guest_data = df[df['Price type'].isin(guest_types)]
        
        if get == 'member':
            return(member_data)
        elif get == 'guest':
            return(guest_data)
        else:
            result = {}
            result['Member tickets'] = member_data.sum()['Tickets']
            result['Member revenue'] = member_data.sum()['Revenue']
            result['Guest tickets'] = guest_data.sum()['Tickets']
            result['Guest revenue'] = guest_data.sum()['Revenue']
            result['Member fraction'] = np.round(result['Member tickets']/(result['Member tickets'] + result['Guest tickets']),3)
            return(result)      
    
    else: # Retreive the proper DataFrame and call this function recursively
        df = get_pricing_data(df, debug=debug)
        return(get_member_data(df, get=get))
    
def load_pricing_data():

    global tt_pricetype
    
    try: tt_pricetype
    except NameError: # Does not exist, load it.
        tt_pricetype = pd.read_pickle('tt_pricetype.pkl')
    else: # Does exist, data is already loaded
        if not 'ttCode' in tt_pricetype:
            tt_pricetype = add_unique_code(tt_pricetype)
    
def _slice_pricing_data(data, to_extract):

    # Function that takes a DataFrame of pricing data and returns values matching the ttCodes given by to_extract. to_extract must be a list of ttCodes.
    
    return(data[data['ttCode'].isin(to_extract)])
    
def get_pricing_data(data, debug=False, retain_zero=False, return_ttCode=True, silent=False):

    # Function to take a DataFrame without a Price type column and look
    # up the pricing data for it.
    # Set retain_zero = True to not drop shows with no ticket sales. These
    # shows basically always return no pricing data.
    
    global tt_pricetype
    load_pricing_data()
    df = data.copy()
    #df = fix_names(df)
    df = add_unique_code(df)
    
    if not retain_zero:
        df = df[df['Tickets'] > 0]
    
    to_match = df['ttCode'].unique()
    
    # Check whether all the data to match can be found in tt_pricetype
    if (silent is False) or (debug is True):
        overlap_check = pd.Series(to_match).isin(tt_pricetype['ttCode'])
        
        if (len(overlap_check) > 0) and (min(overlap_check) is False):
            problem = (df[df['ttCode'].isin(to_match[~overlap_check])])[['Description','Perf date', 'ttCode']]
            if not silent:
                print('Warning: Price type data does not cover every show. Reasons this may occur:\n- The show was prior to 2018\n- The show was canceled/changed with tickets sold\n- The show is in the current or future month. Pricetype data is updated monthly.\nRows missing: ' + str(len(problem)))
            if debug:
                print(problem)
    
    result = tt_pricetype[tt_pricetype['ttCode'].isin(to_match)]
    if not return_ttCode:
        result = result.drop(['ttCode'], axis=1)
    return(result)
    
def add_unique_code(data):

    # Function that adds a new column to the DataFrame that gives each
    # performance a unique code. Combines the Description and Perf date
    # fields.
    
    if len(data) > 0:
        df = data.copy()
        df['ttCode'] = df['Description'] + np.round((df['Perf date'] - pd.to_datetime('2017-01-01')).dt.total_seconds()).astype(str)
        
        return df
    else:
        return(pd.DataFrame())
    
def set_tt_pricetype(data):
    
    # This function sets the internal tt_pricetype global variable
    # to a DataFrame passed from outside the module. Used when 
    # decrypting the data for remote use.
    
    global tt_pricetype
    
    tt_pricetype = data
        
def resolve_string_date(date):

    # Function to take an inputted string date and return a
    # datetime array. Used to expand shortcuts.
    # Available shortcuts: today, yesterday, tomorrow, future,
    # CYXX (calendar year), FYXX (fiscal year)
    
    date = date.lower()
    
    split = date.split(':')
    dates = list()
    
    for date in split: 
        if 'today' in date:
            today = datetime.datetime.today().strftime('%Y-%m-%d')
            dates.append(pd.to_datetime(today))
        elif 'yesterday' in date:
            today = datetime.datetime.today()
            yesteday = (today - pd.Timedelta('1 day')).strftime('%Y-%m-%d')
            dates.append(pd.to_datetime(yesteday))
        elif 'tomorrow' in date:
            today = datetime.datetime.today()
            tomorrow = (today + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
            dates.append(pd.to_datetime(tomorrow))
        elif 'cy' in date: # Calendar year
            if len(date) == 4: # CYXX
                year = int(date[2:4])
            else: # CY20XX
                year = int(date[4:6])
            dates.append(pd.to_datetime('20'+str(year)+'-01-01'))
            dates.append(pd.to_datetime('20'+str(year)+'-12-31'))
        elif 'fy' in date: # Fiscal year
            if len(date) == 4: # FYXX
                year = int(date[2:4])
            else: # FY20XX
                year = int(date[4:6])
            dates.append(pd.to_datetime('20'+str(year-1)+'-10-01'))
            dates.append(pd.to_datetime('20'+str(year)+'-09-30'))
        elif date == 'future':
            today = datetime.datetime.today().strftime('%Y-%m-%d')
            dates.append(pd.to_datetime(today))
            future = (datetime.datetime.today() + pd.Timedelta('365 day')).strftime('%Y-%m-%d')
            dates.append(pd.to_datetime(future))
        else:
            dates.append(pd.to_datetime(date))
        
    if len(dates) < 3:
        return(pd.to_datetime(dates))
    else:
        print('resolve_string_date: Error: Too many dates!')
        return(pd.to_datetime([]))
        
        
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
   
    if len(date) > 0: # We were passed a date 
        if isinstance(date, str):
            date = resolve_string_date(date)
                
        if isinstance(date, list) or isinstance(date, pd.DatetimeIndex): # We were passed datetimes directly
            if date[0] is not None:
                if len(date) == 1: # Specific date
                    data = data[data['Perf date'].dt.date == date[0].date()]
                elif len(date) == 2: # [start_date, end_date]
                    if (date[0] == date[1]) or (date[1] == None):                            
                        data = data[data['Perf date'].dt.date == pd.to_datetime(date[0]).date()]
                    else:   
                        min_date = min(date)
                        max_date = max(date) + pd.to_timedelta('1 day') # Add day to that all times on the given day pass > or <
                        data = data[(data['Perf date'] >= min_date) & (data['Perf date'] <= max_date)]  
        else:
            print('search: Error: date format invalid')

    if weekday:
        data = data[data['Perf date'].dt.weekday < 5]
    elif weekend:
        data = data[data['Perf date'].dt.weekday > 4]

    # Find the unique columns so we can get lost columns back
    unique = data.drop_duplicates(subset=['Description', 'Perf date']).sort_values(['Description', 'Perf date'])
    # Need to group by name and date before checking against summed values
    data = data.groupby(['Description', 'Perf date']).sum().reset_index().sort_values(['Description', 'Perf date'])
    # String columns are lost, so let's re-add them.
    data['Venue'] = unique['Venue'].values
    data['Audience'] = unique['Audience'].values
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
        # Format is (min:max)
        split = tickets.split(':')
        split = [x.strip() for x in split]
        if split[0] != '':
            data = data[data['Tickets'] >= float(split[0])]
        if split[1] != '':
            data = data[data['Tickets'] <= int(split[1])]

    if len(revenue) > 0:   
        # Format is (min:max)    
        split = revenue.split(':')
        split = [x.strip() for x in split]
        if split[0] != '':
            data = data[data['Revenue'] >= float(split[0])]
        if split[1] != '':
            data = data[data['Revenue'] <= int(split[1])]

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
        if group.lower() in ['d', 'w', 'm', 'y', 'a-sep']:
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
       
        
def get_sales_curve(data, 
                    name, 
                    perf_date, 
                    max_tickets=0, 
                    end_on_event=False,
                    pad=False):
    
    # Function to return a cumulative distribution of sales for a show.
    # Set end_on_event=True to truncate the sales curve on the day of the 
    # event. Set pad=True to fill values up until the previous day (useful
    # if you're making a projection chart and there haven't been sales for 
    # several days)
    
    if isinstance(perf_date, str):
        perf_date = pd.to_datetime(perf_date)
    else:
        pass # Hopefully we were passed a Datetime-like object
    
    orders = get_performance(data, name, perf_date)
    if len(orders) == 0:
        # No data found for this performance
        print('Error: performance not found: ' + name + ' ' + str(perf_date))
        return(pd.DataFrame())
        
    # This useless-seeming line adds rows where we have a missing date
    orders = orders.set_index('Order date').groupby(pd.Grouper(freq='d')).sum().reset_index()
    
    # Optionally pad with zero sales for days between the last ticket sale and the current date.
    if pad:
        yesterday = datetime.date.today() - datetime.timedelta(1)
        new_index = pd.date_range(orders['Order date'].iloc[-1] + datetime.timedelta(1), yesterday)
        extension = pd.DataFrame()
        extension['Order date'] = new_index
        extension['Tickets'] = 0
        extension['Revenue'] = 0
        orders = orders.append(extension)
    
    cumsum = orders.Tickets.cumsum()
    if max_tickets == 0:
        frac_sum = cumsum/max(cumsum)
    else:
        frac_sum = cumsum/max_tickets

    diff = orders.reset_index()['Order date'].dt.date.sub(perf_date.date())/np.timedelta64(1, 'D')

    result = pd.DataFrame()
    result['Days before'] = diff.values
    result['Tickets'] = orders.Tickets.values
    result['Total tickets'] = cumsum.values
    result['Frac sold'] = frac_sum.values
    
    if end_on_event:
        result = result[result['Days before'] <= 0]
    
    return(result)

def create_presale_model(data, curve_list, new_err=False):
    
    # Function to create a model of how future presales might
    # look based on the sales curves of past events. curve_list 
    # is an list of tuples of the format [(name, date), (name2, date2), ...]
    # that specifiy which past events to base the model on.
    # Set new_err=True to compute a potentially-better error estimation, which
    # is much slower
    
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
    mean_by_day = np.zeros(len(collapsed))
    i = 0
    for row in collapsed.iterrows():
        day = (row[1])['Days before']
        dslice = combo[combo['Days before'] == day]
        mean_by_day[i] = dslice['Frac sold'].mean()
        mad_by_day[i] = robust.mad(dslice['Frac sold'].values)
        i += 1
        
    # Finalize columns
    collapsed['Frac sold'] = mean_by_day
    collapsed['Uncertainty'] = mad_by_day # median absolute deviation estimating the stdev
    
    # New error estimation    
    if new_err:
        err = np.zeros((len(fixed_curves),len(max_index)))
        err_low = np.zeros((len(fixed_curves),len(max_index))) # Holds errors where the prediction was too low
        err_high = np.zeros((len(fixed_curves),len(max_index))) # Holds errors where the prediction was too high
        for i in range(len(fixed_curves)):
            for j in np.arange(1,len(max_index)):
                err = (_do_fit(collapsed, (fixed_curves[i]).iloc[:-j], project=True) - (fixed_curves[i])['Total tickets']).values[-1]/(fixed_curves[i])['Total tickets'].values[-1]
                
                if err >=0:
                    err_high[i,j-1] = err
                    err_low[i,j-1] = np.nan
                else:
                    err_low[i,j-1] = abs(err)
                    err_high[i,j-1] = np.nan

        high_80 = np.nanpercentile(err_high, 80, axis = 0)
        low_80 = np.nanpercentile(err_low, 80, axis=0)
        
        collapsed['Error high'] = np.flip(high_80,axis=0)
        collapsed['Error low'] = np.flip(low_80,axis=0)
    collapsed = collapsed.drop(['Tickets', 'Total tickets'], axis=1)
    return(collapsed)

def project_sales_with_fit(data, model, name, date,
                           day=1000, max_tickets=0, verbose=True):

    # This function projects sales by fitting the model to the sales path
    # so far. It's a bit slower than project_sales, and *much* slower if
    # the model must also be computed (requires new_err).
    # If day != 1000, the projection will be based only on days up to that day (used when making the projection graph).
    
    if isinstance(model, str):
        model = data.tt.get_model(model)#data.tt._preset_models(model, new_err=True)
    else: 
        pass # Hopefully we have been passed a model DataFrame
    
    sales_path = data.tt.get_sales_curve(name, date, end_on_event=True, pad=True)
    if day != 1000:
        day = min(day, -1*day) # Index with negative numbers
        sales_path = sales_path[sales_path['Days before'] <= day]
        if len(sales_path) == 0:
            if verbose:
                print('project_sales_with_fit: error: no data prior to specified day')
            return(max_tickets/2, 0, max_tickets)
    
    df1 = sales_path.copy()
    df2 = model.copy()
    df1 = df1.set_index('Days before')
    df2 = df2.set_index('Days before')
    df2['Frac expected'] = df2['Frac sold']
    df2 = df2.drop(['Frac sold', 'Uncertainty'], axis=1)
    
    combo = pd.concat([df1, df2], axis=1).reset_index().dropna()
    
    # Log weights to add weight to recent observations
    weights = np.logspace(0,-5,len(combo))

    if len(combo) == 0:
        if verbose:
            print('project_sales_with_fit: error: not enough data!')
        proj = max_tickets/2
        low = 0
        high = max_tickets
    else:
        fit = curve_fit(_f, combo['Frac expected'].values, combo['Total tickets'].values, sigma=weights, absolute_sigma=True)
    
        proj = fit[0]*model['Frac sold'].iloc[-1]
        low = proj - proj*combo['Error low'].iloc[-1]
        high = proj + proj*combo['Error high'].iloc[-1]
    
    if (max_tickets > 0):
        if proj > max_tickets:
            proj = max_tickets
        if high > max_tickets:
            high = max_tickets
    
    if low < (sales_path.iloc[-1])['Total tickets']:
        low = (sales_path.iloc[-1])['Total tickets']
    if low > proj:
        low = proj
    
    return((np.round(np.squeeze(proj),0), np.round(np.squeeze(low),0), np.round(np.squeeze(high),0)))
    
def get_model(data, name):

    # Helper function that returns a model from the known global models.
    
    load_models(data)
    
    try:
        return(globals()['_model_'+name])
    except:
        print('get_model: error: model "'+name+'" does not exist')
        return(pd.DataFrame())
    
def load_models(data, *args, reload=False):

    # This function loads one or more models.
    # If no arguments are passed, all models are loaded.
    # Pass a list of models to load only those.
    # NOTE: if a model exists in memory, it will not be reloaded unless reload=True

    global _model_standard
    global _model_family
    global _model_adult
    global _model_lecture

    if len(args) == 1: # We passed a list to load
        if isinstance(args[0], list):
            models_to_load = args[0]
        else: # Assume we were passed a string
            models_to_load = [args[0]]
    else: # Load them all
        models_to_load = ['standard', 'family', 'adult', 'lecture']
    
    for model in models_to_load:
        if reload:
            load = True
        else:
            load = False
        try: # Check if the model already exists in memory
            eval('_model_'+model)
        except NameError:
            load = True
        if load:
            temp_model = _preset_models(data, model, new_err=True)
            globals()['_model_'+model] = temp_model
    
def _do_fit(model, sales_path, project=False, *kwargs):

    # This internal function fits the model to the data.
    # If project=False, returns the scalar s, where data = s*model
    # If project=True, returns a simple projection, s*model[day=0]
    
    df1 = sales_path.copy()
    df2 = model.copy()
    df1 = df1.set_index('Days before')
    df2 = df2.set_index('Days before')
    df2['Frac expected'] = df2['Frac sold']
    df2 = df2.drop(['Frac sold', 'Uncertainty', 'Total tickets'], axis=1)
    
    combo = pd.concat([df1, df2], axis=1).reset_index().dropna()
    
    if 'weights' not in kwargs:
        weights = np.logspace(0,-4,len(combo))   
    
    fit = curve_fit(_f, combo['Frac expected'].values, combo['Total tickets'].values, sigma=weights, absolute_sigma=True)  
    
    if not project:
        return(fit)
    else:
        return((fit[0]*model['Frac sold'].iloc[-1])[0])
    
def _f(model, scalar):
    
    # This stub function is used in curve fitting as part of project_sales_with_fit
    # model must be a numpy array, not a pandas series.
    
    return scalar*model
    
def project_sales(data, model, input1, input2, max_tickets=0, verbose=True, new_err=False):

    # Function to project how many tickets will ultimately be sold given
    # the number currently sold, the time to the event, and a presale
    # model.
    
    # Create the model, if it's not passed directly in
    if isinstance(model, str): # We were passed a string shorthand
        model = _preset_models(data, model, new_err=new_err)
    elif isinstance(model, list): # We were passed a list of peformances
        model = create_presale_model(data, model, new_err=new_err)
    else: # We were hopefully passed a DataFrame containing the model directly!
        pass
    
    # If we want new_err, make sure our model has new_err in it:
    if new_err:
        if not 'Error high' in model:
            print('Error: model must be computed with new_err=True')
            return((0,0,0))
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
            high = round(proj + proj*day['Error high'].values[0])
        else:
            high = round(sold/(day['Frac sold'].values[0] - day['Uncertainty'].values[0]))      
        if (high < 0) and (max_tickets == 0): 
            high = np.nan
        elif (high < 0) or (not np.isfinite(high)):
            high = max_tickets
        
        if new_err:
            low = round(proj - proj*day['Error low'].values[0])
        else:
            low = round(sold/(day['Frac sold'].values[0] + day['Uncertainty'].values[0]))
        if not np.isfinite(low):
            low = sold

        if max_tickets > 0:
            low = min(low, max_tickets)
            proj = min(proj, max_tickets)
            high = min(high, max_tickets)

        return(proj, low, high)
    
def project_sales_path(data, model, input1, input2, 
                       max_tickets=0, full=False):
    # Function to project what the sales path should look like
    # going forward. This takes into account the possibility
    # of a sellout. 
    # model is the sales model to use
    # max=0 means the number of sales is unbounded
    # full=True means the path will include every day in the model,
    # rather than just from "days_out"
    #
    # For the "dumb" projection, pass input1 as int(sales) and input2 as int(days_out)
    # For the "smart" projection, which uses a fit, pass input1 as str(name), input2 as str(date), and a model with new_err=True
    # The "dumb" approach is much faster
    
    if isinstance(input1, str) and isinstance(input2, str):
        fit = True
    elif isinstance(input1, int) and isinstance(input2, int):
        fit = False
    else:
        print('project_sales_path: error: must pass either name/date as string or sold/days_out as int')
        return()
    
    if fit:
        sales_curve = get_sales_curve(data, input1, input2, pad=True)
        days_out = (sales_curve.iloc[-1])['Days before']
        proj, low, high = project_sales_with_fit(data, model, input1, input2, max_tickets=0)
    else:
        days_out = min(input2, -1*input2) # Want the negative version
        proj, low, high = project_sales(data, model, input1, input2, max_tickets=0)
    
    path = proj*model['Frac sold']
    # correct for a sellout
    if max_tickets > 0:
        path = path.clip(upper=max_tickets)
        
    if full == True:
        return(model['Days before'].values, path.values)
    else:
        day_index = np.where(model['Days before'].values == days_out)[0]
        return(model['Days before'].values[day_index[0]:], path.values[day_index[0]:])

def _preset_models(data, name, list=False, new_err=False):

    # Function to return a pre-defined sales model by name. Setting
    # list=True returns the list of performances included in the 
    # model rather than the computed model itself.

    models = {'standard': [('Science on Tap', '2018-04-13'),
                           ('Science on Tap', '2018-07-20'),
                           ('Science on Tap', '2018-10-12'),
                           ('Science on Tap', '2019-03-30'),
                           ('Give Back Game Night', '2018-08-18'),
                           ('Reel Adventures', '2018-05-12'),
                           ('Reel Adventures', '2018-05-18'),
                           ('Reel Adventures', '2019-04-26'),
                           ('Reel Adventures', '2019-05-10'),
                           ('Reel Adventures', '2019-05-17'),
                           ('FAMapalooza', '2018-06-22'),
                           ('FAMapalooza', '2018-07-14'),
                           ('FAMapalooza', '2018-08-11')],
                           
              'family':   [('Reel Adventures', '2018-05-12'),
                           ('Reel Adventures', '2018-05-18'),
                           ('Reel Adventures', '2019-04-26'),
                           ('Reel Adventures', '2019-05-10'),
                           ('Reel Adventures', '2019-05-17'),
                           ('FAMapalooza', '2018-06-22'),
                           ('FAMapalooza', '2018-07-14'),
                           ('FAMapalooza', '2018-08-11')],
                           
              'adult':    [('Science on Tap', '2018-04-13'),
                           ('Science on Tap', '2018-07-20'),
                           ('Science on Tap', '2018-10-12'),
                           ('Science on Tap', '2019-03-30'),
                           ('Give Back Game Night', '2018-08-18')],
                           
              'lecture':  [('Lecture Series', '2019-04-15'),
                           ('Lecture Series', '2019-05-13'),
                           ('Lecture Series', '2018-05-10'),
                           ('Lecture Series', '2018-03-22'),
                           ('Lecture Series', '2018-02-08 19:00:00')]}
        
    if name.lower() in models:
        if list:
            return(models[name.lower()])
        else:
            return(create_presale_model(data, models[name.lower()], new_err=new_err))
    else:
        print('Model ' + name + ' not found.')
        if list:
            return([])
        else:
            return(pd.DataFrame())
                 
        
def create_model_chart(data, model_name, filename='', active=[], simple=False, title=''):
    
    # Function to plot the model for a given curve list and
    # all of the data that went into it. Curves passed to
    # active do not contribute to the model and are plotted
    # in color. The format is [(name, date, max_tickets)]
    
    # Clear the current figure
    plt.clf()
    
    model = get_model(data, model_name)
    curve_list = _preset_models(data, model_name, list=True)
    
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
        temp = get_sales_curve(data, curve[0], curve[1], max_tickets=int(curve[2]), pad=True)
        plt.plot(temp['Days before'], 100*temp['Frac sold'], label=curve[0] + ' ' + curve[1])

    if len(active) > 0:
        plt.legend()
    plt.ylim([0,100])
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
                       annotate=False,
                       title='',
                       frac=False,
                       end_on_event=True,
                       filename='',
                       max_tickets=0):
    
    # Function to plot the sales curves for given events/
    # curve list should be a list of tuples in the format
    # (perf_name, perf_date, max_tickets) with max_tickets
    # being an optional parameter. Set annotate=True to 
    # label each curve with the current number of sales 
    # (only if frac=False).
    
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
            temp = get_sales_curve(data, curve[0], curve[1], pad=True)

            plt.plot(temp['Days before'], temp['Total tickets'], label=curve[0] + ' ' + str(curve[1]))
            num = temp['Total tickets'].iloc[-1]
            if annotate:
                plt.annotate(str(int(num)),xy=(temp['Days before'].iloc[-1], num), xytext=(4,-8), textcoords='offset points')
        plt.ylabel('Tickets sold')
        
    else:
        for curve in curve_list:
            if len(curve) == 3:
                temp = get_sales_curve(data, curve[0], curve[1], max_tickets=int(curve[2]))
            else: # No max tickets supplied
                temp = get_sales_curve(data, curve[0], curve[1])            
            plt.plot(temp['Days before'], 100*temp['Frac sold'], label=curve[0] + ' ' + str(curve[1]))
        plt.ylabel('Percent of tickets sold')
    
    plt.xlabel('Days until event')
    plt.title(title)
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_facecolor('white')
    frame.set_edgecolor('white')
    
    if end_on_event:
        left, right = plt.xlim()
        plt.xlim(left,0)
        
    if max_tickets > 0:
        plt.ylim(0, 1.05*max_tickets)
    else:
        axes = plt.gca()
        autoy = axes.get_ylim()
        plt.ylim(0, autoy[1])
    
    if len(filename) > 0:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
     
    fig = plt.gcf()
    plt.close()
    return(fig)
    
def create_projection_chart(data, model, name, perf_date,
                            filename='',
                            fit=True,
                            max_tickets=0, 
                            pad=True,
                            #tight=False,
                            simple=False,
                            title=''):
    
    # Function to create a chart that plots the past and predicted sales
    # for a given event. Set simple=True to supress the display of daily
    # projections. Set pad=True to extend projected data through yesterday
    # (useful if there haven't been any ticket sales in several days). Set
    # tight=True to trim the x-axis to only those days where the model is 
    # non-zero (useful when tickets are being sold far before the model is 
    # valid).
    # If fit=True, the "smart" approach using fitting to determine sales
    # projections is used. This is much slower, but should be more stable.
    
    # Clear the current figure
    plt.clf()
    
    # Create the model, if it's not passed directly in
    if isinstance(model, str): # We were passed a string shorthand
        model = _preset_models(data, model, new_err=fit)
        if not 'Days before' in model: # Bad model 
            return
    elif isinstance(model, list): # We were passed a list of peformances
        model = create_presale_model(data, model, new_err=fit)
    else: # We were hopefully passed a DataFrame containing the model directly!
        if fit and ('Error high' not in model):
            print('create_projection_chart: error: if fit=True (default), model must be generated with new_err=True')

    # Set basic plot parameters
    sb.set_context('poster')
    pal = sb.color_palette('colorblind')
    
    perf_curve = get_sales_curve(data, name, perf_date, max_tickets=max_tickets, pad=pad, end_on_event=True)
    
    if len(perf_curve) == 0:
        # Performance not found
        return
    
    if simple:
        if fit:
            proj, low, high = project_sales_with_fit(data, model, 
                                                            name, 
                                                            perf_date, 
                                                            max_tickets=max_tickets,
                                                            verbose=False)
        else:
            proj, low, high = project_sales(data, model, 
                                                perf_curve['Total tickets'].iloc[-1], 
                                                perf_curve['Days before'].iloc[-1], 
                                                max_tickets=max_tickets, 
                                                verbose=False)
    else:
        for i in range(len(perf_curve)):
            if fit:
                proj, low, high = project_sales_with_fit(data, model, 
                                                            name, 
                                                            perf_date,
                                                            day=(perf_curve.iloc[i])['Days before'],
                                                            max_tickets=max_tickets,
                                                            verbose=False)
            else:
                proj, low, high = project_sales(data, model, 
                                                    perf_curve['Total tickets'].iloc[i], 
                                                    perf_curve['Days before'].iloc[i], 
                                                    max_tickets=max_tickets, 
                                                    verbose=False)
                                                                
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
    
    if fit:
        path_days, path_proj = project_sales_path(data, model, 
                                                  name, 
                                                  perf_date, 
                                                  max_tickets=max_tickets, 
                                                  full=full_bool)
    else:
        path_days, path_proj = project_sales_path(data, model, 
                                                  int(perf_curve['Total tickets'].iloc[-1]), 
                                                  int(perf_curve['Days before'].iloc[-1]), 
                                                  max_tickets=max_tickets, 
                                                  full=full_bool)
    # This forces the sales path to connect with the number of current sales. If fit=False,
    # this just sets the same value over again.
    if simple:
        path_proj[0] = (perf_curve.iloc[-1])['Total tickets']
    
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

    plt.ylim((0,1.05*max_tickets))
    plt.xlabel('Days until event')
    plt.ylabel('Projected attendence')
    if title == '':
        plt.title(str(name) + ' ' + str(perf_date))
    else:
        plt.title(title)
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_facecolor('white')
    frame.set_edgecolor('white')
    
    plt.xlim((min(perf_curve['Days before'])-1,1))
    # if tight: # Overrides xlim from above
        # # For events with very long sales histories; set plot lims to match model length.
        # useful = path_days[path_proj > 0]
        # if len(useful) > 0:
            # plt.xlim(min(useful)-1, 1)
        # else:
            # print('create_projection_chart: warning: cannot create tight plot because model is always zero. This will resolve itself closer to the event.')
    
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
        
    def get_member_data(self, **kwargs):
        return(get_member_data(self._obj, **kwargs))
    
    def get_age_data(self, **kwargs):
        return(get_age_data(self._obj, **kwargs))
    
    def get_group_data(self, **kwargs):
        return(get_group_data(self._obj, **kwargs))
    
    def get_school_data(self, **kwargs):
        return(get_school_data(self._obj, **kwargs))
    
    def get_pricing_data(self, **kwargs):
        return(get_pricing_data(self._obj, **kwargs))
    
    def get_yoy(self, **kwargs):
        return(get_yoy(self._obj, **kwargs))
    
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
    
    def get_model(self, *args, **kwargs):
        return(get_model(self._obj, *args, **kwargs))
    
    def load_models(self, *args, **kwargs):
        return(load_models(self._obj, *args, **kwargs))
    
    def project_sales(self, *args, **kwargs):
        return(project_sales(self._obj, *args, **kwargs))
    
    def project_sales_with_fit(self, *args, **kwargs):
        return(project_sales_with_fit(self._obj, *args, **kwargs))
    
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
    
    
    