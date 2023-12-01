
#----------------------------------------------------------------------------------------------------------
# Import Libraries
#----------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import MultipleLocator

#----------------------------------------------------------------------------------------------------------
# DP: Loading & Understanding Dataset
#----------------------------------------------------------------------------------------------------------

pd.set_option('display.float_format',lambda x:'%.2f'%x)
pd.set_option('display.max_columns',None) #Display All Columns
df = pd.read_csv(r'C:\Users\User\Desktop\Projects\Data Science\project_startup_success\data\raw\investments_VC.csv', encoding = "ISO-8859-1")
df.head()
df.tail()

# Understanding Dataset & QC
pd.set_option('display.max_rows',None)
df.shape 
df.nunique() #1
df.info() #2
df.describe() 
df.isnull().sum() #3

pd.reset_option('display.max_rows', silent=True)
# pd.reset_option('display.', silent=True) #Default setting

# Eror Found
# DP1.s column name error, unecesary spacing: market, funding_total_usd
# DP2.s formating: index 5,18-38:float | index 11:int | index 12-17: datetime
# DP3.s null values 

#----------------------------------------------------------------------------------------------------------
# DP: Cleaning Dataset
#----------------------------------------------------------------------------------------------------------

df1 = df.copy()

# Column_name
df1.columns = [col.strip() for col in df1.columns]
df1.head()

# Delete Duplicate
df1 = df1.drop_duplicates()
df1.shape #Compare with df.shape
# There is alot of duplicate data here. (54294-49439)/54294 | 8% deleted 
df[df.duplicated()] #show all duplicate rows
df1.tail(100) 
df1 = df1.drop(index=49438) #All NaN rows are cleared. Error by csv_import

# Null Handling
# Locating 'name' with null value. Key ID should not be null. 
df1[df1['name'].isnull()] #name should be tellitin
df1.loc[df1['name'].isnull(), 'name'] ='tellitin'
df1[df1['name']=='tellitin']

# Delete unwanted column : state_code
df1.drop(columns=['state_code'], inplace=True)
df1.drop(columns=['founded_quarter'], inplace=True)
df1.drop(columns=['founded_month'], inplace=True)
df1.drop(columns=['founded_year'], inplace=True)
#df1 = df1.drop(columns=['state_code']) Optional

# Formating
df1.info()
df1 = df1.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# 'funding_round' to int
columns_to_convert_to_int = ['funding_rounds']
df1[columns_to_convert_to_int] = df1[columns_to_convert_to_int].astype(int)
# 'funding_total_usd' to float
unique_values = df1['funding_total_usd'].unique() #commas from 
unique_values_sorted = pd.Series(unique_values).sort_values(ascending=True)
unique_values_sorted
df1['funding_total_usd'] = df1['funding_total_usd'].str.replace(',','')
df1['funding_total_usd'] = pd.to_numeric(df1['funding_total_usd'],errors='coerce')
# 'founded_at','first_funding_at','last_funding_at'  to date-time
# 'founded_year' to yyyy
# 'founded_month' to yyyy-mm
unique_values = df1['founded_at'].unique() #commas from 
unique_values_sorted = pd.Series(unique_values).sort_values(ascending=True)
unique_values_sorted
# Errors in first_funding_at and last_funding_at
error_date = ['0001-05-14','0001-07-14','0001-11-14','0007-05-13','0011-11-14','0019-11-20','0020-06-14','0026-11-14','0029-09-14','0201-01-01'] 
df1[df1['first_funding_at'].isin(error_date)]
# Assign best possible date to the error, logic: first_funding should be near to founded year
date_mapping = {
    '0001-05-14': '2014-05-10',
    '0001-07-14': '',
    '0001-11-14': '',
    '0007-05-13': '2013-05-07',
    '0011-11-14': '2004-11-11',
    '0019-11-20': '2012-11-20',
    '0020-06-14': '2012-06-14',
    '0026-11-14': '',
    '0029-09-14': '',
    '0201-01-01': '2015-01-01'
}
df1['first_funding_at'] = df1['first_funding_at'].replace(date_mapping)
date_mapping_2 = {
    '0001-05-14': '2014-05-10',
    '0001-07-14': '',
    '0001-11-14': '',
    '0026-11-14': '',
    '0029-09-14': '',
    '0201-01-01': '2015-01-01'
}
df1['last_funding_at'] = df1['last_funding_at'].replace(date_mapping_2)
# convert datetime
df1['founded_at'] =  pd.to_datetime(df1['founded_at'], format='%Y-%m-%d', errors = 'coerce')
df1['first_funding_at'] =  pd.to_datetime(df1['first_funding_at'], format='%Y-%m-%d', errors = 'coerce')
df1['last_funding_at'] =  pd.to_datetime(df1['last_funding_at'], format='%Y-%m-%d', errors = 'coerce')
# Founded_month will be biased to january as converting year to datetime resluting in january as converted month.

df1.head(100)
df1.isnull().sum()
df1.shape 

#----------------------------------------------------------------------------------------------------------
# DP: Preparing Dataset
#---------------------------------------------------------------------------------------------------------- 
# DP4. df1. Null Values is high. "founded_at"
# DP5. df2. For Null in 'founded_at', parsed values from 'first_funding_at'.Clear Null. Delete column: founded_month,founded_year

df2 = df1.copy()
df2['founded_at'].fillna(df2['first_funding_at'], inplace=True)
df2.dropna(subset=['founded_at'], inplace=True)
df2['funding_total_usd'].sort_values(ascending=True)
df2.head()
df2.isnull().sum()
df2.shape
df2.info()

#----------------------------------------------------------------------------------------------------------
# EDA: DA
#----------------------------------------------------------------------------------------------------------
# EDA1. How many startups in the dataset? 49347
df2['name'].nunique() 

# EDA2. Which location has the most startups?
startup_counts_by_country = df2['name'].groupby(df2['country_code']).count()
startup_counts_by_country = startup_counts_by_country.sort_values(ascending=False)
top_50_locations = startup_counts_by_country.head(50)
# EDA2. Visualization
plt.figure(figsize=(10, 6)) 
plt.bar(top_50_locations.index, top_50_locations.values, color='skyblue')
plt.xlabel('Country Code')  # X-axis label
plt.ylabel('Number of Startups')  # Y-axis label
plt.title('Number of Startups by Location') 
plt.xticks(rotation=45)
plt.tight_layout()  # Ensures labels are fully visible
plt.show()

# EDA2.1. Startup hotspot in each continent
# Merge Dataset> continent
country_continent_mapping = pd.read_csv(r'C:\Users\User\Desktop\Projects\Data Science\project_startup_success\data\raw\continents-according-to-our-world-in-data.csv')
country_continent_mapping.info()
country_continent_mapping.rename(columns={'Code':'country_code'},inplace=True)
country_continent_mapping.columns = country_continent_mapping.columns.str.lower()
df3 = df2.merge(country_continent_mapping, on='country_code')

def visualize_top_countries_by_continent(df, title='Number of Startups per Country'):
    # Get the unique continents from the 'continent' column
    continents_to_select = df['continent'].unique()
    # Initialize an empty DataFrame to store the top-performing countries
    top_countries_by_continent = pd.DataFrame()
    # Define the number of top countries to display for each continent
    top_n = 10
    # Iterate over the selected continents
    for continent in continents_to_select:
        # Filter the data for the current continent
        continent_data = df[df['continent'] == continent]
        # Group the data by country and count the startups
        country_counts = continent_data.groupby('country_code')['name'].count().reset_index()
        # Sort the countries by count in descending order and select the top country
        top_countries = country_counts.nlargest(top_n, 'name')
        # Add a column for the continent
        top_countries['Continent'] = continent
        # Concatenate the top country with the result DataFrame
        top_countries_by_continent = pd.concat([top_countries_by_continent, top_countries])
    # Rename the 'name' column to 'number of startups'
    top_countries_by_continent.rename(columns={'name': title}, inplace=True)

    # Create the tree map visualization
    fig = px.treemap(
        top_countries_by_continent,
        path=['Continent', 'country_code'],
        values=title,
        title=title,
        color_continuous_scale='Viridis',
    )
    
    # Customize the layout
    fig.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),  # Adjust margins
        paper_bgcolor='rgba(0,0,0,0)',  # Set background color
        font=dict(family='Droid Serif, monospace', size=14),  # Customize font
        # Customize the title font
        title={
            'text': f'<b>{title}<b>',
            'x': 0.5,  # Centered title
            'font': {'size': 18, 'color': 'white', 'family': 'Droid Serif, monospace'}
        }
    )
    # Customize the font inside the tree map
    fig.update_traces(
        textfont={'family': 'Droid Serif, monospace', 'size': 14}
    )

    # Show the tree map
    fig.show()

visualize_top_countries_by_continent(df3, title='Number of Startups per Country')

#Let's look at the status of big market
def status_overview(df):
    df['status'].fillna('unknown', inplace=True)
    total_count = df['name'].nunique()
    percentage_by_status = (df.groupby('status')['name'].nunique() / total_count) * 100
    percentage_by_status = percentage_by_status.sort_values(ascending=False)
    print(percentage_by_status)

df_USA = df3[df3['country_code']=='USA']
df_CHN = df3[df3['country_code']=='CHN']
df_SGP = df3[df3['country_code']=='SGP']
df_IND = df3[df3['country_code']=='IND']
df_ISR = df3[df3['country_code']=='ISR']
df_GBR = df3[df3['country_code']=='GBR']
df_DEU = df3[df3['country_code']=='DEU']
df_FRA = df3[df3['country_code']=='FRA']
status_overview(df_USA) #highest acquired rate 9.68
status_overview(df_CHN) #lowest closed rate, lowest acquired rate 
status_overview(df_SGP) #Almost same closed rate as US, but low acquired rate
status_overview(df_IND) #second lowest closed rate, lowest acquired rate 
status_overview(df_ISR) #highest closed rate: 6.9, but second highest acquired rate
status_overview(df_GBR) #moderate
status_overview(df_DEU) #highest acquired rate among european peers
status_overview(df_FRA) #moderate


# EDA3.1. Hot Market
df3['market'].nunique() #There are 738 different markets
df3.groupby('market')['name'].count().sort_values(ascending = False)

def scree_plot_market(df,i='market'): 
    n=25
    data = df.groupby(i)['name'].count().sort_values(ascending = False)
    

    top_n_markets = data.head(n)

    # Calculate the cumulative percentage of startups represented by the top n markets
    total_startups = data.sum()
    top_n_percentage = (top_n_markets / total_startups) * 100
    cumulative_percentage = top_n_percentage.cumsum()

    # Create a scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n + 1), cumulative_percentage, marker='o', linestyle='-')
    plt.xlabel('Number of Top Markets')
    plt.ylabel('Cumulative Percentage of Startups (%)')
    plt.title('Scree Plot: Cumulative Percentage of Startups vs. Number of Top Markets')
    plt.grid(True)
    plt.show()
    print(f'{n} most hot {i}, representative power = {cumulative_percentage.iloc[-1]:.2f}%')

# consolidate needed for better analysis
# grouping markets in industries to decrease the number of segments. The list was being taken from here https://support.crunchbase.com/hc/en-us/articles/360043146954-What-Industries-are-included-in-Crunchbase-
admin_services = str('Employer Benefits Programs, Human Resource Automation, Corporate IT, Distribution, Service Providers, Archiving Service, Call Center, Collection Agency, College Recruiting, Courier Service, Debt Collections, Delivery, Document Preparation, Employee Benefits, Extermination Service, Facilities Support Services, Housekeeping Service, Human Resources, Knowledge Management, Office Administration, Packaging Services, Physical Security, Project Management, Staffing Agency, Trade Shows, Virtual Workforce').split(', ')
advertising = str('Creative Industries, Promotional, Advertising Ad Exchange, Ad Network, Ad Retargeting, Ad Server, Ad Targeting, Advertising, Advertising Platforms, Affiliate Marketing, Local Advertising, Mobile Advertising, Outdoor Advertising, SEM, Social Media Advertising, Video Advertising').split(', ')
agriculture = str('Agriculture, AgTech, Animal Feed, Aquaculture, Equestrian, Farming, Forestry, Horticulture, Hydroponics, Livestock').split(', ')
app = str('Application Performance Monitoring, App Stores, Application Platforms, Enterprise Application, App Discovery, Apps, Consumer Applications, Enterprise Applications, Mobile Apps, Reading Apps, Web Apps').split(', ')
artificial_intelli = str('Artificial Intelligence, Intelligent Systems, Machine Learning, Natural Language Processing, Predictive Analytics').split(', ')
biotechnology = str('Synthetic Biology, Bio-Pharm, Bioinformatics, Biometrics, Biopharma, Biotechnology, Genetics, Life Science, Neuroscience, Quantified Self').split(', ')
clothing = str('Fashion, Laundry and Dry-cleaning, Lingerie, Shoes').split(', ')
shopping = str('Consumer Behavior, Customer Support Tools, Discounts, Reviews and Recommendations, Auctions, Classifieds, Collectibles, Consumer Reviews, Coupons, E-Commerce, E-Commerce Platforms, Flash Sale, Gift, Gift Card, Gift Exchange, Gift Registry, Group Buying, Local Shopping, Made to Order, Marketplace, Online Auctions, Personalization, Point of Sale, Price Comparison, Rental, Retail, Retail Technology, Shopping, Shopping Mall, Social Shopping, Sporting Goods, Vending and Concessions, Virtual Goods, Wholesale').split(', ')
community = str("Self Development, Sex, Forums, Match-Making, Babies, Identity, Women, Kids, Entrepreneur, Networking, Adult, Baby, Cannabis, Children, Communities, Dating, Elderly, Family, Funerals, Humanitarian, Leisure, LGBT, Lifestyle, Men's, Online Forums, Parenting, Pet, Private Social Networking, Professional Networking, Q&A, Religion, Retirement, Sex Industry, Sex Tech, Social, Social Entrepreneurship, Teenagers, Virtual World, Wedding, Women's, Young Adults").split(', ')
electronics  = str('Mac, iPod Touch, Tablets, iPad, iPhone, Computer, Consumer Electronics, Drones, Electronics, Google Glass, Mobile Devices, Nintendo, Playstation, Roku, Smart Home, Wearables, Windows Phone, Xbox').split(', ')
consumer_goods= str('Commodities, Sunglasses, Groceries, Batteries, Cars, Beauty, Comics, Consumer Goods, Cosmetics, DIY, Drones, Eyewear, Fast-Moving Consumer Goods, Flowers, Furniture, Green Consumer Goods, Handmade, Jewelry, Lingerie, Shoes, Tobacco, Toys').split(', ')
content = str('E-Books, MicroBlogging, Opinions, Blogging Platforms, Content Delivery Network, Content Discovery, Content Syndication, Creative Agency, DRM, EBooks, Journalism, News, Photo Editing, Photo Sharing, Photography, Printing, Publishing, Social Bookmarking, Video Editing, Video Streaming').split(', ')
data = str('Optimization, A/B Testing, Analytics, Application Performance Management, Artificial Intelligence, Big Data, Bioinformatics, Biometrics, Business Intelligence, Consumer Research, Data Integration, Data Mining, Data Visualization, Database, Facial Recognition, Geospatial, Image Recognition, Intelligent Systems, Location Based Services, Machine Learning, Market Research, Natural Language Processing, Predictive Analytics, Product Research, Quantified Self, Speech Recognition, Test and Measurement, Text Analytics, Usability Testing').split(', ')
design = str('Visualization, Graphics, Design, Designers, CAD, Consumer Research, Data Visualization, Fashion, Graphic Design, Human Computer Interaction, Industrial Design, Interior Design, Market Research, Mechanical Design, Product Design, Product Research, Usability Testing, UX Design, Web Design').split(', ')
education = str('Universities, College Campuses, University Students, High Schools, All Students, Colleges, Alumni, Charter Schools, College Recruiting, Continuing Education, Corporate Training, E-Learning, EdTech, Education, Edutainment, Higher Education, Language Learning, MOOC, Music Education, Personal Development, Primary Education, Secondary Education, Skill Assessment, STEM Education, Textbook, Training, Tutoring, Vocational Education').split(', ')
energy = str('Gas, Natural Gas Uses, Oil, Oil & Gas, Battery, Biofuel, Biomass Energy, Clean Energy, Electrical Distribution, Energy, Energy Efficiency, Energy Management, Energy Storage, Fossil Fuels, Fuel, Fuel Cell, Oil and Gas, Power Grid, Renewable Energy, Solar, Wind Energy').split(', ')
events = str('Concerts, Event Management, Event Promotion, Events, Nightclubs, Nightlife, Reservations, Ticketing, Wedding').split(', ')
financial = str('Debt Collecting, P2P Money Transfer, Investment Management, Trading, Accounting, Angel Investment, Asset Management, Auto Insurance, Banking, Bitcoin, Commercial Insurance, Commercial Lending, Consumer Lending, Credit, Credit Bureau, Credit Cards, Crowdfunding, Cryptocurrency, Debit Cards, Debt Collections, Finance, Financial Exchanges, Financial Services, FinTech, Fraud Detection, Funding Platform, Gift Card, Health Insurance, Hedge Funds, Impact Investing, Incubators, Insurance, InsurTech, Leasing, Lending, Life Insurance, Micro Lending, Mobile Payments, Payments, Personal Finance, Prediction Markets, Property Insurance, Real Estate Investment, Stock Exchanges, Trading Platform, Transaction Processing, Venture Capital, Virtual Currency, Wealth Management').split(', ')
food = str('Specialty Foods, Bakery, Brewing, Cannabis, Catering, Coffee, Confectionery, Cooking, Craft Beer, Dietary Supplements, Distillery, Farmers Market, Food and Beverage, Food Delivery, Food Processing, Food Trucks, Fruit, Grocery, Nutrition, Organic Food, Recipes, Restaurants, Seafood, Snack Food, Tea, Tobacco, Wine And Spirits, Winery').split(', ')
gaming = str('Game, Games, Casual Games, Console Games, Contests, Fantasy Sports, Gambling, Gamification, Gaming, MMO Games, Online Games, PC Games, Serious Games, Video Games').split(', ')
government = str('Polling, Governance, CivicTech, Government, GovTech, Law Enforcement, Military, National Security, Politics, Public Safety, Social Assistance').split(', ')
hardware= str('Cable, 3D, 3D Technology, Application Specific Integrated Circuit (ASIC), Augmented Reality, Cloud Infrastructure, Communication Hardware, Communications Infrastructure, Computer, Computer Vision, Consumer Electronics, Data Center, Data Center Automation, Data Storage, Drone Management, Drones, DSP, Electronic Design Automation (EDA), Electronics, Embedded Systems, Field-Programmable Gate Array (FPGA), Flash Storage, Google Glass, GPS, GPU, Hardware, Industrial Design, Laser, Lighting, Mechanical Design, Mobile Devices, Network Hardware, NFC, Nintendo, Optical Communication, Playstation, Private Cloud, Retail Technology, RFID, RISC, Robotics, Roku, Satellite Communication, Semiconductor, Sensor, Sex Tech, Telecommunications, Video Conferencing, Virtual Reality, Virtualization, Wearables, Windows Phone, Wireless, Xbox').split(', ')
health_care = str('Senior Health, Physicians, Electronic Health Records, Doctors, Healthcare Services, Diagnostics, Alternative Medicine, Assisted Living, Assistive Technology, Biopharma, Cannabis, Child Care, Clinical Trials, Cosmetic Surgery, Dental, Diabetes, Dietary Supplements, Elder Care, Electronic Health Record (EHR), Emergency Medicine, Employee Benefits, Fertility, First Aid, Funerals, Genetics, Health Care, Health Diagnostics, Home Health Care, Hospital, Medical, Medical Device, mHealth, Nursing and Residential Care, Nutraceutical, Nutrition, Outpatient Care, Personal Health, Pharmaceutical, Psychology, Rehabilitation, Therapeutics, Veterinary, Wellness').split(', ')
it = str('Distributors, Algorithms, ICT, M2M, Technology, Business Information Systems, CivicTech, Cloud Data Services, Cloud Management, Cloud Security, CMS, Contact Management, CRM, Cyber Security, Data Center, Data Center Automation, Data Integration, Data Mining, Data Visualization, Document Management, E-Signature, Email, GovTech, Identity Management, Information and Communications Technology (ICT), Information Services, Information Technology, Intrusion Detection, IT Infrastructure, IT Management, Management Information Systems, Messaging, Military, Network Security, Penetration Testing, Private Cloud, Reputation, Sales Automation, Scheduling, Social CRM, Spam Filtering, Technical Support, Unified Communications, Video Chat, Video Conferencing, Virtualization, VoIP').split(', ')
internet = str('Online Identity, Cyber, Portals, Web Presence Management, Domains, Tracking, Web Tools, Curated Web, Search, Cloud Computing, Cloud Data Services, Cloud Infrastructure, Cloud Management, Cloud Storage, Darknet, Domain Registrar, E-Commerce Platforms, Ediscovery, Email, Internet, Internet of Things, ISP, Location Based Services, Messaging, Music Streaming, Online Forums, Online Portals, Private Cloud, Product Search, Search Engine, SEM, Semantic Search, Semantic Web, SEO, SMS, Social Media, Social Media Management, Social Network, Unified Communications, Vertical Search, Video Chat, Video Conferencing, Visual Search, VoIP, Web Browsers, Web Hosting').split(', ')
invest = str('Angel Investment, Banking, Commercial Lending, Consumer Lending, Credit, Credit Cards, Financial Exchanges, Funding Platform, Hedge Funds, Impact Investing, Incubators, Micro Lending, Stock Exchanges, Trading Platform, Venture Capital').split(', ')
manufacturing = str('Innovation Engineering, Civil Engineers, Heavy Industry, Engineering Firms, Systems, 3D Printing, Advanced Materials, Foundries, Industrial, Industrial Automation, Industrial Engineering, Industrial Manufacturing, Machinery Manufacturing, Manufacturing, Paper Manufacturing, Plastics and Rubber Manufacturing, Textiles, Wood Processing').split(', ')
media = str('Writers, Creative, Television, Entertainment, Media, Advice, Animation, Art, Audio, Audiobooks, Blogging Platforms, Broadcasting, Celebrity, Concerts, Content, Content Creators, Content Discovery, Content Syndication, Creative Agency, Digital Entertainment, Digital Media, DRM, EBooks, Edutainment, Event Management, Event Promotion, Events, Film, Film Distribution, Film Production, Guides, In-Flight Entertainment, Independent Music, Internet Radio, Journalism, Media and Entertainment, Motion Capture, Music, Music Education, Music Label, Music Streaming, Music Venues, Musical Instruments, News, Nightclubs, Nightlife, Performing Arts, Photo Editing, Photo Sharing, Photography, Podcast, Printing, Publishing, Reservations, Social Media, Social News, Theatre, Ticketing, TV, TV Production, Video, Video Editing, Video on Demand, Video Streaming, Virtual World').split(', ')
message = str('Unifed Communications, Chat, Email, Meeting Software, Messaging, SMS, Unified Communications, Video Chat, Video Conferencing, VoIP, Wired Telecommunications').split(', ')
mobile = str('Android, Google Glass, iOS, mHealth, Mobile, Mobile Apps, Mobile Devices, Mobile Payments, Windows Phone, Wireless').split(', ')
music = str('Audio, Audiobooks, Independent Music, Internet Radio, Music, Music Education, Music Label, Music Streaming, Musical Instruments, Podcast').split(', ')
resource = str('Biofuel, Biomass Energy, Fossil Fuels, Mineral, Mining, Mining Technology, Natural Resources, Oil and Gas, Precious Metals, Solar, Timber, Water, Wind Energy').split(', ')
navigation = str('Maps, Geospatial, GPS, Indoor Positioning, Location Based Services, Mapping Services, Navigation').split(', ')
other = str('Mass Customization, Monetization, Testing, Subscription Businesses, Mobility, Incentives, Peer-to-Peer, Nonprofits, Alumni, Association, B2B, B2C, Blockchain, Charity, Collaboration, Collaborative Consumption, Commercial, Consumer, Crowdsourcing, Customer Service, Desktop Apps, Emerging Markets, Enterprise, Ethereum, Franchise, Freemium, Generation Y, Generation Z, Homeless Shelter, Infrastructure, Knowledge Management, LGBT Millennials, Non Profit, Peer to Peer, Professional Services, Project Management, Real Time, Retirement, Service Industry, Sharing Economy, Small and Medium Businesses, Social Bookmarking, Social Impact, Subscription Service, Technical Support, Underserved Children, Universities').split(', ')
payment = str('Billing, Bitcoin, Credit Cards, Cryptocurrency, Debit Cards, Fraud Detection, Mobile Payments, Payments, Transaction Processing, Virtual Currency').split(', ')
platforms = str('Development Platforms, Android, Facebook, Google, Google Glass, iOS, Linux, macOS, Nintendo, Operating Systems, Playstation, Roku, Tizen, Twitter, WebOS, Windows, Windows Phone, Xbox').split(', ')
privacy = str('Digital Rights Management, Personal Data, Cloud Security, Corrections Facilities, Cyber Security, DRM, E-Signature, Fraud Detection, Homeland Security, Identity Management, Intrusion Detection, Law Enforcement, Network Security, Penetration Testing, Physical Security, Privacy, Security').split(', ')
services = str('Funeral Industry, English-Speaking, Spas, Plumbers, Service Industries, Staffing Firms, Translation, Career Management, Business Services, Services, Accounting, Business Development, Career Planning, Compliance, Consulting, Customer Service, Employment, Environmental Consulting, Field Support, Freelance, Intellectual Property, Innovation Management, Legal, Legal Tech, Management Consulting, Outsourcing, Professional Networking, Quality Assurance, Recruiting, Risk Management, Social Recruiting, Translation Service').split(', ')
realestate= str('Office Space, Self Storage, Brokers, Storage, Home Owners, Self Storage , Realtors, Home & Garden, Utilities, Home Automation, Architecture, Building Maintenance, Building Material, Commercial Real Estate, Construction, Coworking, Facility Management, Fast-Moving Consumer Goods, Green Building, Home and Garden, Home Decor, Home Improvement, Home Renovation, Home Services, Interior Design, Janitorial Service, Landscaping, Property Development, Property Management, Real Estate, Real Estate Investment, Rental Property, Residential, Self-Storage, Smart Building, Smart Cities, Smart Home, Timeshare, Vacation Rental').split(', ')
sales = str('Advertising, Affiliate Marketing, App Discovery, App Marketing, Brand Marketing, Cause Marketing, Content Marketing, CRM, Digital Marketing, Digital Signage, Direct Marketing, Direct Sales, Email Marketing, Lead Generation, Lead Management, Local, Local Advertising, Local Business, Loyalty Programs, Marketing, Marketing Automation, Mobile Advertising, Multi-level Marketing, Outdoor Advertising, Personal Branding, Public Relations, Sales, Sales Automation, SEM, SEO, Social CRM, Social Media Advertising, Social Media Management, Social Media Marketing, Sponsorship, Video Advertising').split(', ')
science = str('Face Recognition, New Technologies, Advanced Materials, Aerospace, Artificial Intelligence, Bioinformatics, Biometrics, Biopharma, Biotechnology, Chemical, Chemical Engineering, Civil Engineering, Embedded Systems, Environmental Engineering, Human Computer Interaction, Industrial Automation, Industrial Engineering, Intelligent Systems, Laser, Life Science, Marine Technology, Mechanical Engineering, Nanotechnology, Neuroscience, Nuclear, Quantum Computing, Robotics, Semiconductor, Software Engineering, STEM Education').split(', ')
software = str('Business Productivity, 3D Technology, Android, App Discovery, Application Performance Management, Apps, Artificial Intelligence, Augmented Reality, Billing, Bitcoin, Browser Extensions, CAD, Cloud Computing, Cloud Management, CMS, Computer Vision, Consumer Applications, Consumer Software, Contact Management, CRM, Cryptocurrency, Data Center Automation, Data Integration, Data Storage, Data Visualization, Database, Developer APIs, Developer Platform, Developer Tools, Document Management, Drone Management, E-Learning, EdTech, Electronic Design Automation (EDA), Embedded Software, Embedded Systems, Enterprise Applications, Enterprise Resource Planning (ERP), Enterprise Software, Facial Recognition, File Sharing, IaaS, Image Recognition, iOS, Linux, Machine Learning, macOS, Marketing Automation, Meeting Software, Mobile Apps, Mobile Payments, MOOC, Natural Language Processing, Open Source, Operating Systems, PaaS, Predictive Analytics, Presentation Software, Presentations, Private Cloud, Productivity Tools, QR Codes, Reading Apps, Retail Technology, Robotics, SaaS, Sales Automation, Scheduling, Sex Tech, Simulation, SNS, Social CRM, Software, Software Engineering, Speech Recognition, Task Management, Text Analytics, Transaction Processing, Video Conferencing, Virtual Assistant, Virtual Currency, Virtual Desktop, Virtual Goods, Virtual Reality, Virtual World, Virtualization, Web Apps, Web Browsers, Web Development').split(', ')
sports = str('American Football, Baseball, Basketball, Boating, Cricket, Cycling, Diving, eSports, Fantasy Sports, Fitness, Golf, Hockey, Hunting, Outdoors, Racing, Recreation, Rugby, Sailing, Skiing, Soccer, Sporting Goods, Sports, Surfing, Swimming, Table Tennis, Tennis, Ultimate Frisbee, Volley Ball').split(', ')
sustainability = str('Green, Wind, Biomass Power Generation, Renewable Tech, Environmental Innovation, Renewable Energies, Clean Technology, Biofuel, Biomass Energy, Clean Energy, CleanTech, Energy Efficiency, Environmental Engineering, Green Building, Green Consumer Goods, GreenTech, Natural Resources, Organic, Pollution Control, Recycling, Renewable Energy, Solar, Sustainability, Waste Management, Water Purification, Wind Energy').split(', ')
transportation = str('Taxis, Air Transportation, Automotive, Autonomous Vehicles, Car Sharing, Courier Service, Delivery Service, Electric Vehicle, Ferry Service, Fleet Management, Food Delivery, Freight Service, Last Mile Transportation, Limousine Service, Logistics, Marine Transportation, Parking, Ports and Harbors, Procurement, Public Transportation, Railroad, Recreational Vehicles, Ride Sharing, Same Day Delivery, Shipping, Shipping Broker, Space Travel, Supply Chain Management, Taxi Service, Transportation, Warehousing, Water Transportation').split(', ')
travel = str('Adventure Travel, Amusement Park and Arcade, Business Travel, Casino, Hospitality, Hotel, Museums and Historical Sites, Parks, Resorts, Timeshare, Tour Operator, Tourism, Travel, Travel Accommodations, Travel Agency, Vacation Rental').split(', ')
video = str('Animation, Broadcasting, Film, Film Distribution, Film Production, Motion Capture, TV, TV Production, Video, Video Editing, Video on Demand, Video Streaming').split(', ')

# Define a function to categorize the 'market' values
def categorize_market(row):
    if isinstance(row['market'], float) and np.isnan(row['market']):
        return "Other"
    
    for keyword_list, industry_group in [
        (admin_services, "Administrative Services"),
        (software, "Software"),
        (advertising, "Advertising"),
        (agriculture, "Agriculture and Farming"),
        (app, "Apps"),
        (artificial_intelli, "Artificial Intelligence"),
        (biotechnology, "Biotechnology"),
        (clothing, "Clothing and Apparel"),
        (shopping, "Commerce and Shopping"),
        (community, "Community and Lifestyle"),
        (electronics, "Consumer Electronics"),
        (consumer_goods, "Consumer Goods"),
        (content, "Content and Publishing"),
        (data, "Data and Analytics"),
        (design, "Design"),
        (education, "Education"),
        (energy, "Energy"),
        (events, "Events"),
        (financial, "Financial Services"),
        (food, "Food and Beverage"),
        (gaming, "Gaming"),
        (government, "Government and Military"),
        (hardware, "Hardware"),
        (health_care, "Health Care"),
        (it, "Information Technology"),
        (internet, "Internet Services"),
        (invest, "Lending and Investments"),
        (manufacturing, "Manufacturing"),
        (media, "Media and Entertainment"),
        (message, "Messaging and Telecommunication"),
        (mobile, "Mobile"),
        (music, "Music and Audio"),
        (resource, "Natural Resources"),
        (navigation, "Navigation and Mapping"),
        (payment, "Payments"),
        (platforms, "Platforms"),
        (privacy, "Privacy and Security"),
        (services, "Professional Services"),
        (realestate, "Real Estate"),
        (sales, "Sales and Marketing"),
        (science, "Science and Engineering"),
        (sports, "Sports"),
        (sustainability, "Sustainability"),
        (transportation, "Transportation"),
        (travel, "Travel and Tourism"),
        (video, "Video"),
        (other, "Other")
    ]:
        if any(keyword in str(row['market']) for keyword in keyword_list):
            return industry_group
    return "Other"

# Apply the function to create the 'Industry_Group' column
df3['industry_group'] = df3.apply(categorize_market, axis=1)

# Using scree_plot to validate the aggregation
scree_plot_market(df3,i='market')
scree_plot_market(df3,i='industry_group')

# Top 10 industry_group
df3.groupby('industry_group')['name'].count().sort_values(ascending = False).head(10)

def status_analysis(df, status="closed"):
    # Group the DataFrame by 'industry_group' and count the total number of startups in each group
    industry_group_counts = df.groupby('industry_group')['name'].count().reset_index()
    industry_group_counts.rename(columns={'name': 'total_startups'}, inplace=True)

    # Filter the DataFrame to include only startups with the specified status
    status_startups = df[df['status'] == status]

    # Group the filtered DataFrame by 'industry_group' and count the number of startups with the specified status in each group
    status_startups_counts = status_startups.groupby('industry_group')['name'].count().reset_index()
    status_startups_counts.rename(columns={'name': f'{status}_startups'}, inplace=True)

    # Merge the two DataFrames on 'industry_group' to have the total and specified status startup counts together
    status_rate_data = pd.merge(industry_group_counts, status_startups_counts, on='industry_group', how='left')

    # Calculate the fail rate as the ratio of startups with the specified status to total startups, handling NaN values with 0 fail rate
    status_rate_data[f'{status}_rate'] = (status_rate_data[f'{status}_startups'] / status_rate_data['total_startups']).fillna(0)

    # Calculate the correlation between 'status rate' and 'total startups' for each industry group
    correlation = status_rate_data[[f'{status}_rate', 'total_startups']].corr()
    
    # Plot a heatmap to visualize the correlations
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', square=True)
    plt.title(f'Correlation Heatmap between {status.capitalize()} Rate and Total Startups by Industry')
    plt.show()
    
    # Sort the DataFrame by 'fail_rate' in descending order, with total startups as a secondary sorting key
    status_rate_data_sorted = status_rate_data.sort_values(by=[f'{status}_rate', 'total_startups'], ascending=[False, False])

    # Create a dictionary to map industry_group to a unique color
    industry_group_colors = {
        group: plt.cm.viridis(i / len(status_rate_data_sorted['industry_group'].unique()))
        for i, group in enumerate(status_rate_data_sorted['industry_group'].unique())
    }
    
    # Plot a scatter plot for each industry_group and add a legend
    plt.figure(figsize=(10, 6))
    for group in status_rate_data_sorted['industry_group'].unique():
        group_data = status_rate_data_sorted[status_rate_data_sorted['industry_group'] == group]
        plt.scatter(group_data['total_startups'], group_data[f'{status}_rate'], s=50, alpha=0.5, label=group, color=industry_group_colors[group])

    plt.title(f'Scatter Plot of {status.capitalize()} Startup Rate vs. Total Startups By Industry Group')
    plt.xlabel('Total Startups')
    plt.ylabel(f'{status.capitalize()} Startup Rate')
    plt.ylim(0, 0.5)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return status_rate_data_sorted[['industry_group', 'total_startups', f'{status}_rate']].head(10)

# number matters?
status_analysis(df3, status='acquired')
status_analysis(df3, status='closed')

# EDA3. Defining Success
# Status Overview
status_overview(df3)
 
# There are more than 84% still in operation, making it's hard to evaluate the success of the startup. This might due to no update on the dataset or most of them are newly formed. (Media saids that 90% of startup fail, 10% fail within the first year)
df3.groupby('status')['funding_total_usd'].describe() #More Funding, higher chance to get acquisition
df3.groupby('status')['funding_rounds'].describe() #More Funding rounds, higher aacquisition
# Create a histogram

# Get 'years_since_established'
df3['founded_year'] = df3['founded_at'].apply(lambda x: pd.to_datetime(x, format='%d-%m-%Y').strftime('%Y'))
current_year = 2014
df3['founded_year'] = df3['founded_year'].astype(int)
df3['years_since_established'] = current_year - df3['founded_year']
def hist_timeframe(df,i='founded_year'):
    plt.figure(figsize=(10, 6))
    plt.hist(df[i], bins=25, range=(1990, 2014), edgecolor='black')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Startups')
    plt.title('Histogram of Founded Year (1990-2014)')
    plt.grid(True)
    # Set x-axis limits to show the range from 1990 to 2014
    plt.xlim(1990, 2014)
    # Reduce the number of x-tick labels by setting the frequency to display every 2 years
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.xticks(range(1990, 2015, 2))  # Set the x-tick labels
    plt.show()

hist_timeframe(df3,i='founded_year')

# ED3.1 Analyzing 100m_Startup [Valuation might over 1b]
df_100m_startup = df3[(df3['funding_total_usd']>=100000000) & (df3['funding_total_usd'] < 10000000000)]    
status_overview(df_100m_startup) 
visualize_top_countries_by_continent(df_100m_startup, title='Number of 100m USD Startups per Country')
df_100m_startup['years_since_established'].hist()
hist_timeframe(df_100m_startup,i='founded_year')

# ED3.2 Analyzing Unicorn [Valuation over 1b for sure]
df_unicorn = df3[df3['funding_total_usd']>=1000000000]
status_overview(df_unicorn)
visualize_top_countries_by_continent(df_unicorn, title='Number of Unicorns per Country')
df_unicorn['years_since_established'].hist()
hist_timeframe(df_unicorn,i='founded_year')

# ED3.3 Operating timeframe
df3['founded_year'].min() # 1802
df3['founded_year'].max() # latest update =2014
df3['founded_year'] = df3['founded_at'].dt.year
# Most companies are formed around the year of 2009-2013

# company_5years
df_5years = df3[df3['years_since_established']<=5]
df_5years['name'].nunique()
status_overview(df_5years)
# company_10years
df_10years = df3[(df3['years_since_established']<=10) & (df3['years_since_established']>5)]
df_10years['name'].nunique()
status_overview(df_10years)
# company_above_10years
df_above10years = df3[df3['years_since_established']>10]
df_above10years['name'].nunique()
status_overview(df_above10years)
# longer years = higher chance to get acquired, more than 10 years = lowest closed rate

# In conclusion: Success: longer years, funding size, funding round
# However, from investment standpoint, the longer year means the risk reward is no longer the best, we can explore the determining factors for companies<5 years and low funding values

# Calculate the correlation between 'status rate' and 'total startups' for each industry group
# Turn status into numeric  = Status Score
# Define a mapping from status to score
status_to_score = {
    'acquired': 3,
    'closed': 0,
    'operating': 2,
    'unknown': 1
}

# Create the 'status score' column by mapping the 'status' values to scores
df3['status_score'] = df3['status'].map(status_to_score)

# Ensure the 'status score' column is of type float within the scale of 0-3
df3['status_score'] = df3['status_score'].astype(float)

correlation_matrix = df3[['status_score', 'years_since_established','funding_total_usd','funding_rounds', 
        'seed', 'venture', 'equity_crowdfunding',
       'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
       'private_equity', 'post_ipo_equity', 'post_ipo_debt',
       'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',
       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']].corr()

#Plot a heatmap to visualize the correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap between Variables and Total Startups by Industry')
plt.show()

high_correlations = correlation_matrix[correlation_matrix.abs() > 0.5]
# Plot a heatmap to visualize the correlations
plt.figure(figsize=(10, 6))
sns.heatmap(high_correlations, annot=True, cmap='coolwarm', square=True)
plt.title('High Correlation Heatmap between Variables and Total Startups by Industry')
plt.show()
# Most money are get from undisclosed and C,D,E rounds

status_score_corr = correlation_matrix['status_score']

# Display the correlation as a table
status_score_corr = status_score_corr.drop('status_score')  # Remove self-correlation
status_score_corr = status_score_corr.reset_index()
status_score_corr.columns = ['Variable', 'Correlation with Status Score']
status_score_corr = status_score_corr.sort_values(by='Correlation with Status Score', ascending=False)
print(status_score_corr)

# Funding_rounds indicates that the more funding_rounds that startups involves, the more higher chance of success
# This could be due to the small milestones effect, motivating the team to outperform
# This could be due to the expanding network and guidances received from more professional investors

debt_financing_corr = correlation_matrix['debt_financing']
# Display the correlation as a table
debt_financing_corr = debt_financing_corr.drop('debt_financing')  # Remove self-correlation
debt_financing_corr = debt_financing_corr.reset_index()
debt_financing_corr.columns = ['Variable', 'Correlation with Status Score']
debt_financing_corr = debt_financing_corr.sort_values(by='Correlation with Status Score', ascending=False)
print(debt_financing_corr)
# Extremely high score to total_funding. Startup that get into debt financing has less tendency to get financing from other places.
# Is it a good thing?
debt_finance_startup_df = df3[df3['debt_financing'] > 0]
status_overview(debt_finance_startup_df) 
#Unable to evaluate, almost same as the average performance

df3.groupby('status')[['funding_rounds', 'funding_total_usd', 'seed', 'venture', 'equity_crowdfunding',
       'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
       'private_equity', 'post_ipo_equity', 'post_ipo_debt',
       'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',
       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']].mean().T

# As a result, it's quite hard to to determine the success factors based on normal data analysis
# This is because the featured is not dominance, we can proceed into ML model.

# Final_clean
df3.drop(columns=['permalink', 'homepage_url','category_list','city','entity'], inplace=True)
df3.isnull().sum()

df3[df3['funding_total_usd'].isnull()]
# Define a list of columns to be summed
columns_to_sum = ['seed', 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
       'private_equity', 'post_ipo_equity', 'post_ipo_debt', 'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',
       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']
# Replace NULL values in 'total_funding' with the sum of specified columns
df3['funding_total_usd'].fillna(df3[columns_to_sum].sum(axis=1), inplace=True)
df3['market'].fillna(df3['industry_group'], inplace=True) #2836 data into other category
df3.isnull().sum()

df3.to_csv(r'C:\Users\User\Desktop\Projects\Data Science\project_startup_success\data\processed\df_dp.csv', index=False)



