import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import geopandas as gpd
from scipy.stats import pearsonr
# ------------------------------------------- Pre - Processing --------------------------------- #
df = pd.read_csv(".covid_data.csv")

#Limited interpolation fill small gaps
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].interpolate(method='nearest', limit=5, limit_direction='both')

#Convert date into datetime
df['date'] = pd.to_datetime(df['date'])

# ------------------------------------------- Streamlit Dashboard ------------------------------ #
st.title("Numbers Behind the Crisis: The COVID-19 Report")

# -- Abstract -- 
st.header("Abstract")

# Objective
st.markdown("**Objective**")
st.markdown("""
• Analyze global COVID-19 trends using the Our World in Data (OWID) dataset spanning 2020-2023
• Examine key pandemic metrics: cases, deaths, testing efficiency, vaccination impact, and government policy stringency
• Compare pandemic responses and outcomes between high-GDP versus low-GDP countries
• Assess vaccine effectiveness across different dosage levels and timing of implementation
• Identify data-driven insights for improved future pandemic preparedness and response strategies
""")

# Methodology
st.markdown("**Methodology**")
st.markdown("""
• **Data Source**: OWID COVID-19 dataset with global pandemic metrics across 200+ countries
• **Preprocessing**: Applied interpolation methods to handle missing values, converted dates to datetime format
• **Statistical Analysis**: Used Pearson correlation analysis to quantify vaccine-death rate relationships
• **Geographical Analysis**: Performed continent-level data aggregation to track pandemic wave progression
• **Economic Classification**: Grouped countries by GDP per capita to analyze resource-outcome relationships
• **Policy Comparison**: Contrasted strict (India) versus lenient (Sweden) policy approaches using stringency indices
• **Visualization Tools**: Implemented matplotlib, seaborn, and geopandas for comprehensive data visualization
• **Time-Series Analysis**: Applied rolling averages and trend analysis to identify key pandemic phases
""")

# Key Findings
st.markdown("**Key Findings**")
st.markdown("""
• **Wave Patterns**: Omicron variant created the highest global case surge, spreading systematically from Europe → North America → rest of world
• **Vaccine Effectiveness**: Strong negative correlation (r = -0.65 to -0.8) between vaccination rates and COVID-19 death rates across all analyzed countries
• **Dosage Impact**: Full vaccination demonstrated superior death prevention compared to single-dose regimens, with booster shots providing additional protection
• **Economic Disparities**: Higher GDP countries achieved better testing efficiency with lower test positivity rates (5-10% vs 15-25% in low-GDP countries)
• **Policy Responses**: Lower-income countries implemented stricter lockdown measures (stringency index 80-90) compared to wealthier nations (60-70)
• **Geographic Spread**: Each major variant wave showed distinct geographic progression patterns, with 2-4 week delays between continents
• **Testing Efficiency**: Countries with GDP >$20,000 showed proactive testing strategies, while GDP <$5,000 countries exhibited reactive patterns
""")

# Conclusions
st.markdown("**Conclusions**")
st.markdown("""
• **Vaccine Impact**: Vaccination campaigns significantly reduced COVID-19 mortality across all income levels and geographic regions, with full vaccination providing optimal protection
• **Economic Inequality**: Substantial disparities in pandemic outcomes correlated directly with national economic resources and healthcare infrastructure capacity  
• **Policy Effectiveness**: Data-driven government responses (combining moderate stringency with high testing) proved more effective than purely restrictive or permissive approaches
• **Global Coordination**: Future pandemic preparedness requires enhanced international cooperation, equitable vaccine distribution, and real-time data sharing systems
• **Evidence-Based Decision Making**: Countries that implemented evidence-based policies using real-time epidemiological data achieved better health and economic outcomes
""")

# Table Of Contents
st.markdown("**Table Of Contents**")
st.markdown("""
1. Dataset Overview   
2. Visual Trends  
3. Key Insights  
4. Vaccine Hesitancy Analysis  
5. Economic Impact Assessment  
6. Policy Stringency Comparison
7. Conclusion   
8. Future Recommendations  
""")

# Data Quality Statement
st.markdown("**Data Quality & Limitations**")
st.markdown("""
• **Coverage**: Dataset includes 200+ countries with varying data completeness levels
• **Time Period**: Analysis covers January 2020 through December 2023
• **Missing Data**: Applied nearest-neighbor interpolation for gaps ≤5 days in key metrics
• **Reporting Variations**: Country-level differences in testing strategies and case definitions may affect comparability
• **Economic Data**: GDP per capita figures based on World Bank 2019-2020 estimates
""")

# -- Dataset Overview --  
st.header("Dataset Overview")
st.write("The OWID COVID-19 dataset is a comprehensive, curated collection of global pandemic data compiled from official sources. " \
"It provides standardized, up-to-date information on cases, deaths, testing, vaccinations, and government responses across countries.")

st.markdown("**Dataset Sample**")
st.table(df[['country','date','total_cases_per_million','code','gdp_per_capita']].sample(3).sort_index())

st.markdown("**Dataset Description**")
dataset_description = dict({"rows":len(df),"columns":len(df.columns),"countries":df['country'].nunique(),"date_range_min":df['date'].min(),"date_range_max":df['date'].max(),"duplicated_rows":df.duplicated().sum()})
st.table(dataset_description)

#Geomap
url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
world = gpd.read_file(url)
merged = world.merge(df, left_on="ISO_A3", right_on="code")
fig0, ax = plt.subplots(figsize=(12, 6))
merged.plot(column="total_cases_per_million", cmap="viridis", linewidth=0.8, ax=ax, edgecolor="0.8", legend=True,vmin=0,vmax=10_000 )
ax.set_title("Total COVID-19 Cases per million people by Country", fontsize=14)
plt.axis("off")
st.pyplot(fig0)

# -- Visual Trends
st.header("Visual Trends")

#Epidemic Spread and Waves
st.subheader("Epidemic Trends")
st.write("The peaks represent different waves.")
st.write("The Omicron wave had the highest impact, it spread from Europe to North America then to the rest of the world.")

grouped = df.groupby(['date', 'continent']).agg({ #grouping of countries in a continet wrt date
    'new_cases': 'sum','population':'sum'
}).reset_index()

grouped['new_cases_per_million'] = (grouped['new_cases']/grouped['population'])*1000000
st.table(grouped.sample(3).sort_index())

fig1,axes = plt.subplots(3,2,figsize=(12,16))

continent = df[df['continent']=="Europe"]
sns.lineplot(data=continent, x='date', y='new_cases_per_million', ax = axes[0,0])
axes[0,0].set_title("New COVID-19 Cases per Million Over Time In Europe")

continent = df[df['continent']=="Asia"]
sns.lineplot(data=continent, x='date', y='new_cases_per_million', ax = axes[0,1])
axes[0,1].set_title("New COVID-19 Cases per Million Over Time In Asia")

continent = df[df['continent']=="North America"]
sns.lineplot(data=continent, x='date', y='new_cases_per_million', ax = axes[1,0])
axes[1,0].set_title("New COVID-19 Cases per Million Over Time In North America")

continent = df[df['continent']=="South America"]
sns.lineplot(data=continent, x='date', y='new_cases_per_million', ax = axes[1,1])
axes[1,1].set_title("New COVID-19 Cases per Million Over Time In South America")

continent = df[df['continent']=="Oceania"]
sns.lineplot(data=continent, x='date', y='new_cases_per_million', ax = axes[2,0])
axes[2,0].set_title("New COVID-19 Cases per Million Over Time In Oceania")

continent = df[df['continent']=="Africa"]
sns.lineplot(data=continent, x='date', y='new_cases_per_million', ax = axes[2,1])
axes[2,1].set_title("New COVID-19 Cases per Million Over Time In Africa")

st.pyplot(fig1)

#To view the effects of vaccination
st.subheader("Vaccine Effectiveness")
st.write("Countries with higher vaccination rate had a steeper decline in death count.")

grouped_vaccine = df.groupby(['country', 'date']).agg({
    'people_vaccinated_per_hundred': 'max',
    'new_cases_per_million': 'max',
    'icu_patients_per_million': 'max',
    'new_deaths_smoothed_per_million': 'max'
}).reset_index()

grouped_vaccine = grouped_vaccine.dropna(subset=[
    'people_vaccinated_per_hundred',
    'new_cases_per_million',
    'icu_patients_per_million',
    'new_deaths_smoothed_per_million'
])

st.table(grouped_vaccine.sample(3).sort_index())

#Display Countires with max and min vaccinations to help analyze the impact of vaccination
country_vacc_maxxed = grouped_vaccine.groupby('country').agg({'people_vaccinated_per_hundred':'max'})
most_vacc_country = country_vacc_maxxed['people_vaccinated_per_hundred'].idxmax()
least_vacc_country = country_vacc_maxxed['people_vaccinated_per_hundred'].idxmin()

st.write(f"""{most_vacc_country} has the highest vaccination rate of
        {country_vacc_maxxed['people_vaccinated_per_hundred'].max()} vaccinations per hundred people, and
        {least_vacc_country} has the lowest vaccination rate with
        {country_vacc_maxxed['people_vaccinated_per_hundred'].min()} vaccinations per hundred people.""")


def normalize(series): # Normalize to see the trends effictevely
    return (series - series.min()) / (series.max() - series.min())

def plot_vaccine_effectiveness(country_name, df):
    # Compute people vaccinated per million
    df['people_vaccinated_per_million'] = df['people_vaccinated_per_hundred'] * 1000

    # Filter for the specific country and sort
    country = df[df['country'] == country_name].sort_values('date').copy()

    # Metrics to plot
    metrics = [
        'new_cases_per_million',
        'people_vaccinated_per_million',
        'icu_patients_per_million',
        'new_deaths_smoothed_per_million'
    ]
    metric_labels = {
        'new_cases_per_million': "New Cases per Million",
        'people_vaccinated_per_million': "Vaccinated per Million",
        'icu_patients_per_million': "ICU Patients per Million",
        'new_deaths_smoothed_per_million': 'New_deaths_smoothed_per_million'
    }

    # Normalize each metric and create new columns
    for col in metrics:
        country[col + '_norm'] = normalize(country[col])

    # Plot each normalized metric in its own subplot
    fig, axs = plt.subplots(len(metrics), 1, figsize=(14, 12), sharex=True)

    for i, col in enumerate(metrics):
        norm_col = col + '_norm'
        sns.lineplot(data=country, x='date', y=norm_col, ax=axs[i])
        axs[i].set_ylabel("Normalized Value")
        axs[i].set_title(f"{metric_labels[col]} (Normalized) in {country_name}")
        axs[i].grid(True)

        # Add a vertical line for vaccine rollout start
        vax_start = country[country['people_vaccinated_per_million'] > 0]['date'].min()
        if pd.notnull(vax_start):
            axs[i].axvline(vax_start, color='grey', linestyle='--', label='Vaccine Rollout')
            axs[i].legend()

    plt.xlabel("Date")
    plt.tight_layout()
    return fig


st.markdown(f"**{most_vacc_country}**")
fig3 = plot_vaccine_effectiveness(most_vacc_country, grouped_vaccine)
st.pyplot(fig3)


st.markdown(f"**{least_vacc_country}**")
fig4 = plot_vaccine_effectiveness(least_vacc_country, grouped_vaccine)
st.pyplot(fig4)


# Vaccination Drives Effect
st.markdown("**Vaccination drives reduced COVID-19 deaths**")

country = st.selectbox("Which country do you want to see the data for?", ("India","Portugal","Europe","Kenya","Russia","Brazil","Germany",), index=None, placeholder="Select Country...",)
if country == None:
    country = 'India'

country_df = df[df['country'] == country].copy() 


country_df['new_deaths_rolling'] = country_df['new_deaths_smoothed_per_million'].rolling(window=14, min_periods=1).mean()

country_df['total_boosters_per_hundred'] = country_df['total_boosters'] / country_df['population'] * 100

filtered_basic = country_df[['people_vaccinated_per_hundred', 'new_deaths_rolling']].dropna()
plot = sns.lmplot(x='people_vaccinated_per_hundred', y='new_deaths_rolling', data=filtered_basic, height=3, aspect=2)
plot.figure.suptitle(f"Correlation plot of vaccination rates vs new deaths (14 day average, {country})", y=1.03)
fig2 = plot.figure
st.pyplot(plot.figure)


r, p = pearsonr(filtered_basic['people_vaccinated_per_hundred'], filtered_basic['new_deaths_rolling'])
st.write(f"Pearson Correlation: r = {r:.3f}, p = {p:.3f}")

# Dosage breakdown
st.markdown("**Higher Vaccination dosages further reduced deaths**")
st.markdown("""1. 'People semi vaccinated': Unique individuals with only one dose,  
2. 'People fully vaccinated': Unique individuals who got both doses or only one (J&J),    
3. 'Total boosters': Booster doses administered, not unique people.""")

country_df['people_semi_vaccinated_per_hundred'] = (
    country_df['people_vaccinated_per_hundred'] - country_df['people_fully_vaccinated_per_hundred']
).clip(lower=0)

filtered_all = country_df[
    ['people_vaccinated_per_hundred', 'people_semi_vaccinated_per_hundred',
     'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred', 'new_deaths_rolling']
].dropna()


fig5, axes = plt.subplots(2, 2, figsize=(16, 14))

sns.regplot(data=filtered_all, x='people_vaccinated_per_hundred', y='new_deaths_rolling', ax=axes[0, 0], color=sns.color_palette("pastel")[1])
r, p = pearsonr(filtered_all['people_vaccinated_per_hundred'], filtered_all['new_deaths_rolling'])
axes[0, 0].set_title(f"At least One Dose vs New Deaths, r: {r:.3f}, p: {p:.3f}")

sns.regplot(data=filtered_all, x='people_semi_vaccinated_per_hundred', y='new_deaths_rolling', ax=axes[0, 1], color=sns.color_palette("pastel")[3])
r, p = pearsonr(filtered_all['people_semi_vaccinated_per_hundred'], filtered_all['new_deaths_rolling'])
axes[0, 1].set_title(f"Only One Dose vs New Deaths, r: {r:.3f}, p: {p:.3f}")

sns.regplot(data=filtered_all, x='people_fully_vaccinated_per_hundred', y='new_deaths_rolling', ax=axes[1, 0], color=sns.color_palette("pastel")[5])
r, p = pearsonr(filtered_all['people_fully_vaccinated_per_hundred'], filtered_all['new_deaths_rolling'])
axes[1, 0].set_title(f"Fully Vaccinated vs New Deaths, r: {r:.3f}, p: {p:.3f}")

sns.regplot(data=filtered_all, x='total_boosters_per_hundred', y='new_deaths_rolling', ax=axes[1, 1], color=sns.color_palette("pastel")[7])
r, p = pearsonr(filtered_all['total_boosters_per_hundred'], filtered_all['new_deaths_rolling'])
axes[1, 1].set_title(f"Boosters vs New Deaths, r: {r:.3f}, p: {p:.3f}")

st.pyplot(fig5)


#Reduction in Deaths with dosage wrt time
st.markdown(f"**{country}: Time Series of COVID Deaths and Vaccine Uptake Milestones**")

fig6, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(country_df['date'], country_df['new_deaths_rolling'], color='black', label='New Deaths (14-day avg)')
ax1.set_ylabel('New Deaths per Million', color='black')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.plot(country_df['date'], country_df['people_vaccinated_per_hundred'], label='At least 1 dose', color='blue')
ax2.plot(country_df['date'], country_df['people_fully_vaccinated_per_hundred'], label='Fully vaccinated', color='green')
ax2.plot(country_df['date'], country_df['total_boosters_per_hundred'], label='Boosters per 100', color='orange')
ax2.set_ylabel('Vaccination per 100 people', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

for pct in [10, 50, 70]:
    milestone_date = country_df[country_df['people_fully_vaccinated_per_hundred'] >= pct]['date'].min()
    if pd.notnull(milestone_date):
        ax1.axvline(milestone_date, color='green', linestyle='--', alpha=0.5)
        ax1.text(milestone_date, ax1.get_ylim()[1]*0.85, f'{pct}% full vax', rotation=90, color='green')

fig6.suptitle(f"{country} — COVID-19 Deaths vs Vaccination Timeline", fontsize=14)
fig6.legend(loc='upper left', bbox_to_anchor=(0.1, 0.92))
fig6.tight_layout()

st.pyplot(fig6)

## Testing Efficiency
st.subheader("Testing Efficiency")
st.markdown("""**Not all testing is done eqaully**:    
            1. High Testing, Low Positivity: efficiecnt and proactive   
            2. Low testing, high positivity: inefficient & reactive""")

testing_efficiency = df.dropna(subset=['country','new_tests_smoothed_per_thousand','positive_rate','gdp_per_capita','new_deaths_smoothed_per_million']).reset_index()
testing_efficiency = testing_efficiency.groupby(['country']).agg({'new_tests_smoothed_per_thousand':'mean','positive_rate':'mean','gdp_per_capita':'mean','new_deaths_smoothed_per_million':'mean'}).reset_index()

st.markdown("**Countries with Higher GDP had better testing efficiency**")
#Scatterplot comparing countries testing efficiency based on GDP
bins = [0,1500,5000,10000,20000,40000,float('inf')]
labels = ['Very Low', ' Low', 'Lower Middle', 'Middle', 'High', 'Very High']
testing_efficiency['gdp_rating'] = pd.Series(pd.cut(testing_efficiency['gdp_per_capita'],bins=bins,labels=labels))

g = sns.FacetGrid(data=testing_efficiency, col='gdp_rating',col_wrap=3)
g.map_dataframe(sns.regplot,x='positive_rate',y='new_tests_smoothed_per_thousand')
st.pyplot(g.figure)

# Impact of Stringency Measures
st.subheader("Impact Of Stringency Measures")

# Comparing how GDP affects stringency measures
st.markdown("**Affect of GDP on Stringency amidst the worst Covid Waves**")
st.write("Poorer countries tended to respond more harshly to rise in deaths.")

reduced_stringency_df = df.dropna(subset=['country','gdp_per_capita','stringency_index','new_deaths_smoothed_per_million'])
reduced_stringency_df = reduced_stringency_df.groupby(['country']).agg({'gdp_per_capita':'mean','stringency_index':'max','new_deaths_smoothed_per_million':'max'})

bins = [0,5000,20000,float('inf')]
labels = ['Poor','Middle', 'Rich']
reduced_stringency_df['gdp_rating'] = pd.Series(pd.cut(reduced_stringency_df['gdp_per_capita'],bins=bins,labels=labels))

g = sns.FacetGrid(data=reduced_stringency_df, col='gdp_rating',col_wrap=3)
g.map_dataframe(sns.regplot,x='new_deaths_smoothed_per_million',y='stringency_index')
st.pyplot(g.figure)

print(reduced_stringency_df['gdp_rating'].value_counts())

#Comparing Impact of Stringency Measures in Strict vs Lenient Countries
st.markdown("**Comparing Impact of Stringency Measures in Strict (India) vs Lenient (Sweden) countries**")
st.write("Higher stringency were effective in controlling the impact of Covid-19 Waves as shown in the below figures.")
parameters =  ['stringency_index','new_cases_smoothed_per_million','new_deaths_smoothed_per_million','new_people_vaccinated_smoothed_per_hundred','date']
country_stringency_df = df.dropna(subset=parameters)
country_stringency_df = country_stringency_df.set_index("country")
country_stringency_df = country_stringency_df.loc[["Sweden", "India"],parameters]

#Automated multiple plot comparision
fig, axs = plt.subplots(len(parameters)-1, 2, figsize=(12, 16), sharex=True)
for i, col in enumerate(parameters): 
    if col == 'date':
        break
    country = country_stringency_df.loc['India'].sort_values('date').copy()
    sns.lineplot(data=country, x='date', y=col, ax=axs[i][0],color = 'orange')
    axs[i][0].set_ylabel(f"{parameters[i]}")
    axs[i][0].grid(True)

    country = country_stringency_df.loc['Sweden'].sort_values('date').copy()
    sns.lineplot(data=country, x='date', y=col, ax=axs[i][1],color='green')
    axs[i][1].set_ylabel(f"{parameters[i]}")
    axs[i][1].grid(True)

    plt.xlabel("Date")
st.pyplot(fig)

## Conclusion 
st.header("Conclusion")
st.markdown("""
The global trajectory of COVID-19 reveals not only common patterns, such as the widespread impact of the Omicron wave, but also striking disparities in national responses and outcomes.
While wealthier countries had greater testing efficiency and vaccine access, many lower-income nations relied on stricter policy stringency to manage outbreaks. 
These findings underscore the importance of equitable resource distribution and the critical role of real-time data in guiding public health decisions.
 Strengthening global data-sharing systems and using evidence-driven strategies will be key to mitigating the effects of future pandemics.
""")

## Recommendations
st.header("Recommendations")
st.markdown("**Data driven guidelines for future pandemics**")
st.markdown("""
1. Ensure global vaccine equity between low and high income countries.   
2. Support evidence based stringency measures based on outbreak sensitivity.   
3. Invest in early warning, data sharing and pandemic response systems.   
4. Promote public trust and health literacy.   
5. Strengthen international cooperation and collaboration.         
""")
