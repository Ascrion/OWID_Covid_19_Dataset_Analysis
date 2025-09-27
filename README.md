# Numbers Behind the Crisis: The COVID-19 Report

An interactive Streamlit dashboard that transforms complex pandemic data into clear, actionable insights. This project analyzes global COVID-19 trends through data visualization and statistical analysis, helping us understand what worked, what didn't, and how we can better prepare for future health crises.

## What This Project Does

Rather than just presenting raw numbers, this dashboard tells the story of the pandemic through data. It reveals patterns in how different countries responded to COVID-19, shows the real impact of vaccination campaigns, and uncovers the relationship between economic resources and health outcomes.

## Abstract

### Objective

* Analyze global COVID-19 trends using the Our World in Data (OWID) dataset spanning 2020-2023
* Examine key pandemic metrics: cases, deaths, testing efficiency, vaccination impact, and government policy stringency
* Compare pandemic responses and outcomes between high-GDP versus low-GDP countries
* Assess vaccine effectiveness across different dosage levels and timing of implementation
* Identify data-driven insights for improved future pandemic preparedness and response strategies

### Methodology

* **Data Source**: OWID COVID-19 dataset with global pandemic metrics across 200+ countries
* **Preprocessing**: Applied interpolation methods to handle missing values, converted dates to datetime format
* **Statistical Analysis**: Used Pearson correlation analysis to quantify vaccine-death rate relationships
* **Geographical Analysis**: Performed continent-level data aggregation to track pandemic wave progression
* **Economic Classification**: Grouped countries by GDP per capita to analyze resource-outcome relationships
* **Policy Comparison**: Contrasted strict (India) versus lenient (Sweden) policy approaches using stringency indices
* **Visualization Tools**: Implemented matplotlib, seaborn, and geopandas for comprehensive data visualization
* **Time-Series Analysis**: Applied rolling averages and trend analysis to identify key pandemic phases

### Key Findings

* **Wave Patterns**: Omicron variant created the highest global case surge, spreading systematically from Europe → North America → rest of world
* **Vaccine Effectiveness**: Strong negative correlation (r = -0.65 to -0.8) between vaccination rates and COVID-19 death rates across all analyzed countries
* **Dosage Impact**: Full vaccination demonstrated superior death prevention compared to single-dose regimens, with booster shots providing additional protection
* **Economic Disparities**: Higher GDP countries achieved better testing efficiency with lower test positivity rates (5-10% vs 15-25% in low-GDP countries)
* **Policy Responses**: Lower-income countries implemented stricter lockdown measures (stringency index 80-90) compared to wealthier nations (60-70)
* **Geographic Spread**: Each major variant wave showed distinct geographic progression patterns, with 2-4 week delays between continents
* **Testing Efficiency**: Countries with GDP >$20,000 showed proactive testing strategies, while GDP <$5,000 countries exhibited reactive patterns

## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed, then install the required packages:

```bash
pip install streamlit pandas matplotlib seaborn geopandas scipy
```

### Running the Dashboard

1. Clone this repository or download the files
2. Make sure you have your `covid_data.csv` file in the same directory
3. Run the dashboard:

```bash
streamlit run data_analyzer.py
```

4. Open your browser to `http://localhost:8501`

### Data Requirements

You'll need the OWID COVID-19 dataset saved as `covid_data.csv`. The dashboard expects these key columns:
* `country`, `date`, `continent`, `code`
* `new_cases`, `total_cases_per_million`, `new_deaths_smoothed_per_million`
* `people_vaccinated_per_hundred`, `people_fully_vaccinated_per_hundred`
* `gdp_per_capita`, `stringency_index`, `positive_rate`

## What You'll Discover

### Interactive Visualizations
* **Global heat maps** showing case distribution across countries
* **Time-series analysis** revealing pandemic waves and their geographic spread
* **Correlation plots** demonstrating vaccine effectiveness
* **Economic impact comparisons** between different GDP groups

### Key Insights
* How vaccination campaigns actually saved lives (with statistical proof)
* Why some countries handled testing better than others
* The relationship between wealth and pandemic outcomes
* Which policy approaches worked best

### Country-Specific Analysis
Choose from multiple countries to see detailed breakdowns of:
* Vaccination rollout timelines
* Death rate correlations
* Policy stringency impacts

## Technical Features

* **Memory optimization** with efficient data types and caching
* **Error handling** for robust performance with large datasets
* **Interactive elements** including country selection and performance metrics
* **Responsive design** that works across different screen sizes

## Conclusion

The global trajectory of COVID-19 reveals not only common patterns, such as the widespread impact of the Omicron wave, but also striking disparities in national responses and outcomes. While wealthier countries had greater testing efficiency and vaccine access, many lower-income nations relied on stricter policy stringency to manage outbreaks.

These findings underscore the importance of equitable resource distribution and the critical role of real-time data in guiding public health decisions. Strengthening global data-sharing systems and using evidence-driven strategies will be key to mitigating the effects of future pandemics.

### Key Takeaways
* **Vaccine Impact**: Vaccination campaigns significantly reduced COVID-19 mortality across all income levels and geographic regions, with full vaccination providing optimal protection
* **Economic Inequality**: Substantial disparities in pandemic outcomes correlated directly with national economic resources and healthcare infrastructure capacity
* **Policy Effectiveness**: Data-driven government responses (combining moderate stringency with high testing) proved more effective than purely restrictive or permissive approaches
* **Global Coordination**: Future pandemic preparedness requires enhanced international cooperation, equitable vaccine distribution, and real-time data sharing systems

## Data Source & Credits

### Data Source
* **Primary Dataset**: Our World in Data (OWID) COVID-19 Dataset
* **API Documentation**: https://docs.owid.io/projects/etl/api/covid/
* **Data Portal**: https://ourworldindata.org/coronavirus
* **Update Frequency**: Daily updates with comprehensive global coverage

### Citation
Edouard Mathieu, Hannah Ritchie, Lucas Rodés-Guirao, Cameron Appel, Daniel Gavrilov, Charlie Giattino, Joe Hasell, Bobbie Macdonald, Saloni Dattani, Diana Beltekian, Esteban Ortiz-Ospina, and Max Roser (2020) - "COVID-19 Pandemic" Published online at OurWorldinData.org. Retrieved from: 'https://ourworldindata.org/coronavirus' [Online Resource]

### Data License & Usage
* **License**: Creative Commons BY license
* **Attribution**: Our World in Data is committed to making data freely available
* **Quality Assurance**: Data undergoes rigorous quality checks and standardization processes
* **Transparency**: All sources and methodologies are publicly documented

## Contributing

This project is designed to be educational and informative. If you find issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project follows the same open-source spirit as Our World in Data. Feel free to use, modify, and share this code for educational and research purposes.

---