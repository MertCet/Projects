# Denmark Property Valuation Analysis

## Research Question

We are set to explore the relationship between socioeconomic factors and property valuations within the context of Denmark's municipalities. Our research is split into two main questions:
1. How can we predict high-demand areas and property value, both historically and presently?
2. How can we model and forecast property demand trends over the next 5 years?

## Data Sources

Our analysis will leverage data from various sources:

- **Property Listing Data**
    - Sites such as Home.dk, Boliga.dk, and Boligsiden.dk
    - Acquisition method: Webscraping

- **Historical Property Data**
    - Public sources like Vurderingsstyrelsen, BBR, and DAWA
    - Acquisition method: API

- **Social/Economic Data**
    - Historical data from sources such as Danmarks Statistik (DST) and DinGeo

## Data Analysis & Methodology

The project's data analysis will adhere to a typical analytical pipeline:
1. **Data Collection**: Tools such as web-scraping and fetching data via API will be used.
2. **Data Preparation & Processing**: Conducted using Python.
3. **Exploratory Data Analysis (EDA)**: To comprehend the nature of the regressions we'll be dealing with.
4. **Machine Learning (ML) Analysis**: Our primary tool for predicting property values based on aggregated socioeconomic data. Depending on data distribution, we'll pick a suitable regression model. K-fold cross validation will be employed to optimize our training and test sets. Additionally, we'll explore L1, L2, or elastic net regressions based on multicollinearity in our dataset.
5. **Sharing & Visualization**: Our findings will be represented through intuitive visualizations and concise data tables or dashboards.
