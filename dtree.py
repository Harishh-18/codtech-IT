import pandas as pd
import plotly.express as px

# Sample data (replace this with actual census data)
data = {
    'Country': ['China', 'India', 'United States', 'Indonesia', 'Pakistan', 'Brazil', 'Nigeria', 'Bangladesh', 'Russia', 'Mexico'],
    'Population': [1444216107, 1393409038, 331893745, 273523621, 220892331, 212559409, 206139587, 166303498, 145912025, 128932753]
}

df = pd.DataFrame(data)

# Use Plotly Express to generate a choropleth map
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="Population",
    hover_name="Country",
    color_continuous_scale="Viridis",
    title="World Population Census by Country"
)

fig.show()
