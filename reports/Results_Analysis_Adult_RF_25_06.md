```python
import pandas as pd
pd.set_option('display.max_colwidth', None) 
from parser_main import analyze_results 

rules_log = "adult_random_forest_rules_36-20_06-06-2025.txt"
entries_log = "adult_random_forest_entries_36-20_06-06-2025.txt"
analyze_results(rules_log, entries_log, path="../experiments_log/", min_cov = 0.01, min_cov_class = 0.01, min_pre = 0.01)
```


## Instance 5988 (Original: <=50K , Predicted: <=50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>41.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>Some-college</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>10</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Married-civ-spouse</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Farming-fishing</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Husband</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>16.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 5988



<style type="text/css">
</style>
<table id="T_cc918">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cc918_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_cc918_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_cc918_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_cc918_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_cc918_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_cc918_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_cc918_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_cc918_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_cc918_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_cc918_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_cc918_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cc918_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_cc918_row0_col0" class="data row0 col0" >5988</td>
      <td id="T_cc918_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_cc918_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_cc918_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row0_col4" class="data row0 col4" >0.02874</td>
      <td id="T_cc918_row0_col5" class="data row0 col5" >0.03416</td>
      <td id="T_cc918_row0_col6" class="data row0 col6" >0.90229</td>
      <td id="T_cc918_row0_col7" class="data row0 col7" >2</td>
      <td id="T_cc918_row0_col8" class="data row0 col8" >0</td>
      <td id="T_cc918_row0_col9" class="data row0 col9" >1.22434</td>
      <td id="T_cc918_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_cc918_row1_col0" class="data row1 col0" >5988</td>
      <td id="T_cc918_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_cc918_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_cc918_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row1_col4" class="data row1 col4" >0.02751</td>
      <td id="T_cc918_row1_col5" class="data row1 col5" >0.03317</td>
      <td id="T_cc918_row1_col6" class="data row1 col6" >0.91547</td>
      <td id="T_cc918_row1_col7" class="data row1 col7" >2</td>
      <td id="T_cc918_row1_col8" class="data row1 col8" >0</td>
      <td id="T_cc918_row1_col9" class="data row1 col9" >1.39099</td>
      <td id="T_cc918_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_cc918_row2_col0" class="data row2 col0" >5988</td>
      <td id="T_cc918_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_cc918_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_cc918_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_cc918_row2_col4" class="data row2 col4" >0.65843</td>
      <td id="T_cc918_row2_col5" class="data row2 col5" >0.73952</td>
      <td id="T_cc918_row2_col6" class="data row2 col6" >0.85267</td>
      <td id="T_cc918_row2_col7" class="data row2 col7" >2</td>
      <td id="T_cc918_row2_col8" class="data row2 col8" >0</td>
      <td id="T_cc918_row2_col9" class="data row2 col9" >1.23656</td>
      <td id="T_cc918_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_cc918_row3_col0" class="data row3 col0" >5988</td>
      <td id="T_cc918_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_cc918_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_cc918_row3_col3" class="data row3 col3" >IF age <= 47.0 AND education = Some-college THEN class = <=50K</td>
      <td id="T_cc918_row3_col4" class="data row3 col4" >0.17791</td>
      <td id="T_cc918_row3_col5" class="data row3 col5" >0.19835</td>
      <td id="T_cc918_row3_col6" class="data row3 col6" >0.84636</td>
      <td id="T_cc918_row3_col7" class="data row3 col7" >2</td>
      <td id="T_cc918_row3_col8" class="data row3 col8" >1</td>
      <td id="T_cc918_row3_col9" class="data row3 col9" >2.50399</td>
      <td id="T_cc918_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_cc918_row4_col0" class="data row4 col0" >5988</td>
      <td id="T_cc918_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_cc918_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_cc918_row4_col3" class="data row4 col3" >IF education = Some-college AND education.num = 10.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_cc918_row4_col4" class="data row4 col4" >0.16567</td>
      <td id="T_cc918_row4_col5" class="data row4 col5" >0.18748</td>
      <td id="T_cc918_row4_col6" class="data row4 col6" >0.85911</td>
      <td id="T_cc918_row4_col7" class="data row4 col7" >3</td>
      <td id="T_cc918_row4_col8" class="data row4 col8" >0</td>
      <td id="T_cc918_row4_col9" class="data row4 col9" >1.49170</td>
      <td id="T_cc918_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_cc918_row5_col0" class="data row5 col0" >5988</td>
      <td id="T_cc918_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_cc918_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_cc918_row5_col3" class="data row5 col3" >IF age <= 41.0 AND capital.gain <= 7298.0 AND education.num = 10.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row5_col4" class="data row5 col4" >0.00325</td>
      <td id="T_cc918_row5_col5" class="data row5 col5" >0.00381</td>
      <td id="T_cc918_row5_col6" class="data row5 col6" >0.89189</td>
      <td id="T_cc918_row5_col7" class="data row5 col7" >4</td>
      <td id="T_cc918_row5_col8" class="data row5 col8" >0</td>
      <td id="T_cc918_row5_col9" class="data row5 col9" >61.02343</td>
      <td id="T_cc918_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_cc918_row6_col0" class="data row6 col0" >5988</td>
      <td id="T_cc918_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_cc918_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_cc918_row6_col3" class="data row6 col3" >IF age > 40.0 AND capital.gain <= 3103.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row6_col4" class="data row6 col4" >0.01272</td>
      <td id="T_cc918_row6_col5" class="data row6 col5" >0.01474</td>
      <td id="T_cc918_row6_col6" class="data row6 col6" >0.87931</td>
      <td id="T_cc918_row6_col7" class="data row6 col7" >3</td>
      <td id="T_cc918_row6_col8" class="data row6 col8" >0</td>
      <td id="T_cc918_row6_col9" class="data row6 col9" >54.10267</td>
      <td id="T_cc918_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_cc918_row7_col0" class="data row7 col0" >5988</td>
      <td id="T_cc918_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_cc918_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_cc918_row7_col3" class="data row7 col3" >IF capital.gain <= 3418.0 AND race = White THEN class = <=50K</td>
      <td id="T_cc918_row7_col4" class="data row7 col4" >0.79642</td>
      <td id="T_cc918_row7_col5" class="data row7 col5" >0.82026</td>
      <td id="T_cc918_row7_col6" class="data row7 col6" >0.78190</td>
      <td id="T_cc918_row7_col7" class="data row7 col7" >2</td>
      <td id="T_cc918_row7_col8" class="data row7 col8" >0</td>
      <td id="T_cc918_row7_col9" class="data row7 col9" >54.59050</td>
      <td id="T_cc918_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_cc918_row8_col0" class="data row8 col0" >5988</td>
      <td id="T_cc918_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_cc918_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_cc918_row8_col3" class="data row8 col3" >IF capital.gain <= 3103.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_cc918_row8_col4" class="data row8 col4" >0.70788</td>
      <td id="T_cc918_row8_col5" class="data row8 col5" >0.76536</td>
      <td id="T_cc918_row8_col6" class="data row8 col6" >0.82081</td>
      <td id="T_cc918_row8_col7" class="data row8 col7" >2</td>
      <td id="T_cc918_row8_col8" class="data row8 col8" >0</td>
      <td id="T_cc918_row8_col9" class="data row8 col9" >55.85134</td>
      <td id="T_cc918_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_cc918_row9_col0" class="data row9 col0" >5988</td>
      <td id="T_cc918_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_cc918_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_cc918_row9_col3" class="data row9 col3" >IF age > 40.0 AND capital.gain <= 2202.0 AND hours.per.week > 10.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row9_col4" class="data row9 col4" >0.01220</td>
      <td id="T_cc918_row9_col5" class="data row9 col5" >0.01410</td>
      <td id="T_cc918_row9_col6" class="data row9 col6" >0.87770</td>
      <td id="T_cc918_row9_col7" class="data row9 col7" >4</td>
      <td id="T_cc918_row9_col8" class="data row9 col8" >0</td>
      <td id="T_cc918_row9_col9" class="data row9 col9" >55.55250</td>
      <td id="T_cc918_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_cc918_row10_col0" class="data row10 col0" >5988</td>
      <td id="T_cc918_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_cc918_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_cc918_row10_col3" class="data row10 col3" >IF capital.gain <= 5323.3428 AND capital.loss <= 2182.0904 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row10_col4" class="data row10 col4" >0.02861</td>
      <td id="T_cc918_row10_col5" class="data row10 col5" >0.03462</td>
      <td id="T_cc918_row10_col6" class="data row10 col6" >0.91871</td>
      <td id="T_cc918_row10_col7" class="data row10 col7" >3</td>
      <td id="T_cc918_row10_col8" class="data row10 col8" >0</td>
      <td id="T_cc918_row10_col9" class="data row10 col9" >10.82846</td>
      <td id="T_cc918_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_cc918_row11_col0" class="data row11 col0" >5988</td>
      <td id="T_cc918_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_cc918_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_cc918_row11_col3" class="data row11 col3" >IF capital.gain <= 6524.3019 AND capital.loss <= 3446.3383 AND workclass = Private THEN class = <=50K</td>
      <td id="T_cc918_row11_col4" class="data row11 col4" >0.72332</td>
      <td id="T_cc918_row11_col5" class="data row11 col5" >0.77940</td>
      <td id="T_cc918_row11_col6" class="data row11 col6" >0.81803</td>
      <td id="T_cc918_row11_col7" class="data row11 col7" >3</td>
      <td id="T_cc918_row11_col8" class="data row11 col8" >0</td>
      <td id="T_cc918_row11_col9" class="data row11 col9" >11.50588</td>
      <td id="T_cc918_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_cc918_row12_col0" class="data row12 col0" >5988</td>
      <td id="T_cc918_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_cc918_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_cc918_row12_col3" class="data row12 col3" >IF capital.gain <= 3061.4681 AND education != Masters AND education != Doctorate AND occupation = Farming-fishing AND race != Black THEN class = <=50K</td>
      <td id="T_cc918_row12_col4" class="data row12 col4" >0.02659</td>
      <td id="T_cc918_row12_col5" class="data row12 col5" >0.03213</td>
      <td id="T_cc918_row12_col6" class="data row12 col6" >0.91749</td>
      <td id="T_cc918_row12_col7" class="data row12 col7" >5</td>
      <td id="T_cc918_row12_col8" class="data row12 col8" >0</td>
      <td id="T_cc918_row12_col9" class="data row12 col9" >11.16233</td>
      <td id="T_cc918_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_cc918_row13_col0" class="data row13 col0" >5988</td>
      <td id="T_cc918_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_cc918_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_cc918_row13_col3" class="data row13 col3" >IF capital.gain <= 6101.2144 AND education != Masters AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row13_col4" class="data row13 col4" >0.02861</td>
      <td id="T_cc918_row13_col5" class="data row13 col5" >0.03462</td>
      <td id="T_cc918_row13_col6" class="data row13 col6" >0.91871</td>
      <td id="T_cc918_row13_col7" class="data row13 col7" >3</td>
      <td id="T_cc918_row13_col8" class="data row13 col8" >0</td>
      <td id="T_cc918_row13_col9" class="data row13 col9" >10.96197</td>
      <td id="T_cc918_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_cc918_row14_col0" class="data row14 col0" >5988</td>
      <td id="T_cc918_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_cc918_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_cc918_row14_col3" class="data row14 col3" >IF capital.gain <= 7638.066 AND education.num != 7.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_cc918_row14_col4" class="data row14 col4" >0.02821</td>
      <td id="T_cc918_row14_col5" class="data row14 col5" >0.03375</td>
      <td id="T_cc918_row14_col6" class="data row14 col6" >0.90824</td>
      <td id="T_cc918_row14_col7" class="data row14 col7" >3</td>
      <td id="T_cc918_row14_col8" class="data row14 col8" >0</td>
      <td id="T_cc918_row14_col9" class="data row14 col9" >11.18833</td>
      <td id="T_cc918_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_cc918_row15_col0" class="data row15 col0" >5988</td>
      <td id="T_cc918_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_cc918_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_cc918_row15_col3" class="data row15 col3" >IF age <= 46.9906 AND capital.gain <= 0.0 AND capital.loss <= 1763.671 AND hours.per.week <= 35.7909 THEN class = <=50K</td>
      <td id="T_cc918_row15_col4" class="data row15 col4" >0.14567</td>
      <td id="T_cc918_row15_col5" class="data row15 col5" >0.18344</td>
      <td id="T_cc918_row15_col6" class="data row15 col6" >0.95602</td>
      <td id="T_cc918_row15_col7" class="data row15 col7" >4</td>
      <td id="T_cc918_row15_col8" class="data row15 col8" >0</td>
      <td id="T_cc918_row15_col9" class="data row15 col9" >1.94383</td>
      <td id="T_cc918_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_cc918_row16_col0" class="data row16 col0" >5988</td>
      <td id="T_cc918_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_cc918_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_cc918_row16_col3" class="data row16 col3" >IF capital.gain <= 6762.3186 AND hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_cc918_row16_col4" class="data row16 col4" >0.20845</td>
      <td id="T_cc918_row16_col5" class="data row16 col5" >0.25510</td>
      <td id="T_cc918_row16_col6" class="data row16 col6" >0.92907</td>
      <td id="T_cc918_row16_col7" class="data row16 col7" >2</td>
      <td id="T_cc918_row16_col8" class="data row16 col8" >0</td>
      <td id="T_cc918_row16_col9" class="data row16 col9" >0.86854</td>
      <td id="T_cc918_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_cc918_row17_col0" class="data row17 col0" >5988</td>
      <td id="T_cc918_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_cc918_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_cc918_row17_col3" class="data row17 col3" >IF capital.gain <= 3103.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_cc918_row17_col4" class="data row17 col4" >0.16734</td>
      <td id="T_cc918_row17_col5" class="data row17 col5" >0.20783</td>
      <td id="T_cc918_row17_col6" class="data row17 col6" >0.94284</td>
      <td id="T_cc918_row17_col7" class="data row17 col7" >2</td>
      <td id="T_cc918_row17_col8" class="data row17 col8" >0</td>
      <td id="T_cc918_row17_col9" class="data row17 col9" >0.87909</td>
      <td id="T_cc918_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_cc918_row18_col0" class="data row18 col0" >5988</td>
      <td id="T_cc918_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_cc918_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_cc918_row18_col3" class="data row18 col3" >IF capital.gain <= 2580.0 AND hours.per.week <= 30.0 THEN class = <=50K</td>
      <td id="T_cc918_row18_col4" class="data row18 col4" >0.15703</td>
      <td id="T_cc918_row18_col5" class="data row18 col5" >0.19540</td>
      <td id="T_cc918_row18_col6" class="data row18 col6" >0.94468</td>
      <td id="T_cc918_row18_col7" class="data row18 col7" >2</td>
      <td id="T_cc918_row18_col8" class="data row18 col8" >0</td>
      <td id="T_cc918_row18_col9" class="data row18 col9" >0.76491</td>
      <td id="T_cc918_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cc918_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_cc918_row19_col0" class="data row19 col0" >5988</td>
      <td id="T_cc918_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_cc918_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_cc918_row19_col3" class="data row19 col3" >IF capital.gain <= 3137.0 AND hours.per.week <= 32.7865 THEN class = <=50K</td>
      <td id="T_cc918_row19_col4" class="data row19 col4" >0.16620</td>
      <td id="T_cc918_row19_col5" class="data row19 col5" >0.20638</td>
      <td id="T_cc918_row19_col6" class="data row19 col6" >0.94271</td>
      <td id="T_cc918_row19_col7" class="data row19 col7" >2</td>
      <td id="T_cc918_row19_col8" class="data row19 col8" >0</td>
      <td id="T_cc918_row19_col9" class="data row19 col9" >0.87083</td>
      <td id="T_cc918_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5988, Correct Prediction



<style type="text/css">
</style>
<table id="T_b8e3b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b8e3b_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_b8e3b_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_b8e3b_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_b8e3b_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_b8e3b_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_b8e3b_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_b8e3b_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_b8e3b_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_b8e3b_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_b8e3b_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_b8e3b_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b8e3b_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b8e3b_row0_col0" class="data row0 col0" >5988</td>
      <td id="T_b8e3b_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_b8e3b_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_b8e3b_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row0_col4" class="data row0 col4" >0.02874</td>
      <td id="T_b8e3b_row0_col5" class="data row0 col5" >0.03416</td>
      <td id="T_b8e3b_row0_col6" class="data row0 col6" >0.90229</td>
      <td id="T_b8e3b_row0_col7" class="data row0 col7" >2</td>
      <td id="T_b8e3b_row0_col8" class="data row0 col8" >0</td>
      <td id="T_b8e3b_row0_col9" class="data row0 col9" >1.22434</td>
      <td id="T_b8e3b_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_b8e3b_row1_col0" class="data row1 col0" >5988</td>
      <td id="T_b8e3b_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_b8e3b_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_b8e3b_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row1_col4" class="data row1 col4" >0.02751</td>
      <td id="T_b8e3b_row1_col5" class="data row1 col5" >0.03317</td>
      <td id="T_b8e3b_row1_col6" class="data row1 col6" >0.91547</td>
      <td id="T_b8e3b_row1_col7" class="data row1 col7" >2</td>
      <td id="T_b8e3b_row1_col8" class="data row1 col8" >0</td>
      <td id="T_b8e3b_row1_col9" class="data row1 col9" >1.39099</td>
      <td id="T_b8e3b_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_b8e3b_row2_col0" class="data row2 col0" >5988</td>
      <td id="T_b8e3b_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_b8e3b_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_b8e3b_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_b8e3b_row2_col4" class="data row2 col4" >0.65843</td>
      <td id="T_b8e3b_row2_col5" class="data row2 col5" >0.73952</td>
      <td id="T_b8e3b_row2_col6" class="data row2 col6" >0.85267</td>
      <td id="T_b8e3b_row2_col7" class="data row2 col7" >2</td>
      <td id="T_b8e3b_row2_col8" class="data row2 col8" >0</td>
      <td id="T_b8e3b_row2_col9" class="data row2 col9" >1.23656</td>
      <td id="T_b8e3b_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_b8e3b_row3_col0" class="data row3 col0" >5988</td>
      <td id="T_b8e3b_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_b8e3b_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_b8e3b_row3_col3" class="data row3 col3" >IF age <= 47.0 AND education = Some-college THEN class = <=50K</td>
      <td id="T_b8e3b_row3_col4" class="data row3 col4" >0.17791</td>
      <td id="T_b8e3b_row3_col5" class="data row3 col5" >0.19835</td>
      <td id="T_b8e3b_row3_col6" class="data row3 col6" >0.84636</td>
      <td id="T_b8e3b_row3_col7" class="data row3 col7" >2</td>
      <td id="T_b8e3b_row3_col8" class="data row3 col8" >1</td>
      <td id="T_b8e3b_row3_col9" class="data row3 col9" >2.50399</td>
      <td id="T_b8e3b_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_b8e3b_row4_col0" class="data row4 col0" >5988</td>
      <td id="T_b8e3b_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_b8e3b_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_b8e3b_row4_col3" class="data row4 col3" >IF education = Some-college AND education.num = 10.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_b8e3b_row4_col4" class="data row4 col4" >0.16567</td>
      <td id="T_b8e3b_row4_col5" class="data row4 col5" >0.18748</td>
      <td id="T_b8e3b_row4_col6" class="data row4 col6" >0.85911</td>
      <td id="T_b8e3b_row4_col7" class="data row4 col7" >3</td>
      <td id="T_b8e3b_row4_col8" class="data row4 col8" >0</td>
      <td id="T_b8e3b_row4_col9" class="data row4 col9" >1.49170</td>
      <td id="T_b8e3b_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_b8e3b_row5_col0" class="data row5 col0" >5988</td>
      <td id="T_b8e3b_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_b8e3b_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_b8e3b_row5_col3" class="data row5 col3" >IF age <= 41.0 AND capital.gain <= 7298.0 AND education.num = 10.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row5_col4" class="data row5 col4" >0.00325</td>
      <td id="T_b8e3b_row5_col5" class="data row5 col5" >0.00381</td>
      <td id="T_b8e3b_row5_col6" class="data row5 col6" >0.89189</td>
      <td id="T_b8e3b_row5_col7" class="data row5 col7" >4</td>
      <td id="T_b8e3b_row5_col8" class="data row5 col8" >0</td>
      <td id="T_b8e3b_row5_col9" class="data row5 col9" >61.02343</td>
      <td id="T_b8e3b_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_b8e3b_row6_col0" class="data row6 col0" >5988</td>
      <td id="T_b8e3b_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_b8e3b_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_b8e3b_row6_col3" class="data row6 col3" >IF age > 40.0 AND capital.gain <= 3103.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row6_col4" class="data row6 col4" >0.01272</td>
      <td id="T_b8e3b_row6_col5" class="data row6 col5" >0.01474</td>
      <td id="T_b8e3b_row6_col6" class="data row6 col6" >0.87931</td>
      <td id="T_b8e3b_row6_col7" class="data row6 col7" >3</td>
      <td id="T_b8e3b_row6_col8" class="data row6 col8" >0</td>
      <td id="T_b8e3b_row6_col9" class="data row6 col9" >54.10267</td>
      <td id="T_b8e3b_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_b8e3b_row7_col0" class="data row7 col0" >5988</td>
      <td id="T_b8e3b_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_b8e3b_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_b8e3b_row7_col3" class="data row7 col3" >IF capital.gain <= 3418.0 AND race = White THEN class = <=50K</td>
      <td id="T_b8e3b_row7_col4" class="data row7 col4" >0.79642</td>
      <td id="T_b8e3b_row7_col5" class="data row7 col5" >0.82026</td>
      <td id="T_b8e3b_row7_col6" class="data row7 col6" >0.78190</td>
      <td id="T_b8e3b_row7_col7" class="data row7 col7" >2</td>
      <td id="T_b8e3b_row7_col8" class="data row7 col8" >0</td>
      <td id="T_b8e3b_row7_col9" class="data row7 col9" >54.59050</td>
      <td id="T_b8e3b_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_b8e3b_row8_col0" class="data row8 col0" >5988</td>
      <td id="T_b8e3b_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_b8e3b_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_b8e3b_row8_col3" class="data row8 col3" >IF capital.gain <= 3103.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_b8e3b_row8_col4" class="data row8 col4" >0.70788</td>
      <td id="T_b8e3b_row8_col5" class="data row8 col5" >0.76536</td>
      <td id="T_b8e3b_row8_col6" class="data row8 col6" >0.82081</td>
      <td id="T_b8e3b_row8_col7" class="data row8 col7" >2</td>
      <td id="T_b8e3b_row8_col8" class="data row8 col8" >0</td>
      <td id="T_b8e3b_row8_col9" class="data row8 col9" >55.85134</td>
      <td id="T_b8e3b_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_b8e3b_row9_col0" class="data row9 col0" >5988</td>
      <td id="T_b8e3b_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_b8e3b_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_b8e3b_row9_col3" class="data row9 col3" >IF age > 40.0 AND capital.gain <= 2202.0 AND hours.per.week > 10.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row9_col4" class="data row9 col4" >0.01220</td>
      <td id="T_b8e3b_row9_col5" class="data row9 col5" >0.01410</td>
      <td id="T_b8e3b_row9_col6" class="data row9 col6" >0.87770</td>
      <td id="T_b8e3b_row9_col7" class="data row9 col7" >4</td>
      <td id="T_b8e3b_row9_col8" class="data row9 col8" >0</td>
      <td id="T_b8e3b_row9_col9" class="data row9 col9" >55.55250</td>
      <td id="T_b8e3b_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_b8e3b_row10_col0" class="data row10 col0" >5988</td>
      <td id="T_b8e3b_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_b8e3b_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_b8e3b_row10_col3" class="data row10 col3" >IF capital.gain <= 5323.3428 AND capital.loss <= 2182.0904 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row10_col4" class="data row10 col4" >0.02861</td>
      <td id="T_b8e3b_row10_col5" class="data row10 col5" >0.03462</td>
      <td id="T_b8e3b_row10_col6" class="data row10 col6" >0.91871</td>
      <td id="T_b8e3b_row10_col7" class="data row10 col7" >3</td>
      <td id="T_b8e3b_row10_col8" class="data row10 col8" >0</td>
      <td id="T_b8e3b_row10_col9" class="data row10 col9" >10.82846</td>
      <td id="T_b8e3b_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_b8e3b_row11_col0" class="data row11 col0" >5988</td>
      <td id="T_b8e3b_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_b8e3b_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_b8e3b_row11_col3" class="data row11 col3" >IF capital.gain <= 6524.3019 AND capital.loss <= 3446.3383 AND workclass = Private THEN class = <=50K</td>
      <td id="T_b8e3b_row11_col4" class="data row11 col4" >0.72332</td>
      <td id="T_b8e3b_row11_col5" class="data row11 col5" >0.77940</td>
      <td id="T_b8e3b_row11_col6" class="data row11 col6" >0.81803</td>
      <td id="T_b8e3b_row11_col7" class="data row11 col7" >3</td>
      <td id="T_b8e3b_row11_col8" class="data row11 col8" >0</td>
      <td id="T_b8e3b_row11_col9" class="data row11 col9" >11.50588</td>
      <td id="T_b8e3b_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_b8e3b_row12_col0" class="data row12 col0" >5988</td>
      <td id="T_b8e3b_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_b8e3b_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_b8e3b_row12_col3" class="data row12 col3" >IF capital.gain <= 3061.4681 AND education != Masters AND education != Doctorate AND occupation = Farming-fishing AND race != Black THEN class = <=50K</td>
      <td id="T_b8e3b_row12_col4" class="data row12 col4" >0.02659</td>
      <td id="T_b8e3b_row12_col5" class="data row12 col5" >0.03213</td>
      <td id="T_b8e3b_row12_col6" class="data row12 col6" >0.91749</td>
      <td id="T_b8e3b_row12_col7" class="data row12 col7" >5</td>
      <td id="T_b8e3b_row12_col8" class="data row12 col8" >0</td>
      <td id="T_b8e3b_row12_col9" class="data row12 col9" >11.16233</td>
      <td id="T_b8e3b_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_b8e3b_row13_col0" class="data row13 col0" >5988</td>
      <td id="T_b8e3b_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_b8e3b_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_b8e3b_row13_col3" class="data row13 col3" >IF capital.gain <= 6101.2144 AND education != Masters AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row13_col4" class="data row13 col4" >0.02861</td>
      <td id="T_b8e3b_row13_col5" class="data row13 col5" >0.03462</td>
      <td id="T_b8e3b_row13_col6" class="data row13 col6" >0.91871</td>
      <td id="T_b8e3b_row13_col7" class="data row13 col7" >3</td>
      <td id="T_b8e3b_row13_col8" class="data row13 col8" >0</td>
      <td id="T_b8e3b_row13_col9" class="data row13 col9" >10.96197</td>
      <td id="T_b8e3b_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_b8e3b_row14_col0" class="data row14 col0" >5988</td>
      <td id="T_b8e3b_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_b8e3b_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_b8e3b_row14_col3" class="data row14 col3" >IF capital.gain <= 7638.066 AND education.num != 7.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_b8e3b_row14_col4" class="data row14 col4" >0.02821</td>
      <td id="T_b8e3b_row14_col5" class="data row14 col5" >0.03375</td>
      <td id="T_b8e3b_row14_col6" class="data row14 col6" >0.90824</td>
      <td id="T_b8e3b_row14_col7" class="data row14 col7" >3</td>
      <td id="T_b8e3b_row14_col8" class="data row14 col8" >0</td>
      <td id="T_b8e3b_row14_col9" class="data row14 col9" >11.18833</td>
      <td id="T_b8e3b_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_b8e3b_row15_col0" class="data row15 col0" >5988</td>
      <td id="T_b8e3b_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_b8e3b_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_b8e3b_row15_col3" class="data row15 col3" >IF age <= 46.9906 AND capital.gain <= 0.0 AND capital.loss <= 1763.671 AND hours.per.week <= 35.7909 THEN class = <=50K</td>
      <td id="T_b8e3b_row15_col4" class="data row15 col4" >0.14567</td>
      <td id="T_b8e3b_row15_col5" class="data row15 col5" >0.18344</td>
      <td id="T_b8e3b_row15_col6" class="data row15 col6" >0.95602</td>
      <td id="T_b8e3b_row15_col7" class="data row15 col7" >4</td>
      <td id="T_b8e3b_row15_col8" class="data row15 col8" >0</td>
      <td id="T_b8e3b_row15_col9" class="data row15 col9" >1.94383</td>
      <td id="T_b8e3b_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_b8e3b_row16_col0" class="data row16 col0" >5988</td>
      <td id="T_b8e3b_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_b8e3b_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_b8e3b_row16_col3" class="data row16 col3" >IF capital.gain <= 6762.3186 AND hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_b8e3b_row16_col4" class="data row16 col4" >0.20845</td>
      <td id="T_b8e3b_row16_col5" class="data row16 col5" >0.25510</td>
      <td id="T_b8e3b_row16_col6" class="data row16 col6" >0.92907</td>
      <td id="T_b8e3b_row16_col7" class="data row16 col7" >2</td>
      <td id="T_b8e3b_row16_col8" class="data row16 col8" >0</td>
      <td id="T_b8e3b_row16_col9" class="data row16 col9" >0.86854</td>
      <td id="T_b8e3b_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_b8e3b_row17_col0" class="data row17 col0" >5988</td>
      <td id="T_b8e3b_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_b8e3b_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_b8e3b_row17_col3" class="data row17 col3" >IF capital.gain <= 3103.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_b8e3b_row17_col4" class="data row17 col4" >0.16734</td>
      <td id="T_b8e3b_row17_col5" class="data row17 col5" >0.20783</td>
      <td id="T_b8e3b_row17_col6" class="data row17 col6" >0.94284</td>
      <td id="T_b8e3b_row17_col7" class="data row17 col7" >2</td>
      <td id="T_b8e3b_row17_col8" class="data row17 col8" >0</td>
      <td id="T_b8e3b_row17_col9" class="data row17 col9" >0.87909</td>
      <td id="T_b8e3b_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_b8e3b_row18_col0" class="data row18 col0" >5988</td>
      <td id="T_b8e3b_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_b8e3b_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_b8e3b_row18_col3" class="data row18 col3" >IF capital.gain <= 2580.0 AND hours.per.week <= 30.0 THEN class = <=50K</td>
      <td id="T_b8e3b_row18_col4" class="data row18 col4" >0.15703</td>
      <td id="T_b8e3b_row18_col5" class="data row18 col5" >0.19540</td>
      <td id="T_b8e3b_row18_col6" class="data row18 col6" >0.94468</td>
      <td id="T_b8e3b_row18_col7" class="data row18 col7" >2</td>
      <td id="T_b8e3b_row18_col8" class="data row18 col8" >0</td>
      <td id="T_b8e3b_row18_col9" class="data row18 col9" >0.76491</td>
      <td id="T_b8e3b_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b8e3b_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_b8e3b_row19_col0" class="data row19 col0" >5988</td>
      <td id="T_b8e3b_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_b8e3b_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_b8e3b_row19_col3" class="data row19 col3" >IF capital.gain <= 3137.0 AND hours.per.week <= 32.7865 THEN class = <=50K</td>
      <td id="T_b8e3b_row19_col4" class="data row19 col4" >0.16620</td>
      <td id="T_b8e3b_row19_col5" class="data row19 col5" >0.20638</td>
      <td id="T_b8e3b_row19_col6" class="data row19 col6" >0.94271</td>
      <td id="T_b8e3b_row19_col7" class="data row19 col7" >2</td>
      <td id="T_b8e3b_row19_col8" class="data row19 col8" >0</td>
      <td id="T_b8e3b_row19_col9" class="data row19 col9" >0.87083</td>
      <td id="T_b8e3b_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5988, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_f5c33">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f5c33_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_f5c33_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_f5c33_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_f5c33_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_f5c33_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_f5c33_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_f5c33_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_f5c33_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_f5c33_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_f5c33_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_f5c33_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f5c33_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f5c33_row0_col0" class="data row0 col0" >5988</td>
      <td id="T_f5c33_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_f5c33_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_f5c33_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_f5c33_row0_col4" class="data row0 col4" >0.02874</td>
      <td id="T_f5c33_row0_col5" class="data row0 col5" >0.03416</td>
      <td id="T_f5c33_row0_col6" class="data row0 col6" >0.90229</td>
      <td id="T_f5c33_row0_col7" class="data row0 col7" >2</td>
      <td id="T_f5c33_row0_col8" class="data row0 col8" >0</td>
      <td id="T_f5c33_row0_col9" class="data row0 col9" >1.22434</td>
      <td id="T_f5c33_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f5c33_row1_col0" class="data row1 col0" >5988</td>
      <td id="T_f5c33_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_f5c33_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_f5c33_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_f5c33_row1_col4" class="data row1 col4" >0.02751</td>
      <td id="T_f5c33_row1_col5" class="data row1 col5" >0.03317</td>
      <td id="T_f5c33_row1_col6" class="data row1 col6" >0.91547</td>
      <td id="T_f5c33_row1_col7" class="data row1 col7" >2</td>
      <td id="T_f5c33_row1_col8" class="data row1 col8" >0</td>
      <td id="T_f5c33_row1_col9" class="data row1 col9" >1.39099</td>
      <td id="T_f5c33_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f5c33_row2_col0" class="data row2 col0" >5988</td>
      <td id="T_f5c33_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_f5c33_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_f5c33_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_f5c33_row2_col4" class="data row2 col4" >0.65843</td>
      <td id="T_f5c33_row2_col5" class="data row2 col5" >0.73952</td>
      <td id="T_f5c33_row2_col6" class="data row2 col6" >0.85267</td>
      <td id="T_f5c33_row2_col7" class="data row2 col7" >2</td>
      <td id="T_f5c33_row2_col8" class="data row2 col8" >0</td>
      <td id="T_f5c33_row2_col9" class="data row2 col9" >1.23656</td>
      <td id="T_f5c33_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f5c33_row3_col0" class="data row3 col0" >5988</td>
      <td id="T_f5c33_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_f5c33_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_f5c33_row3_col3" class="data row3 col3" >IF age <= 47.0 AND education = Some-college THEN class = <=50K</td>
      <td id="T_f5c33_row3_col4" class="data row3 col4" >0.17791</td>
      <td id="T_f5c33_row3_col5" class="data row3 col5" >0.19835</td>
      <td id="T_f5c33_row3_col6" class="data row3 col6" >0.84636</td>
      <td id="T_f5c33_row3_col7" class="data row3 col7" >2</td>
      <td id="T_f5c33_row3_col8" class="data row3 col8" >1</td>
      <td id="T_f5c33_row3_col9" class="data row3 col9" >2.50399</td>
      <td id="T_f5c33_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f5c33_row4_col0" class="data row4 col0" >5988</td>
      <td id="T_f5c33_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_f5c33_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_f5c33_row4_col3" class="data row4 col3" >IF education = Some-college AND education.num = 10.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_f5c33_row4_col4" class="data row4 col4" >0.16567</td>
      <td id="T_f5c33_row4_col5" class="data row4 col5" >0.18748</td>
      <td id="T_f5c33_row4_col6" class="data row4 col6" >0.85911</td>
      <td id="T_f5c33_row4_col7" class="data row4 col7" >3</td>
      <td id="T_f5c33_row4_col8" class="data row4 col8" >0</td>
      <td id="T_f5c33_row4_col9" class="data row4 col9" >1.49170</td>
      <td id="T_f5c33_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f5c33_row5_col0" class="data row5 col0" >5988</td>
      <td id="T_f5c33_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_f5c33_row5_col2" class="data row5 col2" >LORE2</td>
      <td id="T_f5c33_row5_col3" class="data row5 col3" >IF age > 40.0 AND capital.gain <= 3103.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_f5c33_row5_col4" class="data row5 col4" >0.01272</td>
      <td id="T_f5c33_row5_col5" class="data row5 col5" >0.01474</td>
      <td id="T_f5c33_row5_col6" class="data row5 col6" >0.87931</td>
      <td id="T_f5c33_row5_col7" class="data row5 col7" >3</td>
      <td id="T_f5c33_row5_col8" class="data row5 col8" >0</td>
      <td id="T_f5c33_row5_col9" class="data row5 col9" >54.10267</td>
      <td id="T_f5c33_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_f5c33_row6_col0" class="data row6 col0" >5988</td>
      <td id="T_f5c33_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_f5c33_row6_col2" class="data row6 col2" >LORE3</td>
      <td id="T_f5c33_row6_col3" class="data row6 col3" >IF capital.gain <= 3418.0 AND race = White THEN class = <=50K</td>
      <td id="T_f5c33_row6_col4" class="data row6 col4" >0.79642</td>
      <td id="T_f5c33_row6_col5" class="data row6 col5" >0.82026</td>
      <td id="T_f5c33_row6_col6" class="data row6 col6" >0.78190</td>
      <td id="T_f5c33_row6_col7" class="data row6 col7" >2</td>
      <td id="T_f5c33_row6_col8" class="data row6 col8" >0</td>
      <td id="T_f5c33_row6_col9" class="data row6 col9" >54.59050</td>
      <td id="T_f5c33_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_f5c33_row7_col0" class="data row7 col0" >5988</td>
      <td id="T_f5c33_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_f5c33_row7_col2" class="data row7 col2" >LORE4</td>
      <td id="T_f5c33_row7_col3" class="data row7 col3" >IF capital.gain <= 3103.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_f5c33_row7_col4" class="data row7 col4" >0.70788</td>
      <td id="T_f5c33_row7_col5" class="data row7 col5" >0.76536</td>
      <td id="T_f5c33_row7_col6" class="data row7 col6" >0.82081</td>
      <td id="T_f5c33_row7_col7" class="data row7 col7" >2</td>
      <td id="T_f5c33_row7_col8" class="data row7 col8" >0</td>
      <td id="T_f5c33_row7_col9" class="data row7 col9" >55.85134</td>
      <td id="T_f5c33_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_f5c33_row8_col0" class="data row8 col0" >5988</td>
      <td id="T_f5c33_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_f5c33_row8_col2" class="data row8 col2" >LORE5</td>
      <td id="T_f5c33_row8_col3" class="data row8 col3" >IF age > 40.0 AND capital.gain <= 2202.0 AND hours.per.week > 10.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_f5c33_row8_col4" class="data row8 col4" >0.01220</td>
      <td id="T_f5c33_row8_col5" class="data row8 col5" >0.01410</td>
      <td id="T_f5c33_row8_col6" class="data row8 col6" >0.87770</td>
      <td id="T_f5c33_row8_col7" class="data row8 col7" >4</td>
      <td id="T_f5c33_row8_col8" class="data row8 col8" >0</td>
      <td id="T_f5c33_row8_col9" class="data row8 col9" >55.55250</td>
      <td id="T_f5c33_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_f5c33_row9_col0" class="data row9 col0" >5988</td>
      <td id="T_f5c33_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_f5c33_row9_col2" class="data row9 col2" >LORE_SA1</td>
      <td id="T_f5c33_row9_col3" class="data row9 col3" >IF capital.gain <= 5323.3428 AND capital.loss <= 2182.0904 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_f5c33_row9_col4" class="data row9 col4" >0.02861</td>
      <td id="T_f5c33_row9_col5" class="data row9 col5" >0.03462</td>
      <td id="T_f5c33_row9_col6" class="data row9 col6" >0.91871</td>
      <td id="T_f5c33_row9_col7" class="data row9 col7" >3</td>
      <td id="T_f5c33_row9_col8" class="data row9 col8" >0</td>
      <td id="T_f5c33_row9_col9" class="data row9 col9" >10.82846</td>
      <td id="T_f5c33_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_f5c33_row10_col0" class="data row10 col0" >5988</td>
      <td id="T_f5c33_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_f5c33_row10_col2" class="data row10 col2" >LORE_SA2</td>
      <td id="T_f5c33_row10_col3" class="data row10 col3" >IF capital.gain <= 6524.3019 AND capital.loss <= 3446.3383 AND workclass = Private THEN class = <=50K</td>
      <td id="T_f5c33_row10_col4" class="data row10 col4" >0.72332</td>
      <td id="T_f5c33_row10_col5" class="data row10 col5" >0.77940</td>
      <td id="T_f5c33_row10_col6" class="data row10 col6" >0.81803</td>
      <td id="T_f5c33_row10_col7" class="data row10 col7" >3</td>
      <td id="T_f5c33_row10_col8" class="data row10 col8" >0</td>
      <td id="T_f5c33_row10_col9" class="data row10 col9" >11.50588</td>
      <td id="T_f5c33_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_f5c33_row11_col0" class="data row11 col0" >5988</td>
      <td id="T_f5c33_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_f5c33_row11_col2" class="data row11 col2" >LORE_SA3</td>
      <td id="T_f5c33_row11_col3" class="data row11 col3" >IF capital.gain <= 3061.4681 AND education != Masters AND education != Doctorate AND occupation = Farming-fishing AND race != Black THEN class = <=50K</td>
      <td id="T_f5c33_row11_col4" class="data row11 col4" >0.02659</td>
      <td id="T_f5c33_row11_col5" class="data row11 col5" >0.03213</td>
      <td id="T_f5c33_row11_col6" class="data row11 col6" >0.91749</td>
      <td id="T_f5c33_row11_col7" class="data row11 col7" >5</td>
      <td id="T_f5c33_row11_col8" class="data row11 col8" >0</td>
      <td id="T_f5c33_row11_col9" class="data row11 col9" >11.16233</td>
      <td id="T_f5c33_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_f5c33_row12_col0" class="data row12 col0" >5988</td>
      <td id="T_f5c33_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_f5c33_row12_col2" class="data row12 col2" >LORE_SA4</td>
      <td id="T_f5c33_row12_col3" class="data row12 col3" >IF capital.gain <= 6101.2144 AND education != Masters AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_f5c33_row12_col4" class="data row12 col4" >0.02861</td>
      <td id="T_f5c33_row12_col5" class="data row12 col5" >0.03462</td>
      <td id="T_f5c33_row12_col6" class="data row12 col6" >0.91871</td>
      <td id="T_f5c33_row12_col7" class="data row12 col7" >3</td>
      <td id="T_f5c33_row12_col8" class="data row12 col8" >0</td>
      <td id="T_f5c33_row12_col9" class="data row12 col9" >10.96197</td>
      <td id="T_f5c33_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_f5c33_row13_col0" class="data row13 col0" >5988</td>
      <td id="T_f5c33_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_f5c33_row13_col2" class="data row13 col2" >LORE_SA5</td>
      <td id="T_f5c33_row13_col3" class="data row13 col3" >IF capital.gain <= 7638.066 AND education.num != 7.0 AND occupation = Farming-fishing THEN class = <=50K</td>
      <td id="T_f5c33_row13_col4" class="data row13 col4" >0.02821</td>
      <td id="T_f5c33_row13_col5" class="data row13 col5" >0.03375</td>
      <td id="T_f5c33_row13_col6" class="data row13 col6" >0.90824</td>
      <td id="T_f5c33_row13_col7" class="data row13 col7" >3</td>
      <td id="T_f5c33_row13_col8" class="data row13 col8" >0</td>
      <td id="T_f5c33_row13_col9" class="data row13 col9" >11.18833</td>
      <td id="T_f5c33_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_f5c33_row14_col0" class="data row14 col0" >5988</td>
      <td id="T_f5c33_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_f5c33_row14_col2" class="data row14 col2" >EXPLAN1</td>
      <td id="T_f5c33_row14_col3" class="data row14 col3" >IF age <= 46.9906 AND capital.gain <= 0.0 AND capital.loss <= 1763.671 AND hours.per.week <= 35.7909 THEN class = <=50K</td>
      <td id="T_f5c33_row14_col4" class="data row14 col4" >0.14567</td>
      <td id="T_f5c33_row14_col5" class="data row14 col5" >0.18344</td>
      <td id="T_f5c33_row14_col6" class="data row14 col6" >0.95602</td>
      <td id="T_f5c33_row14_col7" class="data row14 col7" >4</td>
      <td id="T_f5c33_row14_col8" class="data row14 col8" >0</td>
      <td id="T_f5c33_row14_col9" class="data row14 col9" >1.94383</td>
      <td id="T_f5c33_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_f5c33_row15_col0" class="data row15 col0" >5988</td>
      <td id="T_f5c33_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_f5c33_row15_col2" class="data row15 col2" >EXPLAN2</td>
      <td id="T_f5c33_row15_col3" class="data row15 col3" >IF capital.gain <= 6762.3186 AND hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_f5c33_row15_col4" class="data row15 col4" >0.20845</td>
      <td id="T_f5c33_row15_col5" class="data row15 col5" >0.25510</td>
      <td id="T_f5c33_row15_col6" class="data row15 col6" >0.92907</td>
      <td id="T_f5c33_row15_col7" class="data row15 col7" >2</td>
      <td id="T_f5c33_row15_col8" class="data row15 col8" >0</td>
      <td id="T_f5c33_row15_col9" class="data row15 col9" >0.86854</td>
      <td id="T_f5c33_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_f5c33_row16_col0" class="data row16 col0" >5988</td>
      <td id="T_f5c33_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_f5c33_row16_col2" class="data row16 col2" >EXPLAN3</td>
      <td id="T_f5c33_row16_col3" class="data row16 col3" >IF capital.gain <= 3103.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_f5c33_row16_col4" class="data row16 col4" >0.16734</td>
      <td id="T_f5c33_row16_col5" class="data row16 col5" >0.20783</td>
      <td id="T_f5c33_row16_col6" class="data row16 col6" >0.94284</td>
      <td id="T_f5c33_row16_col7" class="data row16 col7" >2</td>
      <td id="T_f5c33_row16_col8" class="data row16 col8" >0</td>
      <td id="T_f5c33_row16_col9" class="data row16 col9" >0.87909</td>
      <td id="T_f5c33_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_f5c33_row17_col0" class="data row17 col0" >5988</td>
      <td id="T_f5c33_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_f5c33_row17_col2" class="data row17 col2" >EXPLAN4</td>
      <td id="T_f5c33_row17_col3" class="data row17 col3" >IF capital.gain <= 2580.0 AND hours.per.week <= 30.0 THEN class = <=50K</td>
      <td id="T_f5c33_row17_col4" class="data row17 col4" >0.15703</td>
      <td id="T_f5c33_row17_col5" class="data row17 col5" >0.19540</td>
      <td id="T_f5c33_row17_col6" class="data row17 col6" >0.94468</td>
      <td id="T_f5c33_row17_col7" class="data row17 col7" >2</td>
      <td id="T_f5c33_row17_col8" class="data row17 col8" >0</td>
      <td id="T_f5c33_row17_col9" class="data row17 col9" >0.76491</td>
      <td id="T_f5c33_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_f5c33_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_f5c33_row18_col0" class="data row18 col0" >5988</td>
      <td id="T_f5c33_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_f5c33_row18_col2" class="data row18 col2" >EXPLAN5</td>
      <td id="T_f5c33_row18_col3" class="data row18 col3" >IF capital.gain <= 3137.0 AND hours.per.week <= 32.7865 THEN class = <=50K</td>
      <td id="T_f5c33_row18_col4" class="data row18 col4" >0.16620</td>
      <td id="T_f5c33_row18_col5" class="data row18 col5" >0.20638</td>
      <td id="T_f5c33_row18_col6" class="data row18 col6" >0.94271</td>
      <td id="T_f5c33_row18_col7" class="data row18 col7" >2</td>
      <td id="T_f5c33_row18_col8" class="data row18 col8" >0</td>
      <td id="T_f5c33_row18_col9" class="data row18 col9" >0.87083</td>
      <td id="T_f5c33_row18_col10" class="data row18 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5988, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.79642, Pre: 0.95602)



<style type="text/css">
#T_e967e_row3_col0, #T_e967e_row3_col1, #T_e967e_row3_col2, #T_e967e_row3_col3, #T_e967e_row3_col4, #T_e967e_row3_col5, #T_e967e_row3_col6, #T_e967e_row3_col7, #T_e967e_row3_col8, #T_e967e_row3_col9, #T_e967e_row3_col10, #T_e967e_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_e967e">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e967e_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e967e_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e967e_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e967e_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e967e_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e967e_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e967e_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e967e_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e967e_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e967e_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e967e_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_e967e_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e967e_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e967e_row0_col0" class="data row0 col0" >5988</td>
      <td id="T_e967e_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_e967e_row0_col2" class="data row0 col2" >ANCHOR3</td>
      <td id="T_e967e_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_e967e_row0_col4" class="data row0 col4" >0.65843</td>
      <td id="T_e967e_row0_col5" class="data row0 col5" >0.73952</td>
      <td id="T_e967e_row0_col6" class="data row0 col6" >0.85267</td>
      <td id="T_e967e_row0_col7" class="data row0 col7" >2</td>
      <td id="T_e967e_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e967e_row0_col9" class="data row0 col9" >1.23656</td>
      <td id="T_e967e_row0_col10" class="data row0 col10" >False</td>
      <td id="T_e967e_row0_col11" class="data row0 col11" >0.17240</td>
    </tr>
    <tr>
      <th id="T_e967e_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e967e_row1_col0" class="data row1 col0" >5988</td>
      <td id="T_e967e_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_e967e_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_e967e_row1_col3" class="data row1 col3" >IF capital.gain <= 3418.0 AND race = White THEN class = <=50K</td>
      <td id="T_e967e_row1_col4" class="data row1 col4" >0.79642</td>
      <td id="T_e967e_row1_col5" class="data row1 col5" >0.82026</td>
      <td id="T_e967e_row1_col6" class="data row1 col6" >0.78190</td>
      <td id="T_e967e_row1_col7" class="data row1 col7" >2</td>
      <td id="T_e967e_row1_col8" class="data row1 col8" >0</td>
      <td id="T_e967e_row1_col9" class="data row1 col9" >54.59050</td>
      <td id="T_e967e_row1_col10" class="data row1 col10" >False</td>
      <td id="T_e967e_row1_col11" class="data row1 col11" >0.17412</td>
    </tr>
    <tr>
      <th id="T_e967e_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e967e_row2_col0" class="data row2 col0" >5988</td>
      <td id="T_e967e_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_e967e_row2_col2" class="data row2 col2" >LORE4</td>
      <td id="T_e967e_row2_col3" class="data row2 col3" >IF capital.gain <= 3103.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e967e_row2_col4" class="data row2 col4" >0.70788</td>
      <td id="T_e967e_row2_col5" class="data row2 col5" >0.76536</td>
      <td id="T_e967e_row2_col6" class="data row2 col6" >0.82081</td>
      <td id="T_e967e_row2_col7" class="data row2 col7" >2</td>
      <td id="T_e967e_row2_col8" class="data row2 col8" >0</td>
      <td id="T_e967e_row2_col9" class="data row2 col9" >55.85134</td>
      <td id="T_e967e_row2_col10" class="data row2 col10" >False</td>
      <td id="T_e967e_row2_col11" class="data row2 col11" >0.16162</td>
    </tr>
    <tr>
      <th id="T_e967e_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e967e_row3_col0" class="data row3 col0" >5988</td>
      <td id="T_e967e_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_e967e_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_e967e_row3_col3" class="data row3 col3" >IF capital.gain <= 6524.3019 AND capital.loss <= 3446.3383 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e967e_row3_col4" class="data row3 col4" >0.72332</td>
      <td id="T_e967e_row3_col5" class="data row3 col5" >0.77940</td>
      <td id="T_e967e_row3_col6" class="data row3 col6" >0.81803</td>
      <td id="T_e967e_row3_col7" class="data row3 col7" >3</td>
      <td id="T_e967e_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e967e_row3_col9" class="data row3 col9" >11.50588</td>
      <td id="T_e967e_row3_col10" class="data row3 col10" >False</td>
      <td id="T_e967e_row3_col11" class="data row3 col11" >0.15616</td>
    </tr>
    <tr>
      <th id="T_e967e_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e967e_row4_col0" class="data row4 col0" >5988</td>
      <td id="T_e967e_row4_col1" class="data row4 col1" >EXPLAN</td>
      <td id="T_e967e_row4_col2" class="data row4 col2" >EXPLAN1</td>
      <td id="T_e967e_row4_col3" class="data row4 col3" >IF age <= 46.9906 AND capital.gain <= 0.0 AND capital.loss <= 1763.671 AND hours.per.week <= 35.7909 THEN class = <=50K</td>
      <td id="T_e967e_row4_col4" class="data row4 col4" >0.14567</td>
      <td id="T_e967e_row4_col5" class="data row4 col5" >0.18344</td>
      <td id="T_e967e_row4_col6" class="data row4 col6" >0.95602</td>
      <td id="T_e967e_row4_col7" class="data row4 col7" >4</td>
      <td id="T_e967e_row4_col8" class="data row4 col8" >0</td>
      <td id="T_e967e_row4_col9" class="data row4 col9" >1.94383</td>
      <td id="T_e967e_row4_col10" class="data row4 col10" >False</td>
      <td id="T_e967e_row4_col11" class="data row4 col11" >0.65075</td>
    </tr>
    <tr>
      <th id="T_e967e_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e967e_row5_col0" class="data row5 col0" >5988</td>
      <td id="T_e967e_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_e967e_row5_col2" class="data row5 col2" >EXPLAN2</td>
      <td id="T_e967e_row5_col3" class="data row5 col3" >IF capital.gain <= 6762.3186 AND hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_e967e_row5_col4" class="data row5 col4" >0.20845</td>
      <td id="T_e967e_row5_col5" class="data row5 col5" >0.25510</td>
      <td id="T_e967e_row5_col6" class="data row5 col6" >0.92907</td>
      <td id="T_e967e_row5_col7" class="data row5 col7" >2</td>
      <td id="T_e967e_row5_col8" class="data row5 col8" >0</td>
      <td id="T_e967e_row5_col9" class="data row5 col9" >0.86854</td>
      <td id="T_e967e_row5_col10" class="data row5 col10" >False</td>
      <td id="T_e967e_row5_col11" class="data row5 col11" >0.58859</td>
    </tr>
    <tr>
      <th id="T_e967e_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_e967e_row6_col0" class="data row6 col0" >5988</td>
      <td id="T_e967e_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_e967e_row6_col2" class="data row6 col2" >EXPLAN3</td>
      <td id="T_e967e_row6_col3" class="data row6 col3" >IF capital.gain <= 3103.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_e967e_row6_col4" class="data row6 col4" >0.16734</td>
      <td id="T_e967e_row6_col5" class="data row6 col5" >0.20783</td>
      <td id="T_e967e_row6_col6" class="data row6 col6" >0.94284</td>
      <td id="T_e967e_row6_col7" class="data row6 col7" >2</td>
      <td id="T_e967e_row6_col8" class="data row6 col8" >0</td>
      <td id="T_e967e_row6_col9" class="data row6 col9" >0.87909</td>
      <td id="T_e967e_row6_col10" class="data row6 col10" >False</td>
      <td id="T_e967e_row6_col11" class="data row6 col11" >0.62922</td>
    </tr>
    <tr>
      <th id="T_e967e_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_e967e_row7_col0" class="data row7 col0" >5988</td>
      <td id="T_e967e_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_e967e_row7_col2" class="data row7 col2" >EXPLAN4</td>
      <td id="T_e967e_row7_col3" class="data row7 col3" >IF capital.gain <= 2580.0 AND hours.per.week <= 30.0 THEN class = <=50K</td>
      <td id="T_e967e_row7_col4" class="data row7 col4" >0.15703</td>
      <td id="T_e967e_row7_col5" class="data row7 col5" >0.19540</td>
      <td id="T_e967e_row7_col6" class="data row7 col6" >0.94468</td>
      <td id="T_e967e_row7_col7" class="data row7 col7" >2</td>
      <td id="T_e967e_row7_col8" class="data row7 col8" >0</td>
      <td id="T_e967e_row7_col9" class="data row7 col9" >0.76491</td>
      <td id="T_e967e_row7_col10" class="data row7 col10" >False</td>
      <td id="T_e967e_row7_col11" class="data row7 col11" >0.63949</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_10.png)
    



### Rules for Instance 5988, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.79642, Pre: 0.95602), Unique rules (diffrent features)



<style type="text/css">
#T_bb578_row3_col0, #T_bb578_row3_col1, #T_bb578_row3_col2, #T_bb578_row3_col3, #T_bb578_row3_col4, #T_bb578_row3_col5, #T_bb578_row3_col6, #T_bb578_row3_col7, #T_bb578_row3_col8, #T_bb578_row3_col9, #T_bb578_row3_col10, #T_bb578_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_bb578">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_bb578_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_bb578_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_bb578_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_bb578_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_bb578_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_bb578_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_bb578_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_bb578_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_bb578_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_bb578_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_bb578_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_bb578_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_bb578_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_bb578_row0_col0" class="data row0 col0" >5988</td>
      <td id="T_bb578_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_bb578_row0_col2" class="data row0 col2" >ANCHOR3</td>
      <td id="T_bb578_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_bb578_row0_col4" class="data row0 col4" >0.65843</td>
      <td id="T_bb578_row0_col5" class="data row0 col5" >0.73952</td>
      <td id="T_bb578_row0_col6" class="data row0 col6" >0.85267</td>
      <td id="T_bb578_row0_col7" class="data row0 col7" >2</td>
      <td id="T_bb578_row0_col8" class="data row0 col8" >0</td>
      <td id="T_bb578_row0_col9" class="data row0 col9" >1.23656</td>
      <td id="T_bb578_row0_col10" class="data row0 col10" >False</td>
      <td id="T_bb578_row0_col11" class="data row0 col11" >0.17240</td>
    </tr>
    <tr>
      <th id="T_bb578_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_bb578_row1_col0" class="data row1 col0" >5988</td>
      <td id="T_bb578_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_bb578_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_bb578_row1_col3" class="data row1 col3" >IF capital.gain <= 3418.0 AND race = White THEN class = <=50K</td>
      <td id="T_bb578_row1_col4" class="data row1 col4" >0.79642</td>
      <td id="T_bb578_row1_col5" class="data row1 col5" >0.82026</td>
      <td id="T_bb578_row1_col6" class="data row1 col6" >0.78190</td>
      <td id="T_bb578_row1_col7" class="data row1 col7" >2</td>
      <td id="T_bb578_row1_col8" class="data row1 col8" >0</td>
      <td id="T_bb578_row1_col9" class="data row1 col9" >54.59050</td>
      <td id="T_bb578_row1_col10" class="data row1 col10" >False</td>
      <td id="T_bb578_row1_col11" class="data row1 col11" >0.17412</td>
    </tr>
    <tr>
      <th id="T_bb578_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_bb578_row2_col0" class="data row2 col0" >5988</td>
      <td id="T_bb578_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_bb578_row2_col2" class="data row2 col2" >LORE4</td>
      <td id="T_bb578_row2_col3" class="data row2 col3" >IF capital.gain <= 3103.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_bb578_row2_col4" class="data row2 col4" >0.70788</td>
      <td id="T_bb578_row2_col5" class="data row2 col5" >0.76536</td>
      <td id="T_bb578_row2_col6" class="data row2 col6" >0.82081</td>
      <td id="T_bb578_row2_col7" class="data row2 col7" >2</td>
      <td id="T_bb578_row2_col8" class="data row2 col8" >0</td>
      <td id="T_bb578_row2_col9" class="data row2 col9" >55.85134</td>
      <td id="T_bb578_row2_col10" class="data row2 col10" >False</td>
      <td id="T_bb578_row2_col11" class="data row2 col11" >0.16162</td>
    </tr>
    <tr>
      <th id="T_bb578_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_bb578_row3_col0" class="data row3 col0" >5988</td>
      <td id="T_bb578_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_bb578_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_bb578_row3_col3" class="data row3 col3" >IF capital.gain <= 6524.3019 AND capital.loss <= 3446.3383 AND workclass = Private THEN class = <=50K</td>
      <td id="T_bb578_row3_col4" class="data row3 col4" >0.72332</td>
      <td id="T_bb578_row3_col5" class="data row3 col5" >0.77940</td>
      <td id="T_bb578_row3_col6" class="data row3 col6" >0.81803</td>
      <td id="T_bb578_row3_col7" class="data row3 col7" >3</td>
      <td id="T_bb578_row3_col8" class="data row3 col8" >0</td>
      <td id="T_bb578_row3_col9" class="data row3 col9" >11.50588</td>
      <td id="T_bb578_row3_col10" class="data row3 col10" >False</td>
      <td id="T_bb578_row3_col11" class="data row3 col11" >0.15616</td>
    </tr>
    <tr>
      <th id="T_bb578_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_bb578_row4_col0" class="data row4 col0" >5988</td>
      <td id="T_bb578_row4_col1" class="data row4 col1" >EXPLAN</td>
      <td id="T_bb578_row4_col2" class="data row4 col2" >EXPLAN1</td>
      <td id="T_bb578_row4_col3" class="data row4 col3" >IF age <= 46.9906 AND capital.gain <= 0.0 AND capital.loss <= 1763.671 AND hours.per.week <= 35.7909 THEN class = <=50K</td>
      <td id="T_bb578_row4_col4" class="data row4 col4" >0.14567</td>
      <td id="T_bb578_row4_col5" class="data row4 col5" >0.18344</td>
      <td id="T_bb578_row4_col6" class="data row4 col6" >0.95602</td>
      <td id="T_bb578_row4_col7" class="data row4 col7" >4</td>
      <td id="T_bb578_row4_col8" class="data row4 col8" >0</td>
      <td id="T_bb578_row4_col9" class="data row4 col9" >1.94383</td>
      <td id="T_bb578_row4_col10" class="data row4 col10" >False</td>
      <td id="T_bb578_row4_col11" class="data row4 col11" >0.65075</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_13.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_14.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_15.png)
    



### Rules for Instance 5988, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.82026, Pre: 0.95602, Len: 0.7819)



<style type="text/css">
#T_e8988_row0_col0, #T_e8988_row0_col1, #T_e8988_row0_col2, #T_e8988_row0_col3, #T_e8988_row0_col4, #T_e8988_row0_col5, #T_e8988_row0_col6, #T_e8988_row0_col7, #T_e8988_row0_col8, #T_e8988_row0_col9, #T_e8988_row0_col10, #T_e8988_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_e8988">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e8988_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e8988_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e8988_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e8988_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e8988_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e8988_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e8988_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e8988_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e8988_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e8988_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e8988_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_e8988_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e8988_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e8988_row0_col0" class="data row0 col0" >5988</td>
      <td id="T_e8988_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_e8988_row0_col2" class="data row0 col2" >ANCHOR3</td>
      <td id="T_e8988_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_e8988_row0_col4" class="data row0 col4" >0.65843</td>
      <td id="T_e8988_row0_col5" class="data row0 col5" >0.73952</td>
      <td id="T_e8988_row0_col6" class="data row0 col6" >0.85267</td>
      <td id="T_e8988_row0_col7" class="data row0 col7" >2</td>
      <td id="T_e8988_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e8988_row0_col9" class="data row0 col9" >1.23656</td>
      <td id="T_e8988_row0_col10" class="data row0 col10" >False</td>
      <td id="T_e8988_row0_col11" class="data row0 col11" >1.22514</td>
    </tr>
    <tr>
      <th id="T_e8988_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e8988_row1_col0" class="data row1 col0" >5988</td>
      <td id="T_e8988_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_e8988_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_e8988_row1_col3" class="data row1 col3" >IF capital.gain <= 3418.0 AND race = White THEN class = <=50K</td>
      <td id="T_e8988_row1_col4" class="data row1 col4" >0.79642</td>
      <td id="T_e8988_row1_col5" class="data row1 col5" >0.82026</td>
      <td id="T_e8988_row1_col6" class="data row1 col6" >0.78190</td>
      <td id="T_e8988_row1_col7" class="data row1 col7" >2</td>
      <td id="T_e8988_row1_col8" class="data row1 col8" >0</td>
      <td id="T_e8988_row1_col9" class="data row1 col9" >54.59050</td>
      <td id="T_e8988_row1_col10" class="data row1 col10" >False</td>
      <td id="T_e8988_row1_col11" class="data row1 col11" >1.23048</td>
    </tr>
    <tr>
      <th id="T_e8988_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e8988_row2_col0" class="data row2 col0" >5988</td>
      <td id="T_e8988_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_e8988_row2_col2" class="data row2 col2" >LORE4</td>
      <td id="T_e8988_row2_col3" class="data row2 col3" >IF capital.gain <= 3103.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e8988_row2_col4" class="data row2 col4" >0.70788</td>
      <td id="T_e8988_row2_col5" class="data row2 col5" >0.76536</td>
      <td id="T_e8988_row2_col6" class="data row2 col6" >0.82081</td>
      <td id="T_e8988_row2_col7" class="data row2 col7" >2</td>
      <td id="T_e8988_row2_col8" class="data row2 col8" >0</td>
      <td id="T_e8988_row2_col9" class="data row2 col9" >55.85134</td>
      <td id="T_e8988_row2_col10" class="data row2 col10" >False</td>
      <td id="T_e8988_row2_col11" class="data row2 col11" >1.22681</td>
    </tr>
    <tr>
      <th id="T_e8988_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e8988_row3_col0" class="data row3 col0" >5988</td>
      <td id="T_e8988_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_e8988_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_e8988_row3_col3" class="data row3 col3" >IF capital.gain <= 6524.3019 AND capital.loss <= 3446.3383 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e8988_row3_col4" class="data row3 col4" >0.72332</td>
      <td id="T_e8988_row3_col5" class="data row3 col5" >0.77940</td>
      <td id="T_e8988_row3_col6" class="data row3 col6" >0.81803</td>
      <td id="T_e8988_row3_col7" class="data row3 col7" >3</td>
      <td id="T_e8988_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e8988_row3_col9" class="data row3 col9" >11.50588</td>
      <td id="T_e8988_row3_col10" class="data row3 col10" >False</td>
      <td id="T_e8988_row3_col11" class="data row3 col11" >2.22276</td>
    </tr>
    <tr>
      <th id="T_e8988_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e8988_row4_col0" class="data row4 col0" >5988</td>
      <td id="T_e8988_row4_col1" class="data row4 col1" >EXPLAN</td>
      <td id="T_e8988_row4_col2" class="data row4 col2" >EXPLAN1</td>
      <td id="T_e8988_row4_col3" class="data row4 col3" >IF age <= 46.9906 AND capital.gain <= 0.0 AND capital.loss <= 1763.671 AND hours.per.week <= 35.7909 THEN class = <=50K</td>
      <td id="T_e8988_row4_col4" class="data row4 col4" >0.14567</td>
      <td id="T_e8988_row4_col5" class="data row4 col5" >0.18344</td>
      <td id="T_e8988_row4_col6" class="data row4 col6" >0.95602</td>
      <td id="T_e8988_row4_col7" class="data row4 col7" >4</td>
      <td id="T_e8988_row4_col8" class="data row4 col8" >0</td>
      <td id="T_e8988_row4_col9" class="data row4 col9" >1.94383</td>
      <td id="T_e8988_row4_col10" class="data row4 col10" >False</td>
      <td id="T_e8988_row4_col11" class="data row4 col11" >3.28050</td>
    </tr>
    <tr>
      <th id="T_e8988_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e8988_row5_col0" class="data row5 col0" >5988</td>
      <td id="T_e8988_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_e8988_row5_col2" class="data row5 col2" >EXPLAN2</td>
      <td id="T_e8988_row5_col3" class="data row5 col3" >IF capital.gain <= 6762.3186 AND hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_e8988_row5_col4" class="data row5 col4" >0.20845</td>
      <td id="T_e8988_row5_col5" class="data row5 col5" >0.25510</td>
      <td id="T_e8988_row5_col6" class="data row5 col6" >0.92907</td>
      <td id="T_e8988_row5_col7" class="data row5 col7" >2</td>
      <td id="T_e8988_row5_col8" class="data row5 col8" >0</td>
      <td id="T_e8988_row5_col9" class="data row5 col9" >0.86854</td>
      <td id="T_e8988_row5_col10" class="data row5 col10" >False</td>
      <td id="T_e8988_row5_col11" class="data row5 col11" >1.34309</td>
    </tr>
    <tr>
      <th id="T_e8988_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_e8988_row6_col0" class="data row6 col0" >5988</td>
      <td id="T_e8988_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_e8988_row6_col2" class="data row6 col2" >EXPLAN3</td>
      <td id="T_e8988_row6_col3" class="data row6 col3" >IF capital.gain <= 3103.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_e8988_row6_col4" class="data row6 col4" >0.16734</td>
      <td id="T_e8988_row6_col5" class="data row6 col5" >0.20783</td>
      <td id="T_e8988_row6_col6" class="data row6 col6" >0.94284</td>
      <td id="T_e8988_row6_col7" class="data row6 col7" >2</td>
      <td id="T_e8988_row6_col8" class="data row6 col8" >0</td>
      <td id="T_e8988_row6_col9" class="data row6 col9" >0.87909</td>
      <td id="T_e8988_row6_col10" class="data row6 col10" >False</td>
      <td id="T_e8988_row6_col11" class="data row6 col11" >1.36346</td>
    </tr>
    <tr>
      <th id="T_e8988_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_e8988_row7_col0" class="data row7 col0" >5988</td>
      <td id="T_e8988_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_e8988_row7_col2" class="data row7 col2" >EXPLAN4</td>
      <td id="T_e8988_row7_col3" class="data row7 col3" >IF capital.gain <= 2580.0 AND hours.per.week <= 30.0 THEN class = <=50K</td>
      <td id="T_e8988_row7_col4" class="data row7 col4" >0.15703</td>
      <td id="T_e8988_row7_col5" class="data row7 col5" >0.19540</td>
      <td id="T_e8988_row7_col6" class="data row7 col6" >0.94468</td>
      <td id="T_e8988_row7_col7" class="data row7 col7" >2</td>
      <td id="T_e8988_row7_col8" class="data row7 col8" >0</td>
      <td id="T_e8988_row7_col9" class="data row7 col9" >0.76491</td>
      <td id="T_e8988_row7_col10" class="data row7 col10" >False</td>
      <td id="T_e8988_row7_col11" class="data row7 col11" >1.36907</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5988, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.82026, Pre: 0.95602), Unique rules (diffrent features)



<style type="text/css">
#T_2854b_row0_col0, #T_2854b_row0_col1, #T_2854b_row0_col2, #T_2854b_row0_col3, #T_2854b_row0_col4, #T_2854b_row0_col5, #T_2854b_row0_col6, #T_2854b_row0_col7, #T_2854b_row0_col8, #T_2854b_row0_col9, #T_2854b_row0_col10, #T_2854b_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_2854b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2854b_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_2854b_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_2854b_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_2854b_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_2854b_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_2854b_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_2854b_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_2854b_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_2854b_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_2854b_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_2854b_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_2854b_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2854b_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_2854b_row0_col0" class="data row0 col0" >5988</td>
      <td id="T_2854b_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_2854b_row0_col2" class="data row0 col2" >ANCHOR3</td>
      <td id="T_2854b_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_2854b_row0_col4" class="data row0 col4" >0.65843</td>
      <td id="T_2854b_row0_col5" class="data row0 col5" >0.73952</td>
      <td id="T_2854b_row0_col6" class="data row0 col6" >0.85267</td>
      <td id="T_2854b_row0_col7" class="data row0 col7" >2</td>
      <td id="T_2854b_row0_col8" class="data row0 col8" >0</td>
      <td id="T_2854b_row0_col9" class="data row0 col9" >1.23656</td>
      <td id="T_2854b_row0_col10" class="data row0 col10" >False</td>
      <td id="T_2854b_row0_col11" class="data row0 col11" >1.22514</td>
    </tr>
    <tr>
      <th id="T_2854b_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_2854b_row1_col0" class="data row1 col0" >5988</td>
      <td id="T_2854b_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_2854b_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_2854b_row1_col3" class="data row1 col3" >IF capital.gain <= 3418.0 AND race = White THEN class = <=50K</td>
      <td id="T_2854b_row1_col4" class="data row1 col4" >0.79642</td>
      <td id="T_2854b_row1_col5" class="data row1 col5" >0.82026</td>
      <td id="T_2854b_row1_col6" class="data row1 col6" >0.78190</td>
      <td id="T_2854b_row1_col7" class="data row1 col7" >2</td>
      <td id="T_2854b_row1_col8" class="data row1 col8" >0</td>
      <td id="T_2854b_row1_col9" class="data row1 col9" >54.59050</td>
      <td id="T_2854b_row1_col10" class="data row1 col10" >False</td>
      <td id="T_2854b_row1_col11" class="data row1 col11" >1.23048</td>
    </tr>
    <tr>
      <th id="T_2854b_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_2854b_row2_col0" class="data row2 col0" >5988</td>
      <td id="T_2854b_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_2854b_row2_col2" class="data row2 col2" >LORE4</td>
      <td id="T_2854b_row2_col3" class="data row2 col3" >IF capital.gain <= 3103.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_2854b_row2_col4" class="data row2 col4" >0.70788</td>
      <td id="T_2854b_row2_col5" class="data row2 col5" >0.76536</td>
      <td id="T_2854b_row2_col6" class="data row2 col6" >0.82081</td>
      <td id="T_2854b_row2_col7" class="data row2 col7" >2</td>
      <td id="T_2854b_row2_col8" class="data row2 col8" >0</td>
      <td id="T_2854b_row2_col9" class="data row2 col9" >55.85134</td>
      <td id="T_2854b_row2_col10" class="data row2 col10" >False</td>
      <td id="T_2854b_row2_col11" class="data row2 col11" >1.22681</td>
    </tr>
    <tr>
      <th id="T_2854b_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_2854b_row3_col0" class="data row3 col0" >5988</td>
      <td id="T_2854b_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_2854b_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_2854b_row3_col3" class="data row3 col3" >IF capital.gain <= 6524.3019 AND capital.loss <= 3446.3383 AND workclass = Private THEN class = <=50K</td>
      <td id="T_2854b_row3_col4" class="data row3 col4" >0.72332</td>
      <td id="T_2854b_row3_col5" class="data row3 col5" >0.77940</td>
      <td id="T_2854b_row3_col6" class="data row3 col6" >0.81803</td>
      <td id="T_2854b_row3_col7" class="data row3 col7" >3</td>
      <td id="T_2854b_row3_col8" class="data row3 col8" >0</td>
      <td id="T_2854b_row3_col9" class="data row3 col9" >11.50588</td>
      <td id="T_2854b_row3_col10" class="data row3 col10" >False</td>
      <td id="T_2854b_row3_col11" class="data row3 col11" >2.22276</td>
    </tr>
    <tr>
      <th id="T_2854b_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_2854b_row4_col0" class="data row4 col0" >5988</td>
      <td id="T_2854b_row4_col1" class="data row4 col1" >EXPLAN</td>
      <td id="T_2854b_row4_col2" class="data row4 col2" >EXPLAN1</td>
      <td id="T_2854b_row4_col3" class="data row4 col3" >IF age <= 46.9906 AND capital.gain <= 0.0 AND capital.loss <= 1763.671 AND hours.per.week <= 35.7909 THEN class = <=50K</td>
      <td id="T_2854b_row4_col4" class="data row4 col4" >0.14567</td>
      <td id="T_2854b_row4_col5" class="data row4 col5" >0.18344</td>
      <td id="T_2854b_row4_col6" class="data row4 col6" >0.95602</td>
      <td id="T_2854b_row4_col7" class="data row4 col7" >4</td>
      <td id="T_2854b_row4_col8" class="data row4 col8" >0</td>
      <td id="T_2854b_row4_col9" class="data row4 col9" >1.94383</td>
      <td id="T_2854b_row4_col10" class="data row4 col10" >False</td>
      <td id="T_2854b_row4_col11" class="data row4 col11" >3.28050</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_20.png)
    



## Instance 18073 (Original: <=50K , Predicted: <=50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>19.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>12th</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>8</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Never-married</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Handlers-cleaners</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Own-child</td>
    </tr>
    <tr>
      <th>race</th>
      <td>Black</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>52.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 18073



<style type="text/css">
</style>
<table id="T_ad00b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ad00b_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_ad00b_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_ad00b_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_ad00b_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_ad00b_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_ad00b_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_ad00b_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_ad00b_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_ad00b_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_ad00b_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_ad00b_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ad00b_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_ad00b_row0_col0" class="data row0 col0" >18073</td>
      <td id="T_ad00b_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_ad00b_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_ad00b_row0_col3" class="data row0 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_ad00b_row0_col4" class="data row0 col4" >0.52422</td>
      <td id="T_ad00b_row0_col5" class="data row0 col5" >0.59186</td>
      <td id="T_ad00b_row0_col6" class="data row0 col6" >0.85713</td>
      <td id="T_ad00b_row0_col7" class="data row0 col7" >4</td>
      <td id="T_ad00b_row0_col8" class="data row0 col8" >0</td>
      <td id="T_ad00b_row0_col9" class="data row0 col9" >0.74556</td>
      <td id="T_ad00b_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_ad00b_row1_col0" class="data row1 col0" >18073</td>
      <td id="T_ad00b_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_ad00b_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_ad00b_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_ad00b_row1_col4" class="data row1 col4" >0.66269</td>
      <td id="T_ad00b_row1_col5" class="data row1 col5" >0.73091</td>
      <td id="T_ad00b_row1_col6" class="data row1 col6" >0.83733</td>
      <td id="T_ad00b_row1_col7" class="data row1 col7" >3</td>
      <td id="T_ad00b_row1_col8" class="data row1 col8" >0</td>
      <td id="T_ad00b_row1_col9" class="data row1 col9" >0.63635</td>
      <td id="T_ad00b_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_ad00b_row2_col0" class="data row2 col0" >18073</td>
      <td id="T_ad00b_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_ad00b_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_ad00b_row2_col3" class="data row2 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_ad00b_row2_col4" class="data row2 col4" >0.69595</td>
      <td id="T_ad00b_row2_col5" class="data row2 col5" >0.75357</td>
      <td id="T_ad00b_row2_col6" class="data row2 col6" >0.82203</td>
      <td id="T_ad00b_row2_col7" class="data row2 col7" >2</td>
      <td id="T_ad00b_row2_col8" class="data row2 col8" >0</td>
      <td id="T_ad00b_row2_col9" class="data row2 col9" >0.64159</td>
      <td id="T_ad00b_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_ad00b_row3_col0" class="data row3 col0" >18073</td>
      <td id="T_ad00b_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_ad00b_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_ad00b_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_ad00b_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_ad00b_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_ad00b_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_ad00b_row3_col7" class="data row3 col7" >2</td>
      <td id="T_ad00b_row3_col8" class="data row3 col8" >0</td>
      <td id="T_ad00b_row3_col9" class="data row3 col9" >0.67650</td>
      <td id="T_ad00b_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_ad00b_row4_col0" class="data row4 col0" >18073</td>
      <td id="T_ad00b_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_ad00b_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_ad00b_row4_col3" class="data row4 col3" >IF age <= 37.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_ad00b_row4_col4" class="data row4 col4" >0.46850</td>
      <td id="T_ad00b_row4_col5" class="data row4 col5" >0.55083</td>
      <td id="T_ad00b_row4_col6" class="data row4 col6" >0.89258</td>
      <td id="T_ad00b_row4_col7" class="data row4 col7" >3</td>
      <td id="T_ad00b_row4_col8" class="data row4 col8" >2</td>
      <td id="T_ad00b_row4_col9" class="data row4 col9" >2.08425</td>
      <td id="T_ad00b_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_ad00b_row5_col0" class="data row5 col0" >18073</td>
      <td id="T_ad00b_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_ad00b_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_ad00b_row5_col3" class="data row5 col3" >IF capital.gain <= 5013.0 THEN class = <=50K</td>
      <td id="T_ad00b_row5_col4" class="data row5 col4" >0.95077</td>
      <td id="T_ad00b_row5_col5" class="data row5 col5" >0.99659</td>
      <td id="T_ad00b_row5_col6" class="data row5 col6" >0.79575</td>
      <td id="T_ad00b_row5_col7" class="data row5 col7" >1</td>
      <td id="T_ad00b_row5_col8" class="data row5 col8" >0</td>
      <td id="T_ad00b_row5_col9" class="data row5 col9" >57.71988</td>
      <td id="T_ad00b_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_ad00b_row6_col0" class="data row6 col0" >18073</td>
      <td id="T_ad00b_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_ad00b_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_ad00b_row6_col3" class="data row6 col3" >IF capital.gain <= 7298.0 THEN class = <=50K</td>
      <td id="T_ad00b_row6_col4" class="data row6 col4" >0.96363</td>
      <td id="T_ad00b_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_ad00b_row6_col6" class="data row6 col6" >0.78696</td>
      <td id="T_ad00b_row6_col7" class="data row6 col7" >1</td>
      <td id="T_ad00b_row6_col8" class="data row6 col8" >0</td>
      <td id="T_ad00b_row6_col9" class="data row6 col9" >56.74821</td>
      <td id="T_ad00b_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_ad00b_row7_col0" class="data row7 col0" >18073</td>
      <td id="T_ad00b_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_ad00b_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_ad00b_row7_col3" class="data row7 col3" >IF capital.gain <= 7688.0 THEN class = <=50K</td>
      <td id="T_ad00b_row7_col4" class="data row7 col4" >0.97306</td>
      <td id="T_ad00b_row7_col5" class="data row7 col5" >0.99919</td>
      <td id="T_ad00b_row7_col6" class="data row7 col6" >0.77956</td>
      <td id="T_ad00b_row7_col7" class="data row7 col7" >1</td>
      <td id="T_ad00b_row7_col8" class="data row7 col8" >0</td>
      <td id="T_ad00b_row7_col9" class="data row7 col9" >55.95721</td>
      <td id="T_ad00b_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_ad00b_row8_col0" class="data row8 col0" >18073</td>
      <td id="T_ad00b_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_ad00b_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_ad00b_row8_col3" class="data row8 col3" >IF capital.gain <= 6696.0568 THEN class = <=50K</td>
      <td id="T_ad00b_row8_col4" class="data row8 col4" >0.95534</td>
      <td id="T_ad00b_row8_col5" class="data row8 col5" >0.99775</td>
      <td id="T_ad00b_row8_col6" class="data row8 col6" >0.79287</td>
      <td id="T_ad00b_row8_col7" class="data row8 col7" >1</td>
      <td id="T_ad00b_row8_col8" class="data row8 col8" >0</td>
      <td id="T_ad00b_row8_col9" class="data row8 col9" >56.41765</td>
      <td id="T_ad00b_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_ad00b_row9_col0" class="data row9 col0" >18073</td>
      <td id="T_ad00b_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_ad00b_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_ad00b_row9_col3" class="data row9 col3" >IF capital.gain <= 3464.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_ad00b_row9_col4" class="data row9 col4" >0.32011</td>
      <td id="T_ad00b_row9_col5" class="data row9 col5" >0.40802</td>
      <td id="T_ad00b_row9_col6" class="data row9 col6" >0.96765</td>
      <td id="T_ad00b_row9_col7" class="data row9 col7" >2</td>
      <td id="T_ad00b_row9_col8" class="data row9 col8" >0</td>
      <td id="T_ad00b_row9_col9" class="data row9 col9" >57.99263</td>
      <td id="T_ad00b_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_ad00b_row10_col0" class="data row10 col0" >18073</td>
      <td id="T_ad00b_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_ad00b_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_ad00b_row10_col3" class="data row10 col3" >IF capital.gain <= 7426.083 THEN class = <=50K</td>
      <td id="T_ad00b_row10_col4" class="data row10 col4" >0.96363</td>
      <td id="T_ad00b_row10_col5" class="data row10 col5" >0.99890</td>
      <td id="T_ad00b_row10_col6" class="data row10 col6" >0.78696</td>
      <td id="T_ad00b_row10_col7" class="data row10 col7" >1</td>
      <td id="T_ad00b_row10_col8" class="data row10 col8" >0</td>
      <td id="T_ad00b_row10_col9" class="data row10 col9" >11.73073</td>
      <td id="T_ad00b_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_ad00b_row11_col0" class="data row11 col0" >18073</td>
      <td id="T_ad00b_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_ad00b_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_ad00b_row11_col3" class="data row11 col3" >IF capital.gain <= 5756.5615 THEN class = <=50K</td>
      <td id="T_ad00b_row11_col4" class="data row11 col4" >0.95450</td>
      <td id="T_ad00b_row11_col5" class="data row11 col5" >0.99717</td>
      <td id="T_ad00b_row11_col6" class="data row11 col6" >0.79311</td>
      <td id="T_ad00b_row11_col7" class="data row11 col7" >1</td>
      <td id="T_ad00b_row11_col8" class="data row11 col8" >0</td>
      <td id="T_ad00b_row11_col9" class="data row11 col9" >11.80187</td>
      <td id="T_ad00b_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_ad00b_row12_col0" class="data row12 col0" >18073</td>
      <td id="T_ad00b_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_ad00b_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_ad00b_row12_col3" class="data row12 col3" >IF capital.gain <= 3299.5553 THEN class = <=50K</td>
      <td id="T_ad00b_row12_col4" class="data row12 col4" >0.93476</td>
      <td id="T_ad00b_row12_col5" class="data row12 col5" >0.97954</td>
      <td id="T_ad00b_row12_col6" class="data row12 col6" >0.79554</td>
      <td id="T_ad00b_row12_col7" class="data row12 col7" >1</td>
      <td id="T_ad00b_row12_col8" class="data row12 col8" >0</td>
      <td id="T_ad00b_row12_col9" class="data row12 col9" >11.79798</td>
      <td id="T_ad00b_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_ad00b_row13_col0" class="data row13 col0" >18073</td>
      <td id="T_ad00b_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_ad00b_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_ad00b_row13_col3" class="data row13 col3" >IF age <= 29.8893 AND education.num != 4.0 AND hours.per.week <= 71.7261 AND occupation != Prof-specialty THEN class = <=50K</td>
      <td id="T_ad00b_row13_col4" class="data row13 col4" >0.24434</td>
      <td id="T_ad00b_row13_col5" class="data row13 col5" >0.30677</td>
      <td id="T_ad00b_row13_col6" class="data row13 col6" >0.95313</td>
      <td id="T_ad00b_row13_col7" class="data row13 col7" >4</td>
      <td id="T_ad00b_row13_col8" class="data row13 col8" >0</td>
      <td id="T_ad00b_row13_col9" class="data row13 col9" >11.62583</td>
      <td id="T_ad00b_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_ad00b_row14_col0" class="data row14 col0" >18073</td>
      <td id="T_ad00b_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_ad00b_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_ad00b_row14_col3" class="data row14 col3" >IF capital.gain <= 6554.2478 THEN class = <=50K</td>
      <td id="T_ad00b_row14_col4" class="data row14 col4" >0.95534</td>
      <td id="T_ad00b_row14_col5" class="data row14 col5" >0.99775</td>
      <td id="T_ad00b_row14_col6" class="data row14 col6" >0.79287</td>
      <td id="T_ad00b_row14_col7" class="data row14 col7" >1</td>
      <td id="T_ad00b_row14_col8" class="data row14 col8" >0</td>
      <td id="T_ad00b_row14_col9" class="data row14 col9" >11.71087</td>
      <td id="T_ad00b_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_ad00b_row15_col0" class="data row15 col0" >18073</td>
      <td id="T_ad00b_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_ad00b_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_ad00b_row15_col3" class="data row15 col3" >IF age <= 32.1452 AND capital.gain <= 6753.9448 AND hours.per.week > 44.3573 THEN class = <=50K</td>
      <td id="T_ad00b_row15_col4" class="data row15 col4" >0.07573</td>
      <td id="T_ad00b_row15_col5" class="data row15 col5" >0.08224</td>
      <td id="T_ad00b_row15_col6" class="data row15 col6" >0.82445</td>
      <td id="T_ad00b_row15_col7" class="data row15 col7" >3</td>
      <td id="T_ad00b_row15_col8" class="data row15 col8" >0</td>
      <td id="T_ad00b_row15_col9" class="data row15 col9" >1.33378</td>
      <td id="T_ad00b_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_ad00b_row16_col0" class="data row16 col0" >18073</td>
      <td id="T_ad00b_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_ad00b_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_ad00b_row16_col3" class="data row16 col3" >IF age <= 38.0 AND capital.gain <= 0.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_ad00b_row16_col4" class="data row16 col4" >0.27431</td>
      <td id="T_ad00b_row16_col5" class="data row16 col5" >0.35358</td>
      <td id="T_ad00b_row16_col6" class="data row16 col6" >0.97857</td>
      <td id="T_ad00b_row16_col7" class="data row16 col7" >3</td>
      <td id="T_ad00b_row16_col8" class="data row16 col8" >0</td>
      <td id="T_ad00b_row16_col9" class="data row16 col9" >0.88249</td>
      <td id="T_ad00b_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_ad00b_row17_col0" class="data row17 col0" >18073</td>
      <td id="T_ad00b_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_ad00b_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_ad00b_row17_col3" class="data row17 col3" >IF age <= 44.4668 AND capital.gain <= 5377.5028 AND hours.per.week > 40.0 THEN class = <=50K</td>
      <td id="T_ad00b_row17_col4" class="data row17 col4" >0.18349</td>
      <td id="T_ad00b_row17_col5" class="data row17 col5" >0.16789</td>
      <td id="T_ad00b_row17_col6" class="data row17 col6" >0.69464</td>
      <td id="T_ad00b_row17_col7" class="data row17 col7" >3</td>
      <td id="T_ad00b_row17_col8" class="data row17 col8" >0</td>
      <td id="T_ad00b_row17_col9" class="data row17 col9" >1.08478</td>
      <td id="T_ad00b_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_ad00b_row18_col0" class="data row18 col0" >18073</td>
      <td id="T_ad00b_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_ad00b_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_ad00b_row18_col3" class="data row18 col3" >IF age <= 34.1577 AND capital.gain <= 403.4566 THEN class = <=50K</td>
      <td id="T_ad00b_row18_col4" class="data row18 col4" >0.41260</td>
      <td id="T_ad00b_row18_col5" class="data row18 col5" >0.49396</td>
      <td id="T_ad00b_row18_col6" class="data row18 col6" >0.90887</td>
      <td id="T_ad00b_row18_col7" class="data row18 col7" >2</td>
      <td id="T_ad00b_row18_col8" class="data row18 col8" >0</td>
      <td id="T_ad00b_row18_col9" class="data row18 col9" >1.05122</td>
      <td id="T_ad00b_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ad00b_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_ad00b_row19_col0" class="data row19 col0" >18073</td>
      <td id="T_ad00b_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_ad00b_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_ad00b_row19_col3" class="data row19 col3" >IF age <= 35.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_ad00b_row19_col4" class="data row19 col4" >0.43726</td>
      <td id="T_ad00b_row19_col5" class="data row19 col5" >0.51916</td>
      <td id="T_ad00b_row19_col6" class="data row19 col6" >0.90136</td>
      <td id="T_ad00b_row19_col7" class="data row19 col7" >2</td>
      <td id="T_ad00b_row19_col8" class="data row19 col8" >0</td>
      <td id="T_ad00b_row19_col9" class="data row19 col9" >1.34529</td>
      <td id="T_ad00b_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 18073, Correct Prediction



<style type="text/css">
</style>
<table id="T_d85d1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d85d1_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_d85d1_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_d85d1_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_d85d1_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_d85d1_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_d85d1_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_d85d1_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_d85d1_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_d85d1_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_d85d1_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_d85d1_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d85d1_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_d85d1_row0_col0" class="data row0 col0" >18073</td>
      <td id="T_d85d1_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_d85d1_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_d85d1_row0_col3" class="data row0 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_d85d1_row0_col4" class="data row0 col4" >0.52422</td>
      <td id="T_d85d1_row0_col5" class="data row0 col5" >0.59186</td>
      <td id="T_d85d1_row0_col6" class="data row0 col6" >0.85713</td>
      <td id="T_d85d1_row0_col7" class="data row0 col7" >4</td>
      <td id="T_d85d1_row0_col8" class="data row0 col8" >0</td>
      <td id="T_d85d1_row0_col9" class="data row0 col9" >0.74556</td>
      <td id="T_d85d1_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_d85d1_row1_col0" class="data row1 col0" >18073</td>
      <td id="T_d85d1_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_d85d1_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_d85d1_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_d85d1_row1_col4" class="data row1 col4" >0.66269</td>
      <td id="T_d85d1_row1_col5" class="data row1 col5" >0.73091</td>
      <td id="T_d85d1_row1_col6" class="data row1 col6" >0.83733</td>
      <td id="T_d85d1_row1_col7" class="data row1 col7" >3</td>
      <td id="T_d85d1_row1_col8" class="data row1 col8" >0</td>
      <td id="T_d85d1_row1_col9" class="data row1 col9" >0.63635</td>
      <td id="T_d85d1_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_d85d1_row2_col0" class="data row2 col0" >18073</td>
      <td id="T_d85d1_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_d85d1_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_d85d1_row2_col3" class="data row2 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_d85d1_row2_col4" class="data row2 col4" >0.69595</td>
      <td id="T_d85d1_row2_col5" class="data row2 col5" >0.75357</td>
      <td id="T_d85d1_row2_col6" class="data row2 col6" >0.82203</td>
      <td id="T_d85d1_row2_col7" class="data row2 col7" >2</td>
      <td id="T_d85d1_row2_col8" class="data row2 col8" >0</td>
      <td id="T_d85d1_row2_col9" class="data row2 col9" >0.64159</td>
      <td id="T_d85d1_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_d85d1_row3_col0" class="data row3 col0" >18073</td>
      <td id="T_d85d1_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_d85d1_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_d85d1_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_d85d1_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_d85d1_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_d85d1_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_d85d1_row3_col7" class="data row3 col7" >2</td>
      <td id="T_d85d1_row3_col8" class="data row3 col8" >0</td>
      <td id="T_d85d1_row3_col9" class="data row3 col9" >0.67650</td>
      <td id="T_d85d1_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_d85d1_row4_col0" class="data row4 col0" >18073</td>
      <td id="T_d85d1_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_d85d1_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_d85d1_row4_col3" class="data row4 col3" >IF age <= 37.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_d85d1_row4_col4" class="data row4 col4" >0.46850</td>
      <td id="T_d85d1_row4_col5" class="data row4 col5" >0.55083</td>
      <td id="T_d85d1_row4_col6" class="data row4 col6" >0.89258</td>
      <td id="T_d85d1_row4_col7" class="data row4 col7" >3</td>
      <td id="T_d85d1_row4_col8" class="data row4 col8" >2</td>
      <td id="T_d85d1_row4_col9" class="data row4 col9" >2.08425</td>
      <td id="T_d85d1_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_d85d1_row5_col0" class="data row5 col0" >18073</td>
      <td id="T_d85d1_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_d85d1_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_d85d1_row5_col3" class="data row5 col3" >IF capital.gain <= 5013.0 THEN class = <=50K</td>
      <td id="T_d85d1_row5_col4" class="data row5 col4" >0.95077</td>
      <td id="T_d85d1_row5_col5" class="data row5 col5" >0.99659</td>
      <td id="T_d85d1_row5_col6" class="data row5 col6" >0.79575</td>
      <td id="T_d85d1_row5_col7" class="data row5 col7" >1</td>
      <td id="T_d85d1_row5_col8" class="data row5 col8" >0</td>
      <td id="T_d85d1_row5_col9" class="data row5 col9" >57.71988</td>
      <td id="T_d85d1_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_d85d1_row6_col0" class="data row6 col0" >18073</td>
      <td id="T_d85d1_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_d85d1_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_d85d1_row6_col3" class="data row6 col3" >IF capital.gain <= 7298.0 THEN class = <=50K</td>
      <td id="T_d85d1_row6_col4" class="data row6 col4" >0.96363</td>
      <td id="T_d85d1_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_d85d1_row6_col6" class="data row6 col6" >0.78696</td>
      <td id="T_d85d1_row6_col7" class="data row6 col7" >1</td>
      <td id="T_d85d1_row6_col8" class="data row6 col8" >0</td>
      <td id="T_d85d1_row6_col9" class="data row6 col9" >56.74821</td>
      <td id="T_d85d1_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_d85d1_row7_col0" class="data row7 col0" >18073</td>
      <td id="T_d85d1_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_d85d1_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_d85d1_row7_col3" class="data row7 col3" >IF capital.gain <= 7688.0 THEN class = <=50K</td>
      <td id="T_d85d1_row7_col4" class="data row7 col4" >0.97306</td>
      <td id="T_d85d1_row7_col5" class="data row7 col5" >0.99919</td>
      <td id="T_d85d1_row7_col6" class="data row7 col6" >0.77956</td>
      <td id="T_d85d1_row7_col7" class="data row7 col7" >1</td>
      <td id="T_d85d1_row7_col8" class="data row7 col8" >0</td>
      <td id="T_d85d1_row7_col9" class="data row7 col9" >55.95721</td>
      <td id="T_d85d1_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_d85d1_row8_col0" class="data row8 col0" >18073</td>
      <td id="T_d85d1_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_d85d1_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_d85d1_row8_col3" class="data row8 col3" >IF capital.gain <= 6696.0568 THEN class = <=50K</td>
      <td id="T_d85d1_row8_col4" class="data row8 col4" >0.95534</td>
      <td id="T_d85d1_row8_col5" class="data row8 col5" >0.99775</td>
      <td id="T_d85d1_row8_col6" class="data row8 col6" >0.79287</td>
      <td id="T_d85d1_row8_col7" class="data row8 col7" >1</td>
      <td id="T_d85d1_row8_col8" class="data row8 col8" >0</td>
      <td id="T_d85d1_row8_col9" class="data row8 col9" >56.41765</td>
      <td id="T_d85d1_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_d85d1_row9_col0" class="data row9 col0" >18073</td>
      <td id="T_d85d1_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_d85d1_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_d85d1_row9_col3" class="data row9 col3" >IF capital.gain <= 3464.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_d85d1_row9_col4" class="data row9 col4" >0.32011</td>
      <td id="T_d85d1_row9_col5" class="data row9 col5" >0.40802</td>
      <td id="T_d85d1_row9_col6" class="data row9 col6" >0.96765</td>
      <td id="T_d85d1_row9_col7" class="data row9 col7" >2</td>
      <td id="T_d85d1_row9_col8" class="data row9 col8" >0</td>
      <td id="T_d85d1_row9_col9" class="data row9 col9" >57.99263</td>
      <td id="T_d85d1_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_d85d1_row10_col0" class="data row10 col0" >18073</td>
      <td id="T_d85d1_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_d85d1_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_d85d1_row10_col3" class="data row10 col3" >IF capital.gain <= 7426.083 THEN class = <=50K</td>
      <td id="T_d85d1_row10_col4" class="data row10 col4" >0.96363</td>
      <td id="T_d85d1_row10_col5" class="data row10 col5" >0.99890</td>
      <td id="T_d85d1_row10_col6" class="data row10 col6" >0.78696</td>
      <td id="T_d85d1_row10_col7" class="data row10 col7" >1</td>
      <td id="T_d85d1_row10_col8" class="data row10 col8" >0</td>
      <td id="T_d85d1_row10_col9" class="data row10 col9" >11.73073</td>
      <td id="T_d85d1_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_d85d1_row11_col0" class="data row11 col0" >18073</td>
      <td id="T_d85d1_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_d85d1_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_d85d1_row11_col3" class="data row11 col3" >IF capital.gain <= 5756.5615 THEN class = <=50K</td>
      <td id="T_d85d1_row11_col4" class="data row11 col4" >0.95450</td>
      <td id="T_d85d1_row11_col5" class="data row11 col5" >0.99717</td>
      <td id="T_d85d1_row11_col6" class="data row11 col6" >0.79311</td>
      <td id="T_d85d1_row11_col7" class="data row11 col7" >1</td>
      <td id="T_d85d1_row11_col8" class="data row11 col8" >0</td>
      <td id="T_d85d1_row11_col9" class="data row11 col9" >11.80187</td>
      <td id="T_d85d1_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_d85d1_row12_col0" class="data row12 col0" >18073</td>
      <td id="T_d85d1_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_d85d1_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_d85d1_row12_col3" class="data row12 col3" >IF capital.gain <= 3299.5553 THEN class = <=50K</td>
      <td id="T_d85d1_row12_col4" class="data row12 col4" >0.93476</td>
      <td id="T_d85d1_row12_col5" class="data row12 col5" >0.97954</td>
      <td id="T_d85d1_row12_col6" class="data row12 col6" >0.79554</td>
      <td id="T_d85d1_row12_col7" class="data row12 col7" >1</td>
      <td id="T_d85d1_row12_col8" class="data row12 col8" >0</td>
      <td id="T_d85d1_row12_col9" class="data row12 col9" >11.79798</td>
      <td id="T_d85d1_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_d85d1_row13_col0" class="data row13 col0" >18073</td>
      <td id="T_d85d1_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_d85d1_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_d85d1_row13_col3" class="data row13 col3" >IF age <= 29.8893 AND education.num != 4.0 AND hours.per.week <= 71.7261 AND occupation != Prof-specialty THEN class = <=50K</td>
      <td id="T_d85d1_row13_col4" class="data row13 col4" >0.24434</td>
      <td id="T_d85d1_row13_col5" class="data row13 col5" >0.30677</td>
      <td id="T_d85d1_row13_col6" class="data row13 col6" >0.95313</td>
      <td id="T_d85d1_row13_col7" class="data row13 col7" >4</td>
      <td id="T_d85d1_row13_col8" class="data row13 col8" >0</td>
      <td id="T_d85d1_row13_col9" class="data row13 col9" >11.62583</td>
      <td id="T_d85d1_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_d85d1_row14_col0" class="data row14 col0" >18073</td>
      <td id="T_d85d1_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_d85d1_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_d85d1_row14_col3" class="data row14 col3" >IF capital.gain <= 6554.2478 THEN class = <=50K</td>
      <td id="T_d85d1_row14_col4" class="data row14 col4" >0.95534</td>
      <td id="T_d85d1_row14_col5" class="data row14 col5" >0.99775</td>
      <td id="T_d85d1_row14_col6" class="data row14 col6" >0.79287</td>
      <td id="T_d85d1_row14_col7" class="data row14 col7" >1</td>
      <td id="T_d85d1_row14_col8" class="data row14 col8" >0</td>
      <td id="T_d85d1_row14_col9" class="data row14 col9" >11.71087</td>
      <td id="T_d85d1_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_d85d1_row15_col0" class="data row15 col0" >18073</td>
      <td id="T_d85d1_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_d85d1_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_d85d1_row15_col3" class="data row15 col3" >IF age <= 32.1452 AND capital.gain <= 6753.9448 AND hours.per.week > 44.3573 THEN class = <=50K</td>
      <td id="T_d85d1_row15_col4" class="data row15 col4" >0.07573</td>
      <td id="T_d85d1_row15_col5" class="data row15 col5" >0.08224</td>
      <td id="T_d85d1_row15_col6" class="data row15 col6" >0.82445</td>
      <td id="T_d85d1_row15_col7" class="data row15 col7" >3</td>
      <td id="T_d85d1_row15_col8" class="data row15 col8" >0</td>
      <td id="T_d85d1_row15_col9" class="data row15 col9" >1.33378</td>
      <td id="T_d85d1_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_d85d1_row16_col0" class="data row16 col0" >18073</td>
      <td id="T_d85d1_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_d85d1_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_d85d1_row16_col3" class="data row16 col3" >IF age <= 38.0 AND capital.gain <= 0.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_d85d1_row16_col4" class="data row16 col4" >0.27431</td>
      <td id="T_d85d1_row16_col5" class="data row16 col5" >0.35358</td>
      <td id="T_d85d1_row16_col6" class="data row16 col6" >0.97857</td>
      <td id="T_d85d1_row16_col7" class="data row16 col7" >3</td>
      <td id="T_d85d1_row16_col8" class="data row16 col8" >0</td>
      <td id="T_d85d1_row16_col9" class="data row16 col9" >0.88249</td>
      <td id="T_d85d1_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_d85d1_row17_col0" class="data row17 col0" >18073</td>
      <td id="T_d85d1_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_d85d1_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_d85d1_row17_col3" class="data row17 col3" >IF age <= 44.4668 AND capital.gain <= 5377.5028 AND hours.per.week > 40.0 THEN class = <=50K</td>
      <td id="T_d85d1_row17_col4" class="data row17 col4" >0.18349</td>
      <td id="T_d85d1_row17_col5" class="data row17 col5" >0.16789</td>
      <td id="T_d85d1_row17_col6" class="data row17 col6" >0.69464</td>
      <td id="T_d85d1_row17_col7" class="data row17 col7" >3</td>
      <td id="T_d85d1_row17_col8" class="data row17 col8" >0</td>
      <td id="T_d85d1_row17_col9" class="data row17 col9" >1.08478</td>
      <td id="T_d85d1_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_d85d1_row18_col0" class="data row18 col0" >18073</td>
      <td id="T_d85d1_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_d85d1_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_d85d1_row18_col3" class="data row18 col3" >IF age <= 34.1577 AND capital.gain <= 403.4566 THEN class = <=50K</td>
      <td id="T_d85d1_row18_col4" class="data row18 col4" >0.41260</td>
      <td id="T_d85d1_row18_col5" class="data row18 col5" >0.49396</td>
      <td id="T_d85d1_row18_col6" class="data row18 col6" >0.90887</td>
      <td id="T_d85d1_row18_col7" class="data row18 col7" >2</td>
      <td id="T_d85d1_row18_col8" class="data row18 col8" >0</td>
      <td id="T_d85d1_row18_col9" class="data row18 col9" >1.05122</td>
      <td id="T_d85d1_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d85d1_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_d85d1_row19_col0" class="data row19 col0" >18073</td>
      <td id="T_d85d1_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_d85d1_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_d85d1_row19_col3" class="data row19 col3" >IF age <= 35.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_d85d1_row19_col4" class="data row19 col4" >0.43726</td>
      <td id="T_d85d1_row19_col5" class="data row19 col5" >0.51916</td>
      <td id="T_d85d1_row19_col6" class="data row19 col6" >0.90136</td>
      <td id="T_d85d1_row19_col7" class="data row19 col7" >2</td>
      <td id="T_d85d1_row19_col8" class="data row19 col8" >0</td>
      <td id="T_d85d1_row19_col9" class="data row19 col9" >1.34529</td>
      <td id="T_d85d1_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 18073, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_a513b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a513b_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_a513b_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_a513b_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_a513b_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_a513b_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_a513b_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_a513b_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_a513b_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_a513b_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_a513b_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_a513b_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a513b_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a513b_row0_col0" class="data row0 col0" >18073</td>
      <td id="T_a513b_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_a513b_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_a513b_row0_col3" class="data row0 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_a513b_row0_col4" class="data row0 col4" >0.52422</td>
      <td id="T_a513b_row0_col5" class="data row0 col5" >0.59186</td>
      <td id="T_a513b_row0_col6" class="data row0 col6" >0.85713</td>
      <td id="T_a513b_row0_col7" class="data row0 col7" >4</td>
      <td id="T_a513b_row0_col8" class="data row0 col8" >0</td>
      <td id="T_a513b_row0_col9" class="data row0 col9" >0.74556</td>
      <td id="T_a513b_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a513b_row1_col0" class="data row1 col0" >18073</td>
      <td id="T_a513b_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_a513b_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_a513b_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_a513b_row1_col4" class="data row1 col4" >0.66269</td>
      <td id="T_a513b_row1_col5" class="data row1 col5" >0.73091</td>
      <td id="T_a513b_row1_col6" class="data row1 col6" >0.83733</td>
      <td id="T_a513b_row1_col7" class="data row1 col7" >3</td>
      <td id="T_a513b_row1_col8" class="data row1 col8" >0</td>
      <td id="T_a513b_row1_col9" class="data row1 col9" >0.63635</td>
      <td id="T_a513b_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a513b_row2_col0" class="data row2 col0" >18073</td>
      <td id="T_a513b_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_a513b_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_a513b_row2_col3" class="data row2 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_a513b_row2_col4" class="data row2 col4" >0.69595</td>
      <td id="T_a513b_row2_col5" class="data row2 col5" >0.75357</td>
      <td id="T_a513b_row2_col6" class="data row2 col6" >0.82203</td>
      <td id="T_a513b_row2_col7" class="data row2 col7" >2</td>
      <td id="T_a513b_row2_col8" class="data row2 col8" >0</td>
      <td id="T_a513b_row2_col9" class="data row2 col9" >0.64159</td>
      <td id="T_a513b_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a513b_row3_col0" class="data row3 col0" >18073</td>
      <td id="T_a513b_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_a513b_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_a513b_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_a513b_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_a513b_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_a513b_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_a513b_row3_col7" class="data row3 col7" >2</td>
      <td id="T_a513b_row3_col8" class="data row3 col8" >0</td>
      <td id="T_a513b_row3_col9" class="data row3 col9" >0.67650</td>
      <td id="T_a513b_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_a513b_row4_col0" class="data row4 col0" >18073</td>
      <td id="T_a513b_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_a513b_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_a513b_row4_col3" class="data row4 col3" >IF age <= 37.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_a513b_row4_col4" class="data row4 col4" >0.46850</td>
      <td id="T_a513b_row4_col5" class="data row4 col5" >0.55083</td>
      <td id="T_a513b_row4_col6" class="data row4 col6" >0.89258</td>
      <td id="T_a513b_row4_col7" class="data row4 col7" >3</td>
      <td id="T_a513b_row4_col8" class="data row4 col8" >2</td>
      <td id="T_a513b_row4_col9" class="data row4 col9" >2.08425</td>
      <td id="T_a513b_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_a513b_row5_col0" class="data row5 col0" >18073</td>
      <td id="T_a513b_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_a513b_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_a513b_row5_col3" class="data row5 col3" >IF capital.gain <= 5013.0 THEN class = <=50K</td>
      <td id="T_a513b_row5_col4" class="data row5 col4" >0.95077</td>
      <td id="T_a513b_row5_col5" class="data row5 col5" >0.99659</td>
      <td id="T_a513b_row5_col6" class="data row5 col6" >0.79575</td>
      <td id="T_a513b_row5_col7" class="data row5 col7" >1</td>
      <td id="T_a513b_row5_col8" class="data row5 col8" >0</td>
      <td id="T_a513b_row5_col9" class="data row5 col9" >57.71988</td>
      <td id="T_a513b_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_a513b_row6_col0" class="data row6 col0" >18073</td>
      <td id="T_a513b_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_a513b_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_a513b_row6_col3" class="data row6 col3" >IF capital.gain <= 7298.0 THEN class = <=50K</td>
      <td id="T_a513b_row6_col4" class="data row6 col4" >0.96363</td>
      <td id="T_a513b_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_a513b_row6_col6" class="data row6 col6" >0.78696</td>
      <td id="T_a513b_row6_col7" class="data row6 col7" >1</td>
      <td id="T_a513b_row6_col8" class="data row6 col8" >0</td>
      <td id="T_a513b_row6_col9" class="data row6 col9" >56.74821</td>
      <td id="T_a513b_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_a513b_row7_col0" class="data row7 col0" >18073</td>
      <td id="T_a513b_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_a513b_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_a513b_row7_col3" class="data row7 col3" >IF capital.gain <= 7688.0 THEN class = <=50K</td>
      <td id="T_a513b_row7_col4" class="data row7 col4" >0.97306</td>
      <td id="T_a513b_row7_col5" class="data row7 col5" >0.99919</td>
      <td id="T_a513b_row7_col6" class="data row7 col6" >0.77956</td>
      <td id="T_a513b_row7_col7" class="data row7 col7" >1</td>
      <td id="T_a513b_row7_col8" class="data row7 col8" >0</td>
      <td id="T_a513b_row7_col9" class="data row7 col9" >55.95721</td>
      <td id="T_a513b_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_a513b_row8_col0" class="data row8 col0" >18073</td>
      <td id="T_a513b_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_a513b_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_a513b_row8_col3" class="data row8 col3" >IF capital.gain <= 6696.0568 THEN class = <=50K</td>
      <td id="T_a513b_row8_col4" class="data row8 col4" >0.95534</td>
      <td id="T_a513b_row8_col5" class="data row8 col5" >0.99775</td>
      <td id="T_a513b_row8_col6" class="data row8 col6" >0.79287</td>
      <td id="T_a513b_row8_col7" class="data row8 col7" >1</td>
      <td id="T_a513b_row8_col8" class="data row8 col8" >0</td>
      <td id="T_a513b_row8_col9" class="data row8 col9" >56.41765</td>
      <td id="T_a513b_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_a513b_row9_col0" class="data row9 col0" >18073</td>
      <td id="T_a513b_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_a513b_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_a513b_row9_col3" class="data row9 col3" >IF capital.gain <= 3464.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_a513b_row9_col4" class="data row9 col4" >0.32011</td>
      <td id="T_a513b_row9_col5" class="data row9 col5" >0.40802</td>
      <td id="T_a513b_row9_col6" class="data row9 col6" >0.96765</td>
      <td id="T_a513b_row9_col7" class="data row9 col7" >2</td>
      <td id="T_a513b_row9_col8" class="data row9 col8" >0</td>
      <td id="T_a513b_row9_col9" class="data row9 col9" >57.99263</td>
      <td id="T_a513b_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_a513b_row10_col0" class="data row10 col0" >18073</td>
      <td id="T_a513b_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_a513b_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_a513b_row10_col3" class="data row10 col3" >IF capital.gain <= 7426.083 THEN class = <=50K</td>
      <td id="T_a513b_row10_col4" class="data row10 col4" >0.96363</td>
      <td id="T_a513b_row10_col5" class="data row10 col5" >0.99890</td>
      <td id="T_a513b_row10_col6" class="data row10 col6" >0.78696</td>
      <td id="T_a513b_row10_col7" class="data row10 col7" >1</td>
      <td id="T_a513b_row10_col8" class="data row10 col8" >0</td>
      <td id="T_a513b_row10_col9" class="data row10 col9" >11.73073</td>
      <td id="T_a513b_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_a513b_row11_col0" class="data row11 col0" >18073</td>
      <td id="T_a513b_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_a513b_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_a513b_row11_col3" class="data row11 col3" >IF capital.gain <= 5756.5615 THEN class = <=50K</td>
      <td id="T_a513b_row11_col4" class="data row11 col4" >0.95450</td>
      <td id="T_a513b_row11_col5" class="data row11 col5" >0.99717</td>
      <td id="T_a513b_row11_col6" class="data row11 col6" >0.79311</td>
      <td id="T_a513b_row11_col7" class="data row11 col7" >1</td>
      <td id="T_a513b_row11_col8" class="data row11 col8" >0</td>
      <td id="T_a513b_row11_col9" class="data row11 col9" >11.80187</td>
      <td id="T_a513b_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_a513b_row12_col0" class="data row12 col0" >18073</td>
      <td id="T_a513b_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_a513b_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_a513b_row12_col3" class="data row12 col3" >IF capital.gain <= 3299.5553 THEN class = <=50K</td>
      <td id="T_a513b_row12_col4" class="data row12 col4" >0.93476</td>
      <td id="T_a513b_row12_col5" class="data row12 col5" >0.97954</td>
      <td id="T_a513b_row12_col6" class="data row12 col6" >0.79554</td>
      <td id="T_a513b_row12_col7" class="data row12 col7" >1</td>
      <td id="T_a513b_row12_col8" class="data row12 col8" >0</td>
      <td id="T_a513b_row12_col9" class="data row12 col9" >11.79798</td>
      <td id="T_a513b_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_a513b_row13_col0" class="data row13 col0" >18073</td>
      <td id="T_a513b_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_a513b_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_a513b_row13_col3" class="data row13 col3" >IF age <= 29.8893 AND education.num != 4.0 AND hours.per.week <= 71.7261 AND occupation != Prof-specialty THEN class = <=50K</td>
      <td id="T_a513b_row13_col4" class="data row13 col4" >0.24434</td>
      <td id="T_a513b_row13_col5" class="data row13 col5" >0.30677</td>
      <td id="T_a513b_row13_col6" class="data row13 col6" >0.95313</td>
      <td id="T_a513b_row13_col7" class="data row13 col7" >4</td>
      <td id="T_a513b_row13_col8" class="data row13 col8" >0</td>
      <td id="T_a513b_row13_col9" class="data row13 col9" >11.62583</td>
      <td id="T_a513b_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_a513b_row14_col0" class="data row14 col0" >18073</td>
      <td id="T_a513b_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_a513b_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_a513b_row14_col3" class="data row14 col3" >IF capital.gain <= 6554.2478 THEN class = <=50K</td>
      <td id="T_a513b_row14_col4" class="data row14 col4" >0.95534</td>
      <td id="T_a513b_row14_col5" class="data row14 col5" >0.99775</td>
      <td id="T_a513b_row14_col6" class="data row14 col6" >0.79287</td>
      <td id="T_a513b_row14_col7" class="data row14 col7" >1</td>
      <td id="T_a513b_row14_col8" class="data row14 col8" >0</td>
      <td id="T_a513b_row14_col9" class="data row14 col9" >11.71087</td>
      <td id="T_a513b_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_a513b_row15_col0" class="data row15 col0" >18073</td>
      <td id="T_a513b_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_a513b_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_a513b_row15_col3" class="data row15 col3" >IF age <= 32.1452 AND capital.gain <= 6753.9448 AND hours.per.week > 44.3573 THEN class = <=50K</td>
      <td id="T_a513b_row15_col4" class="data row15 col4" >0.07573</td>
      <td id="T_a513b_row15_col5" class="data row15 col5" >0.08224</td>
      <td id="T_a513b_row15_col6" class="data row15 col6" >0.82445</td>
      <td id="T_a513b_row15_col7" class="data row15 col7" >3</td>
      <td id="T_a513b_row15_col8" class="data row15 col8" >0</td>
      <td id="T_a513b_row15_col9" class="data row15 col9" >1.33378</td>
      <td id="T_a513b_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_a513b_row16_col0" class="data row16 col0" >18073</td>
      <td id="T_a513b_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_a513b_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_a513b_row16_col3" class="data row16 col3" >IF age <= 38.0 AND capital.gain <= 0.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_a513b_row16_col4" class="data row16 col4" >0.27431</td>
      <td id="T_a513b_row16_col5" class="data row16 col5" >0.35358</td>
      <td id="T_a513b_row16_col6" class="data row16 col6" >0.97857</td>
      <td id="T_a513b_row16_col7" class="data row16 col7" >3</td>
      <td id="T_a513b_row16_col8" class="data row16 col8" >0</td>
      <td id="T_a513b_row16_col9" class="data row16 col9" >0.88249</td>
      <td id="T_a513b_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_a513b_row17_col0" class="data row17 col0" >18073</td>
      <td id="T_a513b_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_a513b_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_a513b_row17_col3" class="data row17 col3" >IF age <= 44.4668 AND capital.gain <= 5377.5028 AND hours.per.week > 40.0 THEN class = <=50K</td>
      <td id="T_a513b_row17_col4" class="data row17 col4" >0.18349</td>
      <td id="T_a513b_row17_col5" class="data row17 col5" >0.16789</td>
      <td id="T_a513b_row17_col6" class="data row17 col6" >0.69464</td>
      <td id="T_a513b_row17_col7" class="data row17 col7" >3</td>
      <td id="T_a513b_row17_col8" class="data row17 col8" >0</td>
      <td id="T_a513b_row17_col9" class="data row17 col9" >1.08478</td>
      <td id="T_a513b_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_a513b_row18_col0" class="data row18 col0" >18073</td>
      <td id="T_a513b_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_a513b_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_a513b_row18_col3" class="data row18 col3" >IF age <= 34.1577 AND capital.gain <= 403.4566 THEN class = <=50K</td>
      <td id="T_a513b_row18_col4" class="data row18 col4" >0.41260</td>
      <td id="T_a513b_row18_col5" class="data row18 col5" >0.49396</td>
      <td id="T_a513b_row18_col6" class="data row18 col6" >0.90887</td>
      <td id="T_a513b_row18_col7" class="data row18 col7" >2</td>
      <td id="T_a513b_row18_col8" class="data row18 col8" >0</td>
      <td id="T_a513b_row18_col9" class="data row18 col9" >1.05122</td>
      <td id="T_a513b_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a513b_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_a513b_row19_col0" class="data row19 col0" >18073</td>
      <td id="T_a513b_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_a513b_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_a513b_row19_col3" class="data row19 col3" >IF age <= 35.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_a513b_row19_col4" class="data row19 col4" >0.43726</td>
      <td id="T_a513b_row19_col5" class="data row19 col5" >0.51916</td>
      <td id="T_a513b_row19_col6" class="data row19 col6" >0.90136</td>
      <td id="T_a513b_row19_col7" class="data row19 col7" >2</td>
      <td id="T_a513b_row19_col8" class="data row19 col8" >0</td>
      <td id="T_a513b_row19_col9" class="data row19 col9" >1.34529</td>
      <td id="T_a513b_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 18073, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.97306, Pre: 0.97857)



<style type="text/css">
#T_cf7e2_row5_col0, #T_cf7e2_row5_col1, #T_cf7e2_row5_col2, #T_cf7e2_row5_col3, #T_cf7e2_row5_col4, #T_cf7e2_row5_col5, #T_cf7e2_row5_col6, #T_cf7e2_row5_col7, #T_cf7e2_row5_col8, #T_cf7e2_row5_col9, #T_cf7e2_row5_col10, #T_cf7e2_row5_col11 {
  font-weight: bold;
}
</style>
<table id="T_cf7e2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cf7e2_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_cf7e2_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_cf7e2_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_cf7e2_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_cf7e2_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_cf7e2_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_cf7e2_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_cf7e2_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_cf7e2_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_cf7e2_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_cf7e2_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_cf7e2_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cf7e2_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_cf7e2_row0_col0" class="data row0 col0" >18073</td>
      <td id="T_cf7e2_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_cf7e2_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_cf7e2_row0_col3" class="data row0 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_cf7e2_row0_col4" class="data row0 col4" >0.52422</td>
      <td id="T_cf7e2_row0_col5" class="data row0 col5" >0.59186</td>
      <td id="T_cf7e2_row0_col6" class="data row0 col6" >0.85713</td>
      <td id="T_cf7e2_row0_col7" class="data row0 col7" >4</td>
      <td id="T_cf7e2_row0_col8" class="data row0 col8" >0</td>
      <td id="T_cf7e2_row0_col9" class="data row0 col9" >0.74556</td>
      <td id="T_cf7e2_row0_col10" class="data row0 col10" >False</td>
      <td id="T_cf7e2_row0_col11" class="data row0 col11" >0.46498</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_cf7e2_row1_col0" class="data row1 col0" >18073</td>
      <td id="T_cf7e2_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_cf7e2_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_cf7e2_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row1_col4" class="data row1 col4" >0.66269</td>
      <td id="T_cf7e2_row1_col5" class="data row1 col5" >0.73091</td>
      <td id="T_cf7e2_row1_col6" class="data row1 col6" >0.83733</td>
      <td id="T_cf7e2_row1_col7" class="data row1 col7" >3</td>
      <td id="T_cf7e2_row1_col8" class="data row1 col8" >0</td>
      <td id="T_cf7e2_row1_col9" class="data row1 col9" >0.63635</td>
      <td id="T_cf7e2_row1_col10" class="data row1 col10" >False</td>
      <td id="T_cf7e2_row1_col11" class="data row1 col11" >0.34100</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_cf7e2_row2_col0" class="data row2 col0" >18073</td>
      <td id="T_cf7e2_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_cf7e2_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_cf7e2_row2_col3" class="data row2 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row2_col4" class="data row2 col4" >0.69595</td>
      <td id="T_cf7e2_row2_col5" class="data row2 col5" >0.75357</td>
      <td id="T_cf7e2_row2_col6" class="data row2 col6" >0.82203</td>
      <td id="T_cf7e2_row2_col7" class="data row2 col7" >2</td>
      <td id="T_cf7e2_row2_col8" class="data row2 col8" >0</td>
      <td id="T_cf7e2_row2_col9" class="data row2 col9" >0.64159</td>
      <td id="T_cf7e2_row2_col10" class="data row2 col10" >False</td>
      <td id="T_cf7e2_row2_col11" class="data row2 col11" >0.31827</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_cf7e2_row3_col0" class="data row3 col0" >18073</td>
      <td id="T_cf7e2_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_cf7e2_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_cf7e2_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_cf7e2_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_cf7e2_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_cf7e2_row3_col7" class="data row3 col7" >2</td>
      <td id="T_cf7e2_row3_col8" class="data row3 col8" >0</td>
      <td id="T_cf7e2_row3_col9" class="data row3 col9" >0.67650</td>
      <td id="T_cf7e2_row3_col10" class="data row3 col10" >False</td>
      <td id="T_cf7e2_row3_col11" class="data row3 col11" >0.49531</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_cf7e2_row4_col0" class="data row4 col0" >18073</td>
      <td id="T_cf7e2_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_cf7e2_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_cf7e2_row4_col3" class="data row4 col3" >IF age <= 37.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row4_col4" class="data row4 col4" >0.46850</td>
      <td id="T_cf7e2_row4_col5" class="data row4 col5" >0.55083</td>
      <td id="T_cf7e2_row4_col6" class="data row4 col6" >0.89258</td>
      <td id="T_cf7e2_row4_col7" class="data row4 col7" >3</td>
      <td id="T_cf7e2_row4_col8" class="data row4 col8" >2</td>
      <td id="T_cf7e2_row4_col9" class="data row4 col9" >2.08425</td>
      <td id="T_cf7e2_row4_col10" class="data row4 col10" >False</td>
      <td id="T_cf7e2_row4_col11" class="data row4 col11" >0.51184</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_cf7e2_row5_col0" class="data row5 col0" >18073</td>
      <td id="T_cf7e2_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_cf7e2_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_cf7e2_row5_col3" class="data row5 col3" >IF capital.gain <= 5013.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row5_col4" class="data row5 col4" >0.95077</td>
      <td id="T_cf7e2_row5_col5" class="data row5 col5" >0.99659</td>
      <td id="T_cf7e2_row5_col6" class="data row5 col6" >0.79575</td>
      <td id="T_cf7e2_row5_col7" class="data row5 col7" >1</td>
      <td id="T_cf7e2_row5_col8" class="data row5 col8" >0</td>
      <td id="T_cf7e2_row5_col9" class="data row5 col9" >57.71988</td>
      <td id="T_cf7e2_row5_col10" class="data row5 col10" >False</td>
      <td id="T_cf7e2_row5_col11" class="data row5 col11" >0.18417</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_cf7e2_row6_col0" class="data row6 col0" >18073</td>
      <td id="T_cf7e2_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_cf7e2_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_cf7e2_row6_col3" class="data row6 col3" >IF capital.gain <= 7298.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row6_col4" class="data row6 col4" >0.96363</td>
      <td id="T_cf7e2_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_cf7e2_row6_col6" class="data row6 col6" >0.78696</td>
      <td id="T_cf7e2_row6_col7" class="data row6 col7" >1</td>
      <td id="T_cf7e2_row6_col8" class="data row6 col8" >0</td>
      <td id="T_cf7e2_row6_col9" class="data row6 col9" >56.74821</td>
      <td id="T_cf7e2_row6_col10" class="data row6 col10" >False</td>
      <td id="T_cf7e2_row6_col11" class="data row6 col11" >0.19184</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_cf7e2_row7_col0" class="data row7 col0" >18073</td>
      <td id="T_cf7e2_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_cf7e2_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_cf7e2_row7_col3" class="data row7 col3" >IF capital.gain <= 7688.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row7_col4" class="data row7 col4" >0.97306</td>
      <td id="T_cf7e2_row7_col5" class="data row7 col5" >0.99919</td>
      <td id="T_cf7e2_row7_col6" class="data row7 col6" >0.77956</td>
      <td id="T_cf7e2_row7_col7" class="data row7 col7" >1</td>
      <td id="T_cf7e2_row7_col8" class="data row7 col8" >0</td>
      <td id="T_cf7e2_row7_col9" class="data row7 col9" >55.95721</td>
      <td id="T_cf7e2_row7_col10" class="data row7 col10" >False</td>
      <td id="T_cf7e2_row7_col11" class="data row7 col11" >0.19901</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_cf7e2_row8_col0" class="data row8 col0" >18073</td>
      <td id="T_cf7e2_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_cf7e2_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_cf7e2_row8_col3" class="data row8 col3" >IF capital.gain <= 6696.0568 THEN class = <=50K</td>
      <td id="T_cf7e2_row8_col4" class="data row8 col4" >0.95534</td>
      <td id="T_cf7e2_row8_col5" class="data row8 col5" >0.99775</td>
      <td id="T_cf7e2_row8_col6" class="data row8 col6" >0.79287</td>
      <td id="T_cf7e2_row8_col7" class="data row8 col7" >1</td>
      <td id="T_cf7e2_row8_col8" class="data row8 col8" >0</td>
      <td id="T_cf7e2_row8_col9" class="data row8 col9" >56.41765</td>
      <td id="T_cf7e2_row8_col10" class="data row8 col10" >False</td>
      <td id="T_cf7e2_row8_col11" class="data row8 col11" >0.18654</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_cf7e2_row9_col0" class="data row9 col0" >18073</td>
      <td id="T_cf7e2_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_cf7e2_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_cf7e2_row9_col3" class="data row9 col3" >IF capital.gain <= 3464.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_cf7e2_row9_col4" class="data row9 col4" >0.32011</td>
      <td id="T_cf7e2_row9_col5" class="data row9 col5" >0.40802</td>
      <td id="T_cf7e2_row9_col6" class="data row9 col6" >0.96765</td>
      <td id="T_cf7e2_row9_col7" class="data row9 col7" >2</td>
      <td id="T_cf7e2_row9_col8" class="data row9 col8" >0</td>
      <td id="T_cf7e2_row9_col9" class="data row9 col9" >57.99263</td>
      <td id="T_cf7e2_row9_col10" class="data row9 col10" >False</td>
      <td id="T_cf7e2_row9_col11" class="data row9 col11" >0.65304</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_cf7e2_row10_col0" class="data row10 col0" >18073</td>
      <td id="T_cf7e2_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_cf7e2_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_cf7e2_row10_col3" class="data row10 col3" >IF capital.gain <= 7426.083 THEN class = <=50K</td>
      <td id="T_cf7e2_row10_col4" class="data row10 col4" >0.96363</td>
      <td id="T_cf7e2_row10_col5" class="data row10 col5" >0.99890</td>
      <td id="T_cf7e2_row10_col6" class="data row10 col6" >0.78696</td>
      <td id="T_cf7e2_row10_col7" class="data row10 col7" >1</td>
      <td id="T_cf7e2_row10_col8" class="data row10 col8" >0</td>
      <td id="T_cf7e2_row10_col9" class="data row10 col9" >11.73073</td>
      <td id="T_cf7e2_row10_col10" class="data row10 col10" >False</td>
      <td id="T_cf7e2_row10_col11" class="data row10 col11" >0.19184</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_cf7e2_row11_col0" class="data row11 col0" >18073</td>
      <td id="T_cf7e2_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_cf7e2_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_cf7e2_row11_col3" class="data row11 col3" >IF capital.gain <= 5756.5615 THEN class = <=50K</td>
      <td id="T_cf7e2_row11_col4" class="data row11 col4" >0.95450</td>
      <td id="T_cf7e2_row11_col5" class="data row11 col5" >0.99717</td>
      <td id="T_cf7e2_row11_col6" class="data row11 col6" >0.79311</td>
      <td id="T_cf7e2_row11_col7" class="data row11 col7" >1</td>
      <td id="T_cf7e2_row11_col8" class="data row11 col8" >0</td>
      <td id="T_cf7e2_row11_col9" class="data row11 col9" >11.80187</td>
      <td id="T_cf7e2_row11_col10" class="data row11 col10" >False</td>
      <td id="T_cf7e2_row11_col11" class="data row11 col11" >0.18639</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_cf7e2_row12_col0" class="data row12 col0" >18073</td>
      <td id="T_cf7e2_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_cf7e2_row12_col2" class="data row12 col2" >LORE_SA5</td>
      <td id="T_cf7e2_row12_col3" class="data row12 col3" >IF capital.gain <= 6554.2478 THEN class = <=50K</td>
      <td id="T_cf7e2_row12_col4" class="data row12 col4" >0.95534</td>
      <td id="T_cf7e2_row12_col5" class="data row12 col5" >0.99775</td>
      <td id="T_cf7e2_row12_col6" class="data row12 col6" >0.79287</td>
      <td id="T_cf7e2_row12_col7" class="data row12 col7" >1</td>
      <td id="T_cf7e2_row12_col8" class="data row12 col8" >0</td>
      <td id="T_cf7e2_row12_col9" class="data row12 col9" >11.71087</td>
      <td id="T_cf7e2_row12_col10" class="data row12 col10" >False</td>
      <td id="T_cf7e2_row12_col11" class="data row12 col11" >0.18654</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_cf7e2_row13_col0" class="data row13 col0" >18073</td>
      <td id="T_cf7e2_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_cf7e2_row13_col2" class="data row13 col2" >EXPLAN2</td>
      <td id="T_cf7e2_row13_col3" class="data row13 col3" >IF age <= 38.0 AND capital.gain <= 0.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_cf7e2_row13_col4" class="data row13 col4" >0.27431</td>
      <td id="T_cf7e2_row13_col5" class="data row13 col5" >0.35358</td>
      <td id="T_cf7e2_row13_col6" class="data row13 col6" >0.97857</td>
      <td id="T_cf7e2_row13_col7" class="data row13 col7" >3</td>
      <td id="T_cf7e2_row13_col8" class="data row13 col8" >0</td>
      <td id="T_cf7e2_row13_col9" class="data row13 col9" >0.88249</td>
      <td id="T_cf7e2_row13_col10" class="data row13 col10" >False</td>
      <td id="T_cf7e2_row13_col11" class="data row13 col11" >0.69875</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_cf7e2_row14_col0" class="data row14 col0" >18073</td>
      <td id="T_cf7e2_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_cf7e2_row14_col2" class="data row14 col2" >EXPLAN4</td>
      <td id="T_cf7e2_row14_col3" class="data row14 col3" >IF age <= 34.1577 AND capital.gain <= 403.4566 THEN class = <=50K</td>
      <td id="T_cf7e2_row14_col4" class="data row14 col4" >0.41260</td>
      <td id="T_cf7e2_row14_col5" class="data row14 col5" >0.49396</td>
      <td id="T_cf7e2_row14_col6" class="data row14 col6" >0.90887</td>
      <td id="T_cf7e2_row14_col7" class="data row14 col7" >2</td>
      <td id="T_cf7e2_row14_col8" class="data row14 col8" >0</td>
      <td id="T_cf7e2_row14_col9" class="data row14 col9" >1.05122</td>
      <td id="T_cf7e2_row14_col10" class="data row14 col10" >False</td>
      <td id="T_cf7e2_row14_col11" class="data row14 col11" >0.56478</td>
    </tr>
    <tr>
      <th id="T_cf7e2_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_cf7e2_row15_col0" class="data row15 col0" >18073</td>
      <td id="T_cf7e2_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_cf7e2_row15_col2" class="data row15 col2" >EXPLAN5</td>
      <td id="T_cf7e2_row15_col3" class="data row15 col3" >IF age <= 35.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_cf7e2_row15_col4" class="data row15 col4" >0.43726</td>
      <td id="T_cf7e2_row15_col5" class="data row15 col5" >0.51916</td>
      <td id="T_cf7e2_row15_col6" class="data row15 col6" >0.90136</td>
      <td id="T_cf7e2_row15_col7" class="data row15 col7" >2</td>
      <td id="T_cf7e2_row15_col8" class="data row15 col8" >0</td>
      <td id="T_cf7e2_row15_col9" class="data row15 col9" >1.34529</td>
      <td id="T_cf7e2_row15_col10" class="data row15 col10" >False</td>
      <td id="T_cf7e2_row15_col11" class="data row15 col11" >0.54133</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_31.png)
    



### Rules for Instance 18073, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.97306, Pre: 0.97857), Unique rules (diffrent features)



<style type="text/css">
#T_e14de_row3_col0, #T_e14de_row3_col1, #T_e14de_row3_col2, #T_e14de_row3_col3, #T_e14de_row3_col4, #T_e14de_row3_col5, #T_e14de_row3_col6, #T_e14de_row3_col7, #T_e14de_row3_col8, #T_e14de_row3_col9, #T_e14de_row3_col10, #T_e14de_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_e14de">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e14de_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e14de_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e14de_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e14de_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e14de_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e14de_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e14de_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e14de_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e14de_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e14de_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e14de_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_e14de_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e14de_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e14de_row0_col0" class="data row0 col0" >18073</td>
      <td id="T_e14de_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_e14de_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_e14de_row0_col3" class="data row0 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e14de_row0_col4" class="data row0 col4" >0.52422</td>
      <td id="T_e14de_row0_col5" class="data row0 col5" >0.59186</td>
      <td id="T_e14de_row0_col6" class="data row0 col6" >0.85713</td>
      <td id="T_e14de_row0_col7" class="data row0 col7" >4</td>
      <td id="T_e14de_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e14de_row0_col9" class="data row0 col9" >0.74556</td>
      <td id="T_e14de_row0_col10" class="data row0 col10" >False</td>
      <td id="T_e14de_row0_col11" class="data row0 col11" >0.46498</td>
    </tr>
    <tr>
      <th id="T_e14de_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e14de_row1_col0" class="data row1 col0" >18073</td>
      <td id="T_e14de_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_e14de_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_e14de_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_e14de_row1_col4" class="data row1 col4" >0.66269</td>
      <td id="T_e14de_row1_col5" class="data row1 col5" >0.73091</td>
      <td id="T_e14de_row1_col6" class="data row1 col6" >0.83733</td>
      <td id="T_e14de_row1_col7" class="data row1 col7" >3</td>
      <td id="T_e14de_row1_col8" class="data row1 col8" >0</td>
      <td id="T_e14de_row1_col9" class="data row1 col9" >0.63635</td>
      <td id="T_e14de_row1_col10" class="data row1 col10" >False</td>
      <td id="T_e14de_row1_col11" class="data row1 col11" >0.34100</td>
    </tr>
    <tr>
      <th id="T_e14de_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e14de_row2_col0" class="data row2 col0" >18073</td>
      <td id="T_e14de_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_e14de_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_e14de_row2_col3" class="data row2 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_e14de_row2_col4" class="data row2 col4" >0.69595</td>
      <td id="T_e14de_row2_col5" class="data row2 col5" >0.75357</td>
      <td id="T_e14de_row2_col6" class="data row2 col6" >0.82203</td>
      <td id="T_e14de_row2_col7" class="data row2 col7" >2</td>
      <td id="T_e14de_row2_col8" class="data row2 col8" >0</td>
      <td id="T_e14de_row2_col9" class="data row2 col9" >0.64159</td>
      <td id="T_e14de_row2_col10" class="data row2 col10" >False</td>
      <td id="T_e14de_row2_col11" class="data row2 col11" >0.31827</td>
    </tr>
    <tr>
      <th id="T_e14de_level0_row3" class="row_heading level0 row3" >5</th>
      <td id="T_e14de_row3_col0" class="data row3 col0" >18073</td>
      <td id="T_e14de_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_e14de_row3_col2" class="data row3 col2" >LORE1</td>
      <td id="T_e14de_row3_col3" class="data row3 col3" >IF capital.gain <= 5013.0 THEN class = <=50K</td>
      <td id="T_e14de_row3_col4" class="data row3 col4" >0.95077</td>
      <td id="T_e14de_row3_col5" class="data row3 col5" >0.99659</td>
      <td id="T_e14de_row3_col6" class="data row3 col6" >0.79575</td>
      <td id="T_e14de_row3_col7" class="data row3 col7" >1</td>
      <td id="T_e14de_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e14de_row3_col9" class="data row3 col9" >57.71988</td>
      <td id="T_e14de_row3_col10" class="data row3 col10" >False</td>
      <td id="T_e14de_row3_col11" class="data row3 col11" >0.18417</td>
    </tr>
    <tr>
      <th id="T_e14de_level0_row4" class="row_heading level0 row4" >9</th>
      <td id="T_e14de_row4_col0" class="data row4 col0" >18073</td>
      <td id="T_e14de_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_e14de_row4_col2" class="data row4 col2" >LORE5</td>
      <td id="T_e14de_row4_col3" class="data row4 col3" >IF capital.gain <= 3464.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_e14de_row4_col4" class="data row4 col4" >0.32011</td>
      <td id="T_e14de_row4_col5" class="data row4 col5" >0.40802</td>
      <td id="T_e14de_row4_col6" class="data row4 col6" >0.96765</td>
      <td id="T_e14de_row4_col7" class="data row4 col7" >2</td>
      <td id="T_e14de_row4_col8" class="data row4 col8" >0</td>
      <td id="T_e14de_row4_col9" class="data row4 col9" >57.99263</td>
      <td id="T_e14de_row4_col10" class="data row4 col10" >False</td>
      <td id="T_e14de_row4_col11" class="data row4 col11" >0.65304</td>
    </tr>
    <tr>
      <th id="T_e14de_level0_row5" class="row_heading level0 row5" >13</th>
      <td id="T_e14de_row5_col0" class="data row5 col0" >18073</td>
      <td id="T_e14de_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_e14de_row5_col2" class="data row5 col2" >EXPLAN2</td>
      <td id="T_e14de_row5_col3" class="data row5 col3" >IF age <= 38.0 AND capital.gain <= 0.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_e14de_row5_col4" class="data row5 col4" >0.27431</td>
      <td id="T_e14de_row5_col5" class="data row5 col5" >0.35358</td>
      <td id="T_e14de_row5_col6" class="data row5 col6" >0.97857</td>
      <td id="T_e14de_row5_col7" class="data row5 col7" >3</td>
      <td id="T_e14de_row5_col8" class="data row5 col8" >0</td>
      <td id="T_e14de_row5_col9" class="data row5 col9" >0.88249</td>
      <td id="T_e14de_row5_col10" class="data row5 col10" >False</td>
      <td id="T_e14de_row5_col11" class="data row5 col11" >0.69875</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_34.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_35.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_36.png)
    



### Rules for Instance 18073, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99919, Pre: 0.97857, Len: 0.77956)



<style type="text/css">
#T_31acb_row5_col0, #T_31acb_row5_col1, #T_31acb_row5_col2, #T_31acb_row5_col3, #T_31acb_row5_col4, #T_31acb_row5_col5, #T_31acb_row5_col6, #T_31acb_row5_col7, #T_31acb_row5_col8, #T_31acb_row5_col9, #T_31acb_row5_col10, #T_31acb_row5_col11 {
  font-weight: bold;
}
</style>
<table id="T_31acb">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_31acb_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_31acb_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_31acb_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_31acb_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_31acb_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_31acb_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_31acb_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_31acb_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_31acb_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_31acb_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_31acb_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_31acb_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_31acb_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_31acb_row0_col0" class="data row0 col0" >18073</td>
      <td id="T_31acb_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_31acb_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_31acb_row0_col3" class="data row0 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_31acb_row0_col4" class="data row0 col4" >0.52422</td>
      <td id="T_31acb_row0_col5" class="data row0 col5" >0.59186</td>
      <td id="T_31acb_row0_col6" class="data row0 col6" >0.85713</td>
      <td id="T_31acb_row0_col7" class="data row0 col7" >4</td>
      <td id="T_31acb_row0_col8" class="data row0 col8" >0</td>
      <td id="T_31acb_row0_col9" class="data row0 col9" >0.74556</td>
      <td id="T_31acb_row0_col10" class="data row0 col10" >False</td>
      <td id="T_31acb_row0_col11" class="data row0 col11" >3.24837</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_31acb_row1_col0" class="data row1 col0" >18073</td>
      <td id="T_31acb_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_31acb_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_31acb_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_31acb_row1_col4" class="data row1 col4" >0.66269</td>
      <td id="T_31acb_row1_col5" class="data row1 col5" >0.73091</td>
      <td id="T_31acb_row1_col6" class="data row1 col6" >0.83733</td>
      <td id="T_31acb_row1_col7" class="data row1 col7" >3</td>
      <td id="T_31acb_row1_col8" class="data row1 col8" >0</td>
      <td id="T_31acb_row1_col9" class="data row1 col9" >0.63635</td>
      <td id="T_31acb_row1_col10" class="data row1 col10" >False</td>
      <td id="T_31acb_row1_col11" class="data row1 col11" >2.24104</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_31acb_row2_col0" class="data row2 col0" >18073</td>
      <td id="T_31acb_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_31acb_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_31acb_row2_col3" class="data row2 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_31acb_row2_col4" class="data row2 col4" >0.69595</td>
      <td id="T_31acb_row2_col5" class="data row2 col5" >0.75357</td>
      <td id="T_31acb_row2_col6" class="data row2 col6" >0.82203</td>
      <td id="T_31acb_row2_col7" class="data row2 col7" >2</td>
      <td id="T_31acb_row2_col8" class="data row2 col8" >0</td>
      <td id="T_31acb_row2_col9" class="data row2 col9" >0.64159</td>
      <td id="T_31acb_row2_col10" class="data row2 col10" >False</td>
      <td id="T_31acb_row2_col11" class="data row2 col11" >1.25471</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_31acb_row3_col0" class="data row3 col0" >18073</td>
      <td id="T_31acb_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_31acb_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_31acb_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_31acb_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_31acb_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_31acb_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_31acb_row3_col7" class="data row3 col7" >2</td>
      <td id="T_31acb_row3_col8" class="data row3 col8" >0</td>
      <td id="T_31acb_row3_col9" class="data row3 col9" >0.67650</td>
      <td id="T_31acb_row3_col10" class="data row3 col10" >False</td>
      <td id="T_31acb_row3_col11" class="data row3 col11" >1.29843</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_31acb_row4_col0" class="data row4 col0" >18073</td>
      <td id="T_31acb_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_31acb_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_31acb_row4_col3" class="data row4 col3" >IF age <= 37.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_31acb_row4_col4" class="data row4 col4" >0.46850</td>
      <td id="T_31acb_row4_col5" class="data row4 col5" >0.55083</td>
      <td id="T_31acb_row4_col6" class="data row4 col6" >0.89258</td>
      <td id="T_31acb_row4_col7" class="data row4 col7" >3</td>
      <td id="T_31acb_row4_col8" class="data row4 col8" >2</td>
      <td id="T_31acb_row4_col9" class="data row4 col9" >2.08425</td>
      <td id="T_31acb_row4_col10" class="data row4 col10" >False</td>
      <td id="T_31acb_row4_col11" class="data row4 col11" >2.26689</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_31acb_row5_col0" class="data row5 col0" >18073</td>
      <td id="T_31acb_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_31acb_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_31acb_row5_col3" class="data row5 col3" >IF capital.gain <= 5013.0 THEN class = <=50K</td>
      <td id="T_31acb_row5_col4" class="data row5 col4" >0.95077</td>
      <td id="T_31acb_row5_col5" class="data row5 col5" >0.99659</td>
      <td id="T_31acb_row5_col6" class="data row5 col6" >0.79575</td>
      <td id="T_31acb_row5_col7" class="data row5 col7" >1</td>
      <td id="T_31acb_row5_col8" class="data row5 col8" >0</td>
      <td id="T_31acb_row5_col9" class="data row5 col9" >57.71988</td>
      <td id="T_31acb_row5_col10" class="data row5 col10" >False</td>
      <td id="T_31acb_row5_col11" class="data row5 col11" >0.28640</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_31acb_row6_col0" class="data row6 col0" >18073</td>
      <td id="T_31acb_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_31acb_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_31acb_row6_col3" class="data row6 col3" >IF capital.gain <= 7298.0 THEN class = <=50K</td>
      <td id="T_31acb_row6_col4" class="data row6 col4" >0.96363</td>
      <td id="T_31acb_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_31acb_row6_col6" class="data row6 col6" >0.78696</td>
      <td id="T_31acb_row6_col7" class="data row6 col7" >1</td>
      <td id="T_31acb_row6_col8" class="data row6 col8" >0</td>
      <td id="T_31acb_row6_col9" class="data row6 col9" >56.74821</td>
      <td id="T_31acb_row6_col10" class="data row6 col10" >False</td>
      <td id="T_31acb_row6_col11" class="data row6 col11" >0.29208</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_31acb_row7_col0" class="data row7 col0" >18073</td>
      <td id="T_31acb_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_31acb_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_31acb_row7_col3" class="data row7 col3" >IF capital.gain <= 7688.0 THEN class = <=50K</td>
      <td id="T_31acb_row7_col4" class="data row7 col4" >0.97306</td>
      <td id="T_31acb_row7_col5" class="data row7 col5" >0.99919</td>
      <td id="T_31acb_row7_col6" class="data row7 col6" >0.77956</td>
      <td id="T_31acb_row7_col7" class="data row7 col7" >1</td>
      <td id="T_31acb_row7_col8" class="data row7 col8" >0</td>
      <td id="T_31acb_row7_col9" class="data row7 col9" >55.95721</td>
      <td id="T_31acb_row7_col10" class="data row7 col10" >False</td>
      <td id="T_31acb_row7_col11" class="data row7 col11" >0.29698</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_31acb_row8_col0" class="data row8 col0" >18073</td>
      <td id="T_31acb_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_31acb_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_31acb_row8_col3" class="data row8 col3" >IF capital.gain <= 6696.0568 THEN class = <=50K</td>
      <td id="T_31acb_row8_col4" class="data row8 col4" >0.95534</td>
      <td id="T_31acb_row8_col5" class="data row8 col5" >0.99775</td>
      <td id="T_31acb_row8_col6" class="data row8 col6" >0.79287</td>
      <td id="T_31acb_row8_col7" class="data row8 col7" >1</td>
      <td id="T_31acb_row8_col8" class="data row8 col8" >0</td>
      <td id="T_31acb_row8_col9" class="data row8 col9" >56.41765</td>
      <td id="T_31acb_row8_col10" class="data row8 col10" >False</td>
      <td id="T_31acb_row8_col11" class="data row8 col11" >0.28824</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_31acb_row9_col0" class="data row9 col0" >18073</td>
      <td id="T_31acb_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_31acb_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_31acb_row9_col3" class="data row9 col3" >IF capital.gain <= 3464.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_31acb_row9_col4" class="data row9 col4" >0.32011</td>
      <td id="T_31acb_row9_col5" class="data row9 col5" >0.40802</td>
      <td id="T_31acb_row9_col6" class="data row9 col6" >0.96765</td>
      <td id="T_31acb_row9_col7" class="data row9 col7" >2</td>
      <td id="T_31acb_row9_col8" class="data row9 col8" >0</td>
      <td id="T_31acb_row9_col9" class="data row9 col9" >57.99263</td>
      <td id="T_31acb_row9_col10" class="data row9 col10" >False</td>
      <td id="T_31acb_row9_col11" class="data row9 col11" >1.35612</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_31acb_row10_col0" class="data row10 col0" >18073</td>
      <td id="T_31acb_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_31acb_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_31acb_row10_col3" class="data row10 col3" >IF capital.gain <= 7426.083 THEN class = <=50K</td>
      <td id="T_31acb_row10_col4" class="data row10 col4" >0.96363</td>
      <td id="T_31acb_row10_col5" class="data row10 col5" >0.99890</td>
      <td id="T_31acb_row10_col6" class="data row10 col6" >0.78696</td>
      <td id="T_31acb_row10_col7" class="data row10 col7" >1</td>
      <td id="T_31acb_row10_col8" class="data row10 col8" >0</td>
      <td id="T_31acb_row10_col9" class="data row10 col9" >11.73073</td>
      <td id="T_31acb_row10_col10" class="data row10 col10" >False</td>
      <td id="T_31acb_row10_col11" class="data row10 col11" >0.29208</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_31acb_row11_col0" class="data row11 col0" >18073</td>
      <td id="T_31acb_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_31acb_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_31acb_row11_col3" class="data row11 col3" >IF capital.gain <= 5756.5615 THEN class = <=50K</td>
      <td id="T_31acb_row11_col4" class="data row11 col4" >0.95450</td>
      <td id="T_31acb_row11_col5" class="data row11 col5" >0.99717</td>
      <td id="T_31acb_row11_col6" class="data row11 col6" >0.79311</td>
      <td id="T_31acb_row11_col7" class="data row11 col7" >1</td>
      <td id="T_31acb_row11_col8" class="data row11 col8" >0</td>
      <td id="T_31acb_row11_col9" class="data row11 col9" >11.80187</td>
      <td id="T_31acb_row11_col10" class="data row11 col10" >False</td>
      <td id="T_31acb_row11_col11" class="data row11 col11" >0.28809</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_31acb_row12_col0" class="data row12 col0" >18073</td>
      <td id="T_31acb_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_31acb_row12_col2" class="data row12 col2" >LORE_SA5</td>
      <td id="T_31acb_row12_col3" class="data row12 col3" >IF capital.gain <= 6554.2478 THEN class = <=50K</td>
      <td id="T_31acb_row12_col4" class="data row12 col4" >0.95534</td>
      <td id="T_31acb_row12_col5" class="data row12 col5" >0.99775</td>
      <td id="T_31acb_row12_col6" class="data row12 col6" >0.79287</td>
      <td id="T_31acb_row12_col7" class="data row12 col7" >1</td>
      <td id="T_31acb_row12_col8" class="data row12 col8" >0</td>
      <td id="T_31acb_row12_col9" class="data row12 col9" >11.71087</td>
      <td id="T_31acb_row12_col10" class="data row12 col10" >False</td>
      <td id="T_31acb_row12_col11" class="data row12 col11" >0.28824</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_31acb_row13_col0" class="data row13 col0" >18073</td>
      <td id="T_31acb_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_31acb_row13_col2" class="data row13 col2" >EXPLAN2</td>
      <td id="T_31acb_row13_col3" class="data row13 col3" >IF age <= 38.0 AND capital.gain <= 0.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_31acb_row13_col4" class="data row13 col4" >0.27431</td>
      <td id="T_31acb_row13_col5" class="data row13 col5" >0.35358</td>
      <td id="T_31acb_row13_col6" class="data row13 col6" >0.97857</td>
      <td id="T_31acb_row13_col7" class="data row13 col7" >3</td>
      <td id="T_31acb_row13_col8" class="data row13 col8" >0</td>
      <td id="T_31acb_row13_col9" class="data row13 col9" >0.88249</td>
      <td id="T_31acb_row13_col10" class="data row13 col10" >False</td>
      <td id="T_31acb_row13_col11" class="data row13 col11" >2.31239</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_31acb_row14_col0" class="data row14 col0" >18073</td>
      <td id="T_31acb_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_31acb_row14_col2" class="data row14 col2" >EXPLAN4</td>
      <td id="T_31acb_row14_col3" class="data row14 col3" >IF age <= 34.1577 AND capital.gain <= 403.4566 THEN class = <=50K</td>
      <td id="T_31acb_row14_col4" class="data row14 col4" >0.41260</td>
      <td id="T_31acb_row14_col5" class="data row14 col5" >0.49396</td>
      <td id="T_31acb_row14_col6" class="data row14 col6" >0.90887</td>
      <td id="T_31acb_row14_col7" class="data row14 col7" >2</td>
      <td id="T_31acb_row14_col8" class="data row14 col8" >0</td>
      <td id="T_31acb_row14_col9" class="data row14 col9" >1.05122</td>
      <td id="T_31acb_row14_col10" class="data row14 col10" >False</td>
      <td id="T_31acb_row14_col11" class="data row14 col11" >1.32272</td>
    </tr>
    <tr>
      <th id="T_31acb_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_31acb_row15_col0" class="data row15 col0" >18073</td>
      <td id="T_31acb_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_31acb_row15_col2" class="data row15 col2" >EXPLAN5</td>
      <td id="T_31acb_row15_col3" class="data row15 col3" >IF age <= 35.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_31acb_row15_col4" class="data row15 col4" >0.43726</td>
      <td id="T_31acb_row15_col5" class="data row15 col5" >0.51916</td>
      <td id="T_31acb_row15_col6" class="data row15 col6" >0.90136</td>
      <td id="T_31acb_row15_col7" class="data row15 col7" >2</td>
      <td id="T_31acb_row15_col8" class="data row15 col8" >0</td>
      <td id="T_31acb_row15_col9" class="data row15 col9" >1.34529</td>
      <td id="T_31acb_row15_col10" class="data row15 col10" >False</td>
      <td id="T_31acb_row15_col11" class="data row15 col11" >1.31372</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 18073, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99919, Pre: 0.97857), Unique rules (diffrent features)



<style type="text/css">
#T_1029e_row3_col0, #T_1029e_row3_col1, #T_1029e_row3_col2, #T_1029e_row3_col3, #T_1029e_row3_col4, #T_1029e_row3_col5, #T_1029e_row3_col6, #T_1029e_row3_col7, #T_1029e_row3_col8, #T_1029e_row3_col9, #T_1029e_row3_col10, #T_1029e_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_1029e">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1029e_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_1029e_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_1029e_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_1029e_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_1029e_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_1029e_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_1029e_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_1029e_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_1029e_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_1029e_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_1029e_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_1029e_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1029e_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_1029e_row0_col0" class="data row0 col0" >18073</td>
      <td id="T_1029e_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_1029e_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_1029e_row0_col3" class="data row0 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_1029e_row0_col4" class="data row0 col4" >0.52422</td>
      <td id="T_1029e_row0_col5" class="data row0 col5" >0.59186</td>
      <td id="T_1029e_row0_col6" class="data row0 col6" >0.85713</td>
      <td id="T_1029e_row0_col7" class="data row0 col7" >4</td>
      <td id="T_1029e_row0_col8" class="data row0 col8" >0</td>
      <td id="T_1029e_row0_col9" class="data row0 col9" >0.74556</td>
      <td id="T_1029e_row0_col10" class="data row0 col10" >False</td>
      <td id="T_1029e_row0_col11" class="data row0 col11" >3.24837</td>
    </tr>
    <tr>
      <th id="T_1029e_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_1029e_row1_col0" class="data row1 col0" >18073</td>
      <td id="T_1029e_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_1029e_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_1029e_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_1029e_row1_col4" class="data row1 col4" >0.66269</td>
      <td id="T_1029e_row1_col5" class="data row1 col5" >0.73091</td>
      <td id="T_1029e_row1_col6" class="data row1 col6" >0.83733</td>
      <td id="T_1029e_row1_col7" class="data row1 col7" >3</td>
      <td id="T_1029e_row1_col8" class="data row1 col8" >0</td>
      <td id="T_1029e_row1_col9" class="data row1 col9" >0.63635</td>
      <td id="T_1029e_row1_col10" class="data row1 col10" >False</td>
      <td id="T_1029e_row1_col11" class="data row1 col11" >2.24104</td>
    </tr>
    <tr>
      <th id="T_1029e_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_1029e_row2_col0" class="data row2 col0" >18073</td>
      <td id="T_1029e_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_1029e_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_1029e_row2_col3" class="data row2 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_1029e_row2_col4" class="data row2 col4" >0.69595</td>
      <td id="T_1029e_row2_col5" class="data row2 col5" >0.75357</td>
      <td id="T_1029e_row2_col6" class="data row2 col6" >0.82203</td>
      <td id="T_1029e_row2_col7" class="data row2 col7" >2</td>
      <td id="T_1029e_row2_col8" class="data row2 col8" >0</td>
      <td id="T_1029e_row2_col9" class="data row2 col9" >0.64159</td>
      <td id="T_1029e_row2_col10" class="data row2 col10" >False</td>
      <td id="T_1029e_row2_col11" class="data row2 col11" >1.25471</td>
    </tr>
    <tr>
      <th id="T_1029e_level0_row3" class="row_heading level0 row3" >5</th>
      <td id="T_1029e_row3_col0" class="data row3 col0" >18073</td>
      <td id="T_1029e_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_1029e_row3_col2" class="data row3 col2" >LORE1</td>
      <td id="T_1029e_row3_col3" class="data row3 col3" >IF capital.gain <= 5013.0 THEN class = <=50K</td>
      <td id="T_1029e_row3_col4" class="data row3 col4" >0.95077</td>
      <td id="T_1029e_row3_col5" class="data row3 col5" >0.99659</td>
      <td id="T_1029e_row3_col6" class="data row3 col6" >0.79575</td>
      <td id="T_1029e_row3_col7" class="data row3 col7" >1</td>
      <td id="T_1029e_row3_col8" class="data row3 col8" >0</td>
      <td id="T_1029e_row3_col9" class="data row3 col9" >57.71988</td>
      <td id="T_1029e_row3_col10" class="data row3 col10" >False</td>
      <td id="T_1029e_row3_col11" class="data row3 col11" >0.28640</td>
    </tr>
    <tr>
      <th id="T_1029e_level0_row4" class="row_heading level0 row4" >9</th>
      <td id="T_1029e_row4_col0" class="data row4 col0" >18073</td>
      <td id="T_1029e_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_1029e_row4_col2" class="data row4 col2" >LORE5</td>
      <td id="T_1029e_row4_col3" class="data row4 col3" >IF capital.gain <= 3464.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_1029e_row4_col4" class="data row4 col4" >0.32011</td>
      <td id="T_1029e_row4_col5" class="data row4 col5" >0.40802</td>
      <td id="T_1029e_row4_col6" class="data row4 col6" >0.96765</td>
      <td id="T_1029e_row4_col7" class="data row4 col7" >2</td>
      <td id="T_1029e_row4_col8" class="data row4 col8" >0</td>
      <td id="T_1029e_row4_col9" class="data row4 col9" >57.99263</td>
      <td id="T_1029e_row4_col10" class="data row4 col10" >False</td>
      <td id="T_1029e_row4_col11" class="data row4 col11" >1.35612</td>
    </tr>
    <tr>
      <th id="T_1029e_level0_row5" class="row_heading level0 row5" >13</th>
      <td id="T_1029e_row5_col0" class="data row5 col0" >18073</td>
      <td id="T_1029e_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_1029e_row5_col2" class="data row5 col2" >EXPLAN2</td>
      <td id="T_1029e_row5_col3" class="data row5 col3" >IF age <= 38.0 AND capital.gain <= 0.0 AND marital.status = Never-married THEN class = <=50K</td>
      <td id="T_1029e_row5_col4" class="data row5 col4" >0.27431</td>
      <td id="T_1029e_row5_col5" class="data row5 col5" >0.35358</td>
      <td id="T_1029e_row5_col6" class="data row5 col6" >0.97857</td>
      <td id="T_1029e_row5_col7" class="data row5 col7" >3</td>
      <td id="T_1029e_row5_col8" class="data row5 col8" >0</td>
      <td id="T_1029e_row5_col9" class="data row5 col9" >0.88249</td>
      <td id="T_1029e_row5_col10" class="data row5 col10" >False</td>
      <td id="T_1029e_row5_col11" class="data row5 col11" >2.31239</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_41.png)
    



## Instance 652 (Original: >50K , Predicted: >50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>44.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>Bachelors</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>13</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Married-civ-spouse</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Exec-managerial</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Husband</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>1902.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>50.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 652



<style type="text/css">
</style>
<table id="T_65b03">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_65b03_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_65b03_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_65b03_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_65b03_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_65b03_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_65b03_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_65b03_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_65b03_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_65b03_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_65b03_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_65b03_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_65b03_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_65b03_row0_col0" class="data row0 col0" >652</td>
      <td id="T_65b03_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_65b03_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_65b03_row0_col3" class="data row0 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row0_col4" class="data row0 col4" >0.00584</td>
      <td id="T_65b03_row0_col5" class="data row0 col5" >0.02113</td>
      <td id="T_65b03_row0_col6" class="data row0 col6" >0.87218</td>
      <td id="T_65b03_row0_col7" class="data row0 col7" >6</td>
      <td id="T_65b03_row0_col8" class="data row0 col8" >0</td>
      <td id="T_65b03_row0_col9" class="data row0 col9" >5.08110</td>
      <td id="T_65b03_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_65b03_row1_col0" class="data row1 col0" >652</td>
      <td id="T_65b03_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_65b03_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_65b03_row1_col3" class="data row1 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row1_col4" class="data row1 col4" >0.01329</td>
      <td id="T_65b03_row1_col5" class="data row1 col5" >0.04828</td>
      <td id="T_65b03_row1_col6" class="data row1 col6" >0.87459</td>
      <td id="T_65b03_row1_col7" class="data row1 col7" >5</td>
      <td id="T_65b03_row1_col8" class="data row1 col8" >0</td>
      <td id="T_65b03_row1_col9" class="data row1 col9" >5.47437</td>
      <td id="T_65b03_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_65b03_row2_col0" class="data row2 col0" >652</td>
      <td id="T_65b03_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_65b03_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_65b03_row2_col3" class="data row2 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_65b03_row2_col4" class="data row2 col4" >0.01386</td>
      <td id="T_65b03_row2_col5" class="data row2 col5" >0.04992</td>
      <td id="T_65b03_row2_col6" class="data row2 col6" >0.86709</td>
      <td id="T_65b03_row2_col7" class="data row2 col7" >5</td>
      <td id="T_65b03_row2_col8" class="data row2 col8" >1</td>
      <td id="T_65b03_row2_col9" class="data row2 col9" >11.39965</td>
      <td id="T_65b03_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_65b03_row3_col0" class="data row3 col0" >652</td>
      <td id="T_65b03_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_65b03_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_65b03_row3_col3" class="data row3 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 45.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row3_col4" class="data row3 col4" >0.01075</td>
      <td id="T_65b03_row3_col5" class="data row3 col5" >0.03917</td>
      <td id="T_65b03_row3_col6" class="data row3 col6" >0.87755</td>
      <td id="T_65b03_row3_col7" class="data row3 col7" >5</td>
      <td id="T_65b03_row3_col8" class="data row3 col8" >0</td>
      <td id="T_65b03_row3_col9" class="data row3 col9" >5.98049</td>
      <td id="T_65b03_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_65b03_row4_col0" class="data row4 col0" >652</td>
      <td id="T_65b03_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_65b03_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_65b03_row4_col3" class="data row4 col3" >IF age > 28.0 AND education.num = 13.0 AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_65b03_row4_col4" class="data row4 col4" >0.01386</td>
      <td id="T_65b03_row4_col5" class="data row4 col5" >0.04992</td>
      <td id="T_65b03_row4_col6" class="data row4 col6" >0.86709</td>
      <td id="T_65b03_row4_col7" class="data row4 col7" >5</td>
      <td id="T_65b03_row4_col8" class="data row4 col8" >0</td>
      <td id="T_65b03_row4_col9" class="data row4 col9" >7.24741</td>
      <td id="T_65b03_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_65b03_row5_col0" class="data row5 col0" >652</td>
      <td id="T_65b03_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_65b03_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_65b03_row5_col3" class="data row5 col3" >IF hours.per.week > 42.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_65b03_row5_col4" class="data row5 col4" >0.16519</td>
      <td id="T_65b03_row5_col5" class="data row5 col5" >0.38422</td>
      <td id="T_65b03_row5_col6" class="data row5 col6" >0.56016</td>
      <td id="T_65b03_row5_col7" class="data row5 col7" >4</td>
      <td id="T_65b03_row5_col8" class="data row5 col8" >0</td>
      <td id="T_65b03_row5_col9" class="data row5 col9" >57.73337</td>
      <td id="T_65b03_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_65b03_row6_col0" class="data row6 col0" >652</td>
      <td id="T_65b03_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_65b03_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_65b03_row6_col3" class="data row6 col3" >IF age > 43.0 AND age <= 44.0 AND hours.per.week > 49.0481 THEN class = <=50K</td>
      <td id="T_65b03_row6_col4" class="data row6 col4" >0.00562</td>
      <td id="T_65b03_row6_col5" class="data row6 col5" >0.00387</td>
      <td id="T_65b03_row6_col6" class="data row6 col6" >0.52344</td>
      <td id="T_65b03_row6_col7" class="data row6 col7" >3</td>
      <td id="T_65b03_row6_col8" class="data row6 col8" >0</td>
      <td id="T_65b03_row6_col9" class="data row6 col9" >56.85025</td>
      <td id="T_65b03_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_65b03_row7_col0" class="data row7 col0" >652</td>
      <td id="T_65b03_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_65b03_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_65b03_row7_col3" class="data row7 col3" >IF hours.per.week > 30.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row7_col4" class="data row7 col4" >0.37575</td>
      <td id="T_65b03_row7_col5" class="data row7 col5" >0.73529</td>
      <td id="T_65b03_row7_col6" class="data row7 col6" >0.47128</td>
      <td id="T_65b03_row7_col7" class="data row7 col7" >3</td>
      <td id="T_65b03_row7_col8" class="data row7 col8" >0</td>
      <td id="T_65b03_row7_col9" class="data row7 col9" >57.10518</td>
      <td id="T_65b03_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_65b03_row8_col0" class="data row8 col0" >652</td>
      <td id="T_65b03_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_65b03_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_65b03_row8_col3" class="data row8 col3" >IF marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row8_col4" class="data row8 col4" >0.40453</td>
      <td id="T_65b03_row8_col5" class="data row8 col5" >0.75715</td>
      <td id="T_65b03_row8_col6" class="data row8 col6" >0.45076</td>
      <td id="T_65b03_row8_col7" class="data row8 col7" >2</td>
      <td id="T_65b03_row8_col8" class="data row8 col8" >0</td>
      <td id="T_65b03_row8_col9" class="data row8 col9" >56.87405</td>
      <td id="T_65b03_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_65b03_row9_col0" class="data row9 col0" >652</td>
      <td id="T_65b03_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_65b03_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_65b03_row9_col3" class="data row9 col3" >IF hours.per.week <= 50.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row9_col4" class="data row9 col4" >0.33591</td>
      <td id="T_65b03_row9_col5" class="data row9 col5" >0.59883</td>
      <td id="T_65b03_row9_col6" class="data row9 col6" >0.42934</td>
      <td id="T_65b03_row9_col7" class="data row9 col7" >3</td>
      <td id="T_65b03_row9_col8" class="data row9 col8" >0</td>
      <td id="T_65b03_row9_col9" class="data row9 col9" >55.86448</td>
      <td id="T_65b03_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_65b03_row10_col0" class="data row10 col0" >652</td>
      <td id="T_65b03_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_65b03_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_65b03_row10_col3" class="data row10 col3" >IF education != 10th AND education.num != 11.0 AND hours.per.week > 19.3595 AND marital.status = Married-civ-spouse AND relationship != Own-child THEN class = >50K</td>
      <td id="T_65b03_row10_col4" class="data row10 col4" >0.41234</td>
      <td id="T_65b03_row10_col5" class="data row10 col5" >0.79395</td>
      <td id="T_65b03_row10_col6" class="data row10 col6" >0.46372</td>
      <td id="T_65b03_row10_col7" class="data row10 col7" >5</td>
      <td id="T_65b03_row10_col8" class="data row10 col8" >0</td>
      <td id="T_65b03_row10_col9" class="data row10 col9" >14.09076</td>
      <td id="T_65b03_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_65b03_row11_col0" class="data row11 col0" >652</td>
      <td id="T_65b03_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_65b03_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_65b03_row11_col3" class="data row11 col3" >IF age > 36.8148 AND education.num != 6.0 AND marital.status != Never-married AND marital.status != Married-spouse-absent AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row11_col4" class="data row11 col4" >0.27067</td>
      <td id="T_65b03_row11_col5" class="data row11 col5" >0.57807</td>
      <td id="T_65b03_row11_col6" class="data row11 col6" >0.51435</td>
      <td id="T_65b03_row11_col7" class="data row11 col7" >5</td>
      <td id="T_65b03_row11_col8" class="data row11 col8" >0</td>
      <td id="T_65b03_row11_col9" class="data row11 col9" >13.75957</td>
      <td id="T_65b03_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_65b03_row12_col0" class="data row12 col0" >652</td>
      <td id="T_65b03_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_65b03_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_65b03_row12_col3" class="data row12 col3" >IF capital.loss > 260.2145 AND marital.status != Never-married AND marital.status = Married-civ-spouse AND occupation != Transport-moving AND relationship != Own-child THEN class = >50K</td>
      <td id="T_65b03_row12_col4" class="data row12 col4" >0.02808</td>
      <td id="T_65b03_row12_col5" class="data row12 col5" >0.08362</td>
      <td id="T_65b03_row12_col6" class="data row12 col6" >0.71719</td>
      <td id="T_65b03_row12_col7" class="data row12 col7" >5</td>
      <td id="T_65b03_row12_col8" class="data row12 col8" >0</td>
      <td id="T_65b03_row12_col9" class="data row12 col9" >13.74144</td>
      <td id="T_65b03_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_65b03_row13_col0" class="data row13 col0" >652</td>
      <td id="T_65b03_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_65b03_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_65b03_row13_col3" class="data row13 col3" >IF age > 22.6687 AND hours.per.week > 28.5804 AND marital.status != Never-married AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row13_col4" class="data row13 col4" >0.38044</td>
      <td id="T_65b03_row13_col5" class="data row13 col5" >0.74094</td>
      <td id="T_65b03_row13_col6" class="data row13 col6" >0.46903</td>
      <td id="T_65b03_row13_col7" class="data row13 col7" >4</td>
      <td id="T_65b03_row13_col8" class="data row13 col8" >0</td>
      <td id="T_65b03_row13_col9" class="data row13 col9" >13.66720</td>
      <td id="T_65b03_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_65b03_row14_col0" class="data row14 col0" >652</td>
      <td id="T_65b03_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_65b03_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_65b03_row14_col3" class="data row14 col3" >IF education != Assoc-acdm AND education.num != 8.0 AND education.num != 9.0 AND marital.status != Never-married AND occupation != Farming-fishing AND relationship != Own-child THEN class = >50K</td>
      <td id="T_65b03_row14_col4" class="data row14 col4" >0.39523</td>
      <td id="T_65b03_row14_col5" class="data row14 col5" >0.68373</td>
      <td id="T_65b03_row14_col6" class="data row14 col6" >0.41663</td>
      <td id="T_65b03_row14_col7" class="data row14 col7" >6</td>
      <td id="T_65b03_row14_col8" class="data row14 col8" >0</td>
      <td id="T_65b03_row14_col9" class="data row14 col9" >13.07513</td>
      <td id="T_65b03_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_65b03_row15_col0" class="data row15 col0" >652</td>
      <td id="T_65b03_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_65b03_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_65b03_row15_col3" class="data row15 col3" >IF age > 27.0 AND capital.gain <= 1173.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row15_col4" class="data row15 col4" >0.02365</td>
      <td id="T_65b03_row15_col5" class="data row15 col5" >0.07196</td>
      <td id="T_65b03_row15_col6" class="data row15 col6" >0.73284</td>
      <td id="T_65b03_row15_col7" class="data row15 col7" >6</td>
      <td id="T_65b03_row15_col8" class="data row15 col8" >0</td>
      <td id="T_65b03_row15_col9" class="data row15 col9" >4.72258</td>
      <td id="T_65b03_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_65b03_row16_col0" class="data row16 col0" >652</td>
      <td id="T_65b03_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_65b03_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_65b03_row16_col3" class="data row16 col3" >IF age > 36.0 AND age <= 45.9777 AND capital.gain <= 1506.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_65b03_row16_col4" class="data row16 col4" >0.01759</td>
      <td id="T_65b03_row16_col5" class="data row16 col5" >0.05119</td>
      <td id="T_65b03_row16_col6" class="data row16 col6" >0.70075</td>
      <td id="T_65b03_row16_col7" class="data row16 col7" >7</td>
      <td id="T_65b03_row16_col8" class="data row16 col8" >0</td>
      <td id="T_65b03_row16_col9" class="data row16 col9" >4.92132</td>
      <td id="T_65b03_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_65b03_row17_col0" class="data row17 col0" >652</td>
      <td id="T_65b03_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_65b03_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_65b03_row17_col3" class="data row17 col3" >IF age > 32.2285 AND capital.gain <= 1173.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND race = White AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row17_col4" class="data row17 col4" >0.04414</td>
      <td id="T_65b03_row17_col5" class="data row17 col5" >0.12461</td>
      <td id="T_65b03_row17_col6" class="data row17 col6" >0.67992</td>
      <td id="T_65b03_row17_col7" class="data row17 col7" >6</td>
      <td id="T_65b03_row17_col8" class="data row17 col8" >0</td>
      <td id="T_65b03_row17_col9" class="data row17 col9" >3.81111</td>
      <td id="T_65b03_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_65b03_row18_col0" class="data row18 col0" >652</td>
      <td id="T_65b03_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_65b03_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_65b03_row18_col3" class="data row18 col3" >IF age > 34.0 AND capital.gain <= 657.9029 AND capital.loss > 1881.0024 AND hours.per.week > 35.1456 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row18_col4" class="data row18 col4" >0.01509</td>
      <td id="T_65b03_row18_col5" class="data row18 col5" >0.05429</td>
      <td id="T_65b03_row18_col6" class="data row18 col6" >0.86628</td>
      <td id="T_65b03_row18_col7" class="data row18 col7" >6</td>
      <td id="T_65b03_row18_col8" class="data row18 col8" >0</td>
      <td id="T_65b03_row18_col9" class="data row18 col9" >3.61566</td>
      <td id="T_65b03_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_65b03_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_65b03_row19_col0" class="data row19 col0" >652</td>
      <td id="T_65b03_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_65b03_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_65b03_row19_col3" class="data row19 col3" >IF age > 33.0 AND capital.loss > 1775.7448 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_65b03_row19_col4" class="data row19 col4" >0.01803</td>
      <td id="T_65b03_row19_col5" class="data row19 col5" >0.06504</td>
      <td id="T_65b03_row19_col6" class="data row19 col6" >0.86861</td>
      <td id="T_65b03_row19_col7" class="data row19 col7" >4</td>
      <td id="T_65b03_row19_col8" class="data row19 col8" >0</td>
      <td id="T_65b03_row19_col9" class="data row19 col9" >5.70397</td>
      <td id="T_65b03_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 652, Correct Prediction



<style type="text/css">
</style>
<table id="T_cde8a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cde8a_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_cde8a_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_cde8a_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_cde8a_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_cde8a_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_cde8a_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_cde8a_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_cde8a_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_cde8a_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_cde8a_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_cde8a_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cde8a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_cde8a_row0_col0" class="data row0 col0" >652</td>
      <td id="T_cde8a_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_cde8a_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_cde8a_row0_col3" class="data row0 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row0_col4" class="data row0 col4" >0.00584</td>
      <td id="T_cde8a_row0_col5" class="data row0 col5" >0.02113</td>
      <td id="T_cde8a_row0_col6" class="data row0 col6" >0.87218</td>
      <td id="T_cde8a_row0_col7" class="data row0 col7" >6</td>
      <td id="T_cde8a_row0_col8" class="data row0 col8" >0</td>
      <td id="T_cde8a_row0_col9" class="data row0 col9" >5.08110</td>
      <td id="T_cde8a_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_cde8a_row1_col0" class="data row1 col0" >652</td>
      <td id="T_cde8a_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_cde8a_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_cde8a_row1_col3" class="data row1 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row1_col4" class="data row1 col4" >0.01329</td>
      <td id="T_cde8a_row1_col5" class="data row1 col5" >0.04828</td>
      <td id="T_cde8a_row1_col6" class="data row1 col6" >0.87459</td>
      <td id="T_cde8a_row1_col7" class="data row1 col7" >5</td>
      <td id="T_cde8a_row1_col8" class="data row1 col8" >0</td>
      <td id="T_cde8a_row1_col9" class="data row1 col9" >5.47437</td>
      <td id="T_cde8a_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_cde8a_row2_col0" class="data row2 col0" >652</td>
      <td id="T_cde8a_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_cde8a_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_cde8a_row2_col3" class="data row2 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_cde8a_row2_col4" class="data row2 col4" >0.01386</td>
      <td id="T_cde8a_row2_col5" class="data row2 col5" >0.04992</td>
      <td id="T_cde8a_row2_col6" class="data row2 col6" >0.86709</td>
      <td id="T_cde8a_row2_col7" class="data row2 col7" >5</td>
      <td id="T_cde8a_row2_col8" class="data row2 col8" >1</td>
      <td id="T_cde8a_row2_col9" class="data row2 col9" >11.39965</td>
      <td id="T_cde8a_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_cde8a_row3_col0" class="data row3 col0" >652</td>
      <td id="T_cde8a_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_cde8a_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_cde8a_row3_col3" class="data row3 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 45.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row3_col4" class="data row3 col4" >0.01075</td>
      <td id="T_cde8a_row3_col5" class="data row3 col5" >0.03917</td>
      <td id="T_cde8a_row3_col6" class="data row3 col6" >0.87755</td>
      <td id="T_cde8a_row3_col7" class="data row3 col7" >5</td>
      <td id="T_cde8a_row3_col8" class="data row3 col8" >0</td>
      <td id="T_cde8a_row3_col9" class="data row3 col9" >5.98049</td>
      <td id="T_cde8a_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_cde8a_row4_col0" class="data row4 col0" >652</td>
      <td id="T_cde8a_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_cde8a_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_cde8a_row4_col3" class="data row4 col3" >IF age > 28.0 AND education.num = 13.0 AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_cde8a_row4_col4" class="data row4 col4" >0.01386</td>
      <td id="T_cde8a_row4_col5" class="data row4 col5" >0.04992</td>
      <td id="T_cde8a_row4_col6" class="data row4 col6" >0.86709</td>
      <td id="T_cde8a_row4_col7" class="data row4 col7" >5</td>
      <td id="T_cde8a_row4_col8" class="data row4 col8" >0</td>
      <td id="T_cde8a_row4_col9" class="data row4 col9" >7.24741</td>
      <td id="T_cde8a_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_cde8a_row5_col0" class="data row5 col0" >652</td>
      <td id="T_cde8a_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_cde8a_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_cde8a_row5_col3" class="data row5 col3" >IF hours.per.week > 42.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_cde8a_row5_col4" class="data row5 col4" >0.16519</td>
      <td id="T_cde8a_row5_col5" class="data row5 col5" >0.38422</td>
      <td id="T_cde8a_row5_col6" class="data row5 col6" >0.56016</td>
      <td id="T_cde8a_row5_col7" class="data row5 col7" >4</td>
      <td id="T_cde8a_row5_col8" class="data row5 col8" >0</td>
      <td id="T_cde8a_row5_col9" class="data row5 col9" >57.73337</td>
      <td id="T_cde8a_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_cde8a_row6_col0" class="data row6 col0" >652</td>
      <td id="T_cde8a_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_cde8a_row6_col2" class="data row6 col2" >LORE3</td>
      <td id="T_cde8a_row6_col3" class="data row6 col3" >IF hours.per.week > 30.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row6_col4" class="data row6 col4" >0.37575</td>
      <td id="T_cde8a_row6_col5" class="data row6 col5" >0.73529</td>
      <td id="T_cde8a_row6_col6" class="data row6 col6" >0.47128</td>
      <td id="T_cde8a_row6_col7" class="data row6 col7" >3</td>
      <td id="T_cde8a_row6_col8" class="data row6 col8" >0</td>
      <td id="T_cde8a_row6_col9" class="data row6 col9" >57.10518</td>
      <td id="T_cde8a_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_cde8a_row7_col0" class="data row7 col0" >652</td>
      <td id="T_cde8a_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_cde8a_row7_col2" class="data row7 col2" >LORE4</td>
      <td id="T_cde8a_row7_col3" class="data row7 col3" >IF marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row7_col4" class="data row7 col4" >0.40453</td>
      <td id="T_cde8a_row7_col5" class="data row7 col5" >0.75715</td>
      <td id="T_cde8a_row7_col6" class="data row7 col6" >0.45076</td>
      <td id="T_cde8a_row7_col7" class="data row7 col7" >2</td>
      <td id="T_cde8a_row7_col8" class="data row7 col8" >0</td>
      <td id="T_cde8a_row7_col9" class="data row7 col9" >56.87405</td>
      <td id="T_cde8a_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_cde8a_row8_col0" class="data row8 col0" >652</td>
      <td id="T_cde8a_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_cde8a_row8_col2" class="data row8 col2" >LORE5</td>
      <td id="T_cde8a_row8_col3" class="data row8 col3" >IF hours.per.week <= 50.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row8_col4" class="data row8 col4" >0.33591</td>
      <td id="T_cde8a_row8_col5" class="data row8 col5" >0.59883</td>
      <td id="T_cde8a_row8_col6" class="data row8 col6" >0.42934</td>
      <td id="T_cde8a_row8_col7" class="data row8 col7" >3</td>
      <td id="T_cde8a_row8_col8" class="data row8 col8" >0</td>
      <td id="T_cde8a_row8_col9" class="data row8 col9" >55.86448</td>
      <td id="T_cde8a_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_cde8a_row9_col0" class="data row9 col0" >652</td>
      <td id="T_cde8a_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_cde8a_row9_col2" class="data row9 col2" >LORE_SA1</td>
      <td id="T_cde8a_row9_col3" class="data row9 col3" >IF education != 10th AND education.num != 11.0 AND hours.per.week > 19.3595 AND marital.status = Married-civ-spouse AND relationship != Own-child THEN class = >50K</td>
      <td id="T_cde8a_row9_col4" class="data row9 col4" >0.41234</td>
      <td id="T_cde8a_row9_col5" class="data row9 col5" >0.79395</td>
      <td id="T_cde8a_row9_col6" class="data row9 col6" >0.46372</td>
      <td id="T_cde8a_row9_col7" class="data row9 col7" >5</td>
      <td id="T_cde8a_row9_col8" class="data row9 col8" >0</td>
      <td id="T_cde8a_row9_col9" class="data row9 col9" >14.09076</td>
      <td id="T_cde8a_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_cde8a_row10_col0" class="data row10 col0" >652</td>
      <td id="T_cde8a_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_cde8a_row10_col2" class="data row10 col2" >LORE_SA2</td>
      <td id="T_cde8a_row10_col3" class="data row10 col3" >IF age > 36.8148 AND education.num != 6.0 AND marital.status != Never-married AND marital.status != Married-spouse-absent AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row10_col4" class="data row10 col4" >0.27067</td>
      <td id="T_cde8a_row10_col5" class="data row10 col5" >0.57807</td>
      <td id="T_cde8a_row10_col6" class="data row10 col6" >0.51435</td>
      <td id="T_cde8a_row10_col7" class="data row10 col7" >5</td>
      <td id="T_cde8a_row10_col8" class="data row10 col8" >0</td>
      <td id="T_cde8a_row10_col9" class="data row10 col9" >13.75957</td>
      <td id="T_cde8a_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_cde8a_row11_col0" class="data row11 col0" >652</td>
      <td id="T_cde8a_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_cde8a_row11_col2" class="data row11 col2" >LORE_SA3</td>
      <td id="T_cde8a_row11_col3" class="data row11 col3" >IF capital.loss > 260.2145 AND marital.status != Never-married AND marital.status = Married-civ-spouse AND occupation != Transport-moving AND relationship != Own-child THEN class = >50K</td>
      <td id="T_cde8a_row11_col4" class="data row11 col4" >0.02808</td>
      <td id="T_cde8a_row11_col5" class="data row11 col5" >0.08362</td>
      <td id="T_cde8a_row11_col6" class="data row11 col6" >0.71719</td>
      <td id="T_cde8a_row11_col7" class="data row11 col7" >5</td>
      <td id="T_cde8a_row11_col8" class="data row11 col8" >0</td>
      <td id="T_cde8a_row11_col9" class="data row11 col9" >13.74144</td>
      <td id="T_cde8a_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_cde8a_row12_col0" class="data row12 col0" >652</td>
      <td id="T_cde8a_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_cde8a_row12_col2" class="data row12 col2" >LORE_SA4</td>
      <td id="T_cde8a_row12_col3" class="data row12 col3" >IF age > 22.6687 AND hours.per.week > 28.5804 AND marital.status != Never-married AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row12_col4" class="data row12 col4" >0.38044</td>
      <td id="T_cde8a_row12_col5" class="data row12 col5" >0.74094</td>
      <td id="T_cde8a_row12_col6" class="data row12 col6" >0.46903</td>
      <td id="T_cde8a_row12_col7" class="data row12 col7" >4</td>
      <td id="T_cde8a_row12_col8" class="data row12 col8" >0</td>
      <td id="T_cde8a_row12_col9" class="data row12 col9" >13.66720</td>
      <td id="T_cde8a_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_cde8a_row13_col0" class="data row13 col0" >652</td>
      <td id="T_cde8a_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_cde8a_row13_col2" class="data row13 col2" >LORE_SA5</td>
      <td id="T_cde8a_row13_col3" class="data row13 col3" >IF education != Assoc-acdm AND education.num != 8.0 AND education.num != 9.0 AND marital.status != Never-married AND occupation != Farming-fishing AND relationship != Own-child THEN class = >50K</td>
      <td id="T_cde8a_row13_col4" class="data row13 col4" >0.39523</td>
      <td id="T_cde8a_row13_col5" class="data row13 col5" >0.68373</td>
      <td id="T_cde8a_row13_col6" class="data row13 col6" >0.41663</td>
      <td id="T_cde8a_row13_col7" class="data row13 col7" >6</td>
      <td id="T_cde8a_row13_col8" class="data row13 col8" >0</td>
      <td id="T_cde8a_row13_col9" class="data row13 col9" >13.07513</td>
      <td id="T_cde8a_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_cde8a_row14_col0" class="data row14 col0" >652</td>
      <td id="T_cde8a_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_cde8a_row14_col2" class="data row14 col2" >EXPLAN1</td>
      <td id="T_cde8a_row14_col3" class="data row14 col3" >IF age > 27.0 AND capital.gain <= 1173.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row14_col4" class="data row14 col4" >0.02365</td>
      <td id="T_cde8a_row14_col5" class="data row14 col5" >0.07196</td>
      <td id="T_cde8a_row14_col6" class="data row14 col6" >0.73284</td>
      <td id="T_cde8a_row14_col7" class="data row14 col7" >6</td>
      <td id="T_cde8a_row14_col8" class="data row14 col8" >0</td>
      <td id="T_cde8a_row14_col9" class="data row14 col9" >4.72258</td>
      <td id="T_cde8a_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_cde8a_row15_col0" class="data row15 col0" >652</td>
      <td id="T_cde8a_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_cde8a_row15_col2" class="data row15 col2" >EXPLAN2</td>
      <td id="T_cde8a_row15_col3" class="data row15 col3" >IF age > 36.0 AND age <= 45.9777 AND capital.gain <= 1506.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_cde8a_row15_col4" class="data row15 col4" >0.01759</td>
      <td id="T_cde8a_row15_col5" class="data row15 col5" >0.05119</td>
      <td id="T_cde8a_row15_col6" class="data row15 col6" >0.70075</td>
      <td id="T_cde8a_row15_col7" class="data row15 col7" >7</td>
      <td id="T_cde8a_row15_col8" class="data row15 col8" >0</td>
      <td id="T_cde8a_row15_col9" class="data row15 col9" >4.92132</td>
      <td id="T_cde8a_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_cde8a_row16_col0" class="data row16 col0" >652</td>
      <td id="T_cde8a_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_cde8a_row16_col2" class="data row16 col2" >EXPLAN3</td>
      <td id="T_cde8a_row16_col3" class="data row16 col3" >IF age > 32.2285 AND capital.gain <= 1173.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND race = White AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row16_col4" class="data row16 col4" >0.04414</td>
      <td id="T_cde8a_row16_col5" class="data row16 col5" >0.12461</td>
      <td id="T_cde8a_row16_col6" class="data row16 col6" >0.67992</td>
      <td id="T_cde8a_row16_col7" class="data row16 col7" >6</td>
      <td id="T_cde8a_row16_col8" class="data row16 col8" >0</td>
      <td id="T_cde8a_row16_col9" class="data row16 col9" >3.81111</td>
      <td id="T_cde8a_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_cde8a_row17_col0" class="data row17 col0" >652</td>
      <td id="T_cde8a_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_cde8a_row17_col2" class="data row17 col2" >EXPLAN4</td>
      <td id="T_cde8a_row17_col3" class="data row17 col3" >IF age > 34.0 AND capital.gain <= 657.9029 AND capital.loss > 1881.0024 AND hours.per.week > 35.1456 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row17_col4" class="data row17 col4" >0.01509</td>
      <td id="T_cde8a_row17_col5" class="data row17 col5" >0.05429</td>
      <td id="T_cde8a_row17_col6" class="data row17 col6" >0.86628</td>
      <td id="T_cde8a_row17_col7" class="data row17 col7" >6</td>
      <td id="T_cde8a_row17_col8" class="data row17 col8" >0</td>
      <td id="T_cde8a_row17_col9" class="data row17 col9" >3.61566</td>
      <td id="T_cde8a_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_cde8a_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_cde8a_row18_col0" class="data row18 col0" >652</td>
      <td id="T_cde8a_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_cde8a_row18_col2" class="data row18 col2" >EXPLAN5</td>
      <td id="T_cde8a_row18_col3" class="data row18 col3" >IF age > 33.0 AND capital.loss > 1775.7448 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_cde8a_row18_col4" class="data row18 col4" >0.01803</td>
      <td id="T_cde8a_row18_col5" class="data row18 col5" >0.06504</td>
      <td id="T_cde8a_row18_col6" class="data row18 col6" >0.86861</td>
      <td id="T_cde8a_row18_col7" class="data row18 col7" >4</td>
      <td id="T_cde8a_row18_col8" class="data row18 col8" >0</td>
      <td id="T_cde8a_row18_col9" class="data row18 col9" >5.70397</td>
      <td id="T_cde8a_row18_col10" class="data row18 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 652, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_9a689">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9a689_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_9a689_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_9a689_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_9a689_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_9a689_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_9a689_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_9a689_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_9a689_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_9a689_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_9a689_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_9a689_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9a689_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_9a689_row0_col0" class="data row0 col0" >652</td>
      <td id="T_9a689_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_9a689_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_9a689_row0_col3" class="data row0 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row0_col4" class="data row0 col4" >0.01329</td>
      <td id="T_9a689_row0_col5" class="data row0 col5" >0.04828</td>
      <td id="T_9a689_row0_col6" class="data row0 col6" >0.87459</td>
      <td id="T_9a689_row0_col7" class="data row0 col7" >5</td>
      <td id="T_9a689_row0_col8" class="data row0 col8" >0</td>
      <td id="T_9a689_row0_col9" class="data row0 col9" >5.47437</td>
      <td id="T_9a689_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_9a689_row1_col0" class="data row1 col0" >652</td>
      <td id="T_9a689_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_9a689_row1_col2" class="data row1 col2" >ANCHOR3</td>
      <td id="T_9a689_row1_col3" class="data row1 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_9a689_row1_col4" class="data row1 col4" >0.01386</td>
      <td id="T_9a689_row1_col5" class="data row1 col5" >0.04992</td>
      <td id="T_9a689_row1_col6" class="data row1 col6" >0.86709</td>
      <td id="T_9a689_row1_col7" class="data row1 col7" >5</td>
      <td id="T_9a689_row1_col8" class="data row1 col8" >1</td>
      <td id="T_9a689_row1_col9" class="data row1 col9" >11.39965</td>
      <td id="T_9a689_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_9a689_row2_col0" class="data row2 col0" >652</td>
      <td id="T_9a689_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_9a689_row2_col2" class="data row2 col2" >ANCHOR4</td>
      <td id="T_9a689_row2_col3" class="data row2 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 45.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row2_col4" class="data row2 col4" >0.01075</td>
      <td id="T_9a689_row2_col5" class="data row2 col5" >0.03917</td>
      <td id="T_9a689_row2_col6" class="data row2 col6" >0.87755</td>
      <td id="T_9a689_row2_col7" class="data row2 col7" >5</td>
      <td id="T_9a689_row2_col8" class="data row2 col8" >0</td>
      <td id="T_9a689_row2_col9" class="data row2 col9" >5.98049</td>
      <td id="T_9a689_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_9a689_row3_col0" class="data row3 col0" >652</td>
      <td id="T_9a689_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_9a689_row3_col2" class="data row3 col2" >ANCHOR5</td>
      <td id="T_9a689_row3_col3" class="data row3 col3" >IF age > 28.0 AND education.num = 13.0 AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_9a689_row3_col4" class="data row3 col4" >0.01386</td>
      <td id="T_9a689_row3_col5" class="data row3 col5" >0.04992</td>
      <td id="T_9a689_row3_col6" class="data row3 col6" >0.86709</td>
      <td id="T_9a689_row3_col7" class="data row3 col7" >5</td>
      <td id="T_9a689_row3_col8" class="data row3 col8" >0</td>
      <td id="T_9a689_row3_col9" class="data row3 col9" >7.24741</td>
      <td id="T_9a689_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_9a689_row4_col0" class="data row4 col0" >652</td>
      <td id="T_9a689_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_9a689_row4_col2" class="data row4 col2" >LORE1</td>
      <td id="T_9a689_row4_col3" class="data row4 col3" >IF hours.per.week > 42.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_9a689_row4_col4" class="data row4 col4" >0.16519</td>
      <td id="T_9a689_row4_col5" class="data row4 col5" >0.38422</td>
      <td id="T_9a689_row4_col6" class="data row4 col6" >0.56016</td>
      <td id="T_9a689_row4_col7" class="data row4 col7" >4</td>
      <td id="T_9a689_row4_col8" class="data row4 col8" >0</td>
      <td id="T_9a689_row4_col9" class="data row4 col9" >57.73337</td>
      <td id="T_9a689_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_9a689_row5_col0" class="data row5 col0" >652</td>
      <td id="T_9a689_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_9a689_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_9a689_row5_col3" class="data row5 col3" >IF hours.per.week > 30.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row5_col4" class="data row5 col4" >0.37575</td>
      <td id="T_9a689_row5_col5" class="data row5 col5" >0.73529</td>
      <td id="T_9a689_row5_col6" class="data row5 col6" >0.47128</td>
      <td id="T_9a689_row5_col7" class="data row5 col7" >3</td>
      <td id="T_9a689_row5_col8" class="data row5 col8" >0</td>
      <td id="T_9a689_row5_col9" class="data row5 col9" >57.10518</td>
      <td id="T_9a689_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_9a689_row6_col0" class="data row6 col0" >652</td>
      <td id="T_9a689_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_9a689_row6_col2" class="data row6 col2" >LORE4</td>
      <td id="T_9a689_row6_col3" class="data row6 col3" >IF marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row6_col4" class="data row6 col4" >0.40453</td>
      <td id="T_9a689_row6_col5" class="data row6 col5" >0.75715</td>
      <td id="T_9a689_row6_col6" class="data row6 col6" >0.45076</td>
      <td id="T_9a689_row6_col7" class="data row6 col7" >2</td>
      <td id="T_9a689_row6_col8" class="data row6 col8" >0</td>
      <td id="T_9a689_row6_col9" class="data row6 col9" >56.87405</td>
      <td id="T_9a689_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_9a689_row7_col0" class="data row7 col0" >652</td>
      <td id="T_9a689_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_9a689_row7_col2" class="data row7 col2" >LORE5</td>
      <td id="T_9a689_row7_col3" class="data row7 col3" >IF hours.per.week <= 50.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row7_col4" class="data row7 col4" >0.33591</td>
      <td id="T_9a689_row7_col5" class="data row7 col5" >0.59883</td>
      <td id="T_9a689_row7_col6" class="data row7 col6" >0.42934</td>
      <td id="T_9a689_row7_col7" class="data row7 col7" >3</td>
      <td id="T_9a689_row7_col8" class="data row7 col8" >0</td>
      <td id="T_9a689_row7_col9" class="data row7 col9" >55.86448</td>
      <td id="T_9a689_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_9a689_row8_col0" class="data row8 col0" >652</td>
      <td id="T_9a689_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_9a689_row8_col2" class="data row8 col2" >LORE_SA1</td>
      <td id="T_9a689_row8_col3" class="data row8 col3" >IF education != 10th AND education.num != 11.0 AND hours.per.week > 19.3595 AND marital.status = Married-civ-spouse AND relationship != Own-child THEN class = >50K</td>
      <td id="T_9a689_row8_col4" class="data row8 col4" >0.41234</td>
      <td id="T_9a689_row8_col5" class="data row8 col5" >0.79395</td>
      <td id="T_9a689_row8_col6" class="data row8 col6" >0.46372</td>
      <td id="T_9a689_row8_col7" class="data row8 col7" >5</td>
      <td id="T_9a689_row8_col8" class="data row8 col8" >0</td>
      <td id="T_9a689_row8_col9" class="data row8 col9" >14.09076</td>
      <td id="T_9a689_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_9a689_row9_col0" class="data row9 col0" >652</td>
      <td id="T_9a689_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_9a689_row9_col2" class="data row9 col2" >LORE_SA2</td>
      <td id="T_9a689_row9_col3" class="data row9 col3" >IF age > 36.8148 AND education.num != 6.0 AND marital.status != Never-married AND marital.status != Married-spouse-absent AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row9_col4" class="data row9 col4" >0.27067</td>
      <td id="T_9a689_row9_col5" class="data row9 col5" >0.57807</td>
      <td id="T_9a689_row9_col6" class="data row9 col6" >0.51435</td>
      <td id="T_9a689_row9_col7" class="data row9 col7" >5</td>
      <td id="T_9a689_row9_col8" class="data row9 col8" >0</td>
      <td id="T_9a689_row9_col9" class="data row9 col9" >13.75957</td>
      <td id="T_9a689_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_9a689_row10_col0" class="data row10 col0" >652</td>
      <td id="T_9a689_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_9a689_row10_col2" class="data row10 col2" >LORE_SA3</td>
      <td id="T_9a689_row10_col3" class="data row10 col3" >IF capital.loss > 260.2145 AND marital.status != Never-married AND marital.status = Married-civ-spouse AND occupation != Transport-moving AND relationship != Own-child THEN class = >50K</td>
      <td id="T_9a689_row10_col4" class="data row10 col4" >0.02808</td>
      <td id="T_9a689_row10_col5" class="data row10 col5" >0.08362</td>
      <td id="T_9a689_row10_col6" class="data row10 col6" >0.71719</td>
      <td id="T_9a689_row10_col7" class="data row10 col7" >5</td>
      <td id="T_9a689_row10_col8" class="data row10 col8" >0</td>
      <td id="T_9a689_row10_col9" class="data row10 col9" >13.74144</td>
      <td id="T_9a689_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_9a689_row11_col0" class="data row11 col0" >652</td>
      <td id="T_9a689_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_9a689_row11_col2" class="data row11 col2" >LORE_SA4</td>
      <td id="T_9a689_row11_col3" class="data row11 col3" >IF age > 22.6687 AND hours.per.week > 28.5804 AND marital.status != Never-married AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row11_col4" class="data row11 col4" >0.38044</td>
      <td id="T_9a689_row11_col5" class="data row11 col5" >0.74094</td>
      <td id="T_9a689_row11_col6" class="data row11 col6" >0.46903</td>
      <td id="T_9a689_row11_col7" class="data row11 col7" >4</td>
      <td id="T_9a689_row11_col8" class="data row11 col8" >0</td>
      <td id="T_9a689_row11_col9" class="data row11 col9" >13.66720</td>
      <td id="T_9a689_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_9a689_row12_col0" class="data row12 col0" >652</td>
      <td id="T_9a689_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_9a689_row12_col2" class="data row12 col2" >LORE_SA5</td>
      <td id="T_9a689_row12_col3" class="data row12 col3" >IF education != Assoc-acdm AND education.num != 8.0 AND education.num != 9.0 AND marital.status != Never-married AND occupation != Farming-fishing AND relationship != Own-child THEN class = >50K</td>
      <td id="T_9a689_row12_col4" class="data row12 col4" >0.39523</td>
      <td id="T_9a689_row12_col5" class="data row12 col5" >0.68373</td>
      <td id="T_9a689_row12_col6" class="data row12 col6" >0.41663</td>
      <td id="T_9a689_row12_col7" class="data row12 col7" >6</td>
      <td id="T_9a689_row12_col8" class="data row12 col8" >0</td>
      <td id="T_9a689_row12_col9" class="data row12 col9" >13.07513</td>
      <td id="T_9a689_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_9a689_row13_col0" class="data row13 col0" >652</td>
      <td id="T_9a689_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_9a689_row13_col2" class="data row13 col2" >EXPLAN1</td>
      <td id="T_9a689_row13_col3" class="data row13 col3" >IF age > 27.0 AND capital.gain <= 1173.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row13_col4" class="data row13 col4" >0.02365</td>
      <td id="T_9a689_row13_col5" class="data row13 col5" >0.07196</td>
      <td id="T_9a689_row13_col6" class="data row13 col6" >0.73284</td>
      <td id="T_9a689_row13_col7" class="data row13 col7" >6</td>
      <td id="T_9a689_row13_col8" class="data row13 col8" >0</td>
      <td id="T_9a689_row13_col9" class="data row13 col9" >4.72258</td>
      <td id="T_9a689_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_9a689_row14_col0" class="data row14 col0" >652</td>
      <td id="T_9a689_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_9a689_row14_col2" class="data row14 col2" >EXPLAN2</td>
      <td id="T_9a689_row14_col3" class="data row14 col3" >IF age > 36.0 AND age <= 45.9777 AND capital.gain <= 1506.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_9a689_row14_col4" class="data row14 col4" >0.01759</td>
      <td id="T_9a689_row14_col5" class="data row14 col5" >0.05119</td>
      <td id="T_9a689_row14_col6" class="data row14 col6" >0.70075</td>
      <td id="T_9a689_row14_col7" class="data row14 col7" >7</td>
      <td id="T_9a689_row14_col8" class="data row14 col8" >0</td>
      <td id="T_9a689_row14_col9" class="data row14 col9" >4.92132</td>
      <td id="T_9a689_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_9a689_row15_col0" class="data row15 col0" >652</td>
      <td id="T_9a689_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_9a689_row15_col2" class="data row15 col2" >EXPLAN3</td>
      <td id="T_9a689_row15_col3" class="data row15 col3" >IF age > 32.2285 AND capital.gain <= 1173.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND race = White AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row15_col4" class="data row15 col4" >0.04414</td>
      <td id="T_9a689_row15_col5" class="data row15 col5" >0.12461</td>
      <td id="T_9a689_row15_col6" class="data row15 col6" >0.67992</td>
      <td id="T_9a689_row15_col7" class="data row15 col7" >6</td>
      <td id="T_9a689_row15_col8" class="data row15 col8" >0</td>
      <td id="T_9a689_row15_col9" class="data row15 col9" >3.81111</td>
      <td id="T_9a689_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_9a689_row16_col0" class="data row16 col0" >652</td>
      <td id="T_9a689_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_9a689_row16_col2" class="data row16 col2" >EXPLAN4</td>
      <td id="T_9a689_row16_col3" class="data row16 col3" >IF age > 34.0 AND capital.gain <= 657.9029 AND capital.loss > 1881.0024 AND hours.per.week > 35.1456 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row16_col4" class="data row16 col4" >0.01509</td>
      <td id="T_9a689_row16_col5" class="data row16 col5" >0.05429</td>
      <td id="T_9a689_row16_col6" class="data row16 col6" >0.86628</td>
      <td id="T_9a689_row16_col7" class="data row16 col7" >6</td>
      <td id="T_9a689_row16_col8" class="data row16 col8" >0</td>
      <td id="T_9a689_row16_col9" class="data row16 col9" >3.61566</td>
      <td id="T_9a689_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_9a689_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_9a689_row17_col0" class="data row17 col0" >652</td>
      <td id="T_9a689_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_9a689_row17_col2" class="data row17 col2" >EXPLAN5</td>
      <td id="T_9a689_row17_col3" class="data row17 col3" >IF age > 33.0 AND capital.loss > 1775.7448 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_9a689_row17_col4" class="data row17 col4" >0.01803</td>
      <td id="T_9a689_row17_col5" class="data row17 col5" >0.06504</td>
      <td id="T_9a689_row17_col6" class="data row17 col6" >0.86861</td>
      <td id="T_9a689_row17_col7" class="data row17 col7" >4</td>
      <td id="T_9a689_row17_col8" class="data row17 col8" >0</td>
      <td id="T_9a689_row17_col9" class="data row17 col9" >5.70397</td>
      <td id="T_9a689_row17_col10" class="data row17 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 652, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.41234, Pre: 0.87755)



<style type="text/css">
#T_3c5be_row5_col0, #T_3c5be_row5_col1, #T_3c5be_row5_col2, #T_3c5be_row5_col3, #T_3c5be_row5_col4, #T_3c5be_row5_col5, #T_3c5be_row5_col6, #T_3c5be_row5_col7, #T_3c5be_row5_col8, #T_3c5be_row5_col9, #T_3c5be_row5_col10, #T_3c5be_row5_col11 {
  font-weight: bold;
}
</style>
<table id="T_3c5be">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3c5be_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_3c5be_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_3c5be_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_3c5be_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_3c5be_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_3c5be_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_3c5be_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_3c5be_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_3c5be_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_3c5be_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_3c5be_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_3c5be_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3c5be_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3c5be_row0_col0" class="data row0 col0" >652</td>
      <td id="T_3c5be_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_3c5be_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_3c5be_row0_col3" class="data row0 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row0_col4" class="data row0 col4" >0.01329</td>
      <td id="T_3c5be_row0_col5" class="data row0 col5" >0.04828</td>
      <td id="T_3c5be_row0_col6" class="data row0 col6" >0.87459</td>
      <td id="T_3c5be_row0_col7" class="data row0 col7" >5</td>
      <td id="T_3c5be_row0_col8" class="data row0 col8" >0</td>
      <td id="T_3c5be_row0_col9" class="data row0 col9" >5.47437</td>
      <td id="T_3c5be_row0_col10" class="data row0 col10" >False</td>
      <td id="T_3c5be_row0_col11" class="data row0 col11" >0.39906</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_3c5be_row1_col0" class="data row1 col0" >652</td>
      <td id="T_3c5be_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_3c5be_row1_col2" class="data row1 col2" >ANCHOR4</td>
      <td id="T_3c5be_row1_col3" class="data row1 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 45.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row1_col4" class="data row1 col4" >0.01075</td>
      <td id="T_3c5be_row1_col5" class="data row1 col5" >0.03917</td>
      <td id="T_3c5be_row1_col6" class="data row1 col6" >0.87755</td>
      <td id="T_3c5be_row1_col7" class="data row1 col7" >5</td>
      <td id="T_3c5be_row1_col8" class="data row1 col8" >0</td>
      <td id="T_3c5be_row1_col9" class="data row1 col9" >5.98049</td>
      <td id="T_3c5be_row1_col10" class="data row1 col10" >False</td>
      <td id="T_3c5be_row1_col11" class="data row1 col11" >0.40159</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_3c5be_row2_col0" class="data row2 col0" >652</td>
      <td id="T_3c5be_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_3c5be_row2_col2" class="data row2 col2" >LORE1</td>
      <td id="T_3c5be_row2_col3" class="data row2 col3" >IF hours.per.week > 42.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_3c5be_row2_col4" class="data row2 col4" >0.16519</td>
      <td id="T_3c5be_row2_col5" class="data row2 col5" >0.38422</td>
      <td id="T_3c5be_row2_col6" class="data row2 col6" >0.56016</td>
      <td id="T_3c5be_row2_col7" class="data row2 col7" >4</td>
      <td id="T_3c5be_row2_col8" class="data row2 col8" >0</td>
      <td id="T_3c5be_row2_col9" class="data row2 col9" >57.73337</td>
      <td id="T_3c5be_row2_col10" class="data row2 col10" >False</td>
      <td id="T_3c5be_row2_col11" class="data row2 col11" >0.40227</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_3c5be_row3_col0" class="data row3 col0" >652</td>
      <td id="T_3c5be_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_3c5be_row3_col2" class="data row3 col2" >LORE3</td>
      <td id="T_3c5be_row3_col3" class="data row3 col3" >IF hours.per.week > 30.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row3_col4" class="data row3 col4" >0.37575</td>
      <td id="T_3c5be_row3_col5" class="data row3 col5" >0.73529</td>
      <td id="T_3c5be_row3_col6" class="data row3 col6" >0.47128</td>
      <td id="T_3c5be_row3_col7" class="data row3 col7" >3</td>
      <td id="T_3c5be_row3_col8" class="data row3 col8" >0</td>
      <td id="T_3c5be_row3_col9" class="data row3 col9" >57.10518</td>
      <td id="T_3c5be_row3_col10" class="data row3 col10" >False</td>
      <td id="T_3c5be_row3_col11" class="data row3 col11" >0.40791</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_3c5be_row4_col0" class="data row4 col0" >652</td>
      <td id="T_3c5be_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_3c5be_row4_col2" class="data row4 col2" >LORE_SA1</td>
      <td id="T_3c5be_row4_col3" class="data row4 col3" >IF education != 10th AND education.num != 11.0 AND hours.per.week > 19.3595 AND marital.status = Married-civ-spouse AND relationship != Own-child THEN class = >50K</td>
      <td id="T_3c5be_row4_col4" class="data row4 col4" >0.41234</td>
      <td id="T_3c5be_row4_col5" class="data row4 col5" >0.79395</td>
      <td id="T_3c5be_row4_col6" class="data row4 col6" >0.46372</td>
      <td id="T_3c5be_row4_col7" class="data row4 col7" >5</td>
      <td id="T_3c5be_row4_col8" class="data row4 col8" >0</td>
      <td id="T_3c5be_row4_col9" class="data row4 col9" >14.09076</td>
      <td id="T_3c5be_row4_col10" class="data row4 col10" >False</td>
      <td id="T_3c5be_row4_col11" class="data row4 col11" >0.41383</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_3c5be_row5_col0" class="data row5 col0" >652</td>
      <td id="T_3c5be_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_3c5be_row5_col2" class="data row5 col2" >LORE_SA2</td>
      <td id="T_3c5be_row5_col3" class="data row5 col3" >IF age > 36.8148 AND education.num != 6.0 AND marital.status != Never-married AND marital.status != Married-spouse-absent AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row5_col4" class="data row5 col4" >0.27067</td>
      <td id="T_3c5be_row5_col5" class="data row5 col5" >0.57807</td>
      <td id="T_3c5be_row5_col6" class="data row5 col6" >0.51435</td>
      <td id="T_3c5be_row5_col7" class="data row5 col7" >5</td>
      <td id="T_3c5be_row5_col8" class="data row5 col8" >0</td>
      <td id="T_3c5be_row5_col9" class="data row5 col9" >13.75957</td>
      <td id="T_3c5be_row5_col10" class="data row5 col10" >False</td>
      <td id="T_3c5be_row5_col11" class="data row5 col11" >0.38985</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_3c5be_row6_col0" class="data row6 col0" >652</td>
      <td id="T_3c5be_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_3c5be_row6_col2" class="data row6 col2" >LORE_SA3</td>
      <td id="T_3c5be_row6_col3" class="data row6 col3" >IF capital.loss > 260.2145 AND marital.status != Never-married AND marital.status = Married-civ-spouse AND occupation != Transport-moving AND relationship != Own-child THEN class = >50K</td>
      <td id="T_3c5be_row6_col4" class="data row6 col4" >0.02808</td>
      <td id="T_3c5be_row6_col5" class="data row6 col5" >0.08362</td>
      <td id="T_3c5be_row6_col6" class="data row6 col6" >0.71719</td>
      <td id="T_3c5be_row6_col7" class="data row6 col7" >5</td>
      <td id="T_3c5be_row6_col8" class="data row6 col8" >0</td>
      <td id="T_3c5be_row6_col9" class="data row6 col9" >13.74144</td>
      <td id="T_3c5be_row6_col10" class="data row6 col10" >False</td>
      <td id="T_3c5be_row6_col11" class="data row6 col11" >0.41638</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_3c5be_row7_col0" class="data row7 col0" >652</td>
      <td id="T_3c5be_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_3c5be_row7_col2" class="data row7 col2" >LORE_SA4</td>
      <td id="T_3c5be_row7_col3" class="data row7 col3" >IF age > 22.6687 AND hours.per.week > 28.5804 AND marital.status != Never-married AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row7_col4" class="data row7 col4" >0.38044</td>
      <td id="T_3c5be_row7_col5" class="data row7 col5" >0.74094</td>
      <td id="T_3c5be_row7_col6" class="data row7 col6" >0.46903</td>
      <td id="T_3c5be_row7_col7" class="data row7 col7" >4</td>
      <td id="T_3c5be_row7_col8" class="data row7 col8" >0</td>
      <td id="T_3c5be_row7_col9" class="data row7 col9" >13.66720</td>
      <td id="T_3c5be_row7_col10" class="data row7 col10" >False</td>
      <td id="T_3c5be_row7_col11" class="data row7 col11" >0.40976</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_3c5be_row8_col0" class="data row8 col0" >652</td>
      <td id="T_3c5be_row8_col1" class="data row8 col1" >EXPLAN</td>
      <td id="T_3c5be_row8_col2" class="data row8 col2" >EXPLAN1</td>
      <td id="T_3c5be_row8_col3" class="data row8 col3" >IF age > 27.0 AND capital.gain <= 1173.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row8_col4" class="data row8 col4" >0.02365</td>
      <td id="T_3c5be_row8_col5" class="data row8 col5" >0.07196</td>
      <td id="T_3c5be_row8_col6" class="data row8 col6" >0.73284</td>
      <td id="T_3c5be_row8_col7" class="data row8 col7" >6</td>
      <td id="T_3c5be_row8_col8" class="data row8 col8" >0</td>
      <td id="T_3c5be_row8_col9" class="data row8 col9" >4.72258</td>
      <td id="T_3c5be_row8_col10" class="data row8 col10" >False</td>
      <td id="T_3c5be_row8_col11" class="data row8 col11" >0.41475</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_3c5be_row9_col0" class="data row9 col0" >652</td>
      <td id="T_3c5be_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_3c5be_row9_col2" class="data row9 col2" >EXPLAN3</td>
      <td id="T_3c5be_row9_col3" class="data row9 col3" >IF age > 32.2285 AND capital.gain <= 1173.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND race = White AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row9_col4" class="data row9 col4" >0.04414</td>
      <td id="T_3c5be_row9_col5" class="data row9 col5" >0.12461</td>
      <td id="T_3c5be_row9_col6" class="data row9 col6" >0.67992</td>
      <td id="T_3c5be_row9_col7" class="data row9 col7" >6</td>
      <td id="T_3c5be_row9_col8" class="data row9 col8" >0</td>
      <td id="T_3c5be_row9_col9" class="data row9 col9" >3.81111</td>
      <td id="T_3c5be_row9_col10" class="data row9 col10" >False</td>
      <td id="T_3c5be_row9_col11" class="data row9 col11" >0.41789</td>
    </tr>
    <tr>
      <th id="T_3c5be_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_3c5be_row10_col0" class="data row10 col0" >652</td>
      <td id="T_3c5be_row10_col1" class="data row10 col1" >EXPLAN</td>
      <td id="T_3c5be_row10_col2" class="data row10 col2" >EXPLAN5</td>
      <td id="T_3c5be_row10_col3" class="data row10 col3" >IF age > 33.0 AND capital.loss > 1775.7448 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_3c5be_row10_col4" class="data row10 col4" >0.01803</td>
      <td id="T_3c5be_row10_col5" class="data row10 col5" >0.06504</td>
      <td id="T_3c5be_row10_col6" class="data row10 col6" >0.86861</td>
      <td id="T_3c5be_row10_col7" class="data row10 col7" >4</td>
      <td id="T_3c5be_row10_col8" class="data row10 col8" >0</td>
      <td id="T_3c5be_row10_col9" class="data row10 col9" >5.70397</td>
      <td id="T_3c5be_row10_col10" class="data row10 col10" >False</td>
      <td id="T_3c5be_row10_col11" class="data row10 col11" >0.39441</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_52.png)
    



### Rules for Instance 652, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.41234, Pre: 0.87755), Unique rules (diffrent features)



<style type="text/css">
#T_bade7_row4_col0, #T_bade7_row4_col1, #T_bade7_row4_col2, #T_bade7_row4_col3, #T_bade7_row4_col4, #T_bade7_row4_col5, #T_bade7_row4_col6, #T_bade7_row4_col7, #T_bade7_row4_col8, #T_bade7_row4_col9, #T_bade7_row4_col10, #T_bade7_row4_col11 {
  font-weight: bold;
}
</style>
<table id="T_bade7">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_bade7_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_bade7_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_bade7_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_bade7_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_bade7_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_bade7_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_bade7_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_bade7_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_bade7_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_bade7_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_bade7_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_bade7_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_bade7_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_bade7_row0_col0" class="data row0 col0" >652</td>
      <td id="T_bade7_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_bade7_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_bade7_row0_col3" class="data row0 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_bade7_row0_col4" class="data row0 col4" >0.01329</td>
      <td id="T_bade7_row0_col5" class="data row0 col5" >0.04828</td>
      <td id="T_bade7_row0_col6" class="data row0 col6" >0.87459</td>
      <td id="T_bade7_row0_col7" class="data row0 col7" >5</td>
      <td id="T_bade7_row0_col8" class="data row0 col8" >0</td>
      <td id="T_bade7_row0_col9" class="data row0 col9" >5.47437</td>
      <td id="T_bade7_row0_col10" class="data row0 col10" >False</td>
      <td id="T_bade7_row0_col11" class="data row0 col11" >0.39906</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_bade7_row1_col0" class="data row1 col0" >652</td>
      <td id="T_bade7_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_bade7_row1_col2" class="data row1 col2" >LORE1</td>
      <td id="T_bade7_row1_col3" class="data row1 col3" >IF hours.per.week > 42.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_bade7_row1_col4" class="data row1 col4" >0.16519</td>
      <td id="T_bade7_row1_col5" class="data row1 col5" >0.38422</td>
      <td id="T_bade7_row1_col6" class="data row1 col6" >0.56016</td>
      <td id="T_bade7_row1_col7" class="data row1 col7" >4</td>
      <td id="T_bade7_row1_col8" class="data row1 col8" >0</td>
      <td id="T_bade7_row1_col9" class="data row1 col9" >57.73337</td>
      <td id="T_bade7_row1_col10" class="data row1 col10" >False</td>
      <td id="T_bade7_row1_col11" class="data row1 col11" >0.40227</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_bade7_row2_col0" class="data row2 col0" >652</td>
      <td id="T_bade7_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_bade7_row2_col2" class="data row2 col2" >LORE3</td>
      <td id="T_bade7_row2_col3" class="data row2 col3" >IF hours.per.week > 30.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_bade7_row2_col4" class="data row2 col4" >0.37575</td>
      <td id="T_bade7_row2_col5" class="data row2 col5" >0.73529</td>
      <td id="T_bade7_row2_col6" class="data row2 col6" >0.47128</td>
      <td id="T_bade7_row2_col7" class="data row2 col7" >3</td>
      <td id="T_bade7_row2_col8" class="data row2 col8" >0</td>
      <td id="T_bade7_row2_col9" class="data row2 col9" >57.10518</td>
      <td id="T_bade7_row2_col10" class="data row2 col10" >False</td>
      <td id="T_bade7_row2_col11" class="data row2 col11" >0.40791</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_bade7_row3_col0" class="data row3 col0" >652</td>
      <td id="T_bade7_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_bade7_row3_col2" class="data row3 col2" >LORE_SA1</td>
      <td id="T_bade7_row3_col3" class="data row3 col3" >IF education != 10th AND education.num != 11.0 AND hours.per.week > 19.3595 AND marital.status = Married-civ-spouse AND relationship != Own-child THEN class = >50K</td>
      <td id="T_bade7_row3_col4" class="data row3 col4" >0.41234</td>
      <td id="T_bade7_row3_col5" class="data row3 col5" >0.79395</td>
      <td id="T_bade7_row3_col6" class="data row3 col6" >0.46372</td>
      <td id="T_bade7_row3_col7" class="data row3 col7" >5</td>
      <td id="T_bade7_row3_col8" class="data row3 col8" >0</td>
      <td id="T_bade7_row3_col9" class="data row3 col9" >14.09076</td>
      <td id="T_bade7_row3_col10" class="data row3 col10" >False</td>
      <td id="T_bade7_row3_col11" class="data row3 col11" >0.41383</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_bade7_row4_col0" class="data row4 col0" >652</td>
      <td id="T_bade7_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_bade7_row4_col2" class="data row4 col2" >LORE_SA2</td>
      <td id="T_bade7_row4_col3" class="data row4 col3" >IF age > 36.8148 AND education.num != 6.0 AND marital.status != Never-married AND marital.status != Married-spouse-absent AND relationship = Husband THEN class = >50K</td>
      <td id="T_bade7_row4_col4" class="data row4 col4" >0.27067</td>
      <td id="T_bade7_row4_col5" class="data row4 col5" >0.57807</td>
      <td id="T_bade7_row4_col6" class="data row4 col6" >0.51435</td>
      <td id="T_bade7_row4_col7" class="data row4 col7" >5</td>
      <td id="T_bade7_row4_col8" class="data row4 col8" >0</td>
      <td id="T_bade7_row4_col9" class="data row4 col9" >13.75957</td>
      <td id="T_bade7_row4_col10" class="data row4 col10" >False</td>
      <td id="T_bade7_row4_col11" class="data row4 col11" >0.38985</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row5" class="row_heading level0 row5" >6</th>
      <td id="T_bade7_row5_col0" class="data row5 col0" >652</td>
      <td id="T_bade7_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_bade7_row5_col2" class="data row5 col2" >LORE_SA3</td>
      <td id="T_bade7_row5_col3" class="data row5 col3" >IF capital.loss > 260.2145 AND marital.status != Never-married AND marital.status = Married-civ-spouse AND occupation != Transport-moving AND relationship != Own-child THEN class = >50K</td>
      <td id="T_bade7_row5_col4" class="data row5 col4" >0.02808</td>
      <td id="T_bade7_row5_col5" class="data row5 col5" >0.08362</td>
      <td id="T_bade7_row5_col6" class="data row5 col6" >0.71719</td>
      <td id="T_bade7_row5_col7" class="data row5 col7" >5</td>
      <td id="T_bade7_row5_col8" class="data row5 col8" >0</td>
      <td id="T_bade7_row5_col9" class="data row5 col9" >13.74144</td>
      <td id="T_bade7_row5_col10" class="data row5 col10" >False</td>
      <td id="T_bade7_row5_col11" class="data row5 col11" >0.41638</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row6" class="row_heading level0 row6" >7</th>
      <td id="T_bade7_row6_col0" class="data row6 col0" >652</td>
      <td id="T_bade7_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_bade7_row6_col2" class="data row6 col2" >LORE_SA4</td>
      <td id="T_bade7_row6_col3" class="data row6 col3" >IF age > 22.6687 AND hours.per.week > 28.5804 AND marital.status != Never-married AND relationship = Husband THEN class = >50K</td>
      <td id="T_bade7_row6_col4" class="data row6 col4" >0.38044</td>
      <td id="T_bade7_row6_col5" class="data row6 col5" >0.74094</td>
      <td id="T_bade7_row6_col6" class="data row6 col6" >0.46903</td>
      <td id="T_bade7_row6_col7" class="data row6 col7" >4</td>
      <td id="T_bade7_row6_col8" class="data row6 col8" >0</td>
      <td id="T_bade7_row6_col9" class="data row6 col9" >13.66720</td>
      <td id="T_bade7_row6_col10" class="data row6 col10" >False</td>
      <td id="T_bade7_row6_col11" class="data row6 col11" >0.40976</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row7" class="row_heading level0 row7" >8</th>
      <td id="T_bade7_row7_col0" class="data row7 col0" >652</td>
      <td id="T_bade7_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_bade7_row7_col2" class="data row7 col2" >EXPLAN1</td>
      <td id="T_bade7_row7_col3" class="data row7 col3" >IF age > 27.0 AND capital.gain <= 1173.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_bade7_row7_col4" class="data row7 col4" >0.02365</td>
      <td id="T_bade7_row7_col5" class="data row7 col5" >0.07196</td>
      <td id="T_bade7_row7_col6" class="data row7 col6" >0.73284</td>
      <td id="T_bade7_row7_col7" class="data row7 col7" >6</td>
      <td id="T_bade7_row7_col8" class="data row7 col8" >0</td>
      <td id="T_bade7_row7_col9" class="data row7 col9" >4.72258</td>
      <td id="T_bade7_row7_col10" class="data row7 col10" >False</td>
      <td id="T_bade7_row7_col11" class="data row7 col11" >0.41475</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row8" class="row_heading level0 row8" >9</th>
      <td id="T_bade7_row8_col0" class="data row8 col0" >652</td>
      <td id="T_bade7_row8_col1" class="data row8 col1" >EXPLAN</td>
      <td id="T_bade7_row8_col2" class="data row8 col2" >EXPLAN3</td>
      <td id="T_bade7_row8_col3" class="data row8 col3" >IF age > 32.2285 AND capital.gain <= 1173.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND race = White AND relationship = Husband THEN class = >50K</td>
      <td id="T_bade7_row8_col4" class="data row8 col4" >0.04414</td>
      <td id="T_bade7_row8_col5" class="data row8 col5" >0.12461</td>
      <td id="T_bade7_row8_col6" class="data row8 col6" >0.67992</td>
      <td id="T_bade7_row8_col7" class="data row8 col7" >6</td>
      <td id="T_bade7_row8_col8" class="data row8 col8" >0</td>
      <td id="T_bade7_row8_col9" class="data row8 col9" >3.81111</td>
      <td id="T_bade7_row8_col10" class="data row8 col10" >False</td>
      <td id="T_bade7_row8_col11" class="data row8 col11" >0.41789</td>
    </tr>
    <tr>
      <th id="T_bade7_level0_row9" class="row_heading level0 row9" >10</th>
      <td id="T_bade7_row9_col0" class="data row9 col0" >652</td>
      <td id="T_bade7_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_bade7_row9_col2" class="data row9 col2" >EXPLAN5</td>
      <td id="T_bade7_row9_col3" class="data row9 col3" >IF age > 33.0 AND capital.loss > 1775.7448 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_bade7_row9_col4" class="data row9 col4" >0.01803</td>
      <td id="T_bade7_row9_col5" class="data row9 col5" >0.06504</td>
      <td id="T_bade7_row9_col6" class="data row9 col6" >0.86861</td>
      <td id="T_bade7_row9_col7" class="data row9 col7" >4</td>
      <td id="T_bade7_row9_col8" class="data row9 col8" >0</td>
      <td id="T_bade7_row9_col9" class="data row9 col9" >5.70397</td>
      <td id="T_bade7_row9_col10" class="data row9 col10" >False</td>
      <td id="T_bade7_row9_col11" class="data row9 col11" >0.39441</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_55.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_56.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_57.png)
    



### Rules for Instance 652, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.79395, Pre: 0.87755, Len: 0.45076)



<style type="text/css">
#T_b3dcb_row4_col0, #T_b3dcb_row4_col1, #T_b3dcb_row4_col2, #T_b3dcb_row4_col3, #T_b3dcb_row4_col4, #T_b3dcb_row4_col5, #T_b3dcb_row4_col6, #T_b3dcb_row4_col7, #T_b3dcb_row4_col8, #T_b3dcb_row4_col9, #T_b3dcb_row4_col10, #T_b3dcb_row4_col11 {
  font-weight: bold;
}
</style>
<table id="T_b3dcb">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b3dcb_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_b3dcb_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_b3dcb_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_b3dcb_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_b3dcb_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_b3dcb_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_b3dcb_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_b3dcb_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_b3dcb_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_b3dcb_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_b3dcb_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_b3dcb_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b3dcb_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b3dcb_row0_col0" class="data row0 col0" >652</td>
      <td id="T_b3dcb_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_b3dcb_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_b3dcb_row0_col3" class="data row0 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row0_col4" class="data row0 col4" >0.01329</td>
      <td id="T_b3dcb_row0_col5" class="data row0 col5" >0.04828</td>
      <td id="T_b3dcb_row0_col6" class="data row0 col6" >0.87459</td>
      <td id="T_b3dcb_row0_col7" class="data row0 col7" >5</td>
      <td id="T_b3dcb_row0_col8" class="data row0 col8" >0</td>
      <td id="T_b3dcb_row0_col9" class="data row0 col9" >5.47437</td>
      <td id="T_b3dcb_row0_col10" class="data row0 col10" >False</td>
      <td id="T_b3dcb_row0_col11" class="data row0 col11" >4.60995</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_b3dcb_row1_col0" class="data row1 col0" >652</td>
      <td id="T_b3dcb_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_b3dcb_row1_col2" class="data row1 col2" >ANCHOR4</td>
      <td id="T_b3dcb_row1_col3" class="data row1 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 45.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row1_col4" class="data row1 col4" >0.01075</td>
      <td id="T_b3dcb_row1_col5" class="data row1 col5" >0.03917</td>
      <td id="T_b3dcb_row1_col6" class="data row1 col6" >0.87755</td>
      <td id="T_b3dcb_row1_col7" class="data row1 col7" >5</td>
      <td id="T_b3dcb_row1_col8" class="data row1 col8" >0</td>
      <td id="T_b3dcb_row1_col9" class="data row1 col9" >5.98049</td>
      <td id="T_b3dcb_row1_col10" class="data row1 col10" >False</td>
      <td id="T_b3dcb_row1_col11" class="data row1 col11" >4.61143</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_b3dcb_row2_col0" class="data row2 col0" >652</td>
      <td id="T_b3dcb_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_b3dcb_row2_col2" class="data row2 col2" >LORE1</td>
      <td id="T_b3dcb_row2_col3" class="data row2 col3" >IF hours.per.week > 42.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_b3dcb_row2_col4" class="data row2 col4" >0.16519</td>
      <td id="T_b3dcb_row2_col5" class="data row2 col5" >0.38422</td>
      <td id="T_b3dcb_row2_col6" class="data row2 col6" >0.56016</td>
      <td id="T_b3dcb_row2_col7" class="data row2 col7" >4</td>
      <td id="T_b3dcb_row2_col8" class="data row2 col8" >0</td>
      <td id="T_b3dcb_row2_col9" class="data row2 col9" >57.73337</td>
      <td id="T_b3dcb_row2_col10" class="data row2 col10" >False</td>
      <td id="T_b3dcb_row2_col11" class="data row2 col11" >3.58688</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_b3dcb_row3_col0" class="data row3 col0" >652</td>
      <td id="T_b3dcb_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_b3dcb_row3_col2" class="data row3 col2" >LORE3</td>
      <td id="T_b3dcb_row3_col3" class="data row3 col3" >IF hours.per.week > 30.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row3_col4" class="data row3 col4" >0.37575</td>
      <td id="T_b3dcb_row3_col5" class="data row3 col5" >0.73529</td>
      <td id="T_b3dcb_row3_col6" class="data row3 col6" >0.47128</td>
      <td id="T_b3dcb_row3_col7" class="data row3 col7" >3</td>
      <td id="T_b3dcb_row3_col8" class="data row3 col8" >0</td>
      <td id="T_b3dcb_row3_col9" class="data row3 col9" >57.10518</td>
      <td id="T_b3dcb_row3_col10" class="data row3 col10" >False</td>
      <td id="T_b3dcb_row3_col11" class="data row3 col11" >2.58208</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_b3dcb_row4_col0" class="data row4 col0" >652</td>
      <td id="T_b3dcb_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_b3dcb_row4_col2" class="data row4 col2" >LORE4</td>
      <td id="T_b3dcb_row4_col3" class="data row4 col3" >IF marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row4_col4" class="data row4 col4" >0.40453</td>
      <td id="T_b3dcb_row4_col5" class="data row4 col5" >0.75715</td>
      <td id="T_b3dcb_row4_col6" class="data row4 col6" >0.45076</td>
      <td id="T_b3dcb_row4_col7" class="data row4 col7" >2</td>
      <td id="T_b3dcb_row4_col8" class="data row4 col8" >0</td>
      <td id="T_b3dcb_row4_col9" class="data row4 col9" >56.87405</td>
      <td id="T_b3dcb_row4_col10" class="data row4 col10" >False</td>
      <td id="T_b3dcb_row4_col11" class="data row4 col11" >1.60737</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_b3dcb_row5_col0" class="data row5 col0" >652</td>
      <td id="T_b3dcb_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_b3dcb_row5_col2" class="data row5 col2" >LORE_SA1</td>
      <td id="T_b3dcb_row5_col3" class="data row5 col3" >IF education != 10th AND education.num != 11.0 AND hours.per.week > 19.3595 AND marital.status = Married-civ-spouse AND relationship != Own-child THEN class = >50K</td>
      <td id="T_b3dcb_row5_col4" class="data row5 col4" >0.41234</td>
      <td id="T_b3dcb_row5_col5" class="data row5 col5" >0.79395</td>
      <td id="T_b3dcb_row5_col6" class="data row5 col6" >0.46372</td>
      <td id="T_b3dcb_row5_col7" class="data row5 col7" >5</td>
      <td id="T_b3dcb_row5_col8" class="data row5 col8" >0</td>
      <td id="T_b3dcb_row5_col9" class="data row5 col9" >14.09076</td>
      <td id="T_b3dcb_row5_col10" class="data row5 col10" >False</td>
      <td id="T_b3dcb_row5_col11" class="data row5 col11" >4.56802</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_b3dcb_row6_col0" class="data row6 col0" >652</td>
      <td id="T_b3dcb_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_b3dcb_row6_col2" class="data row6 col2" >LORE_SA2</td>
      <td id="T_b3dcb_row6_col3" class="data row6 col3" >IF age > 36.8148 AND education.num != 6.0 AND marital.status != Never-married AND marital.status != Married-spouse-absent AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row6_col4" class="data row6 col4" >0.27067</td>
      <td id="T_b3dcb_row6_col5" class="data row6 col5" >0.57807</td>
      <td id="T_b3dcb_row6_col6" class="data row6 col6" >0.51435</td>
      <td id="T_b3dcb_row6_col7" class="data row6 col7" >5</td>
      <td id="T_b3dcb_row6_col8" class="data row6 col8" >0</td>
      <td id="T_b3dcb_row6_col9" class="data row6 col9" >13.75957</td>
      <td id="T_b3dcb_row6_col10" class="data row6 col10" >False</td>
      <td id="T_b3dcb_row6_col11" class="data row6 col11" >4.56882</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_b3dcb_row7_col0" class="data row7 col0" >652</td>
      <td id="T_b3dcb_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_b3dcb_row7_col2" class="data row7 col2" >LORE_SA3</td>
      <td id="T_b3dcb_row7_col3" class="data row7 col3" >IF capital.loss > 260.2145 AND marital.status != Never-married AND marital.status = Married-civ-spouse AND occupation != Transport-moving AND relationship != Own-child THEN class = >50K</td>
      <td id="T_b3dcb_row7_col4" class="data row7 col4" >0.02808</td>
      <td id="T_b3dcb_row7_col5" class="data row7 col5" >0.08362</td>
      <td id="T_b3dcb_row7_col6" class="data row7 col6" >0.71719</td>
      <td id="T_b3dcb_row7_col7" class="data row7 col7" >5</td>
      <td id="T_b3dcb_row7_col8" class="data row7 col8" >0</td>
      <td id="T_b3dcb_row7_col9" class="data row7 col9" >13.74144</td>
      <td id="T_b3dcb_row7_col10" class="data row7 col10" >False</td>
      <td id="T_b3dcb_row7_col11" class="data row7 col11" >4.60715</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_b3dcb_row8_col0" class="data row8 col0" >652</td>
      <td id="T_b3dcb_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_b3dcb_row8_col2" class="data row8 col2" >LORE_SA4</td>
      <td id="T_b3dcb_row8_col3" class="data row8 col3" >IF age > 22.6687 AND hours.per.week > 28.5804 AND marital.status != Never-married AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row8_col4" class="data row8 col4" >0.38044</td>
      <td id="T_b3dcb_row8_col5" class="data row8 col5" >0.74094</td>
      <td id="T_b3dcb_row8_col6" class="data row8 col6" >0.46903</td>
      <td id="T_b3dcb_row8_col7" class="data row8 col7" >4</td>
      <td id="T_b3dcb_row8_col8" class="data row8 col8" >0</td>
      <td id="T_b3dcb_row8_col9" class="data row8 col9" >13.66720</td>
      <td id="T_b3dcb_row8_col10" class="data row8 col10" >False</td>
      <td id="T_b3dcb_row8_col11" class="data row8 col11" >3.57307</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_b3dcb_row9_col0" class="data row9 col0" >652</td>
      <td id="T_b3dcb_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_b3dcb_row9_col2" class="data row9 col2" >EXPLAN1</td>
      <td id="T_b3dcb_row9_col3" class="data row9 col3" >IF age > 27.0 AND capital.gain <= 1173.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row9_col4" class="data row9 col4" >0.02365</td>
      <td id="T_b3dcb_row9_col5" class="data row9 col5" >0.07196</td>
      <td id="T_b3dcb_row9_col6" class="data row9 col6" >0.73284</td>
      <td id="T_b3dcb_row9_col7" class="data row9 col7" >6</td>
      <td id="T_b3dcb_row9_col8" class="data row9 col8" >0</td>
      <td id="T_b3dcb_row9_col9" class="data row9 col9" >4.72258</td>
      <td id="T_b3dcb_row9_col10" class="data row9 col10" >False</td>
      <td id="T_b3dcb_row9_col11" class="data row9 col11" >5.59788</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_b3dcb_row10_col0" class="data row10 col0" >652</td>
      <td id="T_b3dcb_row10_col1" class="data row10 col1" >EXPLAN</td>
      <td id="T_b3dcb_row10_col2" class="data row10 col2" >EXPLAN3</td>
      <td id="T_b3dcb_row10_col3" class="data row10 col3" >IF age > 32.2285 AND capital.gain <= 1173.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND race = White AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row10_col4" class="data row10 col4" >0.04414</td>
      <td id="T_b3dcb_row10_col5" class="data row10 col5" >0.12461</td>
      <td id="T_b3dcb_row10_col6" class="data row10 col6" >0.67992</td>
      <td id="T_b3dcb_row10_col7" class="data row10 col7" >6</td>
      <td id="T_b3dcb_row10_col8" class="data row10 col8" >0</td>
      <td id="T_b3dcb_row10_col9" class="data row10 col9" >3.81111</td>
      <td id="T_b3dcb_row10_col10" class="data row10 col10" >False</td>
      <td id="T_b3dcb_row10_col11" class="data row10 col11" >5.59295</td>
    </tr>
    <tr>
      <th id="T_b3dcb_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_b3dcb_row11_col0" class="data row11 col0" >652</td>
      <td id="T_b3dcb_row11_col1" class="data row11 col1" >EXPLAN</td>
      <td id="T_b3dcb_row11_col2" class="data row11 col2" >EXPLAN5</td>
      <td id="T_b3dcb_row11_col3" class="data row11 col3" >IF age > 33.0 AND capital.loss > 1775.7448 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b3dcb_row11_col4" class="data row11 col4" >0.01803</td>
      <td id="T_b3dcb_row11_col5" class="data row11 col5" >0.06504</td>
      <td id="T_b3dcb_row11_col6" class="data row11 col6" >0.86861</td>
      <td id="T_b3dcb_row11_col7" class="data row11 col7" >4</td>
      <td id="T_b3dcb_row11_col8" class="data row11 col8" >0</td>
      <td id="T_b3dcb_row11_col9" class="data row11 col9" >5.70397</td>
      <td id="T_b3dcb_row11_col10" class="data row11 col10" >False</td>
      <td id="T_b3dcb_row11_col11" class="data row11 col11" >3.62333</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 652, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.79395, Pre: 0.87755), Unique rules (diffrent features)



<style type="text/css">
#T_b7b21_row3_col0, #T_b7b21_row3_col1, #T_b7b21_row3_col2, #T_b7b21_row3_col3, #T_b7b21_row3_col4, #T_b7b21_row3_col5, #T_b7b21_row3_col6, #T_b7b21_row3_col7, #T_b7b21_row3_col8, #T_b7b21_row3_col9, #T_b7b21_row3_col10, #T_b7b21_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_b7b21">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b7b21_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_b7b21_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_b7b21_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_b7b21_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_b7b21_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_b7b21_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_b7b21_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_b7b21_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_b7b21_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_b7b21_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_b7b21_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_b7b21_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b7b21_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b7b21_row0_col0" class="data row0 col0" >652</td>
      <td id="T_b7b21_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_b7b21_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_b7b21_row0_col3" class="data row0 col3" >IF age > 28.0 AND education = Bachelors AND hours.per.week > 40.0 AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row0_col4" class="data row0 col4" >0.01329</td>
      <td id="T_b7b21_row0_col5" class="data row0 col5" >0.04828</td>
      <td id="T_b7b21_row0_col6" class="data row0 col6" >0.87459</td>
      <td id="T_b7b21_row0_col7" class="data row0 col7" >5</td>
      <td id="T_b7b21_row0_col8" class="data row0 col8" >0</td>
      <td id="T_b7b21_row0_col9" class="data row0 col9" >5.47437</td>
      <td id="T_b7b21_row0_col10" class="data row0 col10" >False</td>
      <td id="T_b7b21_row0_col11" class="data row0 col11" >4.60995</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_b7b21_row1_col0" class="data row1 col0" >652</td>
      <td id="T_b7b21_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_b7b21_row1_col2" class="data row1 col2" >LORE1</td>
      <td id="T_b7b21_row1_col3" class="data row1 col3" >IF hours.per.week > 42.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_b7b21_row1_col4" class="data row1 col4" >0.16519</td>
      <td id="T_b7b21_row1_col5" class="data row1 col5" >0.38422</td>
      <td id="T_b7b21_row1_col6" class="data row1 col6" >0.56016</td>
      <td id="T_b7b21_row1_col7" class="data row1 col7" >4</td>
      <td id="T_b7b21_row1_col8" class="data row1 col8" >0</td>
      <td id="T_b7b21_row1_col9" class="data row1 col9" >57.73337</td>
      <td id="T_b7b21_row1_col10" class="data row1 col10" >False</td>
      <td id="T_b7b21_row1_col11" class="data row1 col11" >3.58688</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_b7b21_row2_col0" class="data row2 col0" >652</td>
      <td id="T_b7b21_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_b7b21_row2_col2" class="data row2 col2" >LORE3</td>
      <td id="T_b7b21_row2_col3" class="data row2 col3" >IF hours.per.week > 30.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row2_col4" class="data row2 col4" >0.37575</td>
      <td id="T_b7b21_row2_col5" class="data row2 col5" >0.73529</td>
      <td id="T_b7b21_row2_col6" class="data row2 col6" >0.47128</td>
      <td id="T_b7b21_row2_col7" class="data row2 col7" >3</td>
      <td id="T_b7b21_row2_col8" class="data row2 col8" >0</td>
      <td id="T_b7b21_row2_col9" class="data row2 col9" >57.10518</td>
      <td id="T_b7b21_row2_col10" class="data row2 col10" >False</td>
      <td id="T_b7b21_row2_col11" class="data row2 col11" >2.58208</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_b7b21_row3_col0" class="data row3 col0" >652</td>
      <td id="T_b7b21_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_b7b21_row3_col2" class="data row3 col2" >LORE4</td>
      <td id="T_b7b21_row3_col3" class="data row3 col3" >IF marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row3_col4" class="data row3 col4" >0.40453</td>
      <td id="T_b7b21_row3_col5" class="data row3 col5" >0.75715</td>
      <td id="T_b7b21_row3_col6" class="data row3 col6" >0.45076</td>
      <td id="T_b7b21_row3_col7" class="data row3 col7" >2</td>
      <td id="T_b7b21_row3_col8" class="data row3 col8" >0</td>
      <td id="T_b7b21_row3_col9" class="data row3 col9" >56.87405</td>
      <td id="T_b7b21_row3_col10" class="data row3 col10" >False</td>
      <td id="T_b7b21_row3_col11" class="data row3 col11" >1.60737</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_b7b21_row4_col0" class="data row4 col0" >652</td>
      <td id="T_b7b21_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_b7b21_row4_col2" class="data row4 col2" >LORE_SA1</td>
      <td id="T_b7b21_row4_col3" class="data row4 col3" >IF education != 10th AND education.num != 11.0 AND hours.per.week > 19.3595 AND marital.status = Married-civ-spouse AND relationship != Own-child THEN class = >50K</td>
      <td id="T_b7b21_row4_col4" class="data row4 col4" >0.41234</td>
      <td id="T_b7b21_row4_col5" class="data row4 col5" >0.79395</td>
      <td id="T_b7b21_row4_col6" class="data row4 col6" >0.46372</td>
      <td id="T_b7b21_row4_col7" class="data row4 col7" >5</td>
      <td id="T_b7b21_row4_col8" class="data row4 col8" >0</td>
      <td id="T_b7b21_row4_col9" class="data row4 col9" >14.09076</td>
      <td id="T_b7b21_row4_col10" class="data row4 col10" >False</td>
      <td id="T_b7b21_row4_col11" class="data row4 col11" >4.56802</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row5" class="row_heading level0 row5" >6</th>
      <td id="T_b7b21_row5_col0" class="data row5 col0" >652</td>
      <td id="T_b7b21_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_b7b21_row5_col2" class="data row5 col2" >LORE_SA2</td>
      <td id="T_b7b21_row5_col3" class="data row5 col3" >IF age > 36.8148 AND education.num != 6.0 AND marital.status != Never-married AND marital.status != Married-spouse-absent AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row5_col4" class="data row5 col4" >0.27067</td>
      <td id="T_b7b21_row5_col5" class="data row5 col5" >0.57807</td>
      <td id="T_b7b21_row5_col6" class="data row5 col6" >0.51435</td>
      <td id="T_b7b21_row5_col7" class="data row5 col7" >5</td>
      <td id="T_b7b21_row5_col8" class="data row5 col8" >0</td>
      <td id="T_b7b21_row5_col9" class="data row5 col9" >13.75957</td>
      <td id="T_b7b21_row5_col10" class="data row5 col10" >False</td>
      <td id="T_b7b21_row5_col11" class="data row5 col11" >4.56882</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row6" class="row_heading level0 row6" >7</th>
      <td id="T_b7b21_row6_col0" class="data row6 col0" >652</td>
      <td id="T_b7b21_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_b7b21_row6_col2" class="data row6 col2" >LORE_SA3</td>
      <td id="T_b7b21_row6_col3" class="data row6 col3" >IF capital.loss > 260.2145 AND marital.status != Never-married AND marital.status = Married-civ-spouse AND occupation != Transport-moving AND relationship != Own-child THEN class = >50K</td>
      <td id="T_b7b21_row6_col4" class="data row6 col4" >0.02808</td>
      <td id="T_b7b21_row6_col5" class="data row6 col5" >0.08362</td>
      <td id="T_b7b21_row6_col6" class="data row6 col6" >0.71719</td>
      <td id="T_b7b21_row6_col7" class="data row6 col7" >5</td>
      <td id="T_b7b21_row6_col8" class="data row6 col8" >0</td>
      <td id="T_b7b21_row6_col9" class="data row6 col9" >13.74144</td>
      <td id="T_b7b21_row6_col10" class="data row6 col10" >False</td>
      <td id="T_b7b21_row6_col11" class="data row6 col11" >4.60715</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row7" class="row_heading level0 row7" >8</th>
      <td id="T_b7b21_row7_col0" class="data row7 col0" >652</td>
      <td id="T_b7b21_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_b7b21_row7_col2" class="data row7 col2" >LORE_SA4</td>
      <td id="T_b7b21_row7_col3" class="data row7 col3" >IF age > 22.6687 AND hours.per.week > 28.5804 AND marital.status != Never-married AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row7_col4" class="data row7 col4" >0.38044</td>
      <td id="T_b7b21_row7_col5" class="data row7 col5" >0.74094</td>
      <td id="T_b7b21_row7_col6" class="data row7 col6" >0.46903</td>
      <td id="T_b7b21_row7_col7" class="data row7 col7" >4</td>
      <td id="T_b7b21_row7_col8" class="data row7 col8" >0</td>
      <td id="T_b7b21_row7_col9" class="data row7 col9" >13.66720</td>
      <td id="T_b7b21_row7_col10" class="data row7 col10" >False</td>
      <td id="T_b7b21_row7_col11" class="data row7 col11" >3.57307</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row8" class="row_heading level0 row8" >9</th>
      <td id="T_b7b21_row8_col0" class="data row8 col0" >652</td>
      <td id="T_b7b21_row8_col1" class="data row8 col1" >EXPLAN</td>
      <td id="T_b7b21_row8_col2" class="data row8 col2" >EXPLAN1</td>
      <td id="T_b7b21_row8_col3" class="data row8 col3" >IF age > 27.0 AND capital.gain <= 1173.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row8_col4" class="data row8 col4" >0.02365</td>
      <td id="T_b7b21_row8_col5" class="data row8 col5" >0.07196</td>
      <td id="T_b7b21_row8_col6" class="data row8 col6" >0.73284</td>
      <td id="T_b7b21_row8_col7" class="data row8 col7" >6</td>
      <td id="T_b7b21_row8_col8" class="data row8 col8" >0</td>
      <td id="T_b7b21_row8_col9" class="data row8 col9" >4.72258</td>
      <td id="T_b7b21_row8_col10" class="data row8 col10" >False</td>
      <td id="T_b7b21_row8_col11" class="data row8 col11" >5.59788</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row9" class="row_heading level0 row9" >10</th>
      <td id="T_b7b21_row9_col0" class="data row9 col0" >652</td>
      <td id="T_b7b21_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_b7b21_row9_col2" class="data row9 col2" >EXPLAN3</td>
      <td id="T_b7b21_row9_col3" class="data row9 col3" >IF age > 32.2285 AND capital.gain <= 1173.0 AND marital.status = Married-civ-spouse AND occupation = Exec-managerial AND race = White AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row9_col4" class="data row9 col4" >0.04414</td>
      <td id="T_b7b21_row9_col5" class="data row9 col5" >0.12461</td>
      <td id="T_b7b21_row9_col6" class="data row9 col6" >0.67992</td>
      <td id="T_b7b21_row9_col7" class="data row9 col7" >6</td>
      <td id="T_b7b21_row9_col8" class="data row9 col8" >0</td>
      <td id="T_b7b21_row9_col9" class="data row9 col9" >3.81111</td>
      <td id="T_b7b21_row9_col10" class="data row9 col10" >False</td>
      <td id="T_b7b21_row9_col11" class="data row9 col11" >5.59295</td>
    </tr>
    <tr>
      <th id="T_b7b21_level0_row10" class="row_heading level0 row10" >11</th>
      <td id="T_b7b21_row10_col0" class="data row10 col0" >652</td>
      <td id="T_b7b21_row10_col1" class="data row10 col1" >EXPLAN</td>
      <td id="T_b7b21_row10_col2" class="data row10 col2" >EXPLAN5</td>
      <td id="T_b7b21_row10_col3" class="data row10 col3" >IF age > 33.0 AND capital.loss > 1775.7448 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b7b21_row10_col4" class="data row10 col4" >0.01803</td>
      <td id="T_b7b21_row10_col5" class="data row10 col5" >0.06504</td>
      <td id="T_b7b21_row10_col6" class="data row10 col6" >0.86861</td>
      <td id="T_b7b21_row10_col7" class="data row10 col7" >4</td>
      <td id="T_b7b21_row10_col8" class="data row10 col8" >0</td>
      <td id="T_b7b21_row10_col9" class="data row10 col9" >5.70397</td>
      <td id="T_b7b21_row10_col10" class="data row10 col10" >False</td>
      <td id="T_b7b21_row10_col11" class="data row10 col11" >3.62333</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_62.png)
    



## Instance 2711 (Original: >50K , Predicted: >50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>39.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>Bachelors</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>13</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Married-civ-spouse</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Sales</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Husband</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>7298.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>55.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 2711



<style type="text/css">
</style>
<table id="T_45134">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_45134_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_45134_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_45134_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_45134_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_45134_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_45134_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_45134_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_45134_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_45134_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_45134_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_45134_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_45134_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_45134_row0_col0" class="data row0 col0" >2711</td>
      <td id="T_45134_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_45134_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_45134_row0_col3" class="data row0 col3" >IF capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND hours.per.week > 45.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row0_col4" class="data row0 col4" >0.00531</td>
      <td id="T_45134_row0_col5" class="data row0 col5" >0.02040</td>
      <td id="T_45134_row0_col6" class="data row0 col6" >0.92562</td>
      <td id="T_45134_row0_col7" class="data row0 col7" >5</td>
      <td id="T_45134_row0_col8" class="data row0 col8" >0</td>
      <td id="T_45134_row0_col9" class="data row0 col9" >4.04182</td>
      <td id="T_45134_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_45134_row1_col0" class="data row1 col0" >2711</td>
      <td id="T_45134_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_45134_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_45134_row1_col3" class="data row1 col3" >IF age > 37.0 AND capital.gain > 0.0 AND education = Bachelors AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row1_col4" class="data row1 col4" >0.00873</td>
      <td id="T_45134_row1_col5" class="data row1 col5" >0.03279</td>
      <td id="T_45134_row1_col6" class="data row1 col6" >0.90452</td>
      <td id="T_45134_row1_col7" class="data row1 col7" >5</td>
      <td id="T_45134_row1_col8" class="data row1 col8" >0</td>
      <td id="T_45134_row1_col9" class="data row1 col9" >4.36910</td>
      <td id="T_45134_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_45134_row2_col0" class="data row2 col0" >2711</td>
      <td id="T_45134_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_45134_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_45134_row2_col3" class="data row2 col3" >IF capital.gain > 0.0 AND education.num = 13.0 AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row2_col4" class="data row2 col4" >0.00663</td>
      <td id="T_45134_row2_col5" class="data row2 col5" >0.02514</td>
      <td id="T_45134_row2_col6" class="data row2 col6" >0.91391</td>
      <td id="T_45134_row2_col7" class="data row2 col7" >5</td>
      <td id="T_45134_row2_col8" class="data row2 col8" >0</td>
      <td id="T_45134_row2_col9" class="data row2 col9" >4.00035</td>
      <td id="T_45134_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_45134_row3_col0" class="data row3 col0" >2711</td>
      <td id="T_45134_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_45134_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_45134_row3_col3" class="data row3 col3" >IF capital.gain > 0.0 AND education = Bachelors AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row3_col4" class="data row3 col4" >0.00663</td>
      <td id="T_45134_row3_col5" class="data row3 col5" >0.02514</td>
      <td id="T_45134_row3_col6" class="data row3 col6" >0.91391</td>
      <td id="T_45134_row3_col7" class="data row3 col7" >5</td>
      <td id="T_45134_row3_col8" class="data row3 col8" >1</td>
      <td id="T_45134_row3_col9" class="data row3 col9" >7.97818</td>
      <td id="T_45134_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_45134_row4_col0" class="data row4 col0" >2711</td>
      <td id="T_45134_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_45134_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_45134_row4_col3" class="data row4 col3" >IF age > 37.0 AND capital.gain > 0.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row4_col4" class="data row4 col4" >0.01505</td>
      <td id="T_45134_row4_col5" class="data row4 col5" >0.05465</td>
      <td id="T_45134_row4_col6" class="data row4 col6" >0.87464</td>
      <td id="T_45134_row4_col7" class="data row4 col7" >5</td>
      <td id="T_45134_row4_col8" class="data row4 col8" >1</td>
      <td id="T_45134_row4_col9" class="data row4 col9" >8.05141</td>
      <td id="T_45134_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_45134_row5_col0" class="data row5 col0" >2711</td>
      <td id="T_45134_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_45134_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_45134_row5_col3" class="data row5 col3" >IF capital.gain > 4934.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row5_col4" class="data row5 col4" >0.03549</td>
      <td id="T_45134_row5_col5" class="data row5 col5" >0.13864</td>
      <td id="T_45134_row5_col6" class="data row5 col6" >0.94067</td>
      <td id="T_45134_row5_col7" class="data row5 col7" >2</td>
      <td id="T_45134_row5_col8" class="data row5 col8" >0</td>
      <td id="T_45134_row5_col9" class="data row5 col9" >55.74695</td>
      <td id="T_45134_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_45134_row6_col0" class="data row6 col0" >2711</td>
      <td id="T_45134_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_45134_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_45134_row6_col3" class="data row6 col3" >IF age > 24.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row6_col4" class="data row6 col4" >0.39505</td>
      <td id="T_45134_row6_col5" class="data row6 col5" >0.75424</td>
      <td id="T_45134_row6_col6" class="data row6 col6" >0.45980</td>
      <td id="T_45134_row6_col7" class="data row6 col7" >3</td>
      <td id="T_45134_row6_col8" class="data row6 col8" >0</td>
      <td id="T_45134_row6_col9" class="data row6 col9" >76.22816</td>
      <td id="T_45134_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_45134_row7_col0" class="data row7 col0" >2711</td>
      <td id="T_45134_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_45134_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_45134_row7_col3" class="data row7 col3" >IF age > 25.0 AND capital.gain > 4101.0 THEN class = >50K</td>
      <td id="T_45134_row7_col4" class="data row7 col4" >0.05555</td>
      <td id="T_45134_row7_col5" class="data row7 col5" >0.20259</td>
      <td id="T_45134_row7_col6" class="data row7 col6" >0.87836</td>
      <td id="T_45134_row7_col7" class="data row7 col7" >2</td>
      <td id="T_45134_row7_col8" class="data row7 col8" >0</td>
      <td id="T_45134_row7_col9" class="data row7 col9" >82.61932</td>
      <td id="T_45134_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_45134_row8_col0" class="data row8 col0" >2711</td>
      <td id="T_45134_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_45134_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_45134_row8_col3" class="data row8 col3" >IF capital.gain > 3453.9974 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row8_col4" class="data row8 col4" >0.04690</td>
      <td id="T_45134_row8_col5" class="data row8 col5" >0.16378</td>
      <td id="T_45134_row8_col6" class="data row8 col6" >0.84097</td>
      <td id="T_45134_row8_col7" class="data row8 col7" >2</td>
      <td id="T_45134_row8_col8" class="data row8 col8" >0</td>
      <td id="T_45134_row8_col9" class="data row8 col9" >74.79921</td>
      <td id="T_45134_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_45134_row9_col0" class="data row9 col0" >2711</td>
      <td id="T_45134_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_45134_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_45134_row9_col3" class="data row9 col3" >IF capital.gain > 4263.7872 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row9_col4" class="data row9 col4" >0.04273</td>
      <td id="T_45134_row9_col5" class="data row9 col5" >0.16378</td>
      <td id="T_45134_row9_col6" class="data row9 col6" >0.92300</td>
      <td id="T_45134_row9_col7" class="data row9 col7" >2</td>
      <td id="T_45134_row9_col8" class="data row9 col8" >0</td>
      <td id="T_45134_row9_col9" class="data row9 col9" >83.88141</td>
      <td id="T_45134_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_45134_row10_col0" class="data row10 col0" >2711</td>
      <td id="T_45134_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_45134_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_45134_row10_col3" class="data row10 col3" >IF relationship = Husband THEN class = >50K</td>
      <td id="T_45134_row10_col4" class="data row10 col4" >0.40479</td>
      <td id="T_45134_row10_col5" class="data row10 col5" >0.75752</td>
      <td id="T_45134_row10_col6" class="data row10 col6" >0.45068</td>
      <td id="T_45134_row10_col7" class="data row10 col7" >1</td>
      <td id="T_45134_row10_col8" class="data row10 col8" >0</td>
      <td id="T_45134_row10_col9" class="data row10 col9" >19.54397</td>
      <td id="T_45134_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_45134_row11_col0" class="data row11 col0" >2711</td>
      <td id="T_45134_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_45134_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_45134_row11_col3" class="data row11 col3" >IF marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row11_col4" class="data row11 col4" >0.45920</td>
      <td id="T_45134_row11_col5" class="data row11 col5" >0.85553</td>
      <td id="T_45134_row11_col6" class="data row11 col6" >0.44869</td>
      <td id="T_45134_row11_col7" class="data row11 col7" >1</td>
      <td id="T_45134_row11_col8" class="data row11 col8" >1</td>
      <td id="T_45134_row11_col9" class="data row11 col9" >39.17329</td>
      <td id="T_45134_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_45134_row12_col0" class="data row12 col0" >2711</td>
      <td id="T_45134_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_45134_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_45134_row12_col3" class="data row12 col3" >IF capital.gain > 3664.4158 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row12_col4" class="data row12 col4" >0.04607</td>
      <td id="T_45134_row12_col5" class="data row12 col5" >0.16378</td>
      <td id="T_45134_row12_col6" class="data row12 col6" >0.85619</td>
      <td id="T_45134_row12_col7" class="data row12 col7" >2</td>
      <td id="T_45134_row12_col8" class="data row12 col8" >2</td>
      <td id="T_45134_row12_col9" class="data row12 col9" >57.88563</td>
      <td id="T_45134_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_45134_row13_col0" class="data row13 col0" >2711</td>
      <td id="T_45134_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_45134_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_45134_row13_col3" class="data row13 col3" >IF capital.gain > 5599.881 AND education != Assoc-acdm AND education.num != 15.0 AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_45134_row13_col4" class="data row13 col4" >0.03549</td>
      <td id="T_45134_row13_col5" class="data row13 col5" >0.14083</td>
      <td id="T_45134_row13_col6" class="data row13 col6" >0.95550</td>
      <td id="T_45134_row13_col7" class="data row13 col7" >4</td>
      <td id="T_45134_row13_col8" class="data row13 col8" >1</td>
      <td id="T_45134_row13_col9" class="data row13 col9" >37.80378</td>
      <td id="T_45134_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_45134_row14_col0" class="data row14 col0" >2711</td>
      <td id="T_45134_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_45134_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_45134_row14_col3" class="data row14 col3" >IF capital.gain > 2356.6662 AND education != 12th AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_45134_row14_col4" class="data row14 col4" >0.06467</td>
      <td id="T_45134_row14_col5" class="data row14 col5" >0.19676</td>
      <td id="T_45134_row14_col6" class="data row14 col6" >0.73270</td>
      <td id="T_45134_row14_col7" class="data row14 col7" >3</td>
      <td id="T_45134_row14_col8" class="data row14 col8" >0</td>
      <td id="T_45134_row14_col9" class="data row14 col9" >18.96218</td>
      <td id="T_45134_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_45134_row15_col0" class="data row15 col0" >2711</td>
      <td id="T_45134_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_45134_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_45134_row15_col3" class="data row15 col3" >IF age > 31.0 AND capital.gain > 5013.0 AND capital.gain <= 8525.3219 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row15_col4" class="data row15 col4" >0.01843</td>
      <td id="T_45134_row15_col5" class="data row15 col5" >0.07579</td>
      <td id="T_45134_row15_col6" class="data row15 col6" >0.99048</td>
      <td id="T_45134_row15_col7" class="data row15 col7" >4</td>
      <td id="T_45134_row15_col8" class="data row15 col8" >0</td>
      <td id="T_45134_row15_col9" class="data row15 col9" >5.99136</td>
      <td id="T_45134_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_45134_row16_col0" class="data row16 col0" >2711</td>
      <td id="T_45134_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_45134_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_45134_row16_col3" class="data row16 col3" >IF age > 24.4479 AND capital.gain > 5019.1937 AND capital.gain <= 7870.1952 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row16_col4" class="data row16 col4" >0.02014</td>
      <td id="T_45134_row16_col5" class="data row16 col5" >0.08289</td>
      <td id="T_45134_row16_col6" class="data row16 col6" >0.99129</td>
      <td id="T_45134_row16_col7" class="data row16 col7" >4</td>
      <td id="T_45134_row16_col8" class="data row16 col8" >0</td>
      <td id="T_45134_row16_col9" class="data row16 col9" >7.34034</td>
      <td id="T_45134_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_45134_row17_col0" class="data row17 col0" >2711</td>
      <td id="T_45134_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_45134_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_45134_row17_col3" class="data row17 col3" >IF capital.gain > 7112.1724 AND capital.gain <= 7443.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row17_col4" class="data row17 col4" >0.00741</td>
      <td id="T_45134_row17_col5" class="data row17 col5" >0.03079</td>
      <td id="T_45134_row17_col6" class="data row17 col6" >1.00000</td>
      <td id="T_45134_row17_col7" class="data row17 col7" >3</td>
      <td id="T_45134_row17_col8" class="data row17 col8" >0</td>
      <td id="T_45134_row17_col9" class="data row17 col9" >6.99874</td>
      <td id="T_45134_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_45134_row18_col0" class="data row18 col0" >2711</td>
      <td id="T_45134_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_45134_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_45134_row18_col3" class="data row18 col3" >IF capital.gain > 5065.0654 AND capital.gain <= 7688.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_45134_row18_col4" class="data row18 col4" >0.02027</td>
      <td id="T_45134_row18_col5" class="data row18 col5" >0.08344</td>
      <td id="T_45134_row18_col6" class="data row18 col6" >0.99134</td>
      <td id="T_45134_row18_col7" class="data row18 col7" >3</td>
      <td id="T_45134_row18_col8" class="data row18 col8" >0</td>
      <td id="T_45134_row18_col9" class="data row18 col9" >5.80869</td>
      <td id="T_45134_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_45134_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_45134_row19_col0" class="data row19 col0" >2711</td>
      <td id="T_45134_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_45134_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_45134_row19_col3" class="data row19 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_45134_row19_col4" class="data row19 col4" >0.04923</td>
      <td id="T_45134_row19_col5" class="data row19 col5" >0.19366</td>
      <td id="T_45134_row19_col6" class="data row19 col6" >0.94742</td>
      <td id="T_45134_row19_col7" class="data row19 col7" >1</td>
      <td id="T_45134_row19_col8" class="data row19 col8" >0</td>
      <td id="T_45134_row19_col9" class="data row19 col9" >5.76786</td>
      <td id="T_45134_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 2711, Correct Prediction



<style type="text/css">
</style>
<table id="T_fbad0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_fbad0_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_fbad0_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_fbad0_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_fbad0_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_fbad0_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_fbad0_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_fbad0_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_fbad0_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_fbad0_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_fbad0_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_fbad0_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fbad0_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_fbad0_row0_col0" class="data row0 col0" >2711</td>
      <td id="T_fbad0_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_fbad0_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_fbad0_row0_col3" class="data row0 col3" >IF capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND hours.per.week > 45.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row0_col4" class="data row0 col4" >0.00531</td>
      <td id="T_fbad0_row0_col5" class="data row0 col5" >0.02040</td>
      <td id="T_fbad0_row0_col6" class="data row0 col6" >0.92562</td>
      <td id="T_fbad0_row0_col7" class="data row0 col7" >5</td>
      <td id="T_fbad0_row0_col8" class="data row0 col8" >0</td>
      <td id="T_fbad0_row0_col9" class="data row0 col9" >4.04182</td>
      <td id="T_fbad0_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_fbad0_row1_col0" class="data row1 col0" >2711</td>
      <td id="T_fbad0_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_fbad0_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_fbad0_row1_col3" class="data row1 col3" >IF age > 37.0 AND capital.gain > 0.0 AND education = Bachelors AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row1_col4" class="data row1 col4" >0.00873</td>
      <td id="T_fbad0_row1_col5" class="data row1 col5" >0.03279</td>
      <td id="T_fbad0_row1_col6" class="data row1 col6" >0.90452</td>
      <td id="T_fbad0_row1_col7" class="data row1 col7" >5</td>
      <td id="T_fbad0_row1_col8" class="data row1 col8" >0</td>
      <td id="T_fbad0_row1_col9" class="data row1 col9" >4.36910</td>
      <td id="T_fbad0_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_fbad0_row2_col0" class="data row2 col0" >2711</td>
      <td id="T_fbad0_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_fbad0_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_fbad0_row2_col3" class="data row2 col3" >IF capital.gain > 0.0 AND education.num = 13.0 AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row2_col4" class="data row2 col4" >0.00663</td>
      <td id="T_fbad0_row2_col5" class="data row2 col5" >0.02514</td>
      <td id="T_fbad0_row2_col6" class="data row2 col6" >0.91391</td>
      <td id="T_fbad0_row2_col7" class="data row2 col7" >5</td>
      <td id="T_fbad0_row2_col8" class="data row2 col8" >0</td>
      <td id="T_fbad0_row2_col9" class="data row2 col9" >4.00035</td>
      <td id="T_fbad0_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_fbad0_row3_col0" class="data row3 col0" >2711</td>
      <td id="T_fbad0_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_fbad0_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_fbad0_row3_col3" class="data row3 col3" >IF capital.gain > 0.0 AND education = Bachelors AND hours.per.week > 40.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row3_col4" class="data row3 col4" >0.00663</td>
      <td id="T_fbad0_row3_col5" class="data row3 col5" >0.02514</td>
      <td id="T_fbad0_row3_col6" class="data row3 col6" >0.91391</td>
      <td id="T_fbad0_row3_col7" class="data row3 col7" >5</td>
      <td id="T_fbad0_row3_col8" class="data row3 col8" >1</td>
      <td id="T_fbad0_row3_col9" class="data row3 col9" >7.97818</td>
      <td id="T_fbad0_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_fbad0_row4_col0" class="data row4 col0" >2711</td>
      <td id="T_fbad0_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_fbad0_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_fbad0_row4_col3" class="data row4 col3" >IF age > 37.0 AND capital.gain > 0.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row4_col4" class="data row4 col4" >0.01505</td>
      <td id="T_fbad0_row4_col5" class="data row4 col5" >0.05465</td>
      <td id="T_fbad0_row4_col6" class="data row4 col6" >0.87464</td>
      <td id="T_fbad0_row4_col7" class="data row4 col7" >5</td>
      <td id="T_fbad0_row4_col8" class="data row4 col8" >1</td>
      <td id="T_fbad0_row4_col9" class="data row4 col9" >8.05141</td>
      <td id="T_fbad0_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_fbad0_row5_col0" class="data row5 col0" >2711</td>
      <td id="T_fbad0_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_fbad0_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_fbad0_row5_col3" class="data row5 col3" >IF capital.gain > 4934.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row5_col4" class="data row5 col4" >0.03549</td>
      <td id="T_fbad0_row5_col5" class="data row5 col5" >0.13864</td>
      <td id="T_fbad0_row5_col6" class="data row5 col6" >0.94067</td>
      <td id="T_fbad0_row5_col7" class="data row5 col7" >2</td>
      <td id="T_fbad0_row5_col8" class="data row5 col8" >0</td>
      <td id="T_fbad0_row5_col9" class="data row5 col9" >55.74695</td>
      <td id="T_fbad0_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_fbad0_row6_col0" class="data row6 col0" >2711</td>
      <td id="T_fbad0_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_fbad0_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_fbad0_row6_col3" class="data row6 col3" >IF age > 24.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row6_col4" class="data row6 col4" >0.39505</td>
      <td id="T_fbad0_row6_col5" class="data row6 col5" >0.75424</td>
      <td id="T_fbad0_row6_col6" class="data row6 col6" >0.45980</td>
      <td id="T_fbad0_row6_col7" class="data row6 col7" >3</td>
      <td id="T_fbad0_row6_col8" class="data row6 col8" >0</td>
      <td id="T_fbad0_row6_col9" class="data row6 col9" >76.22816</td>
      <td id="T_fbad0_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_fbad0_row7_col0" class="data row7 col0" >2711</td>
      <td id="T_fbad0_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_fbad0_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_fbad0_row7_col3" class="data row7 col3" >IF age > 25.0 AND capital.gain > 4101.0 THEN class = >50K</td>
      <td id="T_fbad0_row7_col4" class="data row7 col4" >0.05555</td>
      <td id="T_fbad0_row7_col5" class="data row7 col5" >0.20259</td>
      <td id="T_fbad0_row7_col6" class="data row7 col6" >0.87836</td>
      <td id="T_fbad0_row7_col7" class="data row7 col7" >2</td>
      <td id="T_fbad0_row7_col8" class="data row7 col8" >0</td>
      <td id="T_fbad0_row7_col9" class="data row7 col9" >82.61932</td>
      <td id="T_fbad0_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_fbad0_row8_col0" class="data row8 col0" >2711</td>
      <td id="T_fbad0_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_fbad0_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_fbad0_row8_col3" class="data row8 col3" >IF capital.gain > 3453.9974 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row8_col4" class="data row8 col4" >0.04690</td>
      <td id="T_fbad0_row8_col5" class="data row8 col5" >0.16378</td>
      <td id="T_fbad0_row8_col6" class="data row8 col6" >0.84097</td>
      <td id="T_fbad0_row8_col7" class="data row8 col7" >2</td>
      <td id="T_fbad0_row8_col8" class="data row8 col8" >0</td>
      <td id="T_fbad0_row8_col9" class="data row8 col9" >74.79921</td>
      <td id="T_fbad0_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_fbad0_row9_col0" class="data row9 col0" >2711</td>
      <td id="T_fbad0_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_fbad0_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_fbad0_row9_col3" class="data row9 col3" >IF capital.gain > 4263.7872 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row9_col4" class="data row9 col4" >0.04273</td>
      <td id="T_fbad0_row9_col5" class="data row9 col5" >0.16378</td>
      <td id="T_fbad0_row9_col6" class="data row9 col6" >0.92300</td>
      <td id="T_fbad0_row9_col7" class="data row9 col7" >2</td>
      <td id="T_fbad0_row9_col8" class="data row9 col8" >0</td>
      <td id="T_fbad0_row9_col9" class="data row9 col9" >83.88141</td>
      <td id="T_fbad0_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_fbad0_row10_col0" class="data row10 col0" >2711</td>
      <td id="T_fbad0_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_fbad0_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_fbad0_row10_col3" class="data row10 col3" >IF relationship = Husband THEN class = >50K</td>
      <td id="T_fbad0_row10_col4" class="data row10 col4" >0.40479</td>
      <td id="T_fbad0_row10_col5" class="data row10 col5" >0.75752</td>
      <td id="T_fbad0_row10_col6" class="data row10 col6" >0.45068</td>
      <td id="T_fbad0_row10_col7" class="data row10 col7" >1</td>
      <td id="T_fbad0_row10_col8" class="data row10 col8" >0</td>
      <td id="T_fbad0_row10_col9" class="data row10 col9" >19.54397</td>
      <td id="T_fbad0_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_fbad0_row11_col0" class="data row11 col0" >2711</td>
      <td id="T_fbad0_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_fbad0_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_fbad0_row11_col3" class="data row11 col3" >IF marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row11_col4" class="data row11 col4" >0.45920</td>
      <td id="T_fbad0_row11_col5" class="data row11 col5" >0.85553</td>
      <td id="T_fbad0_row11_col6" class="data row11 col6" >0.44869</td>
      <td id="T_fbad0_row11_col7" class="data row11 col7" >1</td>
      <td id="T_fbad0_row11_col8" class="data row11 col8" >1</td>
      <td id="T_fbad0_row11_col9" class="data row11 col9" >39.17329</td>
      <td id="T_fbad0_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_fbad0_row12_col0" class="data row12 col0" >2711</td>
      <td id="T_fbad0_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_fbad0_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_fbad0_row12_col3" class="data row12 col3" >IF capital.gain > 3664.4158 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row12_col4" class="data row12 col4" >0.04607</td>
      <td id="T_fbad0_row12_col5" class="data row12 col5" >0.16378</td>
      <td id="T_fbad0_row12_col6" class="data row12 col6" >0.85619</td>
      <td id="T_fbad0_row12_col7" class="data row12 col7" >2</td>
      <td id="T_fbad0_row12_col8" class="data row12 col8" >2</td>
      <td id="T_fbad0_row12_col9" class="data row12 col9" >57.88563</td>
      <td id="T_fbad0_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_fbad0_row13_col0" class="data row13 col0" >2711</td>
      <td id="T_fbad0_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_fbad0_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_fbad0_row13_col3" class="data row13 col3" >IF capital.gain > 5599.881 AND education != Assoc-acdm AND education.num != 15.0 AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_fbad0_row13_col4" class="data row13 col4" >0.03549</td>
      <td id="T_fbad0_row13_col5" class="data row13 col5" >0.14083</td>
      <td id="T_fbad0_row13_col6" class="data row13 col6" >0.95550</td>
      <td id="T_fbad0_row13_col7" class="data row13 col7" >4</td>
      <td id="T_fbad0_row13_col8" class="data row13 col8" >1</td>
      <td id="T_fbad0_row13_col9" class="data row13 col9" >37.80378</td>
      <td id="T_fbad0_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_fbad0_row14_col0" class="data row14 col0" >2711</td>
      <td id="T_fbad0_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_fbad0_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_fbad0_row14_col3" class="data row14 col3" >IF capital.gain > 2356.6662 AND education != 12th AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_fbad0_row14_col4" class="data row14 col4" >0.06467</td>
      <td id="T_fbad0_row14_col5" class="data row14 col5" >0.19676</td>
      <td id="T_fbad0_row14_col6" class="data row14 col6" >0.73270</td>
      <td id="T_fbad0_row14_col7" class="data row14 col7" >3</td>
      <td id="T_fbad0_row14_col8" class="data row14 col8" >0</td>
      <td id="T_fbad0_row14_col9" class="data row14 col9" >18.96218</td>
      <td id="T_fbad0_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_fbad0_row15_col0" class="data row15 col0" >2711</td>
      <td id="T_fbad0_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_fbad0_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_fbad0_row15_col3" class="data row15 col3" >IF age > 31.0 AND capital.gain > 5013.0 AND capital.gain <= 8525.3219 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row15_col4" class="data row15 col4" >0.01843</td>
      <td id="T_fbad0_row15_col5" class="data row15 col5" >0.07579</td>
      <td id="T_fbad0_row15_col6" class="data row15 col6" >0.99048</td>
      <td id="T_fbad0_row15_col7" class="data row15 col7" >4</td>
      <td id="T_fbad0_row15_col8" class="data row15 col8" >0</td>
      <td id="T_fbad0_row15_col9" class="data row15 col9" >5.99136</td>
      <td id="T_fbad0_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_fbad0_row16_col0" class="data row16 col0" >2711</td>
      <td id="T_fbad0_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_fbad0_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_fbad0_row16_col3" class="data row16 col3" >IF age > 24.4479 AND capital.gain > 5019.1937 AND capital.gain <= 7870.1952 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row16_col4" class="data row16 col4" >0.02014</td>
      <td id="T_fbad0_row16_col5" class="data row16 col5" >0.08289</td>
      <td id="T_fbad0_row16_col6" class="data row16 col6" >0.99129</td>
      <td id="T_fbad0_row16_col7" class="data row16 col7" >4</td>
      <td id="T_fbad0_row16_col8" class="data row16 col8" >0</td>
      <td id="T_fbad0_row16_col9" class="data row16 col9" >7.34034</td>
      <td id="T_fbad0_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_fbad0_row17_col0" class="data row17 col0" >2711</td>
      <td id="T_fbad0_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_fbad0_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_fbad0_row17_col3" class="data row17 col3" >IF capital.gain > 7112.1724 AND capital.gain <= 7443.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row17_col4" class="data row17 col4" >0.00741</td>
      <td id="T_fbad0_row17_col5" class="data row17 col5" >0.03079</td>
      <td id="T_fbad0_row17_col6" class="data row17 col6" >1.00000</td>
      <td id="T_fbad0_row17_col7" class="data row17 col7" >3</td>
      <td id="T_fbad0_row17_col8" class="data row17 col8" >0</td>
      <td id="T_fbad0_row17_col9" class="data row17 col9" >6.99874</td>
      <td id="T_fbad0_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_fbad0_row18_col0" class="data row18 col0" >2711</td>
      <td id="T_fbad0_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_fbad0_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_fbad0_row18_col3" class="data row18 col3" >IF capital.gain > 5065.0654 AND capital.gain <= 7688.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_fbad0_row18_col4" class="data row18 col4" >0.02027</td>
      <td id="T_fbad0_row18_col5" class="data row18 col5" >0.08344</td>
      <td id="T_fbad0_row18_col6" class="data row18 col6" >0.99134</td>
      <td id="T_fbad0_row18_col7" class="data row18 col7" >3</td>
      <td id="T_fbad0_row18_col8" class="data row18 col8" >0</td>
      <td id="T_fbad0_row18_col9" class="data row18 col9" >5.80869</td>
      <td id="T_fbad0_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_fbad0_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_fbad0_row19_col0" class="data row19 col0" >2711</td>
      <td id="T_fbad0_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_fbad0_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_fbad0_row19_col3" class="data row19 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_fbad0_row19_col4" class="data row19 col4" >0.04923</td>
      <td id="T_fbad0_row19_col5" class="data row19 col5" >0.19366</td>
      <td id="T_fbad0_row19_col6" class="data row19 col6" >0.94742</td>
      <td id="T_fbad0_row19_col7" class="data row19 col7" >1</td>
      <td id="T_fbad0_row19_col8" class="data row19 col8" >0</td>
      <td id="T_fbad0_row19_col9" class="data row19 col9" >5.76786</td>
      <td id="T_fbad0_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 2711, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_91ad4">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_91ad4_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_91ad4_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_91ad4_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_91ad4_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_91ad4_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_91ad4_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_91ad4_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_91ad4_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_91ad4_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_91ad4_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_91ad4_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_91ad4_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_91ad4_row0_col0" class="data row0 col0" >2711</td>
      <td id="T_91ad4_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_91ad4_row0_col2" class="data row0 col2" >ANCHOR5</td>
      <td id="T_91ad4_row0_col3" class="data row0 col3" >IF age > 37.0 AND capital.gain > 0.0 AND hours.per.week > 45.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_91ad4_row0_col4" class="data row0 col4" >0.01505</td>
      <td id="T_91ad4_row0_col5" class="data row0 col5" >0.05465</td>
      <td id="T_91ad4_row0_col6" class="data row0 col6" >0.87464</td>
      <td id="T_91ad4_row0_col7" class="data row0 col7" >5</td>
      <td id="T_91ad4_row0_col8" class="data row0 col8" >1</td>
      <td id="T_91ad4_row0_col9" class="data row0 col9" >8.05141</td>
      <td id="T_91ad4_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_91ad4_row1_col0" class="data row1 col0" >2711</td>
      <td id="T_91ad4_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_91ad4_row1_col2" class="data row1 col2" >LORE1</td>
      <td id="T_91ad4_row1_col3" class="data row1 col3" >IF capital.gain > 4934.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_91ad4_row1_col4" class="data row1 col4" >0.03549</td>
      <td id="T_91ad4_row1_col5" class="data row1 col5" >0.13864</td>
      <td id="T_91ad4_row1_col6" class="data row1 col6" >0.94067</td>
      <td id="T_91ad4_row1_col7" class="data row1 col7" >2</td>
      <td id="T_91ad4_row1_col8" class="data row1 col8" >0</td>
      <td id="T_91ad4_row1_col9" class="data row1 col9" >55.74695</td>
      <td id="T_91ad4_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_91ad4_row2_col0" class="data row2 col0" >2711</td>
      <td id="T_91ad4_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_91ad4_row2_col2" class="data row2 col2" >LORE2</td>
      <td id="T_91ad4_row2_col3" class="data row2 col3" >IF age > 24.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_91ad4_row2_col4" class="data row2 col4" >0.39505</td>
      <td id="T_91ad4_row2_col5" class="data row2 col5" >0.75424</td>
      <td id="T_91ad4_row2_col6" class="data row2 col6" >0.45980</td>
      <td id="T_91ad4_row2_col7" class="data row2 col7" >3</td>
      <td id="T_91ad4_row2_col8" class="data row2 col8" >0</td>
      <td id="T_91ad4_row2_col9" class="data row2 col9" >76.22816</td>
      <td id="T_91ad4_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_91ad4_row3_col0" class="data row3 col0" >2711</td>
      <td id="T_91ad4_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_91ad4_row3_col2" class="data row3 col2" >LORE3</td>
      <td id="T_91ad4_row3_col3" class="data row3 col3" >IF age > 25.0 AND capital.gain > 4101.0 THEN class = >50K</td>
      <td id="T_91ad4_row3_col4" class="data row3 col4" >0.05555</td>
      <td id="T_91ad4_row3_col5" class="data row3 col5" >0.20259</td>
      <td id="T_91ad4_row3_col6" class="data row3 col6" >0.87836</td>
      <td id="T_91ad4_row3_col7" class="data row3 col7" >2</td>
      <td id="T_91ad4_row3_col8" class="data row3 col8" >0</td>
      <td id="T_91ad4_row3_col9" class="data row3 col9" >82.61932</td>
      <td id="T_91ad4_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_91ad4_row4_col0" class="data row4 col0" >2711</td>
      <td id="T_91ad4_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_91ad4_row4_col2" class="data row4 col2" >LORE4</td>
      <td id="T_91ad4_row4_col3" class="data row4 col3" >IF capital.gain > 3453.9974 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_91ad4_row4_col4" class="data row4 col4" >0.04690</td>
      <td id="T_91ad4_row4_col5" class="data row4 col5" >0.16378</td>
      <td id="T_91ad4_row4_col6" class="data row4 col6" >0.84097</td>
      <td id="T_91ad4_row4_col7" class="data row4 col7" >2</td>
      <td id="T_91ad4_row4_col8" class="data row4 col8" >0</td>
      <td id="T_91ad4_row4_col9" class="data row4 col9" >74.79921</td>
      <td id="T_91ad4_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_91ad4_row5_col0" class="data row5 col0" >2711</td>
      <td id="T_91ad4_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_91ad4_row5_col2" class="data row5 col2" >LORE5</td>
      <td id="T_91ad4_row5_col3" class="data row5 col3" >IF capital.gain > 4263.7872 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_91ad4_row5_col4" class="data row5 col4" >0.04273</td>
      <td id="T_91ad4_row5_col5" class="data row5 col5" >0.16378</td>
      <td id="T_91ad4_row5_col6" class="data row5 col6" >0.92300</td>
      <td id="T_91ad4_row5_col7" class="data row5 col7" >2</td>
      <td id="T_91ad4_row5_col8" class="data row5 col8" >0</td>
      <td id="T_91ad4_row5_col9" class="data row5 col9" >83.88141</td>
      <td id="T_91ad4_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_91ad4_row6_col0" class="data row6 col0" >2711</td>
      <td id="T_91ad4_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_91ad4_row6_col2" class="data row6 col2" >LORE_SA1</td>
      <td id="T_91ad4_row6_col3" class="data row6 col3" >IF relationship = Husband THEN class = >50K</td>
      <td id="T_91ad4_row6_col4" class="data row6 col4" >0.40479</td>
      <td id="T_91ad4_row6_col5" class="data row6 col5" >0.75752</td>
      <td id="T_91ad4_row6_col6" class="data row6 col6" >0.45068</td>
      <td id="T_91ad4_row6_col7" class="data row6 col7" >1</td>
      <td id="T_91ad4_row6_col8" class="data row6 col8" >0</td>
      <td id="T_91ad4_row6_col9" class="data row6 col9" >19.54397</td>
      <td id="T_91ad4_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_91ad4_row7_col0" class="data row7 col0" >2711</td>
      <td id="T_91ad4_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_91ad4_row7_col2" class="data row7 col2" >LORE_SA2</td>
      <td id="T_91ad4_row7_col3" class="data row7 col3" >IF marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_91ad4_row7_col4" class="data row7 col4" >0.45920</td>
      <td id="T_91ad4_row7_col5" class="data row7 col5" >0.85553</td>
      <td id="T_91ad4_row7_col6" class="data row7 col6" >0.44869</td>
      <td id="T_91ad4_row7_col7" class="data row7 col7" >1</td>
      <td id="T_91ad4_row7_col8" class="data row7 col8" >1</td>
      <td id="T_91ad4_row7_col9" class="data row7 col9" >39.17329</td>
      <td id="T_91ad4_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_91ad4_row8_col0" class="data row8 col0" >2711</td>
      <td id="T_91ad4_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_91ad4_row8_col2" class="data row8 col2" >LORE_SA3</td>
      <td id="T_91ad4_row8_col3" class="data row8 col3" >IF capital.gain > 3664.4158 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_91ad4_row8_col4" class="data row8 col4" >0.04607</td>
      <td id="T_91ad4_row8_col5" class="data row8 col5" >0.16378</td>
      <td id="T_91ad4_row8_col6" class="data row8 col6" >0.85619</td>
      <td id="T_91ad4_row8_col7" class="data row8 col7" >2</td>
      <td id="T_91ad4_row8_col8" class="data row8 col8" >2</td>
      <td id="T_91ad4_row8_col9" class="data row8 col9" >57.88563</td>
      <td id="T_91ad4_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_91ad4_row9_col0" class="data row9 col0" >2711</td>
      <td id="T_91ad4_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_91ad4_row9_col2" class="data row9 col2" >LORE_SA4</td>
      <td id="T_91ad4_row9_col3" class="data row9 col3" >IF capital.gain > 5599.881 AND education != Assoc-acdm AND education.num != 15.0 AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_91ad4_row9_col4" class="data row9 col4" >0.03549</td>
      <td id="T_91ad4_row9_col5" class="data row9 col5" >0.14083</td>
      <td id="T_91ad4_row9_col6" class="data row9 col6" >0.95550</td>
      <td id="T_91ad4_row9_col7" class="data row9 col7" >4</td>
      <td id="T_91ad4_row9_col8" class="data row9 col8" >1</td>
      <td id="T_91ad4_row9_col9" class="data row9 col9" >37.80378</td>
      <td id="T_91ad4_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_91ad4_row10_col0" class="data row10 col0" >2711</td>
      <td id="T_91ad4_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_91ad4_row10_col2" class="data row10 col2" >LORE_SA5</td>
      <td id="T_91ad4_row10_col3" class="data row10 col3" >IF capital.gain > 2356.6662 AND education != 12th AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_91ad4_row10_col4" class="data row10 col4" >0.06467</td>
      <td id="T_91ad4_row10_col5" class="data row10 col5" >0.19676</td>
      <td id="T_91ad4_row10_col6" class="data row10 col6" >0.73270</td>
      <td id="T_91ad4_row10_col7" class="data row10 col7" >3</td>
      <td id="T_91ad4_row10_col8" class="data row10 col8" >0</td>
      <td id="T_91ad4_row10_col9" class="data row10 col9" >18.96218</td>
      <td id="T_91ad4_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_91ad4_row11_col0" class="data row11 col0" >2711</td>
      <td id="T_91ad4_row11_col1" class="data row11 col1" >EXPLAN</td>
      <td id="T_91ad4_row11_col2" class="data row11 col2" >EXPLAN1</td>
      <td id="T_91ad4_row11_col3" class="data row11 col3" >IF age > 31.0 AND capital.gain > 5013.0 AND capital.gain <= 8525.3219 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_91ad4_row11_col4" class="data row11 col4" >0.01843</td>
      <td id="T_91ad4_row11_col5" class="data row11 col5" >0.07579</td>
      <td id="T_91ad4_row11_col6" class="data row11 col6" >0.99048</td>
      <td id="T_91ad4_row11_col7" class="data row11 col7" >4</td>
      <td id="T_91ad4_row11_col8" class="data row11 col8" >0</td>
      <td id="T_91ad4_row11_col9" class="data row11 col9" >5.99136</td>
      <td id="T_91ad4_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_91ad4_row12_col0" class="data row12 col0" >2711</td>
      <td id="T_91ad4_row12_col1" class="data row12 col1" >EXPLAN</td>
      <td id="T_91ad4_row12_col2" class="data row12 col2" >EXPLAN2</td>
      <td id="T_91ad4_row12_col3" class="data row12 col3" >IF age > 24.4479 AND capital.gain > 5019.1937 AND capital.gain <= 7870.1952 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_91ad4_row12_col4" class="data row12 col4" >0.02014</td>
      <td id="T_91ad4_row12_col5" class="data row12 col5" >0.08289</td>
      <td id="T_91ad4_row12_col6" class="data row12 col6" >0.99129</td>
      <td id="T_91ad4_row12_col7" class="data row12 col7" >4</td>
      <td id="T_91ad4_row12_col8" class="data row12 col8" >0</td>
      <td id="T_91ad4_row12_col9" class="data row12 col9" >7.34034</td>
      <td id="T_91ad4_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_91ad4_row13_col0" class="data row13 col0" >2711</td>
      <td id="T_91ad4_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_91ad4_row13_col2" class="data row13 col2" >EXPLAN4</td>
      <td id="T_91ad4_row13_col3" class="data row13 col3" >IF capital.gain > 5065.0654 AND capital.gain <= 7688.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_91ad4_row13_col4" class="data row13 col4" >0.02027</td>
      <td id="T_91ad4_row13_col5" class="data row13 col5" >0.08344</td>
      <td id="T_91ad4_row13_col6" class="data row13 col6" >0.99134</td>
      <td id="T_91ad4_row13_col7" class="data row13 col7" >3</td>
      <td id="T_91ad4_row13_col8" class="data row13 col8" >0</td>
      <td id="T_91ad4_row13_col9" class="data row13 col9" >5.80869</td>
      <td id="T_91ad4_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_91ad4_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_91ad4_row14_col0" class="data row14 col0" >2711</td>
      <td id="T_91ad4_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_91ad4_row14_col2" class="data row14 col2" >EXPLAN5</td>
      <td id="T_91ad4_row14_col3" class="data row14 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_91ad4_row14_col4" class="data row14 col4" >0.04923</td>
      <td id="T_91ad4_row14_col5" class="data row14 col5" >0.19366</td>
      <td id="T_91ad4_row14_col6" class="data row14 col6" >0.94742</td>
      <td id="T_91ad4_row14_col7" class="data row14 col7" >1</td>
      <td id="T_91ad4_row14_col8" class="data row14 col8" >0</td>
      <td id="T_91ad4_row14_col9" class="data row14 col9" >5.76786</td>
      <td id="T_91ad4_row14_col10" class="data row14 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 2711, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.4592, Pre: 0.99134)



<style type="text/css">
#T_83aae_row7_col0, #T_83aae_row7_col1, #T_83aae_row7_col2, #T_83aae_row7_col3, #T_83aae_row7_col4, #T_83aae_row7_col5, #T_83aae_row7_col6, #T_83aae_row7_col7, #T_83aae_row7_col8, #T_83aae_row7_col9, #T_83aae_row7_col10, #T_83aae_row7_col11 {
  font-weight: bold;
}
</style>
<table id="T_83aae">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_83aae_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_83aae_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_83aae_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_83aae_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_83aae_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_83aae_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_83aae_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_83aae_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_83aae_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_83aae_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_83aae_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_83aae_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_83aae_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_83aae_row0_col0" class="data row0 col0" >2711</td>
      <td id="T_83aae_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_83aae_row0_col2" class="data row0 col2" >LORE2</td>
      <td id="T_83aae_row0_col3" class="data row0 col3" >IF age > 24.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_83aae_row0_col4" class="data row0 col4" >0.39505</td>
      <td id="T_83aae_row0_col5" class="data row0 col5" >0.75424</td>
      <td id="T_83aae_row0_col6" class="data row0 col6" >0.45980</td>
      <td id="T_83aae_row0_col7" class="data row0 col7" >3</td>
      <td id="T_83aae_row0_col8" class="data row0 col8" >0</td>
      <td id="T_83aae_row0_col9" class="data row0 col9" >76.22816</td>
      <td id="T_83aae_row0_col10" class="data row0 col10" >False</td>
      <td id="T_83aae_row0_col11" class="data row0 col11" >0.53540</td>
    </tr>
    <tr>
      <th id="T_83aae_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_83aae_row1_col0" class="data row1 col0" >2711</td>
      <td id="T_83aae_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_83aae_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_83aae_row1_col3" class="data row1 col3" >IF age > 25.0 AND capital.gain > 4101.0 THEN class = >50K</td>
      <td id="T_83aae_row1_col4" class="data row1 col4" >0.05555</td>
      <td id="T_83aae_row1_col5" class="data row1 col5" >0.20259</td>
      <td id="T_83aae_row1_col6" class="data row1 col6" >0.87836</td>
      <td id="T_83aae_row1_col7" class="data row1 col7" >2</td>
      <td id="T_83aae_row1_col8" class="data row1 col8" >0</td>
      <td id="T_83aae_row1_col9" class="data row1 col9" >82.61932</td>
      <td id="T_83aae_row1_col10" class="data row1 col10" >False</td>
      <td id="T_83aae_row1_col11" class="data row1 col11" >0.41916</td>
    </tr>
    <tr>
      <th id="T_83aae_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_83aae_row2_col0" class="data row2 col0" >2711</td>
      <td id="T_83aae_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_83aae_row2_col2" class="data row2 col2" >LORE_SA1</td>
      <td id="T_83aae_row2_col3" class="data row2 col3" >IF relationship = Husband THEN class = >50K</td>
      <td id="T_83aae_row2_col4" class="data row2 col4" >0.40479</td>
      <td id="T_83aae_row2_col5" class="data row2 col5" >0.75752</td>
      <td id="T_83aae_row2_col6" class="data row2 col6" >0.45068</td>
      <td id="T_83aae_row2_col7" class="data row2 col7" >1</td>
      <td id="T_83aae_row2_col8" class="data row2 col8" >0</td>
      <td id="T_83aae_row2_col9" class="data row2 col9" >19.54397</td>
      <td id="T_83aae_row2_col10" class="data row2 col10" >False</td>
      <td id="T_83aae_row2_col11" class="data row2 col11" >0.54339</td>
    </tr>
    <tr>
      <th id="T_83aae_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_83aae_row3_col0" class="data row3 col0" >2711</td>
      <td id="T_83aae_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_83aae_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_83aae_row3_col3" class="data row3 col3" >IF marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_83aae_row3_col4" class="data row3 col4" >0.45920</td>
      <td id="T_83aae_row3_col5" class="data row3 col5" >0.85553</td>
      <td id="T_83aae_row3_col6" class="data row3 col6" >0.44869</td>
      <td id="T_83aae_row3_col7" class="data row3 col7" >1</td>
      <td id="T_83aae_row3_col8" class="data row3 col8" >1</td>
      <td id="T_83aae_row3_col9" class="data row3 col9" >39.17329</td>
      <td id="T_83aae_row3_col10" class="data row3 col10" >False</td>
      <td id="T_83aae_row3_col11" class="data row3 col11" >0.54265</td>
    </tr>
    <tr>
      <th id="T_83aae_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_83aae_row4_col0" class="data row4 col0" >2711</td>
      <td id="T_83aae_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_83aae_row4_col2" class="data row4 col2" >LORE_SA4</td>
      <td id="T_83aae_row4_col3" class="data row4 col3" >IF capital.gain > 5599.881 AND education != Assoc-acdm AND education.num != 15.0 AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_83aae_row4_col4" class="data row4 col4" >0.03549</td>
      <td id="T_83aae_row4_col5" class="data row4 col5" >0.14083</td>
      <td id="T_83aae_row4_col6" class="data row4 col6" >0.95550</td>
      <td id="T_83aae_row4_col7" class="data row4 col7" >4</td>
      <td id="T_83aae_row4_col8" class="data row4 col8" >1</td>
      <td id="T_83aae_row4_col9" class="data row4 col9" >37.80378</td>
      <td id="T_83aae_row4_col10" class="data row4 col10" >False</td>
      <td id="T_83aae_row4_col11" class="data row4 col11" >0.42522</td>
    </tr>
    <tr>
      <th id="T_83aae_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_83aae_row5_col0" class="data row5 col0" >2711</td>
      <td id="T_83aae_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_83aae_row5_col2" class="data row5 col2" >LORE_SA5</td>
      <td id="T_83aae_row5_col3" class="data row5 col3" >IF capital.gain > 2356.6662 AND education != 12th AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_83aae_row5_col4" class="data row5 col4" >0.06467</td>
      <td id="T_83aae_row5_col5" class="data row5 col5" >0.19676</td>
      <td id="T_83aae_row5_col6" class="data row5 col6" >0.73270</td>
      <td id="T_83aae_row5_col7" class="data row5 col7" >3</td>
      <td id="T_83aae_row5_col8" class="data row5 col8" >0</td>
      <td id="T_83aae_row5_col9" class="data row5 col9" >18.96218</td>
      <td id="T_83aae_row5_col10" class="data row5 col10" >False</td>
      <td id="T_83aae_row5_col11" class="data row5 col11" >0.47175</td>
    </tr>
    <tr>
      <th id="T_83aae_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_83aae_row6_col0" class="data row6 col0" >2711</td>
      <td id="T_83aae_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_83aae_row6_col2" class="data row6 col2" >EXPLAN4</td>
      <td id="T_83aae_row6_col3" class="data row6 col3" >IF capital.gain > 5065.0654 AND capital.gain <= 7688.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_83aae_row6_col4" class="data row6 col4" >0.02027</td>
      <td id="T_83aae_row6_col5" class="data row6 col5" >0.08344</td>
      <td id="T_83aae_row6_col6" class="data row6 col6" >0.99134</td>
      <td id="T_83aae_row6_col7" class="data row6 col7" >3</td>
      <td id="T_83aae_row6_col8" class="data row6 col8" >0</td>
      <td id="T_83aae_row6_col9" class="data row6 col9" >5.80869</td>
      <td id="T_83aae_row6_col10" class="data row6 col10" >False</td>
      <td id="T_83aae_row6_col11" class="data row6 col11" >0.43893</td>
    </tr>
    <tr>
      <th id="T_83aae_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_83aae_row7_col0" class="data row7 col0" >2711</td>
      <td id="T_83aae_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_83aae_row7_col2" class="data row7 col2" >EXPLAN5</td>
      <td id="T_83aae_row7_col3" class="data row7 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_83aae_row7_col4" class="data row7 col4" >0.04923</td>
      <td id="T_83aae_row7_col5" class="data row7 col5" >0.19366</td>
      <td id="T_83aae_row7_col6" class="data row7 col6" >0.94742</td>
      <td id="T_83aae_row7_col7" class="data row7 col7" >1</td>
      <td id="T_83aae_row7_col8" class="data row7 col8" >0</td>
      <td id="T_83aae_row7_col9" class="data row7 col9" >5.76786</td>
      <td id="T_83aae_row7_col10" class="data row7 col10" >False</td>
      <td id="T_83aae_row7_col11" class="data row7 col11" >0.41232</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_73.png)
    



### Rules for Instance 2711, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.4592, Pre: 0.99134), Unique rules (diffrent features)



<style type="text/css">
#T_4fff4_row7_col0, #T_4fff4_row7_col1, #T_4fff4_row7_col2, #T_4fff4_row7_col3, #T_4fff4_row7_col4, #T_4fff4_row7_col5, #T_4fff4_row7_col6, #T_4fff4_row7_col7, #T_4fff4_row7_col8, #T_4fff4_row7_col9, #T_4fff4_row7_col10, #T_4fff4_row7_col11 {
  font-weight: bold;
}
</style>
<table id="T_4fff4">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_4fff4_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_4fff4_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_4fff4_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_4fff4_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_4fff4_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_4fff4_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_4fff4_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_4fff4_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_4fff4_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_4fff4_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_4fff4_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_4fff4_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_4fff4_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_4fff4_row0_col0" class="data row0 col0" >2711</td>
      <td id="T_4fff4_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_4fff4_row0_col2" class="data row0 col2" >LORE2</td>
      <td id="T_4fff4_row0_col3" class="data row0 col3" >IF age > 24.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_4fff4_row0_col4" class="data row0 col4" >0.39505</td>
      <td id="T_4fff4_row0_col5" class="data row0 col5" >0.75424</td>
      <td id="T_4fff4_row0_col6" class="data row0 col6" >0.45980</td>
      <td id="T_4fff4_row0_col7" class="data row0 col7" >3</td>
      <td id="T_4fff4_row0_col8" class="data row0 col8" >0</td>
      <td id="T_4fff4_row0_col9" class="data row0 col9" >76.22816</td>
      <td id="T_4fff4_row0_col10" class="data row0 col10" >False</td>
      <td id="T_4fff4_row0_col11" class="data row0 col11" >0.53540</td>
    </tr>
    <tr>
      <th id="T_4fff4_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_4fff4_row1_col0" class="data row1 col0" >2711</td>
      <td id="T_4fff4_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_4fff4_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_4fff4_row1_col3" class="data row1 col3" >IF age > 25.0 AND capital.gain > 4101.0 THEN class = >50K</td>
      <td id="T_4fff4_row1_col4" class="data row1 col4" >0.05555</td>
      <td id="T_4fff4_row1_col5" class="data row1 col5" >0.20259</td>
      <td id="T_4fff4_row1_col6" class="data row1 col6" >0.87836</td>
      <td id="T_4fff4_row1_col7" class="data row1 col7" >2</td>
      <td id="T_4fff4_row1_col8" class="data row1 col8" >0</td>
      <td id="T_4fff4_row1_col9" class="data row1 col9" >82.61932</td>
      <td id="T_4fff4_row1_col10" class="data row1 col10" >False</td>
      <td id="T_4fff4_row1_col11" class="data row1 col11" >0.41916</td>
    </tr>
    <tr>
      <th id="T_4fff4_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_4fff4_row2_col0" class="data row2 col0" >2711</td>
      <td id="T_4fff4_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_4fff4_row2_col2" class="data row2 col2" >LORE_SA1</td>
      <td id="T_4fff4_row2_col3" class="data row2 col3" >IF relationship = Husband THEN class = >50K</td>
      <td id="T_4fff4_row2_col4" class="data row2 col4" >0.40479</td>
      <td id="T_4fff4_row2_col5" class="data row2 col5" >0.75752</td>
      <td id="T_4fff4_row2_col6" class="data row2 col6" >0.45068</td>
      <td id="T_4fff4_row2_col7" class="data row2 col7" >1</td>
      <td id="T_4fff4_row2_col8" class="data row2 col8" >0</td>
      <td id="T_4fff4_row2_col9" class="data row2 col9" >19.54397</td>
      <td id="T_4fff4_row2_col10" class="data row2 col10" >False</td>
      <td id="T_4fff4_row2_col11" class="data row2 col11" >0.54339</td>
    </tr>
    <tr>
      <th id="T_4fff4_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_4fff4_row3_col0" class="data row3 col0" >2711</td>
      <td id="T_4fff4_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_4fff4_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_4fff4_row3_col3" class="data row3 col3" >IF marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_4fff4_row3_col4" class="data row3 col4" >0.45920</td>
      <td id="T_4fff4_row3_col5" class="data row3 col5" >0.85553</td>
      <td id="T_4fff4_row3_col6" class="data row3 col6" >0.44869</td>
      <td id="T_4fff4_row3_col7" class="data row3 col7" >1</td>
      <td id="T_4fff4_row3_col8" class="data row3 col8" >1</td>
      <td id="T_4fff4_row3_col9" class="data row3 col9" >39.17329</td>
      <td id="T_4fff4_row3_col10" class="data row3 col10" >False</td>
      <td id="T_4fff4_row3_col11" class="data row3 col11" >0.54265</td>
    </tr>
    <tr>
      <th id="T_4fff4_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_4fff4_row4_col0" class="data row4 col0" >2711</td>
      <td id="T_4fff4_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_4fff4_row4_col2" class="data row4 col2" >LORE_SA4</td>
      <td id="T_4fff4_row4_col3" class="data row4 col3" >IF capital.gain > 5599.881 AND education != Assoc-acdm AND education.num != 15.0 AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_4fff4_row4_col4" class="data row4 col4" >0.03549</td>
      <td id="T_4fff4_row4_col5" class="data row4 col5" >0.14083</td>
      <td id="T_4fff4_row4_col6" class="data row4 col6" >0.95550</td>
      <td id="T_4fff4_row4_col7" class="data row4 col7" >4</td>
      <td id="T_4fff4_row4_col8" class="data row4 col8" >1</td>
      <td id="T_4fff4_row4_col9" class="data row4 col9" >37.80378</td>
      <td id="T_4fff4_row4_col10" class="data row4 col10" >False</td>
      <td id="T_4fff4_row4_col11" class="data row4 col11" >0.42522</td>
    </tr>
    <tr>
      <th id="T_4fff4_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_4fff4_row5_col0" class="data row5 col0" >2711</td>
      <td id="T_4fff4_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_4fff4_row5_col2" class="data row5 col2" >LORE_SA5</td>
      <td id="T_4fff4_row5_col3" class="data row5 col3" >IF capital.gain > 2356.6662 AND education != 12th AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_4fff4_row5_col4" class="data row5 col4" >0.06467</td>
      <td id="T_4fff4_row5_col5" class="data row5 col5" >0.19676</td>
      <td id="T_4fff4_row5_col6" class="data row5 col6" >0.73270</td>
      <td id="T_4fff4_row5_col7" class="data row5 col7" >3</td>
      <td id="T_4fff4_row5_col8" class="data row5 col8" >0</td>
      <td id="T_4fff4_row5_col9" class="data row5 col9" >18.96218</td>
      <td id="T_4fff4_row5_col10" class="data row5 col10" >False</td>
      <td id="T_4fff4_row5_col11" class="data row5 col11" >0.47175</td>
    </tr>
    <tr>
      <th id="T_4fff4_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_4fff4_row6_col0" class="data row6 col0" >2711</td>
      <td id="T_4fff4_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_4fff4_row6_col2" class="data row6 col2" >EXPLAN4</td>
      <td id="T_4fff4_row6_col3" class="data row6 col3" >IF capital.gain > 5065.0654 AND capital.gain <= 7688.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_4fff4_row6_col4" class="data row6 col4" >0.02027</td>
      <td id="T_4fff4_row6_col5" class="data row6 col5" >0.08344</td>
      <td id="T_4fff4_row6_col6" class="data row6 col6" >0.99134</td>
      <td id="T_4fff4_row6_col7" class="data row6 col7" >3</td>
      <td id="T_4fff4_row6_col8" class="data row6 col8" >0</td>
      <td id="T_4fff4_row6_col9" class="data row6 col9" >5.80869</td>
      <td id="T_4fff4_row6_col10" class="data row6 col10" >False</td>
      <td id="T_4fff4_row6_col11" class="data row6 col11" >0.43893</td>
    </tr>
    <tr>
      <th id="T_4fff4_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_4fff4_row7_col0" class="data row7 col0" >2711</td>
      <td id="T_4fff4_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_4fff4_row7_col2" class="data row7 col2" >EXPLAN5</td>
      <td id="T_4fff4_row7_col3" class="data row7 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_4fff4_row7_col4" class="data row7 col4" >0.04923</td>
      <td id="T_4fff4_row7_col5" class="data row7 col5" >0.19366</td>
      <td id="T_4fff4_row7_col6" class="data row7 col6" >0.94742</td>
      <td id="T_4fff4_row7_col7" class="data row7 col7" >1</td>
      <td id="T_4fff4_row7_col8" class="data row7 col8" >0</td>
      <td id="T_4fff4_row7_col9" class="data row7 col9" >5.76786</td>
      <td id="T_4fff4_row7_col10" class="data row7 col10" >False</td>
      <td id="T_4fff4_row7_col11" class="data row7 col11" >0.41232</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_76.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_77.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_78.png)
    



### Rules for Instance 2711, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.85553, Pre: 0.99134, Len: 0.44869)



<style type="text/css">
#T_8ac3d_row3_col0, #T_8ac3d_row3_col1, #T_8ac3d_row3_col2, #T_8ac3d_row3_col3, #T_8ac3d_row3_col4, #T_8ac3d_row3_col5, #T_8ac3d_row3_col6, #T_8ac3d_row3_col7, #T_8ac3d_row3_col8, #T_8ac3d_row3_col9, #T_8ac3d_row3_col10, #T_8ac3d_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_8ac3d">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8ac3d_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_8ac3d_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_8ac3d_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_8ac3d_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_8ac3d_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_8ac3d_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_8ac3d_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_8ac3d_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_8ac3d_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_8ac3d_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_8ac3d_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_8ac3d_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8ac3d_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_8ac3d_row0_col0" class="data row0 col0" >2711</td>
      <td id="T_8ac3d_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_8ac3d_row0_col2" class="data row0 col2" >LORE2</td>
      <td id="T_8ac3d_row0_col3" class="data row0 col3" >IF age > 24.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_8ac3d_row0_col4" class="data row0 col4" >0.39505</td>
      <td id="T_8ac3d_row0_col5" class="data row0 col5" >0.75424</td>
      <td id="T_8ac3d_row0_col6" class="data row0 col6" >0.45980</td>
      <td id="T_8ac3d_row0_col7" class="data row0 col7" >3</td>
      <td id="T_8ac3d_row0_col8" class="data row0 col8" >0</td>
      <td id="T_8ac3d_row0_col9" class="data row0 col9" >76.22816</td>
      <td id="T_8ac3d_row0_col10" class="data row0 col10" >False</td>
      <td id="T_8ac3d_row0_col11" class="data row0 col11" >2.60806</td>
    </tr>
    <tr>
      <th id="T_8ac3d_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_8ac3d_row1_col0" class="data row1 col0" >2711</td>
      <td id="T_8ac3d_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_8ac3d_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_8ac3d_row1_col3" class="data row1 col3" >IF age > 25.0 AND capital.gain > 4101.0 THEN class = >50K</td>
      <td id="T_8ac3d_row1_col4" class="data row1 col4" >0.05555</td>
      <td id="T_8ac3d_row1_col5" class="data row1 col5" >0.20259</td>
      <td id="T_8ac3d_row1_col6" class="data row1 col6" >0.87836</td>
      <td id="T_8ac3d_row1_col7" class="data row1 col7" >2</td>
      <td id="T_8ac3d_row1_col8" class="data row1 col8" >0</td>
      <td id="T_8ac3d_row1_col9" class="data row1 col9" >82.61932</td>
      <td id="T_8ac3d_row1_col10" class="data row1 col10" >False</td>
      <td id="T_8ac3d_row1_col11" class="data row1 col11" >1.68691</td>
    </tr>
    <tr>
      <th id="T_8ac3d_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_8ac3d_row2_col0" class="data row2 col0" >2711</td>
      <td id="T_8ac3d_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_8ac3d_row2_col2" class="data row2 col2" >LORE_SA1</td>
      <td id="T_8ac3d_row2_col3" class="data row2 col3" >IF relationship = Husband THEN class = >50K</td>
      <td id="T_8ac3d_row2_col4" class="data row2 col4" >0.40479</td>
      <td id="T_8ac3d_row2_col5" class="data row2 col5" >0.75752</td>
      <td id="T_8ac3d_row2_col6" class="data row2 col6" >0.45068</td>
      <td id="T_8ac3d_row2_col7" class="data row2 col7" >1</td>
      <td id="T_8ac3d_row2_col8" class="data row2 col8" >0</td>
      <td id="T_8ac3d_row2_col9" class="data row2 col9" >19.54397</td>
      <td id="T_8ac3d_row2_col10" class="data row2 col10" >False</td>
      <td id="T_8ac3d_row2_col11" class="data row2 col11" >0.77837</td>
    </tr>
    <tr>
      <th id="T_8ac3d_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_8ac3d_row3_col0" class="data row3 col0" >2711</td>
      <td id="T_8ac3d_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_8ac3d_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_8ac3d_row3_col3" class="data row3 col3" >IF marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_8ac3d_row3_col4" class="data row3 col4" >0.45920</td>
      <td id="T_8ac3d_row3_col5" class="data row3 col5" >0.85553</td>
      <td id="T_8ac3d_row3_col6" class="data row3 col6" >0.44869</td>
      <td id="T_8ac3d_row3_col7" class="data row3 col7" >1</td>
      <td id="T_8ac3d_row3_col8" class="data row3 col8" >1</td>
      <td id="T_8ac3d_row3_col9" class="data row3 col9" >39.17329</td>
      <td id="T_8ac3d_row3_col10" class="data row3 col10" >False</td>
      <td id="T_8ac3d_row3_col11" class="data row3 col11" >0.77357</td>
    </tr>
    <tr>
      <th id="T_8ac3d_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_8ac3d_row4_col0" class="data row4 col0" >2711</td>
      <td id="T_8ac3d_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_8ac3d_row4_col2" class="data row4 col2" >LORE_SA4</td>
      <td id="T_8ac3d_row4_col3" class="data row4 col3" >IF capital.gain > 5599.881 AND education != Assoc-acdm AND education.num != 15.0 AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_8ac3d_row4_col4" class="data row4 col4" >0.03549</td>
      <td id="T_8ac3d_row4_col5" class="data row4 col5" >0.14083</td>
      <td id="T_8ac3d_row4_col6" class="data row4 col6" >0.95550</td>
      <td id="T_8ac3d_row4_col7" class="data row4 col7" >4</td>
      <td id="T_8ac3d_row4_col8" class="data row4 col8" >1</td>
      <td id="T_8ac3d_row4_col9" class="data row4 col9" >37.80378</td>
      <td id="T_8ac3d_row4_col10" class="data row4 col10" >False</td>
      <td id="T_8ac3d_row4_col11" class="data row4 col11" >3.62269</td>
    </tr>
    <tr>
      <th id="T_8ac3d_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_8ac3d_row5_col0" class="data row5 col0" >2711</td>
      <td id="T_8ac3d_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_8ac3d_row5_col2" class="data row5 col2" >EXPLAN4</td>
      <td id="T_8ac3d_row5_col3" class="data row5 col3" >IF capital.gain > 5065.0654 AND capital.gain <= 7688.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_8ac3d_row5_col4" class="data row5 col4" >0.02027</td>
      <td id="T_8ac3d_row5_col5" class="data row5 col5" >0.08344</td>
      <td id="T_8ac3d_row5_col6" class="data row5 col6" >0.99134</td>
      <td id="T_8ac3d_row5_col7" class="data row5 col7" >3</td>
      <td id="T_8ac3d_row5_col8" class="data row5 col8" >0</td>
      <td id="T_8ac3d_row5_col9" class="data row5 col9" >5.80869</td>
      <td id="T_8ac3d_row5_col10" class="data row5 col10" >False</td>
      <td id="T_8ac3d_row5_col11" class="data row5 col11" >2.66558</td>
    </tr>
    <tr>
      <th id="T_8ac3d_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_8ac3d_row6_col0" class="data row6 col0" >2711</td>
      <td id="T_8ac3d_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_8ac3d_row6_col2" class="data row6 col2" >EXPLAN5</td>
      <td id="T_8ac3d_row6_col3" class="data row6 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_8ac3d_row6_col4" class="data row6 col4" >0.04923</td>
      <td id="T_8ac3d_row6_col5" class="data row6 col5" >0.19366</td>
      <td id="T_8ac3d_row6_col6" class="data row6 col6" >0.94742</td>
      <td id="T_8ac3d_row6_col7" class="data row6 col7" >1</td>
      <td id="T_8ac3d_row6_col8" class="data row6 col8" >0</td>
      <td id="T_8ac3d_row6_col9" class="data row6 col9" >5.76786</td>
      <td id="T_8ac3d_row6_col10" class="data row6 col10" >False</td>
      <td id="T_8ac3d_row6_col11" class="data row6 col11" >0.86252</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 2711, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.85553, Pre: 0.99134), Unique rules (diffrent features)



<style type="text/css">
#T_605c9_row3_col0, #T_605c9_row3_col1, #T_605c9_row3_col2, #T_605c9_row3_col3, #T_605c9_row3_col4, #T_605c9_row3_col5, #T_605c9_row3_col6, #T_605c9_row3_col7, #T_605c9_row3_col8, #T_605c9_row3_col9, #T_605c9_row3_col10, #T_605c9_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_605c9">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_605c9_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_605c9_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_605c9_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_605c9_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_605c9_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_605c9_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_605c9_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_605c9_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_605c9_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_605c9_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_605c9_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_605c9_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_605c9_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_605c9_row0_col0" class="data row0 col0" >2711</td>
      <td id="T_605c9_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_605c9_row0_col2" class="data row0 col2" >LORE2</td>
      <td id="T_605c9_row0_col3" class="data row0 col3" >IF age > 24.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_605c9_row0_col4" class="data row0 col4" >0.39505</td>
      <td id="T_605c9_row0_col5" class="data row0 col5" >0.75424</td>
      <td id="T_605c9_row0_col6" class="data row0 col6" >0.45980</td>
      <td id="T_605c9_row0_col7" class="data row0 col7" >3</td>
      <td id="T_605c9_row0_col8" class="data row0 col8" >0</td>
      <td id="T_605c9_row0_col9" class="data row0 col9" >76.22816</td>
      <td id="T_605c9_row0_col10" class="data row0 col10" >False</td>
      <td id="T_605c9_row0_col11" class="data row0 col11" >2.60806</td>
    </tr>
    <tr>
      <th id="T_605c9_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_605c9_row1_col0" class="data row1 col0" >2711</td>
      <td id="T_605c9_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_605c9_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_605c9_row1_col3" class="data row1 col3" >IF age > 25.0 AND capital.gain > 4101.0 THEN class = >50K</td>
      <td id="T_605c9_row1_col4" class="data row1 col4" >0.05555</td>
      <td id="T_605c9_row1_col5" class="data row1 col5" >0.20259</td>
      <td id="T_605c9_row1_col6" class="data row1 col6" >0.87836</td>
      <td id="T_605c9_row1_col7" class="data row1 col7" >2</td>
      <td id="T_605c9_row1_col8" class="data row1 col8" >0</td>
      <td id="T_605c9_row1_col9" class="data row1 col9" >82.61932</td>
      <td id="T_605c9_row1_col10" class="data row1 col10" >False</td>
      <td id="T_605c9_row1_col11" class="data row1 col11" >1.68691</td>
    </tr>
    <tr>
      <th id="T_605c9_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_605c9_row2_col0" class="data row2 col0" >2711</td>
      <td id="T_605c9_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_605c9_row2_col2" class="data row2 col2" >LORE_SA1</td>
      <td id="T_605c9_row2_col3" class="data row2 col3" >IF relationship = Husband THEN class = >50K</td>
      <td id="T_605c9_row2_col4" class="data row2 col4" >0.40479</td>
      <td id="T_605c9_row2_col5" class="data row2 col5" >0.75752</td>
      <td id="T_605c9_row2_col6" class="data row2 col6" >0.45068</td>
      <td id="T_605c9_row2_col7" class="data row2 col7" >1</td>
      <td id="T_605c9_row2_col8" class="data row2 col8" >0</td>
      <td id="T_605c9_row2_col9" class="data row2 col9" >19.54397</td>
      <td id="T_605c9_row2_col10" class="data row2 col10" >False</td>
      <td id="T_605c9_row2_col11" class="data row2 col11" >0.77837</td>
    </tr>
    <tr>
      <th id="T_605c9_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_605c9_row3_col0" class="data row3 col0" >2711</td>
      <td id="T_605c9_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_605c9_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_605c9_row3_col3" class="data row3 col3" >IF marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_605c9_row3_col4" class="data row3 col4" >0.45920</td>
      <td id="T_605c9_row3_col5" class="data row3 col5" >0.85553</td>
      <td id="T_605c9_row3_col6" class="data row3 col6" >0.44869</td>
      <td id="T_605c9_row3_col7" class="data row3 col7" >1</td>
      <td id="T_605c9_row3_col8" class="data row3 col8" >1</td>
      <td id="T_605c9_row3_col9" class="data row3 col9" >39.17329</td>
      <td id="T_605c9_row3_col10" class="data row3 col10" >False</td>
      <td id="T_605c9_row3_col11" class="data row3 col11" >0.77357</td>
    </tr>
    <tr>
      <th id="T_605c9_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_605c9_row4_col0" class="data row4 col0" >2711</td>
      <td id="T_605c9_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_605c9_row4_col2" class="data row4 col2" >LORE_SA4</td>
      <td id="T_605c9_row4_col3" class="data row4 col3" >IF capital.gain > 5599.881 AND education != Assoc-acdm AND education.num != 15.0 AND marital.status != Never-married THEN class = >50K</td>
      <td id="T_605c9_row4_col4" class="data row4 col4" >0.03549</td>
      <td id="T_605c9_row4_col5" class="data row4 col5" >0.14083</td>
      <td id="T_605c9_row4_col6" class="data row4 col6" >0.95550</td>
      <td id="T_605c9_row4_col7" class="data row4 col7" >4</td>
      <td id="T_605c9_row4_col8" class="data row4 col8" >1</td>
      <td id="T_605c9_row4_col9" class="data row4 col9" >37.80378</td>
      <td id="T_605c9_row4_col10" class="data row4 col10" >False</td>
      <td id="T_605c9_row4_col11" class="data row4 col11" >3.62269</td>
    </tr>
    <tr>
      <th id="T_605c9_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_605c9_row5_col0" class="data row5 col0" >2711</td>
      <td id="T_605c9_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_605c9_row5_col2" class="data row5 col2" >EXPLAN4</td>
      <td id="T_605c9_row5_col3" class="data row5 col3" >IF capital.gain > 5065.0654 AND capital.gain <= 7688.0 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_605c9_row5_col4" class="data row5 col4" >0.02027</td>
      <td id="T_605c9_row5_col5" class="data row5 col5" >0.08344</td>
      <td id="T_605c9_row5_col6" class="data row5 col6" >0.99134</td>
      <td id="T_605c9_row5_col7" class="data row5 col7" >3</td>
      <td id="T_605c9_row5_col8" class="data row5 col8" >0</td>
      <td id="T_605c9_row5_col9" class="data row5 col9" >5.80869</td>
      <td id="T_605c9_row5_col10" class="data row5 col10" >False</td>
      <td id="T_605c9_row5_col11" class="data row5 col11" >2.66558</td>
    </tr>
    <tr>
      <th id="T_605c9_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_605c9_row6_col0" class="data row6 col0" >2711</td>
      <td id="T_605c9_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_605c9_row6_col2" class="data row6 col2" >EXPLAN5</td>
      <td id="T_605c9_row6_col3" class="data row6 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_605c9_row6_col4" class="data row6 col4" >0.04923</td>
      <td id="T_605c9_row6_col5" class="data row6 col5" >0.19366</td>
      <td id="T_605c9_row6_col6" class="data row6 col6" >0.94742</td>
      <td id="T_605c9_row6_col7" class="data row6 col7" >1</td>
      <td id="T_605c9_row6_col8" class="data row6 col8" >0</td>
      <td id="T_605c9_row6_col9" class="data row6 col9" >5.76786</td>
      <td id="T_605c9_row6_col10" class="data row6 col10" >False</td>
      <td id="T_605c9_row6_col11" class="data row6 col11" >0.86252</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_83.png)
    



## Instance 12758 (Original: <=50K , Predicted: <=50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>66.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>HS-grad</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>9</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Widowed</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Priv-house-serv</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Not-in-family</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>50.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>England</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 12758



<style type="text/css">
</style>
<table id="T_d1258">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d1258_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_d1258_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_d1258_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_d1258_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_d1258_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_d1258_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_d1258_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_d1258_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_d1258_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_d1258_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_d1258_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d1258_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_d1258_row0_col0" class="data row0 col0" >12758</td>
      <td id="T_d1258_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_d1258_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_d1258_row0_col3" class="data row0 col3" >IF education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_d1258_row0_col4" class="data row0 col4" >0.32095</td>
      <td id="T_d1258_row0_col5" class="data row0 col5" >0.35508</td>
      <td id="T_d1258_row0_col6" class="data row0 col6" >0.83992</td>
      <td id="T_d1258_row0_col7" class="data row0 col7" >2</td>
      <td id="T_d1258_row0_col8" class="data row0 col8" >0</td>
      <td id="T_d1258_row0_col9" class="data row0 col9" >1.28740</td>
      <td id="T_d1258_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_d1258_row1_col0" class="data row1 col0" >12758</td>
      <td id="T_d1258_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_d1258_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_d1258_row1_col3" class="data row1 col3" >IF capital.loss <= 0.0 AND education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_d1258_row1_col4" class="data row1 col4" >0.30862</td>
      <td id="T_d1258_row1_col5" class="data row1 col5" >0.34445</td>
      <td id="T_d1258_row1_col6" class="data row1 col6" >0.84731</td>
      <td id="T_d1258_row1_col7" class="data row1 col7" >3</td>
      <td id="T_d1258_row1_col8" class="data row1 col8" >1</td>
      <td id="T_d1258_row1_col9" class="data row1 col9" >2.44282</td>
      <td id="T_d1258_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_d1258_row2_col0" class="data row2 col0" >12758</td>
      <td id="T_d1258_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_d1258_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_d1258_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_d1258_row2_col4" class="data row2 col4" >0.66273</td>
      <td id="T_d1258_row2_col5" class="data row2 col5" >0.72849</td>
      <td id="T_d1258_row2_col6" class="data row2 col6" >0.83449</td>
      <td id="T_d1258_row2_col7" class="data row2 col7" >3</td>
      <td id="T_d1258_row2_col8" class="data row2 col8" >0</td>
      <td id="T_d1258_row2_col9" class="data row2 col9" >1.24879</td>
      <td id="T_d1258_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_d1258_row3_col0" class="data row3 col0" >12758</td>
      <td id="T_d1258_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_d1258_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_d1258_row3_col3" class="data row3 col3" >IF capital.gain <= 0.0 AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_d1258_row3_col4" class="data row3 col4" >0.29936</td>
      <td id="T_d1258_row3_col5" class="data row3 col5" >0.33959</td>
      <td id="T_d1258_row3_col6" class="data row3 col6" >0.86120</td>
      <td id="T_d1258_row3_col7" class="data row3 col7" >2</td>
      <td id="T_d1258_row3_col8" class="data row3 col8" >0</td>
      <td id="T_d1258_row3_col9" class="data row3 col9" >1.56445</td>
      <td id="T_d1258_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_d1258_row4_col0" class="data row4 col0" >12758</td>
      <td id="T_d1258_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_d1258_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_d1258_row4_col3" class="data row4 col3" >IF education = HS-grad THEN class = <=50K</td>
      <td id="T_d1258_row4_col4" class="data row4 col4" >0.32095</td>
      <td id="T_d1258_row4_col5" class="data row4 col5" >0.35508</td>
      <td id="T_d1258_row4_col6" class="data row4 col6" >0.83992</td>
      <td id="T_d1258_row4_col7" class="data row4 col7" >1</td>
      <td id="T_d1258_row4_col8" class="data row4 col8" >1</td>
      <td id="T_d1258_row4_col9" class="data row4 col9" >2.61744</td>
      <td id="T_d1258_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_d1258_row5_col0" class="data row5 col0" >12758</td>
      <td id="T_d1258_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_d1258_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_d1258_row5_col3" class="data row5 col3" >IF capital.gain <= 3160.889 THEN class = <=50K</td>
      <td id="T_d1258_row5_col4" class="data row5 col4" >0.93458</td>
      <td id="T_d1258_row5_col5" class="data row5 col5" >0.97931</td>
      <td id="T_d1258_row5_col6" class="data row5 col6" >0.79550</td>
      <td id="T_d1258_row5_col7" class="data row5 col7" >1</td>
      <td id="T_d1258_row5_col8" class="data row5 col8" >0</td>
      <td id="T_d1258_row5_col9" class="data row5 col9" >80.03021</td>
      <td id="T_d1258_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_d1258_row6_col0" class="data row6 col0" >12758</td>
      <td id="T_d1258_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_d1258_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_d1258_row6_col3" class="data row6 col3" >IF capital.gain <= 3950.5089 THEN class = <=50K</td>
      <td id="T_d1258_row6_col4" class="data row6 col4" >0.94094</td>
      <td id="T_d1258_row6_col5" class="data row6 col5" >0.98769</td>
      <td id="T_d1258_row6_col6" class="data row6 col6" >0.79689</td>
      <td id="T_d1258_row6_col7" class="data row6 col7" >1</td>
      <td id="T_d1258_row6_col8" class="data row6 col8" >0</td>
      <td id="T_d1258_row6_col9" class="data row6 col9" >79.47443</td>
      <td id="T_d1258_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_d1258_row7_col0" class="data row7 col0" >12758</td>
      <td id="T_d1258_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_d1258_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_d1258_row7_col3" class="data row7 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_d1258_row7_col4" class="data row7 col4" >0.94226</td>
      <td id="T_d1258_row7_col5" class="data row7 col5" >0.98942</td>
      <td id="T_d1258_row7_col6" class="data row7 col6" >0.79717</td>
      <td id="T_d1258_row7_col7" class="data row7 col7" >1</td>
      <td id="T_d1258_row7_col8" class="data row7 col8" >0</td>
      <td id="T_d1258_row7_col9" class="data row7 col9" >79.87095</td>
      <td id="T_d1258_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_d1258_row8_col0" class="data row8 col0" >12758</td>
      <td id="T_d1258_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_d1258_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_d1258_row8_col3" class="data row8 col3" >IF capital.gain <= 6025.6313 THEN class = <=50K</td>
      <td id="T_d1258_row8_col4" class="data row8 col4" >0.95450</td>
      <td id="T_d1258_row8_col5" class="data row8 col5" >0.99717</td>
      <td id="T_d1258_row8_col6" class="data row8 col6" >0.79311</td>
      <td id="T_d1258_row8_col7" class="data row8 col7" >1</td>
      <td id="T_d1258_row8_col8" class="data row8 col8" >0</td>
      <td id="T_d1258_row8_col9" class="data row8 col9" >80.20908</td>
      <td id="T_d1258_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_d1258_row9_col0" class="data row9 col0" >12758</td>
      <td id="T_d1258_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_d1258_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_d1258_row9_col3" class="data row9 col3" >IF capital.gain <= 4787.0 THEN class = <=50K</td>
      <td id="T_d1258_row9_col4" class="data row9 col4" >0.94805</td>
      <td id="T_d1258_row9_col5" class="data row9 col5" >0.99324</td>
      <td id="T_d1258_row9_col6" class="data row9 col6" >0.79535</td>
      <td id="T_d1258_row9_col7" class="data row9 col7" >1</td>
      <td id="T_d1258_row9_col8" class="data row9 col8" >0</td>
      <td id="T_d1258_row9_col9" class="data row9 col9" >117.75989</td>
      <td id="T_d1258_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_d1258_row10_col0" class="data row10 col0" >12758</td>
      <td id="T_d1258_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_d1258_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_d1258_row10_col3" class="data row10 col3" >IF capital.gain <= 5422.202 AND education != Bachelors AND hours.per.week <= 58.6454 THEN class = <=50K</td>
      <td id="T_d1258_row10_col4" class="data row10 col4" >0.74373</td>
      <td id="T_d1258_row10_col5" class="data row10 col5" >0.81708</td>
      <td id="T_d1258_row10_col6" class="data row10 col6" >0.83405</td>
      <td id="T_d1258_row10_col7" class="data row10 col7" >3</td>
      <td id="T_d1258_row10_col8" class="data row10 col8" >0</td>
      <td id="T_d1258_row10_col9" class="data row10 col9" >22.65043</td>
      <td id="T_d1258_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_d1258_row11_col0" class="data row11 col0" >12758</td>
      <td id="T_d1258_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_d1258_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_d1258_row11_col3" class="data row11 col3" >IF capital.gain <= 4162.1904 AND hours.per.week <= 68.2131 AND hours.per.week > 18.3677 THEN class = <=50K</td>
      <td id="T_d1258_row11_col4" class="data row11 col4" >0.86978</td>
      <td id="T_d1258_row11_col5" class="data row11 col5" >0.90614</td>
      <td id="T_d1258_row11_col6" class="data row11 col6" >0.79091</td>
      <td id="T_d1258_row11_col7" class="data row11 col7" >3</td>
      <td id="T_d1258_row11_col8" class="data row11 col8" >0</td>
      <td id="T_d1258_row11_col9" class="data row11 col9" >22.66559</td>
      <td id="T_d1258_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_d1258_row12_col0" class="data row12 col0" >12758</td>
      <td id="T_d1258_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_d1258_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_d1258_row12_col3" class="data row12 col3" >IF capital.gain <= 9465.493 AND hours.per.week <= 56.6395 AND native.country != Canada THEN class = <=50K</td>
      <td id="T_d1258_row12_col4" class="data row12 col4" >0.89645</td>
      <td id="T_d1258_row12_col5" class="data row12 col5" >0.93284</td>
      <td id="T_d1258_row12_col6" class="data row12 col6" >0.78999</td>
      <td id="T_d1258_row12_col7" class="data row12 col7" >3</td>
      <td id="T_d1258_row12_col8" class="data row12 col8" >0</td>
      <td id="T_d1258_row12_col9" class="data row12 col9" >22.06458</td>
      <td id="T_d1258_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_d1258_row13_col0" class="data row13 col0" >12758</td>
      <td id="T_d1258_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_d1258_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_d1258_row13_col3" class="data row13 col3" >IF capital.gain <= 2083.3142 AND education.num != 5.0 THEN class = <=50K</td>
      <td id="T_d1258_row13_col4" class="data row13 col4" >0.90562</td>
      <td id="T_d1258_row13_col5" class="data row13 col5" >0.94544</td>
      <td id="T_d1258_row13_col6" class="data row13 col6" >0.79255</td>
      <td id="T_d1258_row13_col7" class="data row13 col7" >2</td>
      <td id="T_d1258_row13_col8" class="data row13 col8" >0</td>
      <td id="T_d1258_row13_col9" class="data row13 col9" >20.25562</td>
      <td id="T_d1258_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_d1258_row14_col0" class="data row14 col0" >12758</td>
      <td id="T_d1258_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_d1258_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_d1258_row14_col3" class="data row14 col3" >IF capital.gain <= 5221.0457 AND occupation != Craft-repair THEN class = <=50K</td>
      <td id="T_d1258_row14_col4" class="data row14 col4" >0.83161</td>
      <td id="T_d1258_row14_col5" class="data row14 col5" >0.86823</td>
      <td id="T_d1258_row14_col6" class="data row14 col6" >0.79260</td>
      <td id="T_d1258_row14_col7" class="data row14 col7" >2</td>
      <td id="T_d1258_row14_col8" class="data row14 col8" >0</td>
      <td id="T_d1258_row14_col9" class="data row14 col9" >25.06267</td>
      <td id="T_d1258_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_d1258_row15_col0" class="data row15 col0" >12758</td>
      <td id="T_d1258_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_d1258_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_d1258_row15_col3" class="data row15 col3" >IF age > 35.03 AND capital.gain <= 2796.6085 AND capital.loss <= 1776.9163 AND relationship = Not-in-family THEN class = <=50K</td>
      <td id="T_d1258_row15_col4" class="data row15 col4" >0.11329</td>
      <td id="T_d1258_row15_col5" class="data row15 col5" >0.13379</td>
      <td id="T_d1258_row15_col6" class="data row15 col6" >0.89659</td>
      <td id="T_d1258_row15_col7" class="data row15 col7" >4</td>
      <td id="T_d1258_row15_col8" class="data row15 col8" >0</td>
      <td id="T_d1258_row15_col9" class="data row15 col9" >7.79743</td>
      <td id="T_d1258_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_d1258_row16_col0" class="data row16 col0" >12758</td>
      <td id="T_d1258_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_d1258_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_d1258_row16_col3" class="data row16 col3" >IF marital.status = Widowed THEN class = <=50K</td>
      <td id="T_d1258_row16_col4" class="data row16 col4" >0.03054</td>
      <td id="T_d1258_row16_col5" class="data row16 col5" >0.03681</td>
      <td id="T_d1258_row16_col6" class="data row16 col6" >0.91523</td>
      <td id="T_d1258_row16_col7" class="data row16 col7" >1</td>
      <td id="T_d1258_row16_col8" class="data row16 col8" >0</td>
      <td id="T_d1258_row16_col9" class="data row16 col9" >7.98429</td>
      <td id="T_d1258_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_d1258_row17_col0" class="data row17 col0" >12758</td>
      <td id="T_d1258_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_d1258_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_d1258_row17_col3" class="data row17 col3" >IF capital.gain <= 4954.4995 AND marital.status = Widowed AND sex = Female THEN class = <=50K</td>
      <td id="T_d1258_row17_col4" class="data row17 col4" >0.02488</td>
      <td id="T_d1258_row17_col5" class="data row17 col5" >0.03138</td>
      <td id="T_d1258_row17_col6" class="data row17 col6" >0.95767</td>
      <td id="T_d1258_row17_col7" class="data row17 col7" >3</td>
      <td id="T_d1258_row17_col8" class="data row17 col8" >0</td>
      <td id="T_d1258_row17_col9" class="data row17 col9" >5.40857</td>
      <td id="T_d1258_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_d1258_row18_col0" class="data row18 col0" >12758</td>
      <td id="T_d1258_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_d1258_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_d1258_row18_col3" class="data row18 col3" >IF capital.gain <= 0.0 AND capital.loss <= 813.2081 AND hours.per.week > 45.0165 AND marital.status = Widowed THEN class = <=50K</td>
      <td id="T_d1258_row18_col4" class="data row18 col4" >0.00215</td>
      <td id="T_d1258_row18_col5" class="data row18 col5" >0.00231</td>
      <td id="T_d1258_row18_col6" class="data row18 col6" >0.81633</td>
      <td id="T_d1258_row18_col7" class="data row18 col7" >4</td>
      <td id="T_d1258_row18_col8" class="data row18 col8" >0</td>
      <td id="T_d1258_row18_col9" class="data row18 col9" >3.77216</td>
      <td id="T_d1258_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_d1258_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_d1258_row19_col0" class="data row19 col0" >12758</td>
      <td id="T_d1258_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_d1258_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_d1258_row19_col3" class="data row19 col3" >IF age > 48.7497 AND capital.gain <= 5086.639 THEN class = <=50K</td>
      <td id="T_d1258_row19_col4" class="data row19 col4" >0.21538</td>
      <td id="T_d1258_row19_col5" class="data row19 col5" >0.20326</td>
      <td id="T_d1258_row19_col6" class="data row19 col6" >0.71644</td>
      <td id="T_d1258_row19_col7" class="data row19 col7" >2</td>
      <td id="T_d1258_row19_col8" class="data row19 col8" >0</td>
      <td id="T_d1258_row19_col9" class="data row19 col9" >3.31217</td>
      <td id="T_d1258_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12758, Correct Prediction



<style type="text/css">
</style>
<table id="T_20cc7">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_20cc7_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_20cc7_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_20cc7_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_20cc7_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_20cc7_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_20cc7_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_20cc7_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_20cc7_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_20cc7_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_20cc7_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_20cc7_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_20cc7_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_20cc7_row0_col0" class="data row0 col0" >12758</td>
      <td id="T_20cc7_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_20cc7_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_20cc7_row0_col3" class="data row0 col3" >IF education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_20cc7_row0_col4" class="data row0 col4" >0.32095</td>
      <td id="T_20cc7_row0_col5" class="data row0 col5" >0.35508</td>
      <td id="T_20cc7_row0_col6" class="data row0 col6" >0.83992</td>
      <td id="T_20cc7_row0_col7" class="data row0 col7" >2</td>
      <td id="T_20cc7_row0_col8" class="data row0 col8" >0</td>
      <td id="T_20cc7_row0_col9" class="data row0 col9" >1.28740</td>
      <td id="T_20cc7_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_20cc7_row1_col0" class="data row1 col0" >12758</td>
      <td id="T_20cc7_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_20cc7_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_20cc7_row1_col3" class="data row1 col3" >IF capital.loss <= 0.0 AND education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_20cc7_row1_col4" class="data row1 col4" >0.30862</td>
      <td id="T_20cc7_row1_col5" class="data row1 col5" >0.34445</td>
      <td id="T_20cc7_row1_col6" class="data row1 col6" >0.84731</td>
      <td id="T_20cc7_row1_col7" class="data row1 col7" >3</td>
      <td id="T_20cc7_row1_col8" class="data row1 col8" >1</td>
      <td id="T_20cc7_row1_col9" class="data row1 col9" >2.44282</td>
      <td id="T_20cc7_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_20cc7_row2_col0" class="data row2 col0" >12758</td>
      <td id="T_20cc7_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_20cc7_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_20cc7_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_20cc7_row2_col4" class="data row2 col4" >0.66273</td>
      <td id="T_20cc7_row2_col5" class="data row2 col5" >0.72849</td>
      <td id="T_20cc7_row2_col6" class="data row2 col6" >0.83449</td>
      <td id="T_20cc7_row2_col7" class="data row2 col7" >3</td>
      <td id="T_20cc7_row2_col8" class="data row2 col8" >0</td>
      <td id="T_20cc7_row2_col9" class="data row2 col9" >1.24879</td>
      <td id="T_20cc7_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_20cc7_row3_col0" class="data row3 col0" >12758</td>
      <td id="T_20cc7_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_20cc7_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_20cc7_row3_col3" class="data row3 col3" >IF capital.gain <= 0.0 AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_20cc7_row3_col4" class="data row3 col4" >0.29936</td>
      <td id="T_20cc7_row3_col5" class="data row3 col5" >0.33959</td>
      <td id="T_20cc7_row3_col6" class="data row3 col6" >0.86120</td>
      <td id="T_20cc7_row3_col7" class="data row3 col7" >2</td>
      <td id="T_20cc7_row3_col8" class="data row3 col8" >0</td>
      <td id="T_20cc7_row3_col9" class="data row3 col9" >1.56445</td>
      <td id="T_20cc7_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_20cc7_row4_col0" class="data row4 col0" >12758</td>
      <td id="T_20cc7_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_20cc7_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_20cc7_row4_col3" class="data row4 col3" >IF education = HS-grad THEN class = <=50K</td>
      <td id="T_20cc7_row4_col4" class="data row4 col4" >0.32095</td>
      <td id="T_20cc7_row4_col5" class="data row4 col5" >0.35508</td>
      <td id="T_20cc7_row4_col6" class="data row4 col6" >0.83992</td>
      <td id="T_20cc7_row4_col7" class="data row4 col7" >1</td>
      <td id="T_20cc7_row4_col8" class="data row4 col8" >1</td>
      <td id="T_20cc7_row4_col9" class="data row4 col9" >2.61744</td>
      <td id="T_20cc7_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_20cc7_row5_col0" class="data row5 col0" >12758</td>
      <td id="T_20cc7_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_20cc7_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_20cc7_row5_col3" class="data row5 col3" >IF capital.gain <= 3160.889 THEN class = <=50K</td>
      <td id="T_20cc7_row5_col4" class="data row5 col4" >0.93458</td>
      <td id="T_20cc7_row5_col5" class="data row5 col5" >0.97931</td>
      <td id="T_20cc7_row5_col6" class="data row5 col6" >0.79550</td>
      <td id="T_20cc7_row5_col7" class="data row5 col7" >1</td>
      <td id="T_20cc7_row5_col8" class="data row5 col8" >0</td>
      <td id="T_20cc7_row5_col9" class="data row5 col9" >80.03021</td>
      <td id="T_20cc7_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_20cc7_row6_col0" class="data row6 col0" >12758</td>
      <td id="T_20cc7_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_20cc7_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_20cc7_row6_col3" class="data row6 col3" >IF capital.gain <= 3950.5089 THEN class = <=50K</td>
      <td id="T_20cc7_row6_col4" class="data row6 col4" >0.94094</td>
      <td id="T_20cc7_row6_col5" class="data row6 col5" >0.98769</td>
      <td id="T_20cc7_row6_col6" class="data row6 col6" >0.79689</td>
      <td id="T_20cc7_row6_col7" class="data row6 col7" >1</td>
      <td id="T_20cc7_row6_col8" class="data row6 col8" >0</td>
      <td id="T_20cc7_row6_col9" class="data row6 col9" >79.47443</td>
      <td id="T_20cc7_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_20cc7_row7_col0" class="data row7 col0" >12758</td>
      <td id="T_20cc7_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_20cc7_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_20cc7_row7_col3" class="data row7 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_20cc7_row7_col4" class="data row7 col4" >0.94226</td>
      <td id="T_20cc7_row7_col5" class="data row7 col5" >0.98942</td>
      <td id="T_20cc7_row7_col6" class="data row7 col6" >0.79717</td>
      <td id="T_20cc7_row7_col7" class="data row7 col7" >1</td>
      <td id="T_20cc7_row7_col8" class="data row7 col8" >0</td>
      <td id="T_20cc7_row7_col9" class="data row7 col9" >79.87095</td>
      <td id="T_20cc7_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_20cc7_row8_col0" class="data row8 col0" >12758</td>
      <td id="T_20cc7_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_20cc7_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_20cc7_row8_col3" class="data row8 col3" >IF capital.gain <= 6025.6313 THEN class = <=50K</td>
      <td id="T_20cc7_row8_col4" class="data row8 col4" >0.95450</td>
      <td id="T_20cc7_row8_col5" class="data row8 col5" >0.99717</td>
      <td id="T_20cc7_row8_col6" class="data row8 col6" >0.79311</td>
      <td id="T_20cc7_row8_col7" class="data row8 col7" >1</td>
      <td id="T_20cc7_row8_col8" class="data row8 col8" >0</td>
      <td id="T_20cc7_row8_col9" class="data row8 col9" >80.20908</td>
      <td id="T_20cc7_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_20cc7_row9_col0" class="data row9 col0" >12758</td>
      <td id="T_20cc7_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_20cc7_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_20cc7_row9_col3" class="data row9 col3" >IF capital.gain <= 4787.0 THEN class = <=50K</td>
      <td id="T_20cc7_row9_col4" class="data row9 col4" >0.94805</td>
      <td id="T_20cc7_row9_col5" class="data row9 col5" >0.99324</td>
      <td id="T_20cc7_row9_col6" class="data row9 col6" >0.79535</td>
      <td id="T_20cc7_row9_col7" class="data row9 col7" >1</td>
      <td id="T_20cc7_row9_col8" class="data row9 col8" >0</td>
      <td id="T_20cc7_row9_col9" class="data row9 col9" >117.75989</td>
      <td id="T_20cc7_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_20cc7_row10_col0" class="data row10 col0" >12758</td>
      <td id="T_20cc7_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_20cc7_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_20cc7_row10_col3" class="data row10 col3" >IF capital.gain <= 5422.202 AND education != Bachelors AND hours.per.week <= 58.6454 THEN class = <=50K</td>
      <td id="T_20cc7_row10_col4" class="data row10 col4" >0.74373</td>
      <td id="T_20cc7_row10_col5" class="data row10 col5" >0.81708</td>
      <td id="T_20cc7_row10_col6" class="data row10 col6" >0.83405</td>
      <td id="T_20cc7_row10_col7" class="data row10 col7" >3</td>
      <td id="T_20cc7_row10_col8" class="data row10 col8" >0</td>
      <td id="T_20cc7_row10_col9" class="data row10 col9" >22.65043</td>
      <td id="T_20cc7_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_20cc7_row11_col0" class="data row11 col0" >12758</td>
      <td id="T_20cc7_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_20cc7_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_20cc7_row11_col3" class="data row11 col3" >IF capital.gain <= 4162.1904 AND hours.per.week <= 68.2131 AND hours.per.week > 18.3677 THEN class = <=50K</td>
      <td id="T_20cc7_row11_col4" class="data row11 col4" >0.86978</td>
      <td id="T_20cc7_row11_col5" class="data row11 col5" >0.90614</td>
      <td id="T_20cc7_row11_col6" class="data row11 col6" >0.79091</td>
      <td id="T_20cc7_row11_col7" class="data row11 col7" >3</td>
      <td id="T_20cc7_row11_col8" class="data row11 col8" >0</td>
      <td id="T_20cc7_row11_col9" class="data row11 col9" >22.66559</td>
      <td id="T_20cc7_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_20cc7_row12_col0" class="data row12 col0" >12758</td>
      <td id="T_20cc7_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_20cc7_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_20cc7_row12_col3" class="data row12 col3" >IF capital.gain <= 9465.493 AND hours.per.week <= 56.6395 AND native.country != Canada THEN class = <=50K</td>
      <td id="T_20cc7_row12_col4" class="data row12 col4" >0.89645</td>
      <td id="T_20cc7_row12_col5" class="data row12 col5" >0.93284</td>
      <td id="T_20cc7_row12_col6" class="data row12 col6" >0.78999</td>
      <td id="T_20cc7_row12_col7" class="data row12 col7" >3</td>
      <td id="T_20cc7_row12_col8" class="data row12 col8" >0</td>
      <td id="T_20cc7_row12_col9" class="data row12 col9" >22.06458</td>
      <td id="T_20cc7_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_20cc7_row13_col0" class="data row13 col0" >12758</td>
      <td id="T_20cc7_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_20cc7_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_20cc7_row13_col3" class="data row13 col3" >IF capital.gain <= 2083.3142 AND education.num != 5.0 THEN class = <=50K</td>
      <td id="T_20cc7_row13_col4" class="data row13 col4" >0.90562</td>
      <td id="T_20cc7_row13_col5" class="data row13 col5" >0.94544</td>
      <td id="T_20cc7_row13_col6" class="data row13 col6" >0.79255</td>
      <td id="T_20cc7_row13_col7" class="data row13 col7" >2</td>
      <td id="T_20cc7_row13_col8" class="data row13 col8" >0</td>
      <td id="T_20cc7_row13_col9" class="data row13 col9" >20.25562</td>
      <td id="T_20cc7_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_20cc7_row14_col0" class="data row14 col0" >12758</td>
      <td id="T_20cc7_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_20cc7_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_20cc7_row14_col3" class="data row14 col3" >IF capital.gain <= 5221.0457 AND occupation != Craft-repair THEN class = <=50K</td>
      <td id="T_20cc7_row14_col4" class="data row14 col4" >0.83161</td>
      <td id="T_20cc7_row14_col5" class="data row14 col5" >0.86823</td>
      <td id="T_20cc7_row14_col6" class="data row14 col6" >0.79260</td>
      <td id="T_20cc7_row14_col7" class="data row14 col7" >2</td>
      <td id="T_20cc7_row14_col8" class="data row14 col8" >0</td>
      <td id="T_20cc7_row14_col9" class="data row14 col9" >25.06267</td>
      <td id="T_20cc7_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_20cc7_row15_col0" class="data row15 col0" >12758</td>
      <td id="T_20cc7_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_20cc7_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_20cc7_row15_col3" class="data row15 col3" >IF age > 35.03 AND capital.gain <= 2796.6085 AND capital.loss <= 1776.9163 AND relationship = Not-in-family THEN class = <=50K</td>
      <td id="T_20cc7_row15_col4" class="data row15 col4" >0.11329</td>
      <td id="T_20cc7_row15_col5" class="data row15 col5" >0.13379</td>
      <td id="T_20cc7_row15_col6" class="data row15 col6" >0.89659</td>
      <td id="T_20cc7_row15_col7" class="data row15 col7" >4</td>
      <td id="T_20cc7_row15_col8" class="data row15 col8" >0</td>
      <td id="T_20cc7_row15_col9" class="data row15 col9" >7.79743</td>
      <td id="T_20cc7_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_20cc7_row16_col0" class="data row16 col0" >12758</td>
      <td id="T_20cc7_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_20cc7_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_20cc7_row16_col3" class="data row16 col3" >IF marital.status = Widowed THEN class = <=50K</td>
      <td id="T_20cc7_row16_col4" class="data row16 col4" >0.03054</td>
      <td id="T_20cc7_row16_col5" class="data row16 col5" >0.03681</td>
      <td id="T_20cc7_row16_col6" class="data row16 col6" >0.91523</td>
      <td id="T_20cc7_row16_col7" class="data row16 col7" >1</td>
      <td id="T_20cc7_row16_col8" class="data row16 col8" >0</td>
      <td id="T_20cc7_row16_col9" class="data row16 col9" >7.98429</td>
      <td id="T_20cc7_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_20cc7_row17_col0" class="data row17 col0" >12758</td>
      <td id="T_20cc7_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_20cc7_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_20cc7_row17_col3" class="data row17 col3" >IF capital.gain <= 4954.4995 AND marital.status = Widowed AND sex = Female THEN class = <=50K</td>
      <td id="T_20cc7_row17_col4" class="data row17 col4" >0.02488</td>
      <td id="T_20cc7_row17_col5" class="data row17 col5" >0.03138</td>
      <td id="T_20cc7_row17_col6" class="data row17 col6" >0.95767</td>
      <td id="T_20cc7_row17_col7" class="data row17 col7" >3</td>
      <td id="T_20cc7_row17_col8" class="data row17 col8" >0</td>
      <td id="T_20cc7_row17_col9" class="data row17 col9" >5.40857</td>
      <td id="T_20cc7_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_20cc7_row18_col0" class="data row18 col0" >12758</td>
      <td id="T_20cc7_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_20cc7_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_20cc7_row18_col3" class="data row18 col3" >IF capital.gain <= 0.0 AND capital.loss <= 813.2081 AND hours.per.week > 45.0165 AND marital.status = Widowed THEN class = <=50K</td>
      <td id="T_20cc7_row18_col4" class="data row18 col4" >0.00215</td>
      <td id="T_20cc7_row18_col5" class="data row18 col5" >0.00231</td>
      <td id="T_20cc7_row18_col6" class="data row18 col6" >0.81633</td>
      <td id="T_20cc7_row18_col7" class="data row18 col7" >4</td>
      <td id="T_20cc7_row18_col8" class="data row18 col8" >0</td>
      <td id="T_20cc7_row18_col9" class="data row18 col9" >3.77216</td>
      <td id="T_20cc7_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_20cc7_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_20cc7_row19_col0" class="data row19 col0" >12758</td>
      <td id="T_20cc7_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_20cc7_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_20cc7_row19_col3" class="data row19 col3" >IF age > 48.7497 AND capital.gain <= 5086.639 THEN class = <=50K</td>
      <td id="T_20cc7_row19_col4" class="data row19 col4" >0.21538</td>
      <td id="T_20cc7_row19_col5" class="data row19 col5" >0.20326</td>
      <td id="T_20cc7_row19_col6" class="data row19 col6" >0.71644</td>
      <td id="T_20cc7_row19_col7" class="data row19 col7" >2</td>
      <td id="T_20cc7_row19_col8" class="data row19 col8" >0</td>
      <td id="T_20cc7_row19_col9" class="data row19 col9" >3.31217</td>
      <td id="T_20cc7_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12758, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_7d461">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7d461_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_7d461_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_7d461_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_7d461_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_7d461_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_7d461_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_7d461_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_7d461_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_7d461_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_7d461_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_7d461_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7d461_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_7d461_row0_col0" class="data row0 col0" >12758</td>
      <td id="T_7d461_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_7d461_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_7d461_row0_col3" class="data row0 col3" >IF education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_7d461_row0_col4" class="data row0 col4" >0.32095</td>
      <td id="T_7d461_row0_col5" class="data row0 col5" >0.35508</td>
      <td id="T_7d461_row0_col6" class="data row0 col6" >0.83992</td>
      <td id="T_7d461_row0_col7" class="data row0 col7" >2</td>
      <td id="T_7d461_row0_col8" class="data row0 col8" >0</td>
      <td id="T_7d461_row0_col9" class="data row0 col9" >1.28740</td>
      <td id="T_7d461_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_7d461_row1_col0" class="data row1 col0" >12758</td>
      <td id="T_7d461_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_7d461_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_7d461_row1_col3" class="data row1 col3" >IF capital.loss <= 0.0 AND education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_7d461_row1_col4" class="data row1 col4" >0.30862</td>
      <td id="T_7d461_row1_col5" class="data row1 col5" >0.34445</td>
      <td id="T_7d461_row1_col6" class="data row1 col6" >0.84731</td>
      <td id="T_7d461_row1_col7" class="data row1 col7" >3</td>
      <td id="T_7d461_row1_col8" class="data row1 col8" >1</td>
      <td id="T_7d461_row1_col9" class="data row1 col9" >2.44282</td>
      <td id="T_7d461_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_7d461_row2_col0" class="data row2 col0" >12758</td>
      <td id="T_7d461_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_7d461_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_7d461_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_7d461_row2_col4" class="data row2 col4" >0.66273</td>
      <td id="T_7d461_row2_col5" class="data row2 col5" >0.72849</td>
      <td id="T_7d461_row2_col6" class="data row2 col6" >0.83449</td>
      <td id="T_7d461_row2_col7" class="data row2 col7" >3</td>
      <td id="T_7d461_row2_col8" class="data row2 col8" >0</td>
      <td id="T_7d461_row2_col9" class="data row2 col9" >1.24879</td>
      <td id="T_7d461_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_7d461_row3_col0" class="data row3 col0" >12758</td>
      <td id="T_7d461_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_7d461_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_7d461_row3_col3" class="data row3 col3" >IF capital.gain <= 0.0 AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_7d461_row3_col4" class="data row3 col4" >0.29936</td>
      <td id="T_7d461_row3_col5" class="data row3 col5" >0.33959</td>
      <td id="T_7d461_row3_col6" class="data row3 col6" >0.86120</td>
      <td id="T_7d461_row3_col7" class="data row3 col7" >2</td>
      <td id="T_7d461_row3_col8" class="data row3 col8" >0</td>
      <td id="T_7d461_row3_col9" class="data row3 col9" >1.56445</td>
      <td id="T_7d461_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_7d461_row4_col0" class="data row4 col0" >12758</td>
      <td id="T_7d461_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_7d461_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_7d461_row4_col3" class="data row4 col3" >IF education = HS-grad THEN class = <=50K</td>
      <td id="T_7d461_row4_col4" class="data row4 col4" >0.32095</td>
      <td id="T_7d461_row4_col5" class="data row4 col5" >0.35508</td>
      <td id="T_7d461_row4_col6" class="data row4 col6" >0.83992</td>
      <td id="T_7d461_row4_col7" class="data row4 col7" >1</td>
      <td id="T_7d461_row4_col8" class="data row4 col8" >1</td>
      <td id="T_7d461_row4_col9" class="data row4 col9" >2.61744</td>
      <td id="T_7d461_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_7d461_row5_col0" class="data row5 col0" >12758</td>
      <td id="T_7d461_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_7d461_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_7d461_row5_col3" class="data row5 col3" >IF capital.gain <= 3160.889 THEN class = <=50K</td>
      <td id="T_7d461_row5_col4" class="data row5 col4" >0.93458</td>
      <td id="T_7d461_row5_col5" class="data row5 col5" >0.97931</td>
      <td id="T_7d461_row5_col6" class="data row5 col6" >0.79550</td>
      <td id="T_7d461_row5_col7" class="data row5 col7" >1</td>
      <td id="T_7d461_row5_col8" class="data row5 col8" >0</td>
      <td id="T_7d461_row5_col9" class="data row5 col9" >80.03021</td>
      <td id="T_7d461_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_7d461_row6_col0" class="data row6 col0" >12758</td>
      <td id="T_7d461_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_7d461_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_7d461_row6_col3" class="data row6 col3" >IF capital.gain <= 3950.5089 THEN class = <=50K</td>
      <td id="T_7d461_row6_col4" class="data row6 col4" >0.94094</td>
      <td id="T_7d461_row6_col5" class="data row6 col5" >0.98769</td>
      <td id="T_7d461_row6_col6" class="data row6 col6" >0.79689</td>
      <td id="T_7d461_row6_col7" class="data row6 col7" >1</td>
      <td id="T_7d461_row6_col8" class="data row6 col8" >0</td>
      <td id="T_7d461_row6_col9" class="data row6 col9" >79.47443</td>
      <td id="T_7d461_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_7d461_row7_col0" class="data row7 col0" >12758</td>
      <td id="T_7d461_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_7d461_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_7d461_row7_col3" class="data row7 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_7d461_row7_col4" class="data row7 col4" >0.94226</td>
      <td id="T_7d461_row7_col5" class="data row7 col5" >0.98942</td>
      <td id="T_7d461_row7_col6" class="data row7 col6" >0.79717</td>
      <td id="T_7d461_row7_col7" class="data row7 col7" >1</td>
      <td id="T_7d461_row7_col8" class="data row7 col8" >0</td>
      <td id="T_7d461_row7_col9" class="data row7 col9" >79.87095</td>
      <td id="T_7d461_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_7d461_row8_col0" class="data row8 col0" >12758</td>
      <td id="T_7d461_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_7d461_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_7d461_row8_col3" class="data row8 col3" >IF capital.gain <= 6025.6313 THEN class = <=50K</td>
      <td id="T_7d461_row8_col4" class="data row8 col4" >0.95450</td>
      <td id="T_7d461_row8_col5" class="data row8 col5" >0.99717</td>
      <td id="T_7d461_row8_col6" class="data row8 col6" >0.79311</td>
      <td id="T_7d461_row8_col7" class="data row8 col7" >1</td>
      <td id="T_7d461_row8_col8" class="data row8 col8" >0</td>
      <td id="T_7d461_row8_col9" class="data row8 col9" >80.20908</td>
      <td id="T_7d461_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_7d461_row9_col0" class="data row9 col0" >12758</td>
      <td id="T_7d461_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_7d461_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_7d461_row9_col3" class="data row9 col3" >IF capital.gain <= 4787.0 THEN class = <=50K</td>
      <td id="T_7d461_row9_col4" class="data row9 col4" >0.94805</td>
      <td id="T_7d461_row9_col5" class="data row9 col5" >0.99324</td>
      <td id="T_7d461_row9_col6" class="data row9 col6" >0.79535</td>
      <td id="T_7d461_row9_col7" class="data row9 col7" >1</td>
      <td id="T_7d461_row9_col8" class="data row9 col8" >0</td>
      <td id="T_7d461_row9_col9" class="data row9 col9" >117.75989</td>
      <td id="T_7d461_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_7d461_row10_col0" class="data row10 col0" >12758</td>
      <td id="T_7d461_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_7d461_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_7d461_row10_col3" class="data row10 col3" >IF capital.gain <= 5422.202 AND education != Bachelors AND hours.per.week <= 58.6454 THEN class = <=50K</td>
      <td id="T_7d461_row10_col4" class="data row10 col4" >0.74373</td>
      <td id="T_7d461_row10_col5" class="data row10 col5" >0.81708</td>
      <td id="T_7d461_row10_col6" class="data row10 col6" >0.83405</td>
      <td id="T_7d461_row10_col7" class="data row10 col7" >3</td>
      <td id="T_7d461_row10_col8" class="data row10 col8" >0</td>
      <td id="T_7d461_row10_col9" class="data row10 col9" >22.65043</td>
      <td id="T_7d461_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_7d461_row11_col0" class="data row11 col0" >12758</td>
      <td id="T_7d461_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_7d461_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_7d461_row11_col3" class="data row11 col3" >IF capital.gain <= 4162.1904 AND hours.per.week <= 68.2131 AND hours.per.week > 18.3677 THEN class = <=50K</td>
      <td id="T_7d461_row11_col4" class="data row11 col4" >0.86978</td>
      <td id="T_7d461_row11_col5" class="data row11 col5" >0.90614</td>
      <td id="T_7d461_row11_col6" class="data row11 col6" >0.79091</td>
      <td id="T_7d461_row11_col7" class="data row11 col7" >3</td>
      <td id="T_7d461_row11_col8" class="data row11 col8" >0</td>
      <td id="T_7d461_row11_col9" class="data row11 col9" >22.66559</td>
      <td id="T_7d461_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_7d461_row12_col0" class="data row12 col0" >12758</td>
      <td id="T_7d461_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_7d461_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_7d461_row12_col3" class="data row12 col3" >IF capital.gain <= 9465.493 AND hours.per.week <= 56.6395 AND native.country != Canada THEN class = <=50K</td>
      <td id="T_7d461_row12_col4" class="data row12 col4" >0.89645</td>
      <td id="T_7d461_row12_col5" class="data row12 col5" >0.93284</td>
      <td id="T_7d461_row12_col6" class="data row12 col6" >0.78999</td>
      <td id="T_7d461_row12_col7" class="data row12 col7" >3</td>
      <td id="T_7d461_row12_col8" class="data row12 col8" >0</td>
      <td id="T_7d461_row12_col9" class="data row12 col9" >22.06458</td>
      <td id="T_7d461_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_7d461_row13_col0" class="data row13 col0" >12758</td>
      <td id="T_7d461_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_7d461_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_7d461_row13_col3" class="data row13 col3" >IF capital.gain <= 2083.3142 AND education.num != 5.0 THEN class = <=50K</td>
      <td id="T_7d461_row13_col4" class="data row13 col4" >0.90562</td>
      <td id="T_7d461_row13_col5" class="data row13 col5" >0.94544</td>
      <td id="T_7d461_row13_col6" class="data row13 col6" >0.79255</td>
      <td id="T_7d461_row13_col7" class="data row13 col7" >2</td>
      <td id="T_7d461_row13_col8" class="data row13 col8" >0</td>
      <td id="T_7d461_row13_col9" class="data row13 col9" >20.25562</td>
      <td id="T_7d461_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_7d461_row14_col0" class="data row14 col0" >12758</td>
      <td id="T_7d461_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_7d461_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_7d461_row14_col3" class="data row14 col3" >IF capital.gain <= 5221.0457 AND occupation != Craft-repair THEN class = <=50K</td>
      <td id="T_7d461_row14_col4" class="data row14 col4" >0.83161</td>
      <td id="T_7d461_row14_col5" class="data row14 col5" >0.86823</td>
      <td id="T_7d461_row14_col6" class="data row14 col6" >0.79260</td>
      <td id="T_7d461_row14_col7" class="data row14 col7" >2</td>
      <td id="T_7d461_row14_col8" class="data row14 col8" >0</td>
      <td id="T_7d461_row14_col9" class="data row14 col9" >25.06267</td>
      <td id="T_7d461_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_7d461_row15_col0" class="data row15 col0" >12758</td>
      <td id="T_7d461_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_7d461_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_7d461_row15_col3" class="data row15 col3" >IF age > 35.03 AND capital.gain <= 2796.6085 AND capital.loss <= 1776.9163 AND relationship = Not-in-family THEN class = <=50K</td>
      <td id="T_7d461_row15_col4" class="data row15 col4" >0.11329</td>
      <td id="T_7d461_row15_col5" class="data row15 col5" >0.13379</td>
      <td id="T_7d461_row15_col6" class="data row15 col6" >0.89659</td>
      <td id="T_7d461_row15_col7" class="data row15 col7" >4</td>
      <td id="T_7d461_row15_col8" class="data row15 col8" >0</td>
      <td id="T_7d461_row15_col9" class="data row15 col9" >7.79743</td>
      <td id="T_7d461_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_7d461_row16_col0" class="data row16 col0" >12758</td>
      <td id="T_7d461_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_7d461_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_7d461_row16_col3" class="data row16 col3" >IF marital.status = Widowed THEN class = <=50K</td>
      <td id="T_7d461_row16_col4" class="data row16 col4" >0.03054</td>
      <td id="T_7d461_row16_col5" class="data row16 col5" >0.03681</td>
      <td id="T_7d461_row16_col6" class="data row16 col6" >0.91523</td>
      <td id="T_7d461_row16_col7" class="data row16 col7" >1</td>
      <td id="T_7d461_row16_col8" class="data row16 col8" >0</td>
      <td id="T_7d461_row16_col9" class="data row16 col9" >7.98429</td>
      <td id="T_7d461_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_7d461_row17_col0" class="data row17 col0" >12758</td>
      <td id="T_7d461_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_7d461_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_7d461_row17_col3" class="data row17 col3" >IF capital.gain <= 4954.4995 AND marital.status = Widowed AND sex = Female THEN class = <=50K</td>
      <td id="T_7d461_row17_col4" class="data row17 col4" >0.02488</td>
      <td id="T_7d461_row17_col5" class="data row17 col5" >0.03138</td>
      <td id="T_7d461_row17_col6" class="data row17 col6" >0.95767</td>
      <td id="T_7d461_row17_col7" class="data row17 col7" >3</td>
      <td id="T_7d461_row17_col8" class="data row17 col8" >0</td>
      <td id="T_7d461_row17_col9" class="data row17 col9" >5.40857</td>
      <td id="T_7d461_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_7d461_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_7d461_row18_col0" class="data row18 col0" >12758</td>
      <td id="T_7d461_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_7d461_row18_col2" class="data row18 col2" >EXPLAN5</td>
      <td id="T_7d461_row18_col3" class="data row18 col3" >IF age > 48.7497 AND capital.gain <= 5086.639 THEN class = <=50K</td>
      <td id="T_7d461_row18_col4" class="data row18 col4" >0.21538</td>
      <td id="T_7d461_row18_col5" class="data row18 col5" >0.20326</td>
      <td id="T_7d461_row18_col6" class="data row18 col6" >0.71644</td>
      <td id="T_7d461_row18_col7" class="data row18 col7" >2</td>
      <td id="T_7d461_row18_col8" class="data row18 col8" >0</td>
      <td id="T_7d461_row18_col9" class="data row18 col9" >3.31217</td>
      <td id="T_7d461_row18_col10" class="data row18 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12758, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.9545, Pre: 0.95767)



<style type="text/css">
#T_bea6e_row5_col0, #T_bea6e_row5_col1, #T_bea6e_row5_col2, #T_bea6e_row5_col3, #T_bea6e_row5_col4, #T_bea6e_row5_col5, #T_bea6e_row5_col6, #T_bea6e_row5_col7, #T_bea6e_row5_col8, #T_bea6e_row5_col9, #T_bea6e_row5_col10, #T_bea6e_row5_col11 {
  font-weight: bold;
}
</style>
<table id="T_bea6e">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_bea6e_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_bea6e_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_bea6e_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_bea6e_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_bea6e_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_bea6e_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_bea6e_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_bea6e_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_bea6e_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_bea6e_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_bea6e_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_bea6e_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_bea6e_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_bea6e_row0_col0" class="data row0 col0" >12758</td>
      <td id="T_bea6e_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_bea6e_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_bea6e_row0_col3" class="data row0 col3" >IF education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_bea6e_row0_col4" class="data row0 col4" >0.32095</td>
      <td id="T_bea6e_row0_col5" class="data row0 col5" >0.35508</td>
      <td id="T_bea6e_row0_col6" class="data row0 col6" >0.83992</td>
      <td id="T_bea6e_row0_col7" class="data row0 col7" >2</td>
      <td id="T_bea6e_row0_col8" class="data row0 col8" >0</td>
      <td id="T_bea6e_row0_col9" class="data row0 col9" >1.28740</td>
      <td id="T_bea6e_row0_col10" class="data row0 col10" >False</td>
      <td id="T_bea6e_row0_col11" class="data row0 col11" >0.64440</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_bea6e_row1_col0" class="data row1 col0" >12758</td>
      <td id="T_bea6e_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_bea6e_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_bea6e_row1_col3" class="data row1 col3" >IF capital.loss <= 0.0 AND education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_bea6e_row1_col4" class="data row1 col4" >0.30862</td>
      <td id="T_bea6e_row1_col5" class="data row1 col5" >0.34445</td>
      <td id="T_bea6e_row1_col6" class="data row1 col6" >0.84731</td>
      <td id="T_bea6e_row1_col7" class="data row1 col7" >3</td>
      <td id="T_bea6e_row1_col8" class="data row1 col8" >1</td>
      <td id="T_bea6e_row1_col9" class="data row1 col9" >2.44282</td>
      <td id="T_bea6e_row1_col10" class="data row1 col10" >False</td>
      <td id="T_bea6e_row1_col11" class="data row1 col11" >0.65524</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_bea6e_row2_col0" class="data row2 col0" >12758</td>
      <td id="T_bea6e_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_bea6e_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_bea6e_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_bea6e_row2_col4" class="data row2 col4" >0.66273</td>
      <td id="T_bea6e_row2_col5" class="data row2 col5" >0.72849</td>
      <td id="T_bea6e_row2_col6" class="data row2 col6" >0.83449</td>
      <td id="T_bea6e_row2_col7" class="data row2 col7" >3</td>
      <td id="T_bea6e_row2_col8" class="data row2 col8" >0</td>
      <td id="T_bea6e_row2_col9" class="data row2 col9" >1.24879</td>
      <td id="T_bea6e_row2_col10" class="data row2 col10" >False</td>
      <td id="T_bea6e_row2_col11" class="data row2 col11" >0.31671</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_bea6e_row3_col0" class="data row3 col0" >12758</td>
      <td id="T_bea6e_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_bea6e_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_bea6e_row3_col3" class="data row3 col3" >IF capital.gain <= 0.0 AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_bea6e_row3_col4" class="data row3 col4" >0.29936</td>
      <td id="T_bea6e_row3_col5" class="data row3 col5" >0.33959</td>
      <td id="T_bea6e_row3_col6" class="data row3 col6" >0.86120</td>
      <td id="T_bea6e_row3_col7" class="data row3 col7" >2</td>
      <td id="T_bea6e_row3_col8" class="data row3 col8" >0</td>
      <td id="T_bea6e_row3_col9" class="data row3 col9" >1.56445</td>
      <td id="T_bea6e_row3_col10" class="data row3 col10" >False</td>
      <td id="T_bea6e_row3_col11" class="data row3 col11" >0.66220</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_bea6e_row4_col0" class="data row4 col0" >12758</td>
      <td id="T_bea6e_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_bea6e_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_bea6e_row4_col3" class="data row4 col3" >IF education = HS-grad THEN class = <=50K</td>
      <td id="T_bea6e_row4_col4" class="data row4 col4" >0.32095</td>
      <td id="T_bea6e_row4_col5" class="data row4 col5" >0.35508</td>
      <td id="T_bea6e_row4_col6" class="data row4 col6" >0.83992</td>
      <td id="T_bea6e_row4_col7" class="data row4 col7" >1</td>
      <td id="T_bea6e_row4_col8" class="data row4 col8" >1</td>
      <td id="T_bea6e_row4_col9" class="data row4 col9" >2.61744</td>
      <td id="T_bea6e_row4_col10" class="data row4 col10" >False</td>
      <td id="T_bea6e_row4_col11" class="data row4 col11" >0.64440</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_bea6e_row5_col0" class="data row5 col0" >12758</td>
      <td id="T_bea6e_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_bea6e_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_bea6e_row5_col3" class="data row5 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_bea6e_row5_col4" class="data row5 col4" >0.94226</td>
      <td id="T_bea6e_row5_col5" class="data row5 col5" >0.98942</td>
      <td id="T_bea6e_row5_col6" class="data row5 col6" >0.79717</td>
      <td id="T_bea6e_row5_col7" class="data row5 col7" >1</td>
      <td id="T_bea6e_row5_col8" class="data row5 col8" >0</td>
      <td id="T_bea6e_row5_col9" class="data row5 col9" >79.87095</td>
      <td id="T_bea6e_row5_col10" class="data row5 col10" >False</td>
      <td id="T_bea6e_row5_col11" class="data row5 col11" >0.16097</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_bea6e_row6_col0" class="data row6 col0" >12758</td>
      <td id="T_bea6e_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_bea6e_row6_col2" class="data row6 col2" >LORE4</td>
      <td id="T_bea6e_row6_col3" class="data row6 col3" >IF capital.gain <= 6025.6313 THEN class = <=50K</td>
      <td id="T_bea6e_row6_col4" class="data row6 col4" >0.95450</td>
      <td id="T_bea6e_row6_col5" class="data row6 col5" >0.99717</td>
      <td id="T_bea6e_row6_col6" class="data row6 col6" >0.79311</td>
      <td id="T_bea6e_row6_col7" class="data row6 col7" >1</td>
      <td id="T_bea6e_row6_col8" class="data row6 col8" >0</td>
      <td id="T_bea6e_row6_col9" class="data row6 col9" >80.20908</td>
      <td id="T_bea6e_row6_col10" class="data row6 col10" >False</td>
      <td id="T_bea6e_row6_col11" class="data row6 col11" >0.16456</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_bea6e_row7_col0" class="data row7 col0" >12758</td>
      <td id="T_bea6e_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_bea6e_row7_col2" class="data row7 col2" >LORE5</td>
      <td id="T_bea6e_row7_col3" class="data row7 col3" >IF capital.gain <= 4787.0 THEN class = <=50K</td>
      <td id="T_bea6e_row7_col4" class="data row7 col4" >0.94805</td>
      <td id="T_bea6e_row7_col5" class="data row7 col5" >0.99324</td>
      <td id="T_bea6e_row7_col6" class="data row7 col6" >0.79535</td>
      <td id="T_bea6e_row7_col7" class="data row7 col7" >1</td>
      <td id="T_bea6e_row7_col8" class="data row7 col8" >0</td>
      <td id="T_bea6e_row7_col9" class="data row7 col9" >117.75989</td>
      <td id="T_bea6e_row7_col10" class="data row7 col10" >False</td>
      <td id="T_bea6e_row7_col11" class="data row7 col11" >0.16245</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_bea6e_row8_col0" class="data row8 col0" >12758</td>
      <td id="T_bea6e_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_bea6e_row8_col2" class="data row8 col2" >LORE_SA1</td>
      <td id="T_bea6e_row8_col3" class="data row8 col3" >IF capital.gain <= 5422.202 AND education != Bachelors AND hours.per.week <= 58.6454 THEN class = <=50K</td>
      <td id="T_bea6e_row8_col4" class="data row8 col4" >0.74373</td>
      <td id="T_bea6e_row8_col5" class="data row8 col5" >0.81708</td>
      <td id="T_bea6e_row8_col6" class="data row8 col6" >0.83405</td>
      <td id="T_bea6e_row8_col7" class="data row8 col7" >3</td>
      <td id="T_bea6e_row8_col8" class="data row8 col8" >0</td>
      <td id="T_bea6e_row8_col9" class="data row8 col9" >22.65043</td>
      <td id="T_bea6e_row8_col10" class="data row8 col10" >False</td>
      <td id="T_bea6e_row8_col11" class="data row8 col11" >0.24435</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_bea6e_row9_col0" class="data row9 col0" >12758</td>
      <td id="T_bea6e_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_bea6e_row9_col2" class="data row9 col2" >EXPLAN1</td>
      <td id="T_bea6e_row9_col3" class="data row9 col3" >IF age > 35.03 AND capital.gain <= 2796.6085 AND capital.loss <= 1776.9163 AND relationship = Not-in-family THEN class = <=50K</td>
      <td id="T_bea6e_row9_col4" class="data row9 col4" >0.11329</td>
      <td id="T_bea6e_row9_col5" class="data row9 col5" >0.13379</td>
      <td id="T_bea6e_row9_col6" class="data row9 col6" >0.89659</td>
      <td id="T_bea6e_row9_col7" class="data row9 col7" >4</td>
      <td id="T_bea6e_row9_col8" class="data row9 col8" >0</td>
      <td id="T_bea6e_row9_col9" class="data row9 col9" >7.79743</td>
      <td id="T_bea6e_row9_col10" class="data row9 col10" >False</td>
      <td id="T_bea6e_row9_col11" class="data row9 col11" >0.84342</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_bea6e_row10_col0" class="data row10 col0" >12758</td>
      <td id="T_bea6e_row10_col1" class="data row10 col1" >EXPLAN</td>
      <td id="T_bea6e_row10_col2" class="data row10 col2" >EXPLAN2</td>
      <td id="T_bea6e_row10_col3" class="data row10 col3" >IF marital.status = Widowed THEN class = <=50K</td>
      <td id="T_bea6e_row10_col4" class="data row10 col4" >0.03054</td>
      <td id="T_bea6e_row10_col5" class="data row10 col5" >0.03681</td>
      <td id="T_bea6e_row10_col6" class="data row10 col6" >0.91523</td>
      <td id="T_bea6e_row10_col7" class="data row10 col7" >1</td>
      <td id="T_bea6e_row10_col8" class="data row10 col8" >0</td>
      <td id="T_bea6e_row10_col9" class="data row10 col9" >7.98429</td>
      <td id="T_bea6e_row10_col10" class="data row10 col10" >False</td>
      <td id="T_bea6e_row10_col11" class="data row10 col11" >0.92493</td>
    </tr>
    <tr>
      <th id="T_bea6e_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_bea6e_row11_col0" class="data row11 col0" >12758</td>
      <td id="T_bea6e_row11_col1" class="data row11 col1" >EXPLAN</td>
      <td id="T_bea6e_row11_col2" class="data row11 col2" >EXPLAN3</td>
      <td id="T_bea6e_row11_col3" class="data row11 col3" >IF capital.gain <= 4954.4995 AND marital.status = Widowed AND sex = Female THEN class = <=50K</td>
      <td id="T_bea6e_row11_col4" class="data row11 col4" >0.02488</td>
      <td id="T_bea6e_row11_col5" class="data row11 col5" >0.03138</td>
      <td id="T_bea6e_row11_col6" class="data row11 col6" >0.95767</td>
      <td id="T_bea6e_row11_col7" class="data row11 col7" >3</td>
      <td id="T_bea6e_row11_col8" class="data row11 col8" >0</td>
      <td id="T_bea6e_row11_col9" class="data row11 col9" >5.40857</td>
      <td id="T_bea6e_row11_col10" class="data row11 col10" >False</td>
      <td id="T_bea6e_row11_col11" class="data row11 col11" >0.92962</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_94.png)
    



### Rules for Instance 12758, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.9545, Pre: 0.95767), Unique rules (diffrent features)



<style type="text/css">
#T_1e930_row5_col0, #T_1e930_row5_col1, #T_1e930_row5_col2, #T_1e930_row5_col3, #T_1e930_row5_col4, #T_1e930_row5_col5, #T_1e930_row5_col6, #T_1e930_row5_col7, #T_1e930_row5_col8, #T_1e930_row5_col9, #T_1e930_row5_col10, #T_1e930_row5_col11 {
  font-weight: bold;
}
</style>
<table id="T_1e930">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1e930_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_1e930_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_1e930_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_1e930_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_1e930_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_1e930_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_1e930_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_1e930_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_1e930_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_1e930_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_1e930_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_1e930_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1e930_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_1e930_row0_col0" class="data row0 col0" >12758</td>
      <td id="T_1e930_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_1e930_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_1e930_row0_col3" class="data row0 col3" >IF education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_1e930_row0_col4" class="data row0 col4" >0.32095</td>
      <td id="T_1e930_row0_col5" class="data row0 col5" >0.35508</td>
      <td id="T_1e930_row0_col6" class="data row0 col6" >0.83992</td>
      <td id="T_1e930_row0_col7" class="data row0 col7" >2</td>
      <td id="T_1e930_row0_col8" class="data row0 col8" >0</td>
      <td id="T_1e930_row0_col9" class="data row0 col9" >1.28740</td>
      <td id="T_1e930_row0_col10" class="data row0 col10" >False</td>
      <td id="T_1e930_row0_col11" class="data row0 col11" >0.64440</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_1e930_row1_col0" class="data row1 col0" >12758</td>
      <td id="T_1e930_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_1e930_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_1e930_row1_col3" class="data row1 col3" >IF capital.loss <= 0.0 AND education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_1e930_row1_col4" class="data row1 col4" >0.30862</td>
      <td id="T_1e930_row1_col5" class="data row1 col5" >0.34445</td>
      <td id="T_1e930_row1_col6" class="data row1 col6" >0.84731</td>
      <td id="T_1e930_row1_col7" class="data row1 col7" >3</td>
      <td id="T_1e930_row1_col8" class="data row1 col8" >1</td>
      <td id="T_1e930_row1_col9" class="data row1 col9" >2.44282</td>
      <td id="T_1e930_row1_col10" class="data row1 col10" >False</td>
      <td id="T_1e930_row1_col11" class="data row1 col11" >0.65524</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_1e930_row2_col0" class="data row2 col0" >12758</td>
      <td id="T_1e930_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_1e930_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_1e930_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_1e930_row2_col4" class="data row2 col4" >0.66273</td>
      <td id="T_1e930_row2_col5" class="data row2 col5" >0.72849</td>
      <td id="T_1e930_row2_col6" class="data row2 col6" >0.83449</td>
      <td id="T_1e930_row2_col7" class="data row2 col7" >3</td>
      <td id="T_1e930_row2_col8" class="data row2 col8" >0</td>
      <td id="T_1e930_row2_col9" class="data row2 col9" >1.24879</td>
      <td id="T_1e930_row2_col10" class="data row2 col10" >False</td>
      <td id="T_1e930_row2_col11" class="data row2 col11" >0.31671</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_1e930_row3_col0" class="data row3 col0" >12758</td>
      <td id="T_1e930_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_1e930_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_1e930_row3_col3" class="data row3 col3" >IF capital.gain <= 0.0 AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_1e930_row3_col4" class="data row3 col4" >0.29936</td>
      <td id="T_1e930_row3_col5" class="data row3 col5" >0.33959</td>
      <td id="T_1e930_row3_col6" class="data row3 col6" >0.86120</td>
      <td id="T_1e930_row3_col7" class="data row3 col7" >2</td>
      <td id="T_1e930_row3_col8" class="data row3 col8" >0</td>
      <td id="T_1e930_row3_col9" class="data row3 col9" >1.56445</td>
      <td id="T_1e930_row3_col10" class="data row3 col10" >False</td>
      <td id="T_1e930_row3_col11" class="data row3 col11" >0.66220</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_1e930_row4_col0" class="data row4 col0" >12758</td>
      <td id="T_1e930_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_1e930_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_1e930_row4_col3" class="data row4 col3" >IF education = HS-grad THEN class = <=50K</td>
      <td id="T_1e930_row4_col4" class="data row4 col4" >0.32095</td>
      <td id="T_1e930_row4_col5" class="data row4 col5" >0.35508</td>
      <td id="T_1e930_row4_col6" class="data row4 col6" >0.83992</td>
      <td id="T_1e930_row4_col7" class="data row4 col7" >1</td>
      <td id="T_1e930_row4_col8" class="data row4 col8" >1</td>
      <td id="T_1e930_row4_col9" class="data row4 col9" >2.61744</td>
      <td id="T_1e930_row4_col10" class="data row4 col10" >False</td>
      <td id="T_1e930_row4_col11" class="data row4 col11" >0.64440</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_1e930_row5_col0" class="data row5 col0" >12758</td>
      <td id="T_1e930_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_1e930_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_1e930_row5_col3" class="data row5 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_1e930_row5_col4" class="data row5 col4" >0.94226</td>
      <td id="T_1e930_row5_col5" class="data row5 col5" >0.98942</td>
      <td id="T_1e930_row5_col6" class="data row5 col6" >0.79717</td>
      <td id="T_1e930_row5_col7" class="data row5 col7" >1</td>
      <td id="T_1e930_row5_col8" class="data row5 col8" >0</td>
      <td id="T_1e930_row5_col9" class="data row5 col9" >79.87095</td>
      <td id="T_1e930_row5_col10" class="data row5 col10" >False</td>
      <td id="T_1e930_row5_col11" class="data row5 col11" >0.16097</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row6" class="row_heading level0 row6" >8</th>
      <td id="T_1e930_row6_col0" class="data row6 col0" >12758</td>
      <td id="T_1e930_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_1e930_row6_col2" class="data row6 col2" >LORE_SA1</td>
      <td id="T_1e930_row6_col3" class="data row6 col3" >IF capital.gain <= 5422.202 AND education != Bachelors AND hours.per.week <= 58.6454 THEN class = <=50K</td>
      <td id="T_1e930_row6_col4" class="data row6 col4" >0.74373</td>
      <td id="T_1e930_row6_col5" class="data row6 col5" >0.81708</td>
      <td id="T_1e930_row6_col6" class="data row6 col6" >0.83405</td>
      <td id="T_1e930_row6_col7" class="data row6 col7" >3</td>
      <td id="T_1e930_row6_col8" class="data row6 col8" >0</td>
      <td id="T_1e930_row6_col9" class="data row6 col9" >22.65043</td>
      <td id="T_1e930_row6_col10" class="data row6 col10" >False</td>
      <td id="T_1e930_row6_col11" class="data row6 col11" >0.24435</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row7" class="row_heading level0 row7" >9</th>
      <td id="T_1e930_row7_col0" class="data row7 col0" >12758</td>
      <td id="T_1e930_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_1e930_row7_col2" class="data row7 col2" >EXPLAN1</td>
      <td id="T_1e930_row7_col3" class="data row7 col3" >IF age > 35.03 AND capital.gain <= 2796.6085 AND capital.loss <= 1776.9163 AND relationship = Not-in-family THEN class = <=50K</td>
      <td id="T_1e930_row7_col4" class="data row7 col4" >0.11329</td>
      <td id="T_1e930_row7_col5" class="data row7 col5" >0.13379</td>
      <td id="T_1e930_row7_col6" class="data row7 col6" >0.89659</td>
      <td id="T_1e930_row7_col7" class="data row7 col7" >4</td>
      <td id="T_1e930_row7_col8" class="data row7 col8" >0</td>
      <td id="T_1e930_row7_col9" class="data row7 col9" >7.79743</td>
      <td id="T_1e930_row7_col10" class="data row7 col10" >False</td>
      <td id="T_1e930_row7_col11" class="data row7 col11" >0.84342</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row8" class="row_heading level0 row8" >10</th>
      <td id="T_1e930_row8_col0" class="data row8 col0" >12758</td>
      <td id="T_1e930_row8_col1" class="data row8 col1" >EXPLAN</td>
      <td id="T_1e930_row8_col2" class="data row8 col2" >EXPLAN2</td>
      <td id="T_1e930_row8_col3" class="data row8 col3" >IF marital.status = Widowed THEN class = <=50K</td>
      <td id="T_1e930_row8_col4" class="data row8 col4" >0.03054</td>
      <td id="T_1e930_row8_col5" class="data row8 col5" >0.03681</td>
      <td id="T_1e930_row8_col6" class="data row8 col6" >0.91523</td>
      <td id="T_1e930_row8_col7" class="data row8 col7" >1</td>
      <td id="T_1e930_row8_col8" class="data row8 col8" >0</td>
      <td id="T_1e930_row8_col9" class="data row8 col9" >7.98429</td>
      <td id="T_1e930_row8_col10" class="data row8 col10" >False</td>
      <td id="T_1e930_row8_col11" class="data row8 col11" >0.92493</td>
    </tr>
    <tr>
      <th id="T_1e930_level0_row9" class="row_heading level0 row9" >11</th>
      <td id="T_1e930_row9_col0" class="data row9 col0" >12758</td>
      <td id="T_1e930_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_1e930_row9_col2" class="data row9 col2" >EXPLAN3</td>
      <td id="T_1e930_row9_col3" class="data row9 col3" >IF capital.gain <= 4954.4995 AND marital.status = Widowed AND sex = Female THEN class = <=50K</td>
      <td id="T_1e930_row9_col4" class="data row9 col4" >0.02488</td>
      <td id="T_1e930_row9_col5" class="data row9 col5" >0.03138</td>
      <td id="T_1e930_row9_col6" class="data row9 col6" >0.95767</td>
      <td id="T_1e930_row9_col7" class="data row9 col7" >3</td>
      <td id="T_1e930_row9_col8" class="data row9 col8" >0</td>
      <td id="T_1e930_row9_col9" class="data row9 col9" >5.40857</td>
      <td id="T_1e930_row9_col10" class="data row9 col10" >False</td>
      <td id="T_1e930_row9_col11" class="data row9 col11" >0.92962</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_97.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_98.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_99.png)
    



### Rules for Instance 12758, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99717, Pre: 0.95767, Len: 0.79311)



<style type="text/css">
#T_900c3_row4_col0, #T_900c3_row4_col1, #T_900c3_row4_col2, #T_900c3_row4_col3, #T_900c3_row4_col4, #T_900c3_row4_col5, #T_900c3_row4_col6, #T_900c3_row4_col7, #T_900c3_row4_col8, #T_900c3_row4_col9, #T_900c3_row4_col10, #T_900c3_row4_col11 {
  font-weight: bold;
}
</style>
<table id="T_900c3">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_900c3_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_900c3_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_900c3_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_900c3_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_900c3_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_900c3_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_900c3_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_900c3_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_900c3_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_900c3_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_900c3_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_900c3_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_900c3_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_900c3_row0_col0" class="data row0 col0" >12758</td>
      <td id="T_900c3_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_900c3_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_900c3_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_900c3_row0_col4" class="data row0 col4" >0.30862</td>
      <td id="T_900c3_row0_col5" class="data row0 col5" >0.34445</td>
      <td id="T_900c3_row0_col6" class="data row0 col6" >0.84731</td>
      <td id="T_900c3_row0_col7" class="data row0 col7" >3</td>
      <td id="T_900c3_row0_col8" class="data row0 col8" >1</td>
      <td id="T_900c3_row0_col9" class="data row0 col9" >2.44282</td>
      <td id="T_900c3_row0_col10" class="data row0 col10" >False</td>
      <td id="T_900c3_row0_col11" class="data row0 col11" >2.30404</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_900c3_row1_col0" class="data row1 col0" >12758</td>
      <td id="T_900c3_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_900c3_row1_col2" class="data row1 col2" >ANCHOR3</td>
      <td id="T_900c3_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_900c3_row1_col4" class="data row1 col4" >0.66273</td>
      <td id="T_900c3_row1_col5" class="data row1 col5" >0.72849</td>
      <td id="T_900c3_row1_col6" class="data row1 col6" >0.83449</td>
      <td id="T_900c3_row1_col7" class="data row1 col7" >3</td>
      <td id="T_900c3_row1_col8" class="data row1 col8" >0</td>
      <td id="T_900c3_row1_col9" class="data row1 col9" >1.24879</td>
      <td id="T_900c3_row1_col10" class="data row1 col10" >False</td>
      <td id="T_900c3_row1_col11" class="data row1 col11" >2.22660</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_900c3_row2_col0" class="data row2 col0" >12758</td>
      <td id="T_900c3_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_900c3_row2_col2" class="data row2 col2" >ANCHOR4</td>
      <td id="T_900c3_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_900c3_row2_col4" class="data row2 col4" >0.29936</td>
      <td id="T_900c3_row2_col5" class="data row2 col5" >0.33959</td>
      <td id="T_900c3_row2_col6" class="data row2 col6" >0.86120</td>
      <td id="T_900c3_row2_col7" class="data row2 col7" >2</td>
      <td id="T_900c3_row2_col8" class="data row2 col8" >0</td>
      <td id="T_900c3_row2_col9" class="data row2 col9" >1.56445</td>
      <td id="T_900c3_row2_col10" class="data row2 col10" >False</td>
      <td id="T_900c3_row2_col11" class="data row2 col11" >1.37779</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_900c3_row3_col0" class="data row3 col0" >12758</td>
      <td id="T_900c3_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_900c3_row3_col2" class="data row3 col2" >ANCHOR5</td>
      <td id="T_900c3_row3_col3" class="data row3 col3" >IF education = HS-grad THEN class = <=50K</td>
      <td id="T_900c3_row3_col4" class="data row3 col4" >0.32095</td>
      <td id="T_900c3_row3_col5" class="data row3 col5" >0.35508</td>
      <td id="T_900c3_row3_col6" class="data row3 col6" >0.83992</td>
      <td id="T_900c3_row3_col7" class="data row3 col7" >1</td>
      <td id="T_900c3_row3_col8" class="data row3 col8" >1</td>
      <td id="T_900c3_row3_col9" class="data row3 col9" >2.61744</td>
      <td id="T_900c3_row3_col10" class="data row3 col10" >False</td>
      <td id="T_900c3_row3_col11" class="data row3 col11" >0.68480</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_900c3_row4_col0" class="data row4 col0" >12758</td>
      <td id="T_900c3_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_900c3_row4_col2" class="data row4 col2" >LORE3</td>
      <td id="T_900c3_row4_col3" class="data row4 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_900c3_row4_col4" class="data row4 col4" >0.94226</td>
      <td id="T_900c3_row4_col5" class="data row4 col5" >0.98942</td>
      <td id="T_900c3_row4_col6" class="data row4 col6" >0.79717</td>
      <td id="T_900c3_row4_col7" class="data row4 col7" >1</td>
      <td id="T_900c3_row4_col8" class="data row4 col8" >0</td>
      <td id="T_900c3_row4_col9" class="data row4 col9" >79.87095</td>
      <td id="T_900c3_row4_col10" class="data row4 col10" >False</td>
      <td id="T_900c3_row4_col11" class="data row4 col11" >0.26196</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_900c3_row5_col0" class="data row5 col0" >12758</td>
      <td id="T_900c3_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_900c3_row5_col2" class="data row5 col2" >LORE4</td>
      <td id="T_900c3_row5_col3" class="data row5 col3" >IF capital.gain <= 6025.6313 THEN class = <=50K</td>
      <td id="T_900c3_row5_col4" class="data row5 col4" >0.95450</td>
      <td id="T_900c3_row5_col5" class="data row5 col5" >0.99717</td>
      <td id="T_900c3_row5_col6" class="data row5 col6" >0.79311</td>
      <td id="T_900c3_row5_col7" class="data row5 col7" >1</td>
      <td id="T_900c3_row5_col8" class="data row5 col8" >0</td>
      <td id="T_900c3_row5_col9" class="data row5 col9" >80.20908</td>
      <td id="T_900c3_row5_col10" class="data row5 col10" >False</td>
      <td id="T_900c3_row5_col11" class="data row5 col11" >0.26435</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_900c3_row6_col0" class="data row6 col0" >12758</td>
      <td id="T_900c3_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_900c3_row6_col2" class="data row6 col2" >LORE5</td>
      <td id="T_900c3_row6_col3" class="data row6 col3" >IF capital.gain <= 4787.0 THEN class = <=50K</td>
      <td id="T_900c3_row6_col4" class="data row6 col4" >0.94805</td>
      <td id="T_900c3_row6_col5" class="data row6 col5" >0.99324</td>
      <td id="T_900c3_row6_col6" class="data row6 col6" >0.79535</td>
      <td id="T_900c3_row6_col7" class="data row6 col7" >1</td>
      <td id="T_900c3_row6_col8" class="data row6 col8" >0</td>
      <td id="T_900c3_row6_col9" class="data row6 col9" >117.75989</td>
      <td id="T_900c3_row6_col10" class="data row6 col10" >False</td>
      <td id="T_900c3_row6_col11" class="data row6 col11" >0.26300</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_900c3_row7_col0" class="data row7 col0" >12758</td>
      <td id="T_900c3_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_900c3_row7_col2" class="data row7 col2" >LORE_SA1</td>
      <td id="T_900c3_row7_col3" class="data row7 col3" >IF capital.gain <= 5422.202 AND education != Bachelors AND hours.per.week <= 58.6454 THEN class = <=50K</td>
      <td id="T_900c3_row7_col4" class="data row7 col4" >0.74373</td>
      <td id="T_900c3_row7_col5" class="data row7 col5" >0.81708</td>
      <td id="T_900c3_row7_col6" class="data row7 col6" >0.83405</td>
      <td id="T_900c3_row7_col7" class="data row7 col7" >3</td>
      <td id="T_900c3_row7_col8" class="data row7 col8" >0</td>
      <td id="T_900c3_row7_col9" class="data row7 col9" >22.65043</td>
      <td id="T_900c3_row7_col10" class="data row7 col10" >False</td>
      <td id="T_900c3_row7_col11" class="data row7 col11" >2.21767</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_900c3_row8_col0" class="data row8 col0" >12758</td>
      <td id="T_900c3_row8_col1" class="data row8 col1" >EXPLAN</td>
      <td id="T_900c3_row8_col2" class="data row8 col2" >EXPLAN1</td>
      <td id="T_900c3_row8_col3" class="data row8 col3" >IF age > 35.03 AND capital.gain <= 2796.6085 AND capital.loss <= 1776.9163 AND relationship = Not-in-family THEN class = <=50K</td>
      <td id="T_900c3_row8_col4" class="data row8 col4" >0.11329</td>
      <td id="T_900c3_row8_col5" class="data row8 col5" >0.13379</td>
      <td id="T_900c3_row8_col6" class="data row8 col6" >0.89659</td>
      <td id="T_900c3_row8_col7" class="data row8 col7" >4</td>
      <td id="T_900c3_row8_col8" class="data row8 col8" >0</td>
      <td id="T_900c3_row8_col9" class="data row8 col9" >7.79743</td>
      <td id="T_900c3_row8_col10" class="data row8 col10" >False</td>
      <td id="T_900c3_row8_col11" class="data row8 col11" >3.32164</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_900c3_row9_col0" class="data row9 col0" >12758</td>
      <td id="T_900c3_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_900c3_row9_col2" class="data row9 col2" >EXPLAN2</td>
      <td id="T_900c3_row9_col3" class="data row9 col3" >IF marital.status = Widowed THEN class = <=50K</td>
      <td id="T_900c3_row9_col4" class="data row9 col4" >0.03054</td>
      <td id="T_900c3_row9_col5" class="data row9 col5" >0.03681</td>
      <td id="T_900c3_row9_col6" class="data row9 col6" >0.91523</td>
      <td id="T_900c3_row9_col7" class="data row9 col7" >1</td>
      <td id="T_900c3_row9_col8" class="data row9 col8" >0</td>
      <td id="T_900c3_row9_col9" class="data row9 col9" >7.98429</td>
      <td id="T_900c3_row9_col10" class="data row9 col10" >False</td>
      <td id="T_900c3_row9_col11" class="data row9 col11" >0.98331</td>
    </tr>
    <tr>
      <th id="T_900c3_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_900c3_row10_col0" class="data row10 col0" >12758</td>
      <td id="T_900c3_row10_col1" class="data row10 col1" >EXPLAN</td>
      <td id="T_900c3_row10_col2" class="data row10 col2" >EXPLAN3</td>
      <td id="T_900c3_row10_col3" class="data row10 col3" >IF capital.gain <= 4954.4995 AND marital.status = Widowed AND sex = Female THEN class = <=50K</td>
      <td id="T_900c3_row10_col4" class="data row10 col4" >0.02488</td>
      <td id="T_900c3_row10_col5" class="data row10 col5" >0.03138</td>
      <td id="T_900c3_row10_col6" class="data row10 col6" >0.95767</td>
      <td id="T_900c3_row10_col7" class="data row10 col7" >3</td>
      <td id="T_900c3_row10_col8" class="data row10 col8" >0</td>
      <td id="T_900c3_row10_col9" class="data row10 col9" >5.40857</td>
      <td id="T_900c3_row10_col10" class="data row10 col10" >False</td>
      <td id="T_900c3_row10_col11" class="data row10 col11" >2.40897</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12758, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99717, Pre: 0.95767), Unique rules (diffrent features)



<style type="text/css">
#T_46531_row4_col0, #T_46531_row4_col1, #T_46531_row4_col2, #T_46531_row4_col3, #T_46531_row4_col4, #T_46531_row4_col5, #T_46531_row4_col6, #T_46531_row4_col7, #T_46531_row4_col8, #T_46531_row4_col9, #T_46531_row4_col10, #T_46531_row4_col11 {
  font-weight: bold;
}
</style>
<table id="T_46531">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_46531_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_46531_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_46531_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_46531_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_46531_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_46531_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_46531_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_46531_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_46531_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_46531_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_46531_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_46531_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_46531_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_46531_row0_col0" class="data row0 col0" >12758</td>
      <td id="T_46531_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_46531_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_46531_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND education = HS-grad AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_46531_row0_col4" class="data row0 col4" >0.30862</td>
      <td id="T_46531_row0_col5" class="data row0 col5" >0.34445</td>
      <td id="T_46531_row0_col6" class="data row0 col6" >0.84731</td>
      <td id="T_46531_row0_col7" class="data row0 col7" >3</td>
      <td id="T_46531_row0_col8" class="data row0 col8" >1</td>
      <td id="T_46531_row0_col9" class="data row0 col9" >2.44282</td>
      <td id="T_46531_row0_col10" class="data row0 col10" >False</td>
      <td id="T_46531_row0_col11" class="data row0 col11" >2.30404</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_46531_row1_col0" class="data row1 col0" >12758</td>
      <td id="T_46531_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_46531_row1_col2" class="data row1 col2" >ANCHOR3</td>
      <td id="T_46531_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_46531_row1_col4" class="data row1 col4" >0.66273</td>
      <td id="T_46531_row1_col5" class="data row1 col5" >0.72849</td>
      <td id="T_46531_row1_col6" class="data row1 col6" >0.83449</td>
      <td id="T_46531_row1_col7" class="data row1 col7" >3</td>
      <td id="T_46531_row1_col8" class="data row1 col8" >0</td>
      <td id="T_46531_row1_col9" class="data row1 col9" >1.24879</td>
      <td id="T_46531_row1_col10" class="data row1 col10" >False</td>
      <td id="T_46531_row1_col11" class="data row1 col11" >2.22660</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_46531_row2_col0" class="data row2 col0" >12758</td>
      <td id="T_46531_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_46531_row2_col2" class="data row2 col2" >ANCHOR4</td>
      <td id="T_46531_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND education.num = 9.0 THEN class = <=50K</td>
      <td id="T_46531_row2_col4" class="data row2 col4" >0.29936</td>
      <td id="T_46531_row2_col5" class="data row2 col5" >0.33959</td>
      <td id="T_46531_row2_col6" class="data row2 col6" >0.86120</td>
      <td id="T_46531_row2_col7" class="data row2 col7" >2</td>
      <td id="T_46531_row2_col8" class="data row2 col8" >0</td>
      <td id="T_46531_row2_col9" class="data row2 col9" >1.56445</td>
      <td id="T_46531_row2_col10" class="data row2 col10" >False</td>
      <td id="T_46531_row2_col11" class="data row2 col11" >1.37779</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_46531_row3_col0" class="data row3 col0" >12758</td>
      <td id="T_46531_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_46531_row3_col2" class="data row3 col2" >ANCHOR5</td>
      <td id="T_46531_row3_col3" class="data row3 col3" >IF education = HS-grad THEN class = <=50K</td>
      <td id="T_46531_row3_col4" class="data row3 col4" >0.32095</td>
      <td id="T_46531_row3_col5" class="data row3 col5" >0.35508</td>
      <td id="T_46531_row3_col6" class="data row3 col6" >0.83992</td>
      <td id="T_46531_row3_col7" class="data row3 col7" >1</td>
      <td id="T_46531_row3_col8" class="data row3 col8" >1</td>
      <td id="T_46531_row3_col9" class="data row3 col9" >2.61744</td>
      <td id="T_46531_row3_col10" class="data row3 col10" >False</td>
      <td id="T_46531_row3_col11" class="data row3 col11" >0.68480</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_46531_row4_col0" class="data row4 col0" >12758</td>
      <td id="T_46531_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_46531_row4_col2" class="data row4 col2" >LORE3</td>
      <td id="T_46531_row4_col3" class="data row4 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_46531_row4_col4" class="data row4 col4" >0.94226</td>
      <td id="T_46531_row4_col5" class="data row4 col5" >0.98942</td>
      <td id="T_46531_row4_col6" class="data row4 col6" >0.79717</td>
      <td id="T_46531_row4_col7" class="data row4 col7" >1</td>
      <td id="T_46531_row4_col8" class="data row4 col8" >0</td>
      <td id="T_46531_row4_col9" class="data row4 col9" >79.87095</td>
      <td id="T_46531_row4_col10" class="data row4 col10" >False</td>
      <td id="T_46531_row4_col11" class="data row4 col11" >0.26196</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row5" class="row_heading level0 row5" >7</th>
      <td id="T_46531_row5_col0" class="data row5 col0" >12758</td>
      <td id="T_46531_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_46531_row5_col2" class="data row5 col2" >LORE_SA1</td>
      <td id="T_46531_row5_col3" class="data row5 col3" >IF capital.gain <= 5422.202 AND education != Bachelors AND hours.per.week <= 58.6454 THEN class = <=50K</td>
      <td id="T_46531_row5_col4" class="data row5 col4" >0.74373</td>
      <td id="T_46531_row5_col5" class="data row5 col5" >0.81708</td>
      <td id="T_46531_row5_col6" class="data row5 col6" >0.83405</td>
      <td id="T_46531_row5_col7" class="data row5 col7" >3</td>
      <td id="T_46531_row5_col8" class="data row5 col8" >0</td>
      <td id="T_46531_row5_col9" class="data row5 col9" >22.65043</td>
      <td id="T_46531_row5_col10" class="data row5 col10" >False</td>
      <td id="T_46531_row5_col11" class="data row5 col11" >2.21767</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row6" class="row_heading level0 row6" >8</th>
      <td id="T_46531_row6_col0" class="data row6 col0" >12758</td>
      <td id="T_46531_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_46531_row6_col2" class="data row6 col2" >EXPLAN1</td>
      <td id="T_46531_row6_col3" class="data row6 col3" >IF age > 35.03 AND capital.gain <= 2796.6085 AND capital.loss <= 1776.9163 AND relationship = Not-in-family THEN class = <=50K</td>
      <td id="T_46531_row6_col4" class="data row6 col4" >0.11329</td>
      <td id="T_46531_row6_col5" class="data row6 col5" >0.13379</td>
      <td id="T_46531_row6_col6" class="data row6 col6" >0.89659</td>
      <td id="T_46531_row6_col7" class="data row6 col7" >4</td>
      <td id="T_46531_row6_col8" class="data row6 col8" >0</td>
      <td id="T_46531_row6_col9" class="data row6 col9" >7.79743</td>
      <td id="T_46531_row6_col10" class="data row6 col10" >False</td>
      <td id="T_46531_row6_col11" class="data row6 col11" >3.32164</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row7" class="row_heading level0 row7" >9</th>
      <td id="T_46531_row7_col0" class="data row7 col0" >12758</td>
      <td id="T_46531_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_46531_row7_col2" class="data row7 col2" >EXPLAN2</td>
      <td id="T_46531_row7_col3" class="data row7 col3" >IF marital.status = Widowed THEN class = <=50K</td>
      <td id="T_46531_row7_col4" class="data row7 col4" >0.03054</td>
      <td id="T_46531_row7_col5" class="data row7 col5" >0.03681</td>
      <td id="T_46531_row7_col6" class="data row7 col6" >0.91523</td>
      <td id="T_46531_row7_col7" class="data row7 col7" >1</td>
      <td id="T_46531_row7_col8" class="data row7 col8" >0</td>
      <td id="T_46531_row7_col9" class="data row7 col9" >7.98429</td>
      <td id="T_46531_row7_col10" class="data row7 col10" >False</td>
      <td id="T_46531_row7_col11" class="data row7 col11" >0.98331</td>
    </tr>
    <tr>
      <th id="T_46531_level0_row8" class="row_heading level0 row8" >10</th>
      <td id="T_46531_row8_col0" class="data row8 col0" >12758</td>
      <td id="T_46531_row8_col1" class="data row8 col1" >EXPLAN</td>
      <td id="T_46531_row8_col2" class="data row8 col2" >EXPLAN3</td>
      <td id="T_46531_row8_col3" class="data row8 col3" >IF capital.gain <= 4954.4995 AND marital.status = Widowed AND sex = Female THEN class = <=50K</td>
      <td id="T_46531_row8_col4" class="data row8 col4" >0.02488</td>
      <td id="T_46531_row8_col5" class="data row8 col5" >0.03138</td>
      <td id="T_46531_row8_col6" class="data row8 col6" >0.95767</td>
      <td id="T_46531_row8_col7" class="data row8 col7" >3</td>
      <td id="T_46531_row8_col8" class="data row8 col8" >0</td>
      <td id="T_46531_row8_col9" class="data row8 col9" >5.40857</td>
      <td id="T_46531_row8_col10" class="data row8 col10" >False</td>
      <td id="T_46531_row8_col11" class="data row8 col11" >2.40897</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_104.png)
    



## Instance 5759 (Original: <=50K , Predicted: <=50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>17.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>10th</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>6</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Never-married</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Prof-specialty</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Own-child</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>30.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 5759



<style type="text/css">
</style>
<table id="T_5609d">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_5609d_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_5609d_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_5609d_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_5609d_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_5609d_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_5609d_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_5609d_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_5609d_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_5609d_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_5609d_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_5609d_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5609d_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_5609d_row0_col0" class="data row0 col0" >5759</td>
      <td id="T_5609d_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_5609d_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_5609d_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_5609d_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_5609d_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_5609d_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_5609d_row0_col7" class="data row0 col7" >3</td>
      <td id="T_5609d_row0_col8" class="data row0 col8" >0</td>
      <td id="T_5609d_row0_col9" class="data row0 col9" >2.13079</td>
      <td id="T_5609d_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_5609d_row1_col0" class="data row1 col0" >5759</td>
      <td id="T_5609d_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_5609d_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_5609d_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_5609d_row1_col4" class="data row1 col4" >0.54611</td>
      <td id="T_5609d_row1_col5" class="data row1 col5" >0.62509</td>
      <td id="T_5609d_row1_col6" class="data row1 col6" >0.86896</td>
      <td id="T_5609d_row1_col7" class="data row1 col7" >4</td>
      <td id="T_5609d_row1_col8" class="data row1 col8" >0</td>
      <td id="T_5609d_row1_col9" class="data row1 col9" >2.29826</td>
      <td id="T_5609d_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_5609d_row2_col0" class="data row2 col0" >5759</td>
      <td id="T_5609d_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_5609d_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_5609d_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_5609d_row2_col4" class="data row2 col4" >0.63088</td>
      <td id="T_5609d_row2_col5" class="data row2 col5" >0.71733</td>
      <td id="T_5609d_row2_col6" class="data row2 col6" >0.86320</td>
      <td id="T_5609d_row2_col7" class="data row2 col7" >3</td>
      <td id="T_5609d_row2_col8" class="data row2 col8" >0</td>
      <td id="T_5609d_row2_col9" class="data row2 col9" >2.07695</td>
      <td id="T_5609d_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_5609d_row3_col0" class="data row3 col0" >5759</td>
      <td id="T_5609d_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_5609d_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_5609d_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_5609d_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_5609d_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_5609d_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_5609d_row3_col7" class="data row3 col7" >2</td>
      <td id="T_5609d_row3_col8" class="data row3 col8" >0</td>
      <td id="T_5609d_row3_col9" class="data row3 col9" >1.65381</td>
      <td id="T_5609d_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_5609d_row4_col0" class="data row4 col0" >5759</td>
      <td id="T_5609d_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_5609d_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_5609d_row4_col3" class="data row4 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_5609d_row4_col4" class="data row4 col4" >0.66269</td>
      <td id="T_5609d_row4_col5" class="data row4 col5" >0.73091</td>
      <td id="T_5609d_row4_col6" class="data row4 col6" >0.83733</td>
      <td id="T_5609d_row4_col7" class="data row4 col7" >3</td>
      <td id="T_5609d_row4_col8" class="data row4 col8" >1</td>
      <td id="T_5609d_row4_col9" class="data row4 col9" >4.35856</td>
      <td id="T_5609d_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_5609d_row5_col0" class="data row5 col0" >5759</td>
      <td id="T_5609d_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_5609d_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_5609d_row5_col3" class="data row5 col3" >IF capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_5609d_row5_col4" class="data row5 col4" >0.91598</td>
      <td id="T_5609d_row5_col5" class="data row5 col5" >0.95810</td>
      <td id="T_5609d_row5_col6" class="data row5 col6" >0.79408</td>
      <td id="T_5609d_row5_col7" class="data row5 col7" >1</td>
      <td id="T_5609d_row5_col8" class="data row5 col8" >0</td>
      <td id="T_5609d_row5_col9" class="data row5 col9" >208.22717</td>
      <td id="T_5609d_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_5609d_row6_col0" class="data row6 col0" >5759</td>
      <td id="T_5609d_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_5609d_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_5609d_row6_col3" class="data row6 col3" >IF capital.gain <= 7063.5299 THEN class = <=50K</td>
      <td id="T_5609d_row6_col4" class="data row6 col4" >0.95621</td>
      <td id="T_5609d_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_5609d_row6_col6" class="data row6 col6" >0.79306</td>
      <td id="T_5609d_row6_col7" class="data row6 col7" >1</td>
      <td id="T_5609d_row6_col8" class="data row6 col8" >0</td>
      <td id="T_5609d_row6_col9" class="data row6 col9" >195.11769</td>
      <td id="T_5609d_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_5609d_row7_col0" class="data row7 col0" >5759</td>
      <td id="T_5609d_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_5609d_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_5609d_row7_col3" class="data row7 col3" >IF capital.gain <= 4759.1385 THEN class = <=50K</td>
      <td id="T_5609d_row7_col4" class="data row7 col4" >0.94735</td>
      <td id="T_5609d_row7_col5" class="data row7 col5" >0.99324</td>
      <td id="T_5609d_row7_col6" class="data row7 col6" >0.79594</td>
      <td id="T_5609d_row7_col7" class="data row7 col7" >1</td>
      <td id="T_5609d_row7_col8" class="data row7 col8" >0</td>
      <td id="T_5609d_row7_col9" class="data row7 col9" >132.87754</td>
      <td id="T_5609d_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_5609d_row8_col0" class="data row8 col0" >5759</td>
      <td id="T_5609d_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_5609d_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_5609d_row8_col3" class="data row8 col3" >IF capital.gain <= 3103.0 THEN class = <=50K</td>
      <td id="T_5609d_row8_col4" class="data row8 col4" >0.93357</td>
      <td id="T_5609d_row8_col5" class="data row8 col5" >0.97798</td>
      <td id="T_5609d_row8_col6" class="data row8 col6" >0.79528</td>
      <td id="T_5609d_row8_col7" class="data row8 col7" >1</td>
      <td id="T_5609d_row8_col8" class="data row8 col8" >0</td>
      <td id="T_5609d_row8_col9" class="data row8 col9" >146.14264</td>
      <td id="T_5609d_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_5609d_row9_col0" class="data row9 col0" >5759</td>
      <td id="T_5609d_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_5609d_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_5609d_row9_col3" class="data row9 col3" >IF marital.status = Never-married THEN class = <=50K</td>
      <td id="T_5609d_row9_col4" class="data row9 col4" >0.32757</td>
      <td id="T_5609d_row9_col5" class="data row9 col5" >0.41178</td>
      <td id="T_5609d_row9_col6" class="data row9 col6" >0.95433</td>
      <td id="T_5609d_row9_col7" class="data row9 col7" >1</td>
      <td id="T_5609d_row9_col8" class="data row9 col8" >0</td>
      <td id="T_5609d_row9_col9" class="data row9 col9" >227.17949</td>
      <td id="T_5609d_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_5609d_row10_col0" class="data row10 col0" >5759</td>
      <td id="T_5609d_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_5609d_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_5609d_row10_col3" class="data row10 col3" >IF relationship = Own-child THEN class = <=50K</td>
      <td id="T_5609d_row10_col4" class="data row10 col4" >0.15554</td>
      <td id="T_5609d_row10_col5" class="data row10 col5" >0.20262</td>
      <td id="T_5609d_row10_col6" class="data row10 col6" >0.98900</td>
      <td id="T_5609d_row10_col7" class="data row10 col7" >1</td>
      <td id="T_5609d_row10_col8" class="data row10 col8" >0</td>
      <td id="T_5609d_row10_col9" class="data row10 col9" >30.98065</td>
      <td id="T_5609d_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_5609d_row11_col0" class="data row11 col0" >5759</td>
      <td id="T_5609d_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_5609d_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_5609d_row11_col3" class="data row11 col3" >IF age <= 86.5864 AND hours.per.week <= 76.1273 AND hours.per.week > 27.1607 THEN class = <=50K</td>
      <td id="T_5609d_row11_col4" class="data row11 col4" >0.86469</td>
      <td id="T_5609d_row11_col5" class="data row11 col5" >0.83789</td>
      <td id="T_5609d_row11_col6" class="data row11 col6" >0.73564</td>
      <td id="T_5609d_row11_col7" class="data row11 col7" >3</td>
      <td id="T_5609d_row11_col8" class="data row11 col8" >0</td>
      <td id="T_5609d_row11_col9" class="data row11 col9" >30.22322</td>
      <td id="T_5609d_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_5609d_row12_col0" class="data row12 col0" >5759</td>
      <td id="T_5609d_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_5609d_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_5609d_row12_col3" class="data row12 col3" >IF age <= 45.5703 AND education != Doctorate AND hours.per.week <= 95.2543 AND occupation != Machine-op-inspct THEN class = <=50K</td>
      <td id="T_5609d_row12_col4" class="data row12 col4" >0.65233</td>
      <td id="T_5609d_row12_col5" class="data row12 col5" >0.68820</td>
      <td id="T_5609d_row12_col6" class="data row12 col6" >0.80091</td>
      <td id="T_5609d_row12_col7" class="data row12 col7" >4</td>
      <td id="T_5609d_row12_col8" class="data row12 col8" >0</td>
      <td id="T_5609d_row12_col9" class="data row12 col9" >31.41192</td>
      <td id="T_5609d_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_5609d_row13_col0" class="data row13 col0" >5759</td>
      <td id="T_5609d_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_5609d_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_5609d_row13_col3" class="data row13 col3" >IF age <= 41.1272 AND education != Doctorate THEN class = <=50K</td>
      <td id="T_5609d_row13_col4" class="data row13 col4" >0.61114</td>
      <td id="T_5609d_row13_col5" class="data row13 col5" >0.67040</td>
      <td id="T_5609d_row13_col6" class="data row13 col6" >0.83279</td>
      <td id="T_5609d_row13_col7" class="data row13 col7" >2</td>
      <td id="T_5609d_row13_col8" class="data row13 col8" >0</td>
      <td id="T_5609d_row13_col9" class="data row13 col9" >33.98273</td>
      <td id="T_5609d_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_5609d_row14_col0" class="data row14 col0" >5759</td>
      <td id="T_5609d_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_5609d_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_5609d_row14_col3" class="data row14 col3" >IF age <= 18.6366 THEN class = <=50K</td>
      <td id="T_5609d_row14_col4" class="data row14 col4" >0.02918</td>
      <td id="T_5609d_row14_col5" class="data row14 col5" >0.03843</td>
      <td id="T_5609d_row14_col6" class="data row14 col6" >1.00000</td>
      <td id="T_5609d_row14_col7" class="data row14 col7" >1</td>
      <td id="T_5609d_row14_col8" class="data row14 col8" >0</td>
      <td id="T_5609d_row14_col9" class="data row14 col9" >31.54394</td>
      <td id="T_5609d_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_5609d_row15_col0" class="data row15 col0" >5759</td>
      <td id="T_5609d_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_5609d_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_5609d_row15_col3" class="data row15 col3" >IF age <= 35.5896 AND capital.gain <= 3809.9333 THEN class = <=50K</td>
      <td id="T_5609d_row15_col4" class="data row15 col4" >0.44647</td>
      <td id="T_5609d_row15_col5" class="data row15 col5" >0.53049</td>
      <td id="T_5609d_row15_col6" class="data row15 col6" >0.90202</td>
      <td id="T_5609d_row15_col7" class="data row15 col7" >2</td>
      <td id="T_5609d_row15_col8" class="data row15 col8" >0</td>
      <td id="T_5609d_row15_col9" class="data row15 col9" >2.73231</td>
      <td id="T_5609d_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_5609d_row16_col0" class="data row16 col0" >5759</td>
      <td id="T_5609d_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_5609d_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_5609d_row16_col3" class="data row16 col3" >IF capital.gain <= 498.5037 AND hours.per.week <= 36.8253 THEN class = <=50K</td>
      <td id="T_5609d_row16_col4" class="data row16 col4" >0.20683</td>
      <td id="T_5609d_row16_col5" class="data row16 col5" >0.25279</td>
      <td id="T_5609d_row16_col6" class="data row16 col6" >0.92787</td>
      <td id="T_5609d_row16_col7" class="data row16 col7" >2</td>
      <td id="T_5609d_row16_col8" class="data row16 col8" >0</td>
      <td id="T_5609d_row16_col9" class="data row16 col9" >2.34009</td>
      <td id="T_5609d_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_5609d_row17_col0" class="data row17 col0" >5759</td>
      <td id="T_5609d_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_5609d_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_5609d_row17_col3" class="data row17 col3" >IF capital.gain <= 4900.9955 AND hours.per.week <= 36.1513 THEN class = <=50K</td>
      <td id="T_5609d_row17_col4" class="data row17 col4" >0.21415</td>
      <td id="T_5609d_row17_col5" class="data row17 col5" >0.26175</td>
      <td id="T_5609d_row17_col6" class="data row17 col6" >0.92788</td>
      <td id="T_5609d_row17_col7" class="data row17 col7" >2</td>
      <td id="T_5609d_row17_col8" class="data row17 col8" >0</td>
      <td id="T_5609d_row17_col9" class="data row17 col9" >2.80984</td>
      <td id="T_5609d_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_5609d_row18_col0" class="data row18 col0" >5759</td>
      <td id="T_5609d_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_5609d_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_5609d_row18_col3" class="data row18 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 32.0 THEN class = <=50K</td>
      <td id="T_5609d_row18_col4" class="data row18 col4" >0.16199</td>
      <td id="T_5609d_row18_col5" class="data row18 col5" >0.20095</td>
      <td id="T_5609d_row18_col6" class="data row18 col6" >0.94177</td>
      <td id="T_5609d_row18_col7" class="data row18 col7" >2</td>
      <td id="T_5609d_row18_col8" class="data row18 col8" >0</td>
      <td id="T_5609d_row18_col9" class="data row18 col9" >2.70685</td>
      <td id="T_5609d_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_5609d_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_5609d_row19_col0" class="data row19 col0" >5759</td>
      <td id="T_5609d_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_5609d_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_5609d_row19_col3" class="data row19 col3" >IF hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_5609d_row19_col4" class="data row19 col4" >0.17208</td>
      <td id="T_5609d_row19_col5" class="data row19 col5" >0.21043</td>
      <td id="T_5609d_row19_col6" class="data row19 col6" >0.92835</td>
      <td id="T_5609d_row19_col7" class="data row19 col7" >1</td>
      <td id="T_5609d_row19_col8" class="data row19 col8" >0</td>
      <td id="T_5609d_row19_col9" class="data row19 col9" >2.51105</td>
      <td id="T_5609d_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5759, Correct Prediction



<style type="text/css">
</style>
<table id="T_c32ee">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c32ee_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_c32ee_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_c32ee_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_c32ee_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_c32ee_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_c32ee_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_c32ee_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_c32ee_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_c32ee_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_c32ee_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_c32ee_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c32ee_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_c32ee_row0_col0" class="data row0 col0" >5759</td>
      <td id="T_c32ee_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_c32ee_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_c32ee_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_c32ee_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_c32ee_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_c32ee_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_c32ee_row0_col7" class="data row0 col7" >3</td>
      <td id="T_c32ee_row0_col8" class="data row0 col8" >0</td>
      <td id="T_c32ee_row0_col9" class="data row0 col9" >2.13079</td>
      <td id="T_c32ee_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_c32ee_row1_col0" class="data row1 col0" >5759</td>
      <td id="T_c32ee_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_c32ee_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_c32ee_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_c32ee_row1_col4" class="data row1 col4" >0.54611</td>
      <td id="T_c32ee_row1_col5" class="data row1 col5" >0.62509</td>
      <td id="T_c32ee_row1_col6" class="data row1 col6" >0.86896</td>
      <td id="T_c32ee_row1_col7" class="data row1 col7" >4</td>
      <td id="T_c32ee_row1_col8" class="data row1 col8" >0</td>
      <td id="T_c32ee_row1_col9" class="data row1 col9" >2.29826</td>
      <td id="T_c32ee_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_c32ee_row2_col0" class="data row2 col0" >5759</td>
      <td id="T_c32ee_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_c32ee_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_c32ee_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_c32ee_row2_col4" class="data row2 col4" >0.63088</td>
      <td id="T_c32ee_row2_col5" class="data row2 col5" >0.71733</td>
      <td id="T_c32ee_row2_col6" class="data row2 col6" >0.86320</td>
      <td id="T_c32ee_row2_col7" class="data row2 col7" >3</td>
      <td id="T_c32ee_row2_col8" class="data row2 col8" >0</td>
      <td id="T_c32ee_row2_col9" class="data row2 col9" >2.07695</td>
      <td id="T_c32ee_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_c32ee_row3_col0" class="data row3 col0" >5759</td>
      <td id="T_c32ee_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_c32ee_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_c32ee_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_c32ee_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_c32ee_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_c32ee_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_c32ee_row3_col7" class="data row3 col7" >2</td>
      <td id="T_c32ee_row3_col8" class="data row3 col8" >0</td>
      <td id="T_c32ee_row3_col9" class="data row3 col9" >1.65381</td>
      <td id="T_c32ee_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_c32ee_row4_col0" class="data row4 col0" >5759</td>
      <td id="T_c32ee_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_c32ee_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_c32ee_row4_col3" class="data row4 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_c32ee_row4_col4" class="data row4 col4" >0.66269</td>
      <td id="T_c32ee_row4_col5" class="data row4 col5" >0.73091</td>
      <td id="T_c32ee_row4_col6" class="data row4 col6" >0.83733</td>
      <td id="T_c32ee_row4_col7" class="data row4 col7" >3</td>
      <td id="T_c32ee_row4_col8" class="data row4 col8" >1</td>
      <td id="T_c32ee_row4_col9" class="data row4 col9" >4.35856</td>
      <td id="T_c32ee_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_c32ee_row5_col0" class="data row5 col0" >5759</td>
      <td id="T_c32ee_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_c32ee_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_c32ee_row5_col3" class="data row5 col3" >IF capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_c32ee_row5_col4" class="data row5 col4" >0.91598</td>
      <td id="T_c32ee_row5_col5" class="data row5 col5" >0.95810</td>
      <td id="T_c32ee_row5_col6" class="data row5 col6" >0.79408</td>
      <td id="T_c32ee_row5_col7" class="data row5 col7" >1</td>
      <td id="T_c32ee_row5_col8" class="data row5 col8" >0</td>
      <td id="T_c32ee_row5_col9" class="data row5 col9" >208.22717</td>
      <td id="T_c32ee_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_c32ee_row6_col0" class="data row6 col0" >5759</td>
      <td id="T_c32ee_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_c32ee_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_c32ee_row6_col3" class="data row6 col3" >IF capital.gain <= 7063.5299 THEN class = <=50K</td>
      <td id="T_c32ee_row6_col4" class="data row6 col4" >0.95621</td>
      <td id="T_c32ee_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_c32ee_row6_col6" class="data row6 col6" >0.79306</td>
      <td id="T_c32ee_row6_col7" class="data row6 col7" >1</td>
      <td id="T_c32ee_row6_col8" class="data row6 col8" >0</td>
      <td id="T_c32ee_row6_col9" class="data row6 col9" >195.11769</td>
      <td id="T_c32ee_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_c32ee_row7_col0" class="data row7 col0" >5759</td>
      <td id="T_c32ee_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_c32ee_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_c32ee_row7_col3" class="data row7 col3" >IF capital.gain <= 4759.1385 THEN class = <=50K</td>
      <td id="T_c32ee_row7_col4" class="data row7 col4" >0.94735</td>
      <td id="T_c32ee_row7_col5" class="data row7 col5" >0.99324</td>
      <td id="T_c32ee_row7_col6" class="data row7 col6" >0.79594</td>
      <td id="T_c32ee_row7_col7" class="data row7 col7" >1</td>
      <td id="T_c32ee_row7_col8" class="data row7 col8" >0</td>
      <td id="T_c32ee_row7_col9" class="data row7 col9" >132.87754</td>
      <td id="T_c32ee_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_c32ee_row8_col0" class="data row8 col0" >5759</td>
      <td id="T_c32ee_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_c32ee_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_c32ee_row8_col3" class="data row8 col3" >IF capital.gain <= 3103.0 THEN class = <=50K</td>
      <td id="T_c32ee_row8_col4" class="data row8 col4" >0.93357</td>
      <td id="T_c32ee_row8_col5" class="data row8 col5" >0.97798</td>
      <td id="T_c32ee_row8_col6" class="data row8 col6" >0.79528</td>
      <td id="T_c32ee_row8_col7" class="data row8 col7" >1</td>
      <td id="T_c32ee_row8_col8" class="data row8 col8" >0</td>
      <td id="T_c32ee_row8_col9" class="data row8 col9" >146.14264</td>
      <td id="T_c32ee_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_c32ee_row9_col0" class="data row9 col0" >5759</td>
      <td id="T_c32ee_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_c32ee_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_c32ee_row9_col3" class="data row9 col3" >IF marital.status = Never-married THEN class = <=50K</td>
      <td id="T_c32ee_row9_col4" class="data row9 col4" >0.32757</td>
      <td id="T_c32ee_row9_col5" class="data row9 col5" >0.41178</td>
      <td id="T_c32ee_row9_col6" class="data row9 col6" >0.95433</td>
      <td id="T_c32ee_row9_col7" class="data row9 col7" >1</td>
      <td id="T_c32ee_row9_col8" class="data row9 col8" >0</td>
      <td id="T_c32ee_row9_col9" class="data row9 col9" >227.17949</td>
      <td id="T_c32ee_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_c32ee_row10_col0" class="data row10 col0" >5759</td>
      <td id="T_c32ee_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_c32ee_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_c32ee_row10_col3" class="data row10 col3" >IF relationship = Own-child THEN class = <=50K</td>
      <td id="T_c32ee_row10_col4" class="data row10 col4" >0.15554</td>
      <td id="T_c32ee_row10_col5" class="data row10 col5" >0.20262</td>
      <td id="T_c32ee_row10_col6" class="data row10 col6" >0.98900</td>
      <td id="T_c32ee_row10_col7" class="data row10 col7" >1</td>
      <td id="T_c32ee_row10_col8" class="data row10 col8" >0</td>
      <td id="T_c32ee_row10_col9" class="data row10 col9" >30.98065</td>
      <td id="T_c32ee_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_c32ee_row11_col0" class="data row11 col0" >5759</td>
      <td id="T_c32ee_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_c32ee_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_c32ee_row11_col3" class="data row11 col3" >IF age <= 86.5864 AND hours.per.week <= 76.1273 AND hours.per.week > 27.1607 THEN class = <=50K</td>
      <td id="T_c32ee_row11_col4" class="data row11 col4" >0.86469</td>
      <td id="T_c32ee_row11_col5" class="data row11 col5" >0.83789</td>
      <td id="T_c32ee_row11_col6" class="data row11 col6" >0.73564</td>
      <td id="T_c32ee_row11_col7" class="data row11 col7" >3</td>
      <td id="T_c32ee_row11_col8" class="data row11 col8" >0</td>
      <td id="T_c32ee_row11_col9" class="data row11 col9" >30.22322</td>
      <td id="T_c32ee_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_c32ee_row12_col0" class="data row12 col0" >5759</td>
      <td id="T_c32ee_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_c32ee_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_c32ee_row12_col3" class="data row12 col3" >IF age <= 45.5703 AND education != Doctorate AND hours.per.week <= 95.2543 AND occupation != Machine-op-inspct THEN class = <=50K</td>
      <td id="T_c32ee_row12_col4" class="data row12 col4" >0.65233</td>
      <td id="T_c32ee_row12_col5" class="data row12 col5" >0.68820</td>
      <td id="T_c32ee_row12_col6" class="data row12 col6" >0.80091</td>
      <td id="T_c32ee_row12_col7" class="data row12 col7" >4</td>
      <td id="T_c32ee_row12_col8" class="data row12 col8" >0</td>
      <td id="T_c32ee_row12_col9" class="data row12 col9" >31.41192</td>
      <td id="T_c32ee_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_c32ee_row13_col0" class="data row13 col0" >5759</td>
      <td id="T_c32ee_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_c32ee_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_c32ee_row13_col3" class="data row13 col3" >IF age <= 41.1272 AND education != Doctorate THEN class = <=50K</td>
      <td id="T_c32ee_row13_col4" class="data row13 col4" >0.61114</td>
      <td id="T_c32ee_row13_col5" class="data row13 col5" >0.67040</td>
      <td id="T_c32ee_row13_col6" class="data row13 col6" >0.83279</td>
      <td id="T_c32ee_row13_col7" class="data row13 col7" >2</td>
      <td id="T_c32ee_row13_col8" class="data row13 col8" >0</td>
      <td id="T_c32ee_row13_col9" class="data row13 col9" >33.98273</td>
      <td id="T_c32ee_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_c32ee_row14_col0" class="data row14 col0" >5759</td>
      <td id="T_c32ee_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_c32ee_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_c32ee_row14_col3" class="data row14 col3" >IF age <= 18.6366 THEN class = <=50K</td>
      <td id="T_c32ee_row14_col4" class="data row14 col4" >0.02918</td>
      <td id="T_c32ee_row14_col5" class="data row14 col5" >0.03843</td>
      <td id="T_c32ee_row14_col6" class="data row14 col6" >1.00000</td>
      <td id="T_c32ee_row14_col7" class="data row14 col7" >1</td>
      <td id="T_c32ee_row14_col8" class="data row14 col8" >0</td>
      <td id="T_c32ee_row14_col9" class="data row14 col9" >31.54394</td>
      <td id="T_c32ee_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_c32ee_row15_col0" class="data row15 col0" >5759</td>
      <td id="T_c32ee_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_c32ee_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_c32ee_row15_col3" class="data row15 col3" >IF age <= 35.5896 AND capital.gain <= 3809.9333 THEN class = <=50K</td>
      <td id="T_c32ee_row15_col4" class="data row15 col4" >0.44647</td>
      <td id="T_c32ee_row15_col5" class="data row15 col5" >0.53049</td>
      <td id="T_c32ee_row15_col6" class="data row15 col6" >0.90202</td>
      <td id="T_c32ee_row15_col7" class="data row15 col7" >2</td>
      <td id="T_c32ee_row15_col8" class="data row15 col8" >0</td>
      <td id="T_c32ee_row15_col9" class="data row15 col9" >2.73231</td>
      <td id="T_c32ee_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_c32ee_row16_col0" class="data row16 col0" >5759</td>
      <td id="T_c32ee_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_c32ee_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_c32ee_row16_col3" class="data row16 col3" >IF capital.gain <= 498.5037 AND hours.per.week <= 36.8253 THEN class = <=50K</td>
      <td id="T_c32ee_row16_col4" class="data row16 col4" >0.20683</td>
      <td id="T_c32ee_row16_col5" class="data row16 col5" >0.25279</td>
      <td id="T_c32ee_row16_col6" class="data row16 col6" >0.92787</td>
      <td id="T_c32ee_row16_col7" class="data row16 col7" >2</td>
      <td id="T_c32ee_row16_col8" class="data row16 col8" >0</td>
      <td id="T_c32ee_row16_col9" class="data row16 col9" >2.34009</td>
      <td id="T_c32ee_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_c32ee_row17_col0" class="data row17 col0" >5759</td>
      <td id="T_c32ee_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_c32ee_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_c32ee_row17_col3" class="data row17 col3" >IF capital.gain <= 4900.9955 AND hours.per.week <= 36.1513 THEN class = <=50K</td>
      <td id="T_c32ee_row17_col4" class="data row17 col4" >0.21415</td>
      <td id="T_c32ee_row17_col5" class="data row17 col5" >0.26175</td>
      <td id="T_c32ee_row17_col6" class="data row17 col6" >0.92788</td>
      <td id="T_c32ee_row17_col7" class="data row17 col7" >2</td>
      <td id="T_c32ee_row17_col8" class="data row17 col8" >0</td>
      <td id="T_c32ee_row17_col9" class="data row17 col9" >2.80984</td>
      <td id="T_c32ee_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_c32ee_row18_col0" class="data row18 col0" >5759</td>
      <td id="T_c32ee_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_c32ee_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_c32ee_row18_col3" class="data row18 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 32.0 THEN class = <=50K</td>
      <td id="T_c32ee_row18_col4" class="data row18 col4" >0.16199</td>
      <td id="T_c32ee_row18_col5" class="data row18 col5" >0.20095</td>
      <td id="T_c32ee_row18_col6" class="data row18 col6" >0.94177</td>
      <td id="T_c32ee_row18_col7" class="data row18 col7" >2</td>
      <td id="T_c32ee_row18_col8" class="data row18 col8" >0</td>
      <td id="T_c32ee_row18_col9" class="data row18 col9" >2.70685</td>
      <td id="T_c32ee_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_c32ee_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_c32ee_row19_col0" class="data row19 col0" >5759</td>
      <td id="T_c32ee_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_c32ee_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_c32ee_row19_col3" class="data row19 col3" >IF hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_c32ee_row19_col4" class="data row19 col4" >0.17208</td>
      <td id="T_c32ee_row19_col5" class="data row19 col5" >0.21043</td>
      <td id="T_c32ee_row19_col6" class="data row19 col6" >0.92835</td>
      <td id="T_c32ee_row19_col7" class="data row19 col7" >1</td>
      <td id="T_c32ee_row19_col8" class="data row19 col8" >0</td>
      <td id="T_c32ee_row19_col9" class="data row19 col9" >2.51105</td>
      <td id="T_c32ee_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5759, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_ff643">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ff643_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_ff643_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_ff643_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_ff643_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_ff643_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_ff643_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_ff643_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_ff643_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_ff643_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_ff643_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_ff643_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ff643_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_ff643_row0_col0" class="data row0 col0" >5759</td>
      <td id="T_ff643_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_ff643_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_ff643_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_ff643_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_ff643_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_ff643_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_ff643_row0_col7" class="data row0 col7" >3</td>
      <td id="T_ff643_row0_col8" class="data row0 col8" >0</td>
      <td id="T_ff643_row0_col9" class="data row0 col9" >2.13079</td>
      <td id="T_ff643_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_ff643_row1_col0" class="data row1 col0" >5759</td>
      <td id="T_ff643_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_ff643_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_ff643_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_ff643_row1_col4" class="data row1 col4" >0.54611</td>
      <td id="T_ff643_row1_col5" class="data row1 col5" >0.62509</td>
      <td id="T_ff643_row1_col6" class="data row1 col6" >0.86896</td>
      <td id="T_ff643_row1_col7" class="data row1 col7" >4</td>
      <td id="T_ff643_row1_col8" class="data row1 col8" >0</td>
      <td id="T_ff643_row1_col9" class="data row1 col9" >2.29826</td>
      <td id="T_ff643_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_ff643_row2_col0" class="data row2 col0" >5759</td>
      <td id="T_ff643_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_ff643_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_ff643_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_ff643_row2_col4" class="data row2 col4" >0.63088</td>
      <td id="T_ff643_row2_col5" class="data row2 col5" >0.71733</td>
      <td id="T_ff643_row2_col6" class="data row2 col6" >0.86320</td>
      <td id="T_ff643_row2_col7" class="data row2 col7" >3</td>
      <td id="T_ff643_row2_col8" class="data row2 col8" >0</td>
      <td id="T_ff643_row2_col9" class="data row2 col9" >2.07695</td>
      <td id="T_ff643_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_ff643_row3_col0" class="data row3 col0" >5759</td>
      <td id="T_ff643_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_ff643_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_ff643_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_ff643_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_ff643_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_ff643_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_ff643_row3_col7" class="data row3 col7" >2</td>
      <td id="T_ff643_row3_col8" class="data row3 col8" >0</td>
      <td id="T_ff643_row3_col9" class="data row3 col9" >1.65381</td>
      <td id="T_ff643_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_ff643_row4_col0" class="data row4 col0" >5759</td>
      <td id="T_ff643_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_ff643_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_ff643_row4_col3" class="data row4 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_ff643_row4_col4" class="data row4 col4" >0.66269</td>
      <td id="T_ff643_row4_col5" class="data row4 col5" >0.73091</td>
      <td id="T_ff643_row4_col6" class="data row4 col6" >0.83733</td>
      <td id="T_ff643_row4_col7" class="data row4 col7" >3</td>
      <td id="T_ff643_row4_col8" class="data row4 col8" >1</td>
      <td id="T_ff643_row4_col9" class="data row4 col9" >4.35856</td>
      <td id="T_ff643_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_ff643_row5_col0" class="data row5 col0" >5759</td>
      <td id="T_ff643_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_ff643_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_ff643_row5_col3" class="data row5 col3" >IF capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_ff643_row5_col4" class="data row5 col4" >0.91598</td>
      <td id="T_ff643_row5_col5" class="data row5 col5" >0.95810</td>
      <td id="T_ff643_row5_col6" class="data row5 col6" >0.79408</td>
      <td id="T_ff643_row5_col7" class="data row5 col7" >1</td>
      <td id="T_ff643_row5_col8" class="data row5 col8" >0</td>
      <td id="T_ff643_row5_col9" class="data row5 col9" >208.22717</td>
      <td id="T_ff643_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_ff643_row6_col0" class="data row6 col0" >5759</td>
      <td id="T_ff643_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_ff643_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_ff643_row6_col3" class="data row6 col3" >IF capital.gain <= 7063.5299 THEN class = <=50K</td>
      <td id="T_ff643_row6_col4" class="data row6 col4" >0.95621</td>
      <td id="T_ff643_row6_col5" class="data row6 col5" >0.99890</td>
      <td id="T_ff643_row6_col6" class="data row6 col6" >0.79306</td>
      <td id="T_ff643_row6_col7" class="data row6 col7" >1</td>
      <td id="T_ff643_row6_col8" class="data row6 col8" >0</td>
      <td id="T_ff643_row6_col9" class="data row6 col9" >195.11769</td>
      <td id="T_ff643_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_ff643_row7_col0" class="data row7 col0" >5759</td>
      <td id="T_ff643_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_ff643_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_ff643_row7_col3" class="data row7 col3" >IF capital.gain <= 4759.1385 THEN class = <=50K</td>
      <td id="T_ff643_row7_col4" class="data row7 col4" >0.94735</td>
      <td id="T_ff643_row7_col5" class="data row7 col5" >0.99324</td>
      <td id="T_ff643_row7_col6" class="data row7 col6" >0.79594</td>
      <td id="T_ff643_row7_col7" class="data row7 col7" >1</td>
      <td id="T_ff643_row7_col8" class="data row7 col8" >0</td>
      <td id="T_ff643_row7_col9" class="data row7 col9" >132.87754</td>
      <td id="T_ff643_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_ff643_row8_col0" class="data row8 col0" >5759</td>
      <td id="T_ff643_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_ff643_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_ff643_row8_col3" class="data row8 col3" >IF capital.gain <= 3103.0 THEN class = <=50K</td>
      <td id="T_ff643_row8_col4" class="data row8 col4" >0.93357</td>
      <td id="T_ff643_row8_col5" class="data row8 col5" >0.97798</td>
      <td id="T_ff643_row8_col6" class="data row8 col6" >0.79528</td>
      <td id="T_ff643_row8_col7" class="data row8 col7" >1</td>
      <td id="T_ff643_row8_col8" class="data row8 col8" >0</td>
      <td id="T_ff643_row8_col9" class="data row8 col9" >146.14264</td>
      <td id="T_ff643_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_ff643_row9_col0" class="data row9 col0" >5759</td>
      <td id="T_ff643_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_ff643_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_ff643_row9_col3" class="data row9 col3" >IF marital.status = Never-married THEN class = <=50K</td>
      <td id="T_ff643_row9_col4" class="data row9 col4" >0.32757</td>
      <td id="T_ff643_row9_col5" class="data row9 col5" >0.41178</td>
      <td id="T_ff643_row9_col6" class="data row9 col6" >0.95433</td>
      <td id="T_ff643_row9_col7" class="data row9 col7" >1</td>
      <td id="T_ff643_row9_col8" class="data row9 col8" >0</td>
      <td id="T_ff643_row9_col9" class="data row9 col9" >227.17949</td>
      <td id="T_ff643_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_ff643_row10_col0" class="data row10 col0" >5759</td>
      <td id="T_ff643_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_ff643_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_ff643_row10_col3" class="data row10 col3" >IF relationship = Own-child THEN class = <=50K</td>
      <td id="T_ff643_row10_col4" class="data row10 col4" >0.15554</td>
      <td id="T_ff643_row10_col5" class="data row10 col5" >0.20262</td>
      <td id="T_ff643_row10_col6" class="data row10 col6" >0.98900</td>
      <td id="T_ff643_row10_col7" class="data row10 col7" >1</td>
      <td id="T_ff643_row10_col8" class="data row10 col8" >0</td>
      <td id="T_ff643_row10_col9" class="data row10 col9" >30.98065</td>
      <td id="T_ff643_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_ff643_row11_col0" class="data row11 col0" >5759</td>
      <td id="T_ff643_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_ff643_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_ff643_row11_col3" class="data row11 col3" >IF age <= 86.5864 AND hours.per.week <= 76.1273 AND hours.per.week > 27.1607 THEN class = <=50K</td>
      <td id="T_ff643_row11_col4" class="data row11 col4" >0.86469</td>
      <td id="T_ff643_row11_col5" class="data row11 col5" >0.83789</td>
      <td id="T_ff643_row11_col6" class="data row11 col6" >0.73564</td>
      <td id="T_ff643_row11_col7" class="data row11 col7" >3</td>
      <td id="T_ff643_row11_col8" class="data row11 col8" >0</td>
      <td id="T_ff643_row11_col9" class="data row11 col9" >30.22322</td>
      <td id="T_ff643_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_ff643_row12_col0" class="data row12 col0" >5759</td>
      <td id="T_ff643_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_ff643_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_ff643_row12_col3" class="data row12 col3" >IF age <= 45.5703 AND education != Doctorate AND hours.per.week <= 95.2543 AND occupation != Machine-op-inspct THEN class = <=50K</td>
      <td id="T_ff643_row12_col4" class="data row12 col4" >0.65233</td>
      <td id="T_ff643_row12_col5" class="data row12 col5" >0.68820</td>
      <td id="T_ff643_row12_col6" class="data row12 col6" >0.80091</td>
      <td id="T_ff643_row12_col7" class="data row12 col7" >4</td>
      <td id="T_ff643_row12_col8" class="data row12 col8" >0</td>
      <td id="T_ff643_row12_col9" class="data row12 col9" >31.41192</td>
      <td id="T_ff643_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_ff643_row13_col0" class="data row13 col0" >5759</td>
      <td id="T_ff643_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_ff643_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_ff643_row13_col3" class="data row13 col3" >IF age <= 41.1272 AND education != Doctorate THEN class = <=50K</td>
      <td id="T_ff643_row13_col4" class="data row13 col4" >0.61114</td>
      <td id="T_ff643_row13_col5" class="data row13 col5" >0.67040</td>
      <td id="T_ff643_row13_col6" class="data row13 col6" >0.83279</td>
      <td id="T_ff643_row13_col7" class="data row13 col7" >2</td>
      <td id="T_ff643_row13_col8" class="data row13 col8" >0</td>
      <td id="T_ff643_row13_col9" class="data row13 col9" >33.98273</td>
      <td id="T_ff643_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_ff643_row14_col0" class="data row14 col0" >5759</td>
      <td id="T_ff643_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_ff643_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_ff643_row14_col3" class="data row14 col3" >IF age <= 18.6366 THEN class = <=50K</td>
      <td id="T_ff643_row14_col4" class="data row14 col4" >0.02918</td>
      <td id="T_ff643_row14_col5" class="data row14 col5" >0.03843</td>
      <td id="T_ff643_row14_col6" class="data row14 col6" >1.00000</td>
      <td id="T_ff643_row14_col7" class="data row14 col7" >1</td>
      <td id="T_ff643_row14_col8" class="data row14 col8" >0</td>
      <td id="T_ff643_row14_col9" class="data row14 col9" >31.54394</td>
      <td id="T_ff643_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_ff643_row15_col0" class="data row15 col0" >5759</td>
      <td id="T_ff643_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_ff643_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_ff643_row15_col3" class="data row15 col3" >IF age <= 35.5896 AND capital.gain <= 3809.9333 THEN class = <=50K</td>
      <td id="T_ff643_row15_col4" class="data row15 col4" >0.44647</td>
      <td id="T_ff643_row15_col5" class="data row15 col5" >0.53049</td>
      <td id="T_ff643_row15_col6" class="data row15 col6" >0.90202</td>
      <td id="T_ff643_row15_col7" class="data row15 col7" >2</td>
      <td id="T_ff643_row15_col8" class="data row15 col8" >0</td>
      <td id="T_ff643_row15_col9" class="data row15 col9" >2.73231</td>
      <td id="T_ff643_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_ff643_row16_col0" class="data row16 col0" >5759</td>
      <td id="T_ff643_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_ff643_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_ff643_row16_col3" class="data row16 col3" >IF capital.gain <= 498.5037 AND hours.per.week <= 36.8253 THEN class = <=50K</td>
      <td id="T_ff643_row16_col4" class="data row16 col4" >0.20683</td>
      <td id="T_ff643_row16_col5" class="data row16 col5" >0.25279</td>
      <td id="T_ff643_row16_col6" class="data row16 col6" >0.92787</td>
      <td id="T_ff643_row16_col7" class="data row16 col7" >2</td>
      <td id="T_ff643_row16_col8" class="data row16 col8" >0</td>
      <td id="T_ff643_row16_col9" class="data row16 col9" >2.34009</td>
      <td id="T_ff643_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_ff643_row17_col0" class="data row17 col0" >5759</td>
      <td id="T_ff643_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_ff643_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_ff643_row17_col3" class="data row17 col3" >IF capital.gain <= 4900.9955 AND hours.per.week <= 36.1513 THEN class = <=50K</td>
      <td id="T_ff643_row17_col4" class="data row17 col4" >0.21415</td>
      <td id="T_ff643_row17_col5" class="data row17 col5" >0.26175</td>
      <td id="T_ff643_row17_col6" class="data row17 col6" >0.92788</td>
      <td id="T_ff643_row17_col7" class="data row17 col7" >2</td>
      <td id="T_ff643_row17_col8" class="data row17 col8" >0</td>
      <td id="T_ff643_row17_col9" class="data row17 col9" >2.80984</td>
      <td id="T_ff643_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_ff643_row18_col0" class="data row18 col0" >5759</td>
      <td id="T_ff643_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_ff643_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_ff643_row18_col3" class="data row18 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 32.0 THEN class = <=50K</td>
      <td id="T_ff643_row18_col4" class="data row18 col4" >0.16199</td>
      <td id="T_ff643_row18_col5" class="data row18 col5" >0.20095</td>
      <td id="T_ff643_row18_col6" class="data row18 col6" >0.94177</td>
      <td id="T_ff643_row18_col7" class="data row18 col7" >2</td>
      <td id="T_ff643_row18_col8" class="data row18 col8" >0</td>
      <td id="T_ff643_row18_col9" class="data row18 col9" >2.70685</td>
      <td id="T_ff643_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_ff643_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_ff643_row19_col0" class="data row19 col0" >5759</td>
      <td id="T_ff643_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_ff643_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_ff643_row19_col3" class="data row19 col3" >IF hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_ff643_row19_col4" class="data row19 col4" >0.17208</td>
      <td id="T_ff643_row19_col5" class="data row19 col5" >0.21043</td>
      <td id="T_ff643_row19_col6" class="data row19 col6" >0.92835</td>
      <td id="T_ff643_row19_col7" class="data row19 col7" >1</td>
      <td id="T_ff643_row19_col8" class="data row19 col8" >0</td>
      <td id="T_ff643_row19_col9" class="data row19 col9" >2.51105</td>
      <td id="T_ff643_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5759, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.95621, Pre: 1.0)



<style type="text/css">
#T_a2fbd_row5_col0, #T_a2fbd_row5_col1, #T_a2fbd_row5_col2, #T_a2fbd_row5_col3, #T_a2fbd_row5_col4, #T_a2fbd_row5_col5, #T_a2fbd_row5_col6, #T_a2fbd_row5_col7, #T_a2fbd_row5_col8, #T_a2fbd_row5_col9, #T_a2fbd_row5_col10, #T_a2fbd_row5_col11 {
  font-weight: bold;
}
</style>
<table id="T_a2fbd">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a2fbd_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_a2fbd_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_a2fbd_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_a2fbd_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_a2fbd_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_a2fbd_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_a2fbd_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_a2fbd_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_a2fbd_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_a2fbd_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_a2fbd_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_a2fbd_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a2fbd_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a2fbd_row0_col0" class="data row0 col0" >5759</td>
      <td id="T_a2fbd_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_a2fbd_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_a2fbd_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_a2fbd_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_a2fbd_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_a2fbd_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_a2fbd_row0_col7" class="data row0 col7" >3</td>
      <td id="T_a2fbd_row0_col8" class="data row0 col8" >0</td>
      <td id="T_a2fbd_row0_col9" class="data row0 col9" >2.13079</td>
      <td id="T_a2fbd_row0_col10" class="data row0 col10" >False</td>
      <td id="T_a2fbd_row0_col11" class="data row0 col11" >0.30354</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a2fbd_row1_col0" class="data row1 col0" >5759</td>
      <td id="T_a2fbd_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_a2fbd_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_a2fbd_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_a2fbd_row1_col4" class="data row1 col4" >0.54611</td>
      <td id="T_a2fbd_row1_col5" class="data row1 col5" >0.62509</td>
      <td id="T_a2fbd_row1_col6" class="data row1 col6" >0.86896</td>
      <td id="T_a2fbd_row1_col7" class="data row1 col7" >4</td>
      <td id="T_a2fbd_row1_col8" class="data row1 col8" >0</td>
      <td id="T_a2fbd_row1_col9" class="data row1 col9" >2.29826</td>
      <td id="T_a2fbd_row1_col10" class="data row1 col10" >False</td>
      <td id="T_a2fbd_row1_col11" class="data row1 col11" >0.43053</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a2fbd_row2_col0" class="data row2 col0" >5759</td>
      <td id="T_a2fbd_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_a2fbd_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_a2fbd_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_a2fbd_row2_col4" class="data row2 col4" >0.63088</td>
      <td id="T_a2fbd_row2_col5" class="data row2 col5" >0.71733</td>
      <td id="T_a2fbd_row2_col6" class="data row2 col6" >0.86320</td>
      <td id="T_a2fbd_row2_col7" class="data row2 col7" >3</td>
      <td id="T_a2fbd_row2_col8" class="data row2 col8" >0</td>
      <td id="T_a2fbd_row2_col9" class="data row2 col9" >2.07695</td>
      <td id="T_a2fbd_row2_col10" class="data row2 col10" >False</td>
      <td id="T_a2fbd_row2_col11" class="data row2 col11" >0.35292</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a2fbd_row3_col0" class="data row3 col0" >5759</td>
      <td id="T_a2fbd_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_a2fbd_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_a2fbd_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_a2fbd_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_a2fbd_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_a2fbd_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_a2fbd_row3_col7" class="data row3 col7" >2</td>
      <td id="T_a2fbd_row3_col8" class="data row3 col8" >0</td>
      <td id="T_a2fbd_row3_col9" class="data row3 col9" >1.65381</td>
      <td id="T_a2fbd_row3_col10" class="data row3 col10" >False</td>
      <td id="T_a2fbd_row3_col11" class="data row3 col11" >0.48353</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_a2fbd_row4_col0" class="data row4 col0" >5759</td>
      <td id="T_a2fbd_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_a2fbd_row4_col2" class="data row4 col2" >LORE2</td>
      <td id="T_a2fbd_row4_col3" class="data row4 col3" >IF capital.gain <= 7063.5299 THEN class = <=50K</td>
      <td id="T_a2fbd_row4_col4" class="data row4 col4" >0.95621</td>
      <td id="T_a2fbd_row4_col5" class="data row4 col5" >0.99890</td>
      <td id="T_a2fbd_row4_col6" class="data row4 col6" >0.79306</td>
      <td id="T_a2fbd_row4_col7" class="data row4 col7" >1</td>
      <td id="T_a2fbd_row4_col8" class="data row4 col8" >0</td>
      <td id="T_a2fbd_row4_col9" class="data row4 col9" >195.11769</td>
      <td id="T_a2fbd_row4_col10" class="data row4 col10" >False</td>
      <td id="T_a2fbd_row4_col11" class="data row4 col11" >0.20694</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_a2fbd_row5_col0" class="data row5 col0" >5759</td>
      <td id="T_a2fbd_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_a2fbd_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_a2fbd_row5_col3" class="data row5 col3" >IF capital.gain <= 4759.1385 THEN class = <=50K</td>
      <td id="T_a2fbd_row5_col4" class="data row5 col4" >0.94735</td>
      <td id="T_a2fbd_row5_col5" class="data row5 col5" >0.99324</td>
      <td id="T_a2fbd_row5_col6" class="data row5 col6" >0.79594</td>
      <td id="T_a2fbd_row5_col7" class="data row5 col7" >1</td>
      <td id="T_a2fbd_row5_col8" class="data row5 col8" >0</td>
      <td id="T_a2fbd_row5_col9" class="data row5 col9" >132.87754</td>
      <td id="T_a2fbd_row5_col10" class="data row5 col10" >False</td>
      <td id="T_a2fbd_row5_col11" class="data row5 col11" >0.20425</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_a2fbd_row6_col0" class="data row6 col0" >5759</td>
      <td id="T_a2fbd_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_a2fbd_row6_col2" class="data row6 col2" >LORE5</td>
      <td id="T_a2fbd_row6_col3" class="data row6 col3" >IF marital.status = Never-married THEN class = <=50K</td>
      <td id="T_a2fbd_row6_col4" class="data row6 col4" >0.32757</td>
      <td id="T_a2fbd_row6_col5" class="data row6 col5" >0.41178</td>
      <td id="T_a2fbd_row6_col6" class="data row6 col6" >0.95433</td>
      <td id="T_a2fbd_row6_col7" class="data row6 col7" >1</td>
      <td id="T_a2fbd_row6_col8" class="data row6 col8" >0</td>
      <td id="T_a2fbd_row6_col9" class="data row6 col9" >227.17949</td>
      <td id="T_a2fbd_row6_col10" class="data row6 col10" >False</td>
      <td id="T_a2fbd_row6_col11" class="data row6 col11" >0.63030</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_a2fbd_row7_col0" class="data row7 col0" >5759</td>
      <td id="T_a2fbd_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_a2fbd_row7_col2" class="data row7 col2" >LORE_SA1</td>
      <td id="T_a2fbd_row7_col3" class="data row7 col3" >IF relationship = Own-child THEN class = <=50K</td>
      <td id="T_a2fbd_row7_col4" class="data row7 col4" >0.15554</td>
      <td id="T_a2fbd_row7_col5" class="data row7 col5" >0.20262</td>
      <td id="T_a2fbd_row7_col6" class="data row7 col6" >0.98900</td>
      <td id="T_a2fbd_row7_col7" class="data row7 col7" >1</td>
      <td id="T_a2fbd_row7_col8" class="data row7 col8" >0</td>
      <td id="T_a2fbd_row7_col9" class="data row7 col9" >30.98065</td>
      <td id="T_a2fbd_row7_col10" class="data row7 col10" >False</td>
      <td id="T_a2fbd_row7_col11" class="data row7 col11" >0.80075</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_a2fbd_row8_col0" class="data row8 col0" >5759</td>
      <td id="T_a2fbd_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_a2fbd_row8_col2" class="data row8 col2" >LORE_SA5</td>
      <td id="T_a2fbd_row8_col3" class="data row8 col3" >IF age <= 18.6366 THEN class = <=50K</td>
      <td id="T_a2fbd_row8_col4" class="data row8 col4" >0.02918</td>
      <td id="T_a2fbd_row8_col5" class="data row8 col5" >0.03843</td>
      <td id="T_a2fbd_row8_col6" class="data row8 col6" >1.00000</td>
      <td id="T_a2fbd_row8_col7" class="data row8 col7" >1</td>
      <td id="T_a2fbd_row8_col8" class="data row8 col8" >0</td>
      <td id="T_a2fbd_row8_col9" class="data row8 col9" >31.54394</td>
      <td id="T_a2fbd_row8_col10" class="data row8 col10" >False</td>
      <td id="T_a2fbd_row8_col11" class="data row8 col11" >0.92703</td>
    </tr>
    <tr>
      <th id="T_a2fbd_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_a2fbd_row9_col0" class="data row9 col0" >5759</td>
      <td id="T_a2fbd_row9_col1" class="data row9 col1" >EXPLAN</td>
      <td id="T_a2fbd_row9_col2" class="data row9 col2" >EXPLAN1</td>
      <td id="T_a2fbd_row9_col3" class="data row9 col3" >IF age <= 35.5896 AND capital.gain <= 3809.9333 THEN class = <=50K</td>
      <td id="T_a2fbd_row9_col4" class="data row9 col4" >0.44647</td>
      <td id="T_a2fbd_row9_col5" class="data row9 col5" >0.53049</td>
      <td id="T_a2fbd_row9_col6" class="data row9 col6" >0.90202</td>
      <td id="T_a2fbd_row9_col7" class="data row9 col7" >2</td>
      <td id="T_a2fbd_row9_col8" class="data row9 col8" >0</td>
      <td id="T_a2fbd_row9_col9" class="data row9 col9" >2.73231</td>
      <td id="T_a2fbd_row9_col10" class="data row9 col10" >False</td>
      <td id="T_a2fbd_row9_col11" class="data row9 col11" >0.51907</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_115.png)
    



### Rules for Instance 5759, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.95621, Pre: 1.0), Unique rules (diffrent features)



<style type="text/css">
#T_254df_row3_col0, #T_254df_row3_col1, #T_254df_row3_col2, #T_254df_row3_col3, #T_254df_row3_col4, #T_254df_row3_col5, #T_254df_row3_col6, #T_254df_row3_col7, #T_254df_row3_col8, #T_254df_row3_col9, #T_254df_row3_col10, #T_254df_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_254df">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_254df_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_254df_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_254df_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_254df_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_254df_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_254df_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_254df_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_254df_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_254df_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_254df_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_254df_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_254df_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_254df_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_254df_row0_col0" class="data row0 col0" >5759</td>
      <td id="T_254df_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_254df_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_254df_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_254df_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_254df_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_254df_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_254df_row0_col7" class="data row0 col7" >3</td>
      <td id="T_254df_row0_col8" class="data row0 col8" >0</td>
      <td id="T_254df_row0_col9" class="data row0 col9" >2.13079</td>
      <td id="T_254df_row0_col10" class="data row0 col10" >False</td>
      <td id="T_254df_row0_col11" class="data row0 col11" >0.30354</td>
    </tr>
    <tr>
      <th id="T_254df_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_254df_row1_col0" class="data row1 col0" >5759</td>
      <td id="T_254df_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_254df_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_254df_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_254df_row1_col4" class="data row1 col4" >0.54611</td>
      <td id="T_254df_row1_col5" class="data row1 col5" >0.62509</td>
      <td id="T_254df_row1_col6" class="data row1 col6" >0.86896</td>
      <td id="T_254df_row1_col7" class="data row1 col7" >4</td>
      <td id="T_254df_row1_col8" class="data row1 col8" >0</td>
      <td id="T_254df_row1_col9" class="data row1 col9" >2.29826</td>
      <td id="T_254df_row1_col10" class="data row1 col10" >False</td>
      <td id="T_254df_row1_col11" class="data row1 col11" >0.43053</td>
    </tr>
    <tr>
      <th id="T_254df_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_254df_row2_col0" class="data row2 col0" >5759</td>
      <td id="T_254df_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_254df_row2_col2" class="data row2 col2" >ANCHOR4</td>
      <td id="T_254df_row2_col3" class="data row2 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_254df_row2_col4" class="data row2 col4" >0.48710</td>
      <td id="T_254df_row2_col5" class="data row2 col5" >0.56643</td>
      <td id="T_254df_row2_col6" class="data row2 col6" >0.88281</td>
      <td id="T_254df_row2_col7" class="data row2 col7" >2</td>
      <td id="T_254df_row2_col8" class="data row2 col8" >0</td>
      <td id="T_254df_row2_col9" class="data row2 col9" >1.65381</td>
      <td id="T_254df_row2_col10" class="data row2 col10" >False</td>
      <td id="T_254df_row2_col11" class="data row2 col11" >0.48353</td>
    </tr>
    <tr>
      <th id="T_254df_level0_row3" class="row_heading level0 row3" >5</th>
      <td id="T_254df_row3_col0" class="data row3 col0" >5759</td>
      <td id="T_254df_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_254df_row3_col2" class="data row3 col2" >LORE3</td>
      <td id="T_254df_row3_col3" class="data row3 col3" >IF capital.gain <= 4759.1385 THEN class = <=50K</td>
      <td id="T_254df_row3_col4" class="data row3 col4" >0.94735</td>
      <td id="T_254df_row3_col5" class="data row3 col5" >0.99324</td>
      <td id="T_254df_row3_col6" class="data row3 col6" >0.79594</td>
      <td id="T_254df_row3_col7" class="data row3 col7" >1</td>
      <td id="T_254df_row3_col8" class="data row3 col8" >0</td>
      <td id="T_254df_row3_col9" class="data row3 col9" >132.87754</td>
      <td id="T_254df_row3_col10" class="data row3 col10" >False</td>
      <td id="T_254df_row3_col11" class="data row3 col11" >0.20425</td>
    </tr>
    <tr>
      <th id="T_254df_level0_row4" class="row_heading level0 row4" >6</th>
      <td id="T_254df_row4_col0" class="data row4 col0" >5759</td>
      <td id="T_254df_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_254df_row4_col2" class="data row4 col2" >LORE5</td>
      <td id="T_254df_row4_col3" class="data row4 col3" >IF marital.status = Never-married THEN class = <=50K</td>
      <td id="T_254df_row4_col4" class="data row4 col4" >0.32757</td>
      <td id="T_254df_row4_col5" class="data row4 col5" >0.41178</td>
      <td id="T_254df_row4_col6" class="data row4 col6" >0.95433</td>
      <td id="T_254df_row4_col7" class="data row4 col7" >1</td>
      <td id="T_254df_row4_col8" class="data row4 col8" >0</td>
      <td id="T_254df_row4_col9" class="data row4 col9" >227.17949</td>
      <td id="T_254df_row4_col10" class="data row4 col10" >False</td>
      <td id="T_254df_row4_col11" class="data row4 col11" >0.63030</td>
    </tr>
    <tr>
      <th id="T_254df_level0_row5" class="row_heading level0 row5" >7</th>
      <td id="T_254df_row5_col0" class="data row5 col0" >5759</td>
      <td id="T_254df_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_254df_row5_col2" class="data row5 col2" >LORE_SA1</td>
      <td id="T_254df_row5_col3" class="data row5 col3" >IF relationship = Own-child THEN class = <=50K</td>
      <td id="T_254df_row5_col4" class="data row5 col4" >0.15554</td>
      <td id="T_254df_row5_col5" class="data row5 col5" >0.20262</td>
      <td id="T_254df_row5_col6" class="data row5 col6" >0.98900</td>
      <td id="T_254df_row5_col7" class="data row5 col7" >1</td>
      <td id="T_254df_row5_col8" class="data row5 col8" >0</td>
      <td id="T_254df_row5_col9" class="data row5 col9" >30.98065</td>
      <td id="T_254df_row5_col10" class="data row5 col10" >False</td>
      <td id="T_254df_row5_col11" class="data row5 col11" >0.80075</td>
    </tr>
    <tr>
      <th id="T_254df_level0_row6" class="row_heading level0 row6" >8</th>
      <td id="T_254df_row6_col0" class="data row6 col0" >5759</td>
      <td id="T_254df_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_254df_row6_col2" class="data row6 col2" >LORE_SA5</td>
      <td id="T_254df_row6_col3" class="data row6 col3" >IF age <= 18.6366 THEN class = <=50K</td>
      <td id="T_254df_row6_col4" class="data row6 col4" >0.02918</td>
      <td id="T_254df_row6_col5" class="data row6 col5" >0.03843</td>
      <td id="T_254df_row6_col6" class="data row6 col6" >1.00000</td>
      <td id="T_254df_row6_col7" class="data row6 col7" >1</td>
      <td id="T_254df_row6_col8" class="data row6 col8" >0</td>
      <td id="T_254df_row6_col9" class="data row6 col9" >31.54394</td>
      <td id="T_254df_row6_col10" class="data row6 col10" >False</td>
      <td id="T_254df_row6_col11" class="data row6 col11" >0.92703</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_118.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_119.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_120.png)
    



### Rules for Instance 5759, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.9989, Pre: 1.0, Len: 0.79306)



<style type="text/css">
#T_744f8_row5_col0, #T_744f8_row5_col1, #T_744f8_row5_col2, #T_744f8_row5_col3, #T_744f8_row5_col4, #T_744f8_row5_col5, #T_744f8_row5_col6, #T_744f8_row5_col7, #T_744f8_row5_col8, #T_744f8_row5_col9, #T_744f8_row5_col10, #T_744f8_row5_col11 {
  font-weight: bold;
}
</style>
<table id="T_744f8">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_744f8_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_744f8_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_744f8_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_744f8_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_744f8_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_744f8_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_744f8_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_744f8_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_744f8_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_744f8_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_744f8_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_744f8_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_744f8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_744f8_row0_col0" class="data row0 col0" >5759</td>
      <td id="T_744f8_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_744f8_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_744f8_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_744f8_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_744f8_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_744f8_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_744f8_row0_col7" class="data row0 col7" >3</td>
      <td id="T_744f8_row0_col8" class="data row0 col8" >0</td>
      <td id="T_744f8_row0_col9" class="data row0 col9" >2.13079</td>
      <td id="T_744f8_row0_col10" class="data row0 col10" >False</td>
      <td id="T_744f8_row0_col11" class="data row0 col11" >2.22341</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_744f8_row1_col0" class="data row1 col0" >5759</td>
      <td id="T_744f8_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_744f8_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_744f8_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_744f8_row1_col4" class="data row1 col4" >0.54611</td>
      <td id="T_744f8_row1_col5" class="data row1 col5" >0.62509</td>
      <td id="T_744f8_row1_col6" class="data row1 col6" >0.86896</td>
      <td id="T_744f8_row1_col7" class="data row1 col7" >4</td>
      <td id="T_744f8_row1_col8" class="data row1 col8" >0</td>
      <td id="T_744f8_row1_col9" class="data row1 col9" >2.29826</td>
      <td id="T_744f8_row1_col10" class="data row1 col10" >False</td>
      <td id="T_744f8_row1_col11" class="data row1 col11" >3.23131</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_744f8_row2_col0" class="data row2 col0" >5759</td>
      <td id="T_744f8_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_744f8_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_744f8_row2_col3" class="data row2 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_744f8_row2_col4" class="data row2 col4" >0.63088</td>
      <td id="T_744f8_row2_col5" class="data row2 col5" >0.71733</td>
      <td id="T_744f8_row2_col6" class="data row2 col6" >0.86320</td>
      <td id="T_744f8_row2_col7" class="data row2 col7" >3</td>
      <td id="T_744f8_row2_col8" class="data row2 col8" >0</td>
      <td id="T_744f8_row2_col9" class="data row2 col9" >2.07695</td>
      <td id="T_744f8_row2_col10" class="data row2 col10" >False</td>
      <td id="T_744f8_row2_col11" class="data row2 col11" >2.22903</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_744f8_row3_col0" class="data row3 col0" >5759</td>
      <td id="T_744f8_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_744f8_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_744f8_row3_col3" class="data row3 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_744f8_row3_col4" class="data row3 col4" >0.48710</td>
      <td id="T_744f8_row3_col5" class="data row3 col5" >0.56643</td>
      <td id="T_744f8_row3_col6" class="data row3 col6" >0.88281</td>
      <td id="T_744f8_row3_col7" class="data row3 col7" >2</td>
      <td id="T_744f8_row3_col8" class="data row3 col8" >0</td>
      <td id="T_744f8_row3_col9" class="data row3 col9" >1.65381</td>
      <td id="T_744f8_row3_col10" class="data row3 col10" >False</td>
      <td id="T_744f8_row3_col11" class="data row3 col11" >1.28743</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_744f8_row4_col0" class="data row4 col0" >5759</td>
      <td id="T_744f8_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_744f8_row4_col2" class="data row4 col2" >LORE2</td>
      <td id="T_744f8_row4_col3" class="data row4 col3" >IF capital.gain <= 7063.5299 THEN class = <=50K</td>
      <td id="T_744f8_row4_col4" class="data row4 col4" >0.95621</td>
      <td id="T_744f8_row4_col5" class="data row4 col5" >0.99890</td>
      <td id="T_744f8_row4_col6" class="data row4 col6" >0.79306</td>
      <td id="T_744f8_row4_col7" class="data row4 col7" >1</td>
      <td id="T_744f8_row4_col8" class="data row4 col8" >0</td>
      <td id="T_744f8_row4_col9" class="data row4 col9" >195.11769</td>
      <td id="T_744f8_row4_col10" class="data row4 col10" >False</td>
      <td id="T_744f8_row4_col11" class="data row4 col11" >0.29266</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_744f8_row5_col0" class="data row5 col0" >5759</td>
      <td id="T_744f8_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_744f8_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_744f8_row5_col3" class="data row5 col3" >IF capital.gain <= 4759.1385 THEN class = <=50K</td>
      <td id="T_744f8_row5_col4" class="data row5 col4" >0.94735</td>
      <td id="T_744f8_row5_col5" class="data row5 col5" >0.99324</td>
      <td id="T_744f8_row5_col6" class="data row5 col6" >0.79594</td>
      <td id="T_744f8_row5_col7" class="data row5 col7" >1</td>
      <td id="T_744f8_row5_col8" class="data row5 col8" >0</td>
      <td id="T_744f8_row5_col9" class="data row5 col9" >132.87754</td>
      <td id="T_744f8_row5_col10" class="data row5 col10" >False</td>
      <td id="T_744f8_row5_col11" class="data row5 col11" >0.29068</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_744f8_row6_col0" class="data row6 col0" >5759</td>
      <td id="T_744f8_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_744f8_row6_col2" class="data row6 col2" >LORE5</td>
      <td id="T_744f8_row6_col3" class="data row6 col3" >IF marital.status = Never-married THEN class = <=50K</td>
      <td id="T_744f8_row6_col4" class="data row6 col4" >0.32757</td>
      <td id="T_744f8_row6_col5" class="data row6 col5" >0.41178</td>
      <td id="T_744f8_row6_col6" class="data row6 col6" >0.95433</td>
      <td id="T_744f8_row6_col7" class="data row6 col7" >1</td>
      <td id="T_744f8_row6_col8" class="data row6 col8" >0</td>
      <td id="T_744f8_row6_col9" class="data row6 col9" >227.17949</td>
      <td id="T_744f8_row6_col10" class="data row6 col10" >False</td>
      <td id="T_744f8_row6_col11" class="data row6 col11" >0.62420</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_744f8_row7_col0" class="data row7 col0" >5759</td>
      <td id="T_744f8_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_744f8_row7_col2" class="data row7 col2" >LORE_SA1</td>
      <td id="T_744f8_row7_col3" class="data row7 col3" >IF relationship = Own-child THEN class = <=50K</td>
      <td id="T_744f8_row7_col4" class="data row7 col4" >0.15554</td>
      <td id="T_744f8_row7_col5" class="data row7 col5" >0.20262</td>
      <td id="T_744f8_row7_col6" class="data row7 col6" >0.98900</td>
      <td id="T_744f8_row7_col7" class="data row7 col7" >1</td>
      <td id="T_744f8_row7_col8" class="data row7 col8" >0</td>
      <td id="T_744f8_row7_col9" class="data row7 col9" >30.98065</td>
      <td id="T_744f8_row7_col10" class="data row7 col10" >False</td>
      <td id="T_744f8_row7_col11" class="data row7 col11" >0.82280</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_744f8_row8_col0" class="data row8 col0" >5759</td>
      <td id="T_744f8_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_744f8_row8_col2" class="data row8 col2" >LORE_SA4</td>
      <td id="T_744f8_row8_col3" class="data row8 col3" >IF age <= 41.1272 AND education != Doctorate THEN class = <=50K</td>
      <td id="T_744f8_row8_col4" class="data row8 col4" >0.61114</td>
      <td id="T_744f8_row8_col5" class="data row8 col5" >0.67040</td>
      <td id="T_744f8_row8_col6" class="data row8 col6" >0.83279</td>
      <td id="T_744f8_row8_col7" class="data row8 col7" >2</td>
      <td id="T_744f8_row8_col8" class="data row8 col8" >0</td>
      <td id="T_744f8_row8_col9" class="data row8 col9" >33.98273</td>
      <td id="T_744f8_row8_col10" class="data row8 col10" >False</td>
      <td id="T_744f8_row8_col11" class="data row8 col11" >1.26197</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_744f8_row9_col0" class="data row9 col0" >5759</td>
      <td id="T_744f8_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_744f8_row9_col2" class="data row9 col2" >LORE_SA5</td>
      <td id="T_744f8_row9_col3" class="data row9 col3" >IF age <= 18.6366 THEN class = <=50K</td>
      <td id="T_744f8_row9_col4" class="data row9 col4" >0.02918</td>
      <td id="T_744f8_row9_col5" class="data row9 col5" >0.03843</td>
      <td id="T_744f8_row9_col6" class="data row9 col6" >1.00000</td>
      <td id="T_744f8_row9_col7" class="data row9 col7" >1</td>
      <td id="T_744f8_row9_col8" class="data row9 col8" >0</td>
      <td id="T_744f8_row9_col9" class="data row9 col9" >31.54394</td>
      <td id="T_744f8_row9_col10" class="data row9 col10" >False</td>
      <td id="T_744f8_row9_col11" class="data row9 col11" >0.98251</td>
    </tr>
    <tr>
      <th id="T_744f8_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_744f8_row10_col0" class="data row10 col0" >5759</td>
      <td id="T_744f8_row10_col1" class="data row10 col1" >EXPLAN</td>
      <td id="T_744f8_row10_col2" class="data row10 col2" >EXPLAN1</td>
      <td id="T_744f8_row10_col3" class="data row10 col3" >IF age <= 35.5896 AND capital.gain <= 3809.9333 THEN class = <=50K</td>
      <td id="T_744f8_row10_col4" class="data row10 col4" >0.44647</td>
      <td id="T_744f8_row10_col5" class="data row10 col5" >0.53049</td>
      <td id="T_744f8_row10_col6" class="data row10 col6" >0.90202</td>
      <td id="T_744f8_row10_col7" class="data row10 col7" >2</td>
      <td id="T_744f8_row10_col8" class="data row10 col8" >0</td>
      <td id="T_744f8_row10_col9" class="data row10 col9" >2.73231</td>
      <td id="T_744f8_row10_col10" class="data row10 col10" >False</td>
      <td id="T_744f8_row10_col11" class="data row10 col11" >1.29835</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 5759, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.9989, Pre: 1.0), Unique rules (diffrent features)



<style type="text/css">
#T_97ade_row3_col0, #T_97ade_row3_col1, #T_97ade_row3_col2, #T_97ade_row3_col3, #T_97ade_row3_col4, #T_97ade_row3_col5, #T_97ade_row3_col6, #T_97ade_row3_col7, #T_97ade_row3_col8, #T_97ade_row3_col9, #T_97ade_row3_col10, #T_97ade_row3_col11 {
  font-weight: bold;
}
</style>
<table id="T_97ade">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_97ade_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_97ade_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_97ade_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_97ade_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_97ade_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_97ade_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_97ade_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_97ade_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_97ade_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_97ade_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_97ade_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_97ade_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_97ade_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_97ade_row0_col0" class="data row0 col0" >5759</td>
      <td id="T_97ade_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_97ade_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_97ade_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_97ade_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_97ade_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_97ade_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_97ade_row0_col7" class="data row0 col7" >3</td>
      <td id="T_97ade_row0_col8" class="data row0 col8" >0</td>
      <td id="T_97ade_row0_col9" class="data row0 col9" >2.13079</td>
      <td id="T_97ade_row0_col10" class="data row0 col10" >False</td>
      <td id="T_97ade_row0_col11" class="data row0 col11" >2.22341</td>
    </tr>
    <tr>
      <th id="T_97ade_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_97ade_row1_col0" class="data row1 col0" >5759</td>
      <td id="T_97ade_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_97ade_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_97ade_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_97ade_row1_col4" class="data row1 col4" >0.54611</td>
      <td id="T_97ade_row1_col5" class="data row1 col5" >0.62509</td>
      <td id="T_97ade_row1_col6" class="data row1 col6" >0.86896</td>
      <td id="T_97ade_row1_col7" class="data row1 col7" >4</td>
      <td id="T_97ade_row1_col8" class="data row1 col8" >0</td>
      <td id="T_97ade_row1_col9" class="data row1 col9" >2.29826</td>
      <td id="T_97ade_row1_col10" class="data row1 col10" >False</td>
      <td id="T_97ade_row1_col11" class="data row1 col11" >3.23131</td>
    </tr>
    <tr>
      <th id="T_97ade_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_97ade_row2_col0" class="data row2 col0" >5759</td>
      <td id="T_97ade_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_97ade_row2_col2" class="data row2 col2" >ANCHOR4</td>
      <td id="T_97ade_row2_col3" class="data row2 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_97ade_row2_col4" class="data row2 col4" >0.48710</td>
      <td id="T_97ade_row2_col5" class="data row2 col5" >0.56643</td>
      <td id="T_97ade_row2_col6" class="data row2 col6" >0.88281</td>
      <td id="T_97ade_row2_col7" class="data row2 col7" >2</td>
      <td id="T_97ade_row2_col8" class="data row2 col8" >0</td>
      <td id="T_97ade_row2_col9" class="data row2 col9" >1.65381</td>
      <td id="T_97ade_row2_col10" class="data row2 col10" >False</td>
      <td id="T_97ade_row2_col11" class="data row2 col11" >1.28743</td>
    </tr>
    <tr>
      <th id="T_97ade_level0_row3" class="row_heading level0 row3" >5</th>
      <td id="T_97ade_row3_col0" class="data row3 col0" >5759</td>
      <td id="T_97ade_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_97ade_row3_col2" class="data row3 col2" >LORE3</td>
      <td id="T_97ade_row3_col3" class="data row3 col3" >IF capital.gain <= 4759.1385 THEN class = <=50K</td>
      <td id="T_97ade_row3_col4" class="data row3 col4" >0.94735</td>
      <td id="T_97ade_row3_col5" class="data row3 col5" >0.99324</td>
      <td id="T_97ade_row3_col6" class="data row3 col6" >0.79594</td>
      <td id="T_97ade_row3_col7" class="data row3 col7" >1</td>
      <td id="T_97ade_row3_col8" class="data row3 col8" >0</td>
      <td id="T_97ade_row3_col9" class="data row3 col9" >132.87754</td>
      <td id="T_97ade_row3_col10" class="data row3 col10" >False</td>
      <td id="T_97ade_row3_col11" class="data row3 col11" >0.29068</td>
    </tr>
    <tr>
      <th id="T_97ade_level0_row4" class="row_heading level0 row4" >6</th>
      <td id="T_97ade_row4_col0" class="data row4 col0" >5759</td>
      <td id="T_97ade_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_97ade_row4_col2" class="data row4 col2" >LORE5</td>
      <td id="T_97ade_row4_col3" class="data row4 col3" >IF marital.status = Never-married THEN class = <=50K</td>
      <td id="T_97ade_row4_col4" class="data row4 col4" >0.32757</td>
      <td id="T_97ade_row4_col5" class="data row4 col5" >0.41178</td>
      <td id="T_97ade_row4_col6" class="data row4 col6" >0.95433</td>
      <td id="T_97ade_row4_col7" class="data row4 col7" >1</td>
      <td id="T_97ade_row4_col8" class="data row4 col8" >0</td>
      <td id="T_97ade_row4_col9" class="data row4 col9" >227.17949</td>
      <td id="T_97ade_row4_col10" class="data row4 col10" >False</td>
      <td id="T_97ade_row4_col11" class="data row4 col11" >0.62420</td>
    </tr>
    <tr>
      <th id="T_97ade_level0_row5" class="row_heading level0 row5" >7</th>
      <td id="T_97ade_row5_col0" class="data row5 col0" >5759</td>
      <td id="T_97ade_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_97ade_row5_col2" class="data row5 col2" >LORE_SA1</td>
      <td id="T_97ade_row5_col3" class="data row5 col3" >IF relationship = Own-child THEN class = <=50K</td>
      <td id="T_97ade_row5_col4" class="data row5 col4" >0.15554</td>
      <td id="T_97ade_row5_col5" class="data row5 col5" >0.20262</td>
      <td id="T_97ade_row5_col6" class="data row5 col6" >0.98900</td>
      <td id="T_97ade_row5_col7" class="data row5 col7" >1</td>
      <td id="T_97ade_row5_col8" class="data row5 col8" >0</td>
      <td id="T_97ade_row5_col9" class="data row5 col9" >30.98065</td>
      <td id="T_97ade_row5_col10" class="data row5 col10" >False</td>
      <td id="T_97ade_row5_col11" class="data row5 col11" >0.82280</td>
    </tr>
    <tr>
      <th id="T_97ade_level0_row6" class="row_heading level0 row6" >8</th>
      <td id="T_97ade_row6_col0" class="data row6 col0" >5759</td>
      <td id="T_97ade_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_97ade_row6_col2" class="data row6 col2" >LORE_SA4</td>
      <td id="T_97ade_row6_col3" class="data row6 col3" >IF age <= 41.1272 AND education != Doctorate THEN class = <=50K</td>
      <td id="T_97ade_row6_col4" class="data row6 col4" >0.61114</td>
      <td id="T_97ade_row6_col5" class="data row6 col5" >0.67040</td>
      <td id="T_97ade_row6_col6" class="data row6 col6" >0.83279</td>
      <td id="T_97ade_row6_col7" class="data row6 col7" >2</td>
      <td id="T_97ade_row6_col8" class="data row6 col8" >0</td>
      <td id="T_97ade_row6_col9" class="data row6 col9" >33.98273</td>
      <td id="T_97ade_row6_col10" class="data row6 col10" >False</td>
      <td id="T_97ade_row6_col11" class="data row6 col11" >1.26197</td>
    </tr>
    <tr>
      <th id="T_97ade_level0_row7" class="row_heading level0 row7" >9</th>
      <td id="T_97ade_row7_col0" class="data row7 col0" >5759</td>
      <td id="T_97ade_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_97ade_row7_col2" class="data row7 col2" >LORE_SA5</td>
      <td id="T_97ade_row7_col3" class="data row7 col3" >IF age <= 18.6366 THEN class = <=50K</td>
      <td id="T_97ade_row7_col4" class="data row7 col4" >0.02918</td>
      <td id="T_97ade_row7_col5" class="data row7 col5" >0.03843</td>
      <td id="T_97ade_row7_col6" class="data row7 col6" >1.00000</td>
      <td id="T_97ade_row7_col7" class="data row7 col7" >1</td>
      <td id="T_97ade_row7_col8" class="data row7 col8" >0</td>
      <td id="T_97ade_row7_col9" class="data row7 col9" >31.54394</td>
      <td id="T_97ade_row7_col10" class="data row7 col10" >False</td>
      <td id="T_97ade_row7_col11" class="data row7 col11" >0.98251</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_125.png)
    



## Instance 20 (Original: >50K , Predicted: >50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>37.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>Bachelors</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>13</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Never-married</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Exec-managerial</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Not-in-family</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>2824.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>40.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 20



<style type="text/css">
</style>
<table id="T_eca48">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_eca48_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_eca48_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_eca48_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_eca48_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_eca48_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_eca48_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_eca48_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_eca48_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_eca48_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_eca48_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_eca48_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_eca48_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_eca48_row0_col0" class="data row0 col0" >20</td>
      <td id="T_eca48_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_eca48_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_eca48_row0_col3" class="data row0 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND occupation = Exec-managerial AND race = White AND sex = Male THEN class = >50K</td>
      <td id="T_eca48_row0_col4" class="data row0 col4" >0.00263</td>
      <td id="T_eca48_row0_col5" class="data row0 col5" >0.00874</td>
      <td id="T_eca48_row0_col6" class="data row0 col6" >0.80000</td>
      <td id="T_eca48_row0_col7" class="data row0 col7" >7</td>
      <td id="T_eca48_row0_col8" class="data row0 col8" >0</td>
      <td id="T_eca48_row0_col9" class="data row0 col9" >57.52619</td>
      <td id="T_eca48_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_eca48_row1_col0" class="data row1 col0" >20</td>
      <td id="T_eca48_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_eca48_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_eca48_row1_col3" class="data row1 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND native.country = United-States AND occupation = Exec-managerial AND sex = Male AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row1_col4" class="data row1 col4" >0.00193</td>
      <td id="T_eca48_row1_col5" class="data row1 col5" >0.00674</td>
      <td id="T_eca48_row1_col6" class="data row1 col6" >0.84091</td>
      <td id="T_eca48_row1_col7" class="data row1 col7" >8</td>
      <td id="T_eca48_row1_col8" class="data row1 col8" >0</td>
      <td id="T_eca48_row1_col9" class="data row1 col9" >66.62694</td>
      <td id="T_eca48_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_eca48_row2_col0" class="data row2 col0" >20</td>
      <td id="T_eca48_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_eca48_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_eca48_row2_col3" class="data row2 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND native.country = United-States AND occupation = Exec-managerial AND sex = Male THEN class = >50K</td>
      <td id="T_eca48_row2_col4" class="data row2 col4" >0.00276</td>
      <td id="T_eca48_row2_col5" class="data row2 col5" >0.00929</td>
      <td id="T_eca48_row2_col6" class="data row2 col6" >0.80952</td>
      <td id="T_eca48_row2_col7" class="data row2 col7" >6</td>
      <td id="T_eca48_row2_col8" class="data row2 col8" >0</td>
      <td id="T_eca48_row2_col9" class="data row2 col9" >53.74712</td>
      <td id="T_eca48_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_eca48_row3_col0" class="data row3 col0" >20</td>
      <td id="T_eca48_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_eca48_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_eca48_row3_col3" class="data row3 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND occupation = Exec-managerial AND sex = Male THEN class = >50K</td>
      <td id="T_eca48_row3_col4" class="data row3 col4" >0.00290</td>
      <td id="T_eca48_row3_col5" class="data row3 col5" >0.00966</td>
      <td id="T_eca48_row3_col6" class="data row3 col6" >0.80303</td>
      <td id="T_eca48_row3_col7" class="data row3 col7" >6</td>
      <td id="T_eca48_row3_col8" class="data row3 col8" >0</td>
      <td id="T_eca48_row3_col9" class="data row3 col9" >51.54575</td>
      <td id="T_eca48_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_eca48_row4_col0" class="data row4 col0" >20</td>
      <td id="T_eca48_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_eca48_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_eca48_row4_col3" class="data row4 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND occupation = Exec-managerial AND race = White AND sex = Male AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row4_col4" class="data row4 col4" >0.00180</td>
      <td id="T_eca48_row4_col5" class="data row4 col5" >0.00619</td>
      <td id="T_eca48_row4_col6" class="data row4 col6" >0.82927</td>
      <td id="T_eca48_row4_col7" class="data row4 col7" >8</td>
      <td id="T_eca48_row4_col8" class="data row4 col8" >2</td>
      <td id="T_eca48_row4_col9" class="data row4 col9" >137.20941</td>
      <td id="T_eca48_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_eca48_row5_col0" class="data row5 col0" >20</td>
      <td id="T_eca48_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_eca48_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_eca48_row5_col3" class="data row5 col3" >IF hours.per.week > 38.0 AND occupation = Exec-managerial AND relationship = Not-in-family THEN class = >50K</td>
      <td id="T_eca48_row5_col4" class="data row5 col4" >0.02575</td>
      <td id="T_eca48_row5_col5" class="data row5 col5" >0.02787</td>
      <td id="T_eca48_row5_col6" class="data row5 col6" >0.26065</td>
      <td id="T_eca48_row5_col7" class="data row5 col7" >3</td>
      <td id="T_eca48_row5_col8" class="data row5 col8" >0</td>
      <td id="T_eca48_row5_col9" class="data row5 col9" >237.35692</td>
      <td id="T_eca48_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_eca48_row6_col0" class="data row6 col0" >20</td>
      <td id="T_eca48_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_eca48_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_eca48_row6_col3" class="data row6 col3" >IF hours.per.week > 39.6185 AND hours.per.week <= 44.0 THEN class = <=50K</td>
      <td id="T_eca48_row6_col4" class="data row6 col4" >0.48679</td>
      <td id="T_eca48_row6_col5" class="data row6 col5" >0.50269</td>
      <td id="T_eca48_row6_col6" class="data row6 col6" >0.78396</td>
      <td id="T_eca48_row6_col7" class="data row6 col7" >2</td>
      <td id="T_eca48_row6_col8" class="data row6 col8" >0</td>
      <td id="T_eca48_row6_col9" class="data row6 col9" >311.79412</td>
      <td id="T_eca48_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_eca48_row7_col0" class="data row7 col0" >20</td>
      <td id="T_eca48_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_eca48_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_eca48_row7_col3" class="data row7 col3" >IF education.num = 13.0 AND hours.per.week > 35.0 AND occupation = Exec-managerial AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row7_col4" class="data row7 col4" >0.00715</td>
      <td id="T_eca48_row7_col5" class="data row7 col5" >0.00929</td>
      <td id="T_eca48_row7_col6" class="data row7 col6" >0.31288</td>
      <td id="T_eca48_row7_col7" class="data row7 col7" >5</td>
      <td id="T_eca48_row7_col8" class="data row7 col8" >0</td>
      <td id="T_eca48_row7_col9" class="data row7 col9" >185.63279</td>
      <td id="T_eca48_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_eca48_row8_col0" class="data row8 col0" >20</td>
      <td id="T_eca48_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_eca48_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_eca48_row8_col3" class="data row8 col3" >IF age > 36.0 AND age <= 37.0 AND education = Bachelors AND hours.per.week > 36.0 AND hours.per.week <= 40.0 AND marital.status = Never-married AND sex = Male THEN class = <=50K</td>
      <td id="T_eca48_row8_col4" class="data row8 col4" >0.00026</td>
      <td id="T_eca48_row8_col5" class="data row8 col5" >0.00029</td>
      <td id="T_eca48_row8_col6" class="data row8 col6" >0.83333</td>
      <td id="T_eca48_row8_col7" class="data row8 col7" >7</td>
      <td id="T_eca48_row8_col8" class="data row8 col8" >0</td>
      <td id="T_eca48_row8_col9" class="data row8 col9" >149.55021</td>
      <td id="T_eca48_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_eca48_row9_col0" class="data row9 col0" >20</td>
      <td id="T_eca48_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_eca48_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_eca48_row9_col3" class="data row9 col3" >IF hours.per.week > 35.0 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row9_col4" class="data row9 col4" >0.07586</td>
      <td id="T_eca48_row9_col5" class="data row9 col5" >0.16160</td>
      <td id="T_eca48_row9_col6" class="data row9 col6" >0.51301</td>
      <td id="T_eca48_row9_col7" class="data row9 col7" >3</td>
      <td id="T_eca48_row9_col8" class="data row9 col8" >0</td>
      <td id="T_eca48_row9_col9" class="data row9 col9" >160.11351</td>
      <td id="T_eca48_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_eca48_row10_col0" class="data row10 col0" >20</td>
      <td id="T_eca48_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_eca48_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_eca48_row10_col3" class="data row10 col3" >IF age > 23.0678 AND capital.loss <= 3051.6125 AND education.num != 8.0 AND education.num != 3.0 AND hours.per.week > 30.3276 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row10_col4" class="data row10 col4" >0.01768</td>
      <td id="T_eca48_row10_col5" class="data row10 col5" >0.01986</td>
      <td id="T_eca48_row10_col6" class="data row10 col6" >0.27047</td>
      <td id="T_eca48_row10_col7" class="data row10 col7" >10</td>
      <td id="T_eca48_row10_col8" class="data row10 col8" >0</td>
      <td id="T_eca48_row10_col9" class="data row10 col9" >39.31556</td>
      <td id="T_eca48_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_eca48_row11_col0" class="data row11 col0" >20</td>
      <td id="T_eca48_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_eca48_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_eca48_row11_col3" class="data row11 col3" >IF age > 30.8268 AND capital.loss > 2182.3465 AND education.num != 9.0 AND education.num != 6.0 AND education.num != 8.0 AND hours.per.week > 28.6128 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row11_col4" class="data row11 col4" >0.00026</td>
      <td id="T_eca48_row11_col5" class="data row11 col5" >0.00109</td>
      <td id="T_eca48_row11_col6" class="data row11 col6" >1.00000</td>
      <td id="T_eca48_row11_col7" class="data row11 col7" >11</td>
      <td id="T_eca48_row11_col8" class="data row11 col8" >0</td>
      <td id="T_eca48_row11_col9" class="data row11 col9" >45.59706</td>
      <td id="T_eca48_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_eca48_row12_col0" class="data row12 col0" >20</td>
      <td id="T_eca48_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_eca48_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_eca48_row12_col3" class="data row12 col3" >IF age <= 45.6925 AND age > 21.8182 AND capital.loss <= 3425.0138 AND capital.loss > 2713.2085 AND education.num != 3.0 AND education.num != 6.0 AND hours.per.week > 32.677 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row12_col4" class="data row12 col4" >0.00004</td>
      <td id="T_eca48_row12_col5" class="data row12 col5" >0.00018</td>
      <td id="T_eca48_row12_col6" class="data row12 col6" >1.00000</td>
      <td id="T_eca48_row12_col7" class="data row12 col7" >12</td>
      <td id="T_eca48_row12_col8" class="data row12 col8" >0</td>
      <td id="T_eca48_row12_col9" class="data row12 col9" >45.43260</td>
      <td id="T_eca48_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_eca48_row13_col0" class="data row13 col0" >20</td>
      <td id="T_eca48_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_eca48_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_eca48_row13_col3" class="data row13 col3" >IF age > 24.3848 AND capital.loss > 1437.6834 AND education.num != 6.0 AND education.num != 8.0 AND hours.per.week > 31.1491 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row13_col4" class="data row13 col4" >0.00101</td>
      <td id="T_eca48_row13_col5" class="data row13 col5" >0.00200</td>
      <td id="T_eca48_row13_col6" class="data row13 col6" >0.47826</td>
      <td id="T_eca48_row13_col7" class="data row13 col7" >10</td>
      <td id="T_eca48_row13_col8" class="data row13 col8" >0</td>
      <td id="T_eca48_row13_col9" class="data row13 col9" >48.81059</td>
      <td id="T_eca48_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_eca48_row14_col0" class="data row14 col0" >20</td>
      <td id="T_eca48_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_eca48_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_eca48_row14_col3" class="data row14 col3" >IF capital.loss <= 3041.1064 AND capital.loss > 2654.4749 AND education.num != 3.0 AND education.num != 9.0 AND hours.per.week > 32.6111 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row14_col4" class="data row14 col4" >0.00004</td>
      <td id="T_eca48_row14_col5" class="data row14 col5" >0.00018</td>
      <td id="T_eca48_row14_col6" class="data row14 col6" >1.00000</td>
      <td id="T_eca48_row14_col7" class="data row14 col7" >10</td>
      <td id="T_eca48_row14_col8" class="data row14 col8" >0</td>
      <td id="T_eca48_row14_col9" class="data row14 col9" >46.78653</td>
      <td id="T_eca48_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_eca48_row15_col0" class="data row15 col0" >20</td>
      <td id="T_eca48_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_eca48_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_eca48_row15_col3" class="data row15 col3" >IF age > 36.0257 AND capital.gain <= 1790.0889 AND capital.loss > 2267.0 AND relationship = Not-in-family THEN class = >50K</td>
      <td id="T_eca48_row15_col4" class="data row15 col4" >0.00127</td>
      <td id="T_eca48_row15_col5" class="data row15 col5" >0.00346</td>
      <td id="T_eca48_row15_col6" class="data row15 col6" >0.65517</td>
      <td id="T_eca48_row15_col7" class="data row15 col7" >4</td>
      <td id="T_eca48_row15_col8" class="data row15 col8" >0</td>
      <td id="T_eca48_row15_col9" class="data row15 col9" >11.13073</td>
      <td id="T_eca48_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_eca48_row16_col0" class="data row16 col0" >20</td>
      <td id="T_eca48_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_eca48_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_eca48_row16_col3" class="data row16 col3" >IF age > 28.495 AND capital.gain <= 5023.4699 AND capital.loss > 2593.5262 AND marital.status = Never-married AND relationship = Not-in-family THEN class = >50K</td>
      <td id="T_eca48_row16_col4" class="data row16 col4" >0.00009</td>
      <td id="T_eca48_row16_col5" class="data row16 col5" >0.00036</td>
      <td id="T_eca48_row16_col6" class="data row16 col6" >1.00000</td>
      <td id="T_eca48_row16_col7" class="data row16 col7" >5</td>
      <td id="T_eca48_row16_col8" class="data row16 col8" >0</td>
      <td id="T_eca48_row16_col9" class="data row16 col9" >12.15413</td>
      <td id="T_eca48_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_eca48_row17_col0" class="data row17 col0" >20</td>
      <td id="T_eca48_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_eca48_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_eca48_row17_col3" class="data row17 col3" >IF age > 31.662 AND capital.gain <= 4949.028 AND capital.loss > 2307.7524 AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row17_col4" class="data row17 col4" >0.00097</td>
      <td id="T_eca48_row17_col5" class="data row17 col5" >0.00273</td>
      <td id="T_eca48_row17_col6" class="data row17 col6" >0.68182</td>
      <td id="T_eca48_row17_col7" class="data row17 col7" >5</td>
      <td id="T_eca48_row17_col8" class="data row17 col8" >0</td>
      <td id="T_eca48_row17_col9" class="data row17 col9" >11.98204</td>
      <td id="T_eca48_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_eca48_row18_col0" class="data row18 col0" >20</td>
      <td id="T_eca48_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_eca48_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_eca48_row18_col3" class="data row18 col3" >IF capital.gain <= 5153.3158 AND capital.loss > 2562.1699 AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_eca48_row18_col4" class="data row18 col4" >0.00004</td>
      <td id="T_eca48_row18_col5" class="data row18 col5" >0.00018</td>
      <td id="T_eca48_row18_col6" class="data row18 col6" >1.00000</td>
      <td id="T_eca48_row18_col7" class="data row18 col7" >3</td>
      <td id="T_eca48_row18_col8" class="data row18 col8" >0</td>
      <td id="T_eca48_row18_col9" class="data row18 col9" >12.52057</td>
      <td id="T_eca48_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_eca48_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_eca48_row19_col0" class="data row19 col0" >20</td>
      <td id="T_eca48_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_eca48_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_eca48_row19_col3" class="data row19 col3" >IF capital.gain <= 5178.0 AND capital.loss > 2724.9772 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_eca48_row19_col4" class="data row19 col4" >0.00004</td>
      <td id="T_eca48_row19_col5" class="data row19 col5" >0.00018</td>
      <td id="T_eca48_row19_col6" class="data row19 col6" >1.00000</td>
      <td id="T_eca48_row19_col7" class="data row19 col7" >4</td>
      <td id="T_eca48_row19_col8" class="data row19 col8" >0</td>
      <td id="T_eca48_row19_col9" class="data row19 col9" >9.95007</td>
      <td id="T_eca48_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 20, Correct Prediction



<style type="text/css">
</style>
<table id="T_e9c4f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e9c4f_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e9c4f_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e9c4f_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e9c4f_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e9c4f_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e9c4f_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e9c4f_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e9c4f_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e9c4f_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e9c4f_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e9c4f_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e9c4f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e9c4f_row0_col0" class="data row0 col0" >20</td>
      <td id="T_e9c4f_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_e9c4f_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_e9c4f_row0_col3" class="data row0 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND occupation = Exec-managerial AND race = White AND sex = Male THEN class = >50K</td>
      <td id="T_e9c4f_row0_col4" class="data row0 col4" >0.00263</td>
      <td id="T_e9c4f_row0_col5" class="data row0 col5" >0.00874</td>
      <td id="T_e9c4f_row0_col6" class="data row0 col6" >0.80000</td>
      <td id="T_e9c4f_row0_col7" class="data row0 col7" >7</td>
      <td id="T_e9c4f_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e9c4f_row0_col9" class="data row0 col9" >57.52619</td>
      <td id="T_e9c4f_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e9c4f_row1_col0" class="data row1 col0" >20</td>
      <td id="T_e9c4f_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_e9c4f_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_e9c4f_row1_col3" class="data row1 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND native.country = United-States AND occupation = Exec-managerial AND sex = Male AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row1_col4" class="data row1 col4" >0.00193</td>
      <td id="T_e9c4f_row1_col5" class="data row1 col5" >0.00674</td>
      <td id="T_e9c4f_row1_col6" class="data row1 col6" >0.84091</td>
      <td id="T_e9c4f_row1_col7" class="data row1 col7" >8</td>
      <td id="T_e9c4f_row1_col8" class="data row1 col8" >0</td>
      <td id="T_e9c4f_row1_col9" class="data row1 col9" >66.62694</td>
      <td id="T_e9c4f_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e9c4f_row2_col0" class="data row2 col0" >20</td>
      <td id="T_e9c4f_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_e9c4f_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_e9c4f_row2_col3" class="data row2 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND native.country = United-States AND occupation = Exec-managerial AND sex = Male THEN class = >50K</td>
      <td id="T_e9c4f_row2_col4" class="data row2 col4" >0.00276</td>
      <td id="T_e9c4f_row2_col5" class="data row2 col5" >0.00929</td>
      <td id="T_e9c4f_row2_col6" class="data row2 col6" >0.80952</td>
      <td id="T_e9c4f_row2_col7" class="data row2 col7" >6</td>
      <td id="T_e9c4f_row2_col8" class="data row2 col8" >0</td>
      <td id="T_e9c4f_row2_col9" class="data row2 col9" >53.74712</td>
      <td id="T_e9c4f_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e9c4f_row3_col0" class="data row3 col0" >20</td>
      <td id="T_e9c4f_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_e9c4f_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_e9c4f_row3_col3" class="data row3 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND occupation = Exec-managerial AND sex = Male THEN class = >50K</td>
      <td id="T_e9c4f_row3_col4" class="data row3 col4" >0.00290</td>
      <td id="T_e9c4f_row3_col5" class="data row3 col5" >0.00966</td>
      <td id="T_e9c4f_row3_col6" class="data row3 col6" >0.80303</td>
      <td id="T_e9c4f_row3_col7" class="data row3 col7" >6</td>
      <td id="T_e9c4f_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e9c4f_row3_col9" class="data row3 col9" >51.54575</td>
      <td id="T_e9c4f_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e9c4f_row4_col0" class="data row4 col0" >20</td>
      <td id="T_e9c4f_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_e9c4f_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_e9c4f_row4_col3" class="data row4 col3" >IF age > 28.0 AND capital.loss > 0.0 AND education = Bachelors AND education.num = 13.0 AND occupation = Exec-managerial AND race = White AND sex = Male AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row4_col4" class="data row4 col4" >0.00180</td>
      <td id="T_e9c4f_row4_col5" class="data row4 col5" >0.00619</td>
      <td id="T_e9c4f_row4_col6" class="data row4 col6" >0.82927</td>
      <td id="T_e9c4f_row4_col7" class="data row4 col7" >8</td>
      <td id="T_e9c4f_row4_col8" class="data row4 col8" >2</td>
      <td id="T_e9c4f_row4_col9" class="data row4 col9" >137.20941</td>
      <td id="T_e9c4f_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e9c4f_row5_col0" class="data row5 col0" >20</td>
      <td id="T_e9c4f_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_e9c4f_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_e9c4f_row5_col3" class="data row5 col3" >IF hours.per.week > 38.0 AND occupation = Exec-managerial AND relationship = Not-in-family THEN class = >50K</td>
      <td id="T_e9c4f_row5_col4" class="data row5 col4" >0.02575</td>
      <td id="T_e9c4f_row5_col5" class="data row5 col5" >0.02787</td>
      <td id="T_e9c4f_row5_col6" class="data row5 col6" >0.26065</td>
      <td id="T_e9c4f_row5_col7" class="data row5 col7" >3</td>
      <td id="T_e9c4f_row5_col8" class="data row5 col8" >0</td>
      <td id="T_e9c4f_row5_col9" class="data row5 col9" >237.35692</td>
      <td id="T_e9c4f_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_e9c4f_row6_col0" class="data row6 col0" >20</td>
      <td id="T_e9c4f_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_e9c4f_row6_col2" class="data row6 col2" >LORE3</td>
      <td id="T_e9c4f_row6_col3" class="data row6 col3" >IF education.num = 13.0 AND hours.per.week > 35.0 AND occupation = Exec-managerial AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row6_col4" class="data row6 col4" >0.00715</td>
      <td id="T_e9c4f_row6_col5" class="data row6 col5" >0.00929</td>
      <td id="T_e9c4f_row6_col6" class="data row6 col6" >0.31288</td>
      <td id="T_e9c4f_row6_col7" class="data row6 col7" >5</td>
      <td id="T_e9c4f_row6_col8" class="data row6 col8" >0</td>
      <td id="T_e9c4f_row6_col9" class="data row6 col9" >185.63279</td>
      <td id="T_e9c4f_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_e9c4f_row7_col0" class="data row7 col0" >20</td>
      <td id="T_e9c4f_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_e9c4f_row7_col2" class="data row7 col2" >LORE5</td>
      <td id="T_e9c4f_row7_col3" class="data row7 col3" >IF hours.per.week > 35.0 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row7_col4" class="data row7 col4" >0.07586</td>
      <td id="T_e9c4f_row7_col5" class="data row7 col5" >0.16160</td>
      <td id="T_e9c4f_row7_col6" class="data row7 col6" >0.51301</td>
      <td id="T_e9c4f_row7_col7" class="data row7 col7" >3</td>
      <td id="T_e9c4f_row7_col8" class="data row7 col8" >0</td>
      <td id="T_e9c4f_row7_col9" class="data row7 col9" >160.11351</td>
      <td id="T_e9c4f_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_e9c4f_row8_col0" class="data row8 col0" >20</td>
      <td id="T_e9c4f_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_e9c4f_row8_col2" class="data row8 col2" >LORE_SA1</td>
      <td id="T_e9c4f_row8_col3" class="data row8 col3" >IF age > 23.0678 AND capital.loss <= 3051.6125 AND education.num != 8.0 AND education.num != 3.0 AND hours.per.week > 30.3276 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row8_col4" class="data row8 col4" >0.01768</td>
      <td id="T_e9c4f_row8_col5" class="data row8 col5" >0.01986</td>
      <td id="T_e9c4f_row8_col6" class="data row8 col6" >0.27047</td>
      <td id="T_e9c4f_row8_col7" class="data row8 col7" >10</td>
      <td id="T_e9c4f_row8_col8" class="data row8 col8" >0</td>
      <td id="T_e9c4f_row8_col9" class="data row8 col9" >39.31556</td>
      <td id="T_e9c4f_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_e9c4f_row9_col0" class="data row9 col0" >20</td>
      <td id="T_e9c4f_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_e9c4f_row9_col2" class="data row9 col2" >LORE_SA2</td>
      <td id="T_e9c4f_row9_col3" class="data row9 col3" >IF age > 30.8268 AND capital.loss > 2182.3465 AND education.num != 9.0 AND education.num != 6.0 AND education.num != 8.0 AND hours.per.week > 28.6128 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row9_col4" class="data row9 col4" >0.00026</td>
      <td id="T_e9c4f_row9_col5" class="data row9 col5" >0.00109</td>
      <td id="T_e9c4f_row9_col6" class="data row9 col6" >1.00000</td>
      <td id="T_e9c4f_row9_col7" class="data row9 col7" >11</td>
      <td id="T_e9c4f_row9_col8" class="data row9 col8" >0</td>
      <td id="T_e9c4f_row9_col9" class="data row9 col9" >45.59706</td>
      <td id="T_e9c4f_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_e9c4f_row10_col0" class="data row10 col0" >20</td>
      <td id="T_e9c4f_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_e9c4f_row10_col2" class="data row10 col2" >LORE_SA3</td>
      <td id="T_e9c4f_row10_col3" class="data row10 col3" >IF age <= 45.6925 AND age > 21.8182 AND capital.loss <= 3425.0138 AND capital.loss > 2713.2085 AND education.num != 3.0 AND education.num != 6.0 AND hours.per.week > 32.677 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row10_col4" class="data row10 col4" >0.00004</td>
      <td id="T_e9c4f_row10_col5" class="data row10 col5" >0.00018</td>
      <td id="T_e9c4f_row10_col6" class="data row10 col6" >1.00000</td>
      <td id="T_e9c4f_row10_col7" class="data row10 col7" >12</td>
      <td id="T_e9c4f_row10_col8" class="data row10 col8" >0</td>
      <td id="T_e9c4f_row10_col9" class="data row10 col9" >45.43260</td>
      <td id="T_e9c4f_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_e9c4f_row11_col0" class="data row11 col0" >20</td>
      <td id="T_e9c4f_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_e9c4f_row11_col2" class="data row11 col2" >LORE_SA4</td>
      <td id="T_e9c4f_row11_col3" class="data row11 col3" >IF age > 24.3848 AND capital.loss > 1437.6834 AND education.num != 6.0 AND education.num != 8.0 AND hours.per.week > 31.1491 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row11_col4" class="data row11 col4" >0.00101</td>
      <td id="T_e9c4f_row11_col5" class="data row11 col5" >0.00200</td>
      <td id="T_e9c4f_row11_col6" class="data row11 col6" >0.47826</td>
      <td id="T_e9c4f_row11_col7" class="data row11 col7" >10</td>
      <td id="T_e9c4f_row11_col8" class="data row11 col8" >0</td>
      <td id="T_e9c4f_row11_col9" class="data row11 col9" >48.81059</td>
      <td id="T_e9c4f_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_e9c4f_row12_col0" class="data row12 col0" >20</td>
      <td id="T_e9c4f_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_e9c4f_row12_col2" class="data row12 col2" >LORE_SA5</td>
      <td id="T_e9c4f_row12_col3" class="data row12 col3" >IF capital.loss <= 3041.1064 AND capital.loss > 2654.4749 AND education.num != 3.0 AND education.num != 9.0 AND hours.per.week > 32.6111 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row12_col4" class="data row12 col4" >0.00004</td>
      <td id="T_e9c4f_row12_col5" class="data row12 col5" >0.00018</td>
      <td id="T_e9c4f_row12_col6" class="data row12 col6" >1.00000</td>
      <td id="T_e9c4f_row12_col7" class="data row12 col7" >10</td>
      <td id="T_e9c4f_row12_col8" class="data row12 col8" >0</td>
      <td id="T_e9c4f_row12_col9" class="data row12 col9" >46.78653</td>
      <td id="T_e9c4f_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_e9c4f_row13_col0" class="data row13 col0" >20</td>
      <td id="T_e9c4f_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_e9c4f_row13_col2" class="data row13 col2" >EXPLAN1</td>
      <td id="T_e9c4f_row13_col3" class="data row13 col3" >IF age > 36.0257 AND capital.gain <= 1790.0889 AND capital.loss > 2267.0 AND relationship = Not-in-family THEN class = >50K</td>
      <td id="T_e9c4f_row13_col4" class="data row13 col4" >0.00127</td>
      <td id="T_e9c4f_row13_col5" class="data row13 col5" >0.00346</td>
      <td id="T_e9c4f_row13_col6" class="data row13 col6" >0.65517</td>
      <td id="T_e9c4f_row13_col7" class="data row13 col7" >4</td>
      <td id="T_e9c4f_row13_col8" class="data row13 col8" >0</td>
      <td id="T_e9c4f_row13_col9" class="data row13 col9" >11.13073</td>
      <td id="T_e9c4f_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_e9c4f_row14_col0" class="data row14 col0" >20</td>
      <td id="T_e9c4f_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_e9c4f_row14_col2" class="data row14 col2" >EXPLAN2</td>
      <td id="T_e9c4f_row14_col3" class="data row14 col3" >IF age > 28.495 AND capital.gain <= 5023.4699 AND capital.loss > 2593.5262 AND marital.status = Never-married AND relationship = Not-in-family THEN class = >50K</td>
      <td id="T_e9c4f_row14_col4" class="data row14 col4" >0.00009</td>
      <td id="T_e9c4f_row14_col5" class="data row14 col5" >0.00036</td>
      <td id="T_e9c4f_row14_col6" class="data row14 col6" >1.00000</td>
      <td id="T_e9c4f_row14_col7" class="data row14 col7" >5</td>
      <td id="T_e9c4f_row14_col8" class="data row14 col8" >0</td>
      <td id="T_e9c4f_row14_col9" class="data row14 col9" >12.15413</td>
      <td id="T_e9c4f_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_e9c4f_row15_col0" class="data row15 col0" >20</td>
      <td id="T_e9c4f_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_e9c4f_row15_col2" class="data row15 col2" >EXPLAN3</td>
      <td id="T_e9c4f_row15_col3" class="data row15 col3" >IF age > 31.662 AND capital.gain <= 4949.028 AND capital.loss > 2307.7524 AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row15_col4" class="data row15 col4" >0.00097</td>
      <td id="T_e9c4f_row15_col5" class="data row15 col5" >0.00273</td>
      <td id="T_e9c4f_row15_col6" class="data row15 col6" >0.68182</td>
      <td id="T_e9c4f_row15_col7" class="data row15 col7" >5</td>
      <td id="T_e9c4f_row15_col8" class="data row15 col8" >0</td>
      <td id="T_e9c4f_row15_col9" class="data row15 col9" >11.98204</td>
      <td id="T_e9c4f_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_e9c4f_row16_col0" class="data row16 col0" >20</td>
      <td id="T_e9c4f_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_e9c4f_row16_col2" class="data row16 col2" >EXPLAN4</td>
      <td id="T_e9c4f_row16_col3" class="data row16 col3" >IF capital.gain <= 5153.3158 AND capital.loss > 2562.1699 AND occupation = Exec-managerial THEN class = >50K</td>
      <td id="T_e9c4f_row16_col4" class="data row16 col4" >0.00004</td>
      <td id="T_e9c4f_row16_col5" class="data row16 col5" >0.00018</td>
      <td id="T_e9c4f_row16_col6" class="data row16 col6" >1.00000</td>
      <td id="T_e9c4f_row16_col7" class="data row16 col7" >3</td>
      <td id="T_e9c4f_row16_col8" class="data row16 col8" >0</td>
      <td id="T_e9c4f_row16_col9" class="data row16 col9" >12.52057</td>
      <td id="T_e9c4f_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e9c4f_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_e9c4f_row17_col0" class="data row17 col0" >20</td>
      <td id="T_e9c4f_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_e9c4f_row17_col2" class="data row17 col2" >EXPLAN5</td>
      <td id="T_e9c4f_row17_col3" class="data row17 col3" >IF capital.gain <= 5178.0 AND capital.loss > 2724.9772 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_e9c4f_row17_col4" class="data row17 col4" >0.00004</td>
      <td id="T_e9c4f_row17_col5" class="data row17 col5" >0.00018</td>
      <td id="T_e9c4f_row17_col6" class="data row17 col6" >1.00000</td>
      <td id="T_e9c4f_row17_col7" class="data row17 col7" >4</td>
      <td id="T_e9c4f_row17_col8" class="data row17 col8" >0</td>
      <td id="T_e9c4f_row17_col9" class="data row17 col9" >9.95007</td>
      <td id="T_e9c4f_row17_col10" class="data row17 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 20, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_18b7f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_18b7f_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_18b7f_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_18b7f_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_18b7f_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_18b7f_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_18b7f_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_18b7f_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_18b7f_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_18b7f_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_18b7f_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_18b7f_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_18b7f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_18b7f_row0_col0" class="data row0 col0" >20</td>
      <td id="T_18b7f_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_18b7f_row0_col2" class="data row0 col2" >LORE1</td>
      <td id="T_18b7f_row0_col3" class="data row0 col3" >IF hours.per.week > 38.0 AND occupation = Exec-managerial AND relationship = Not-in-family THEN class = >50K</td>
      <td id="T_18b7f_row0_col4" class="data row0 col4" >0.02575</td>
      <td id="T_18b7f_row0_col5" class="data row0 col5" >0.02787</td>
      <td id="T_18b7f_row0_col6" class="data row0 col6" >0.26065</td>
      <td id="T_18b7f_row0_col7" class="data row0 col7" >3</td>
      <td id="T_18b7f_row0_col8" class="data row0 col8" >0</td>
      <td id="T_18b7f_row0_col9" class="data row0 col9" >237.35692</td>
      <td id="T_18b7f_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_18b7f_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_18b7f_row1_col0" class="data row1 col0" >20</td>
      <td id="T_18b7f_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_18b7f_row1_col2" class="data row1 col2" >LORE5</td>
      <td id="T_18b7f_row1_col3" class="data row1 col3" >IF hours.per.week > 35.0 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_18b7f_row1_col4" class="data row1 col4" >0.07586</td>
      <td id="T_18b7f_row1_col5" class="data row1 col5" >0.16160</td>
      <td id="T_18b7f_row1_col6" class="data row1 col6" >0.51301</td>
      <td id="T_18b7f_row1_col7" class="data row1 col7" >3</td>
      <td id="T_18b7f_row1_col8" class="data row1 col8" >0</td>
      <td id="T_18b7f_row1_col9" class="data row1 col9" >160.11351</td>
      <td id="T_18b7f_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_18b7f_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_18b7f_row2_col0" class="data row2 col0" >20</td>
      <td id="T_18b7f_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_18b7f_row2_col2" class="data row2 col2" >LORE_SA1</td>
      <td id="T_18b7f_row2_col3" class="data row2 col3" >IF age > 23.0678 AND capital.loss <= 3051.6125 AND education.num != 8.0 AND education.num != 3.0 AND hours.per.week > 30.3276 AND native.country = United-States AND occupation = Exec-managerial AND race != Asian-Pac-Islander AND relationship = Not-in-family AND workclass = Private THEN class = >50K</td>
      <td id="T_18b7f_row2_col4" class="data row2 col4" >0.01768</td>
      <td id="T_18b7f_row2_col5" class="data row2 col5" >0.01986</td>
      <td id="T_18b7f_row2_col6" class="data row2 col6" >0.27047</td>
      <td id="T_18b7f_row2_col7" class="data row2 col7" >10</td>
      <td id="T_18b7f_row2_col8" class="data row2 col8" >0</td>
      <td id="T_18b7f_row2_col9" class="data row2 col9" >39.31556</td>
      <td id="T_18b7f_row2_col10" class="data row2 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 20, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.07586, Pre: 0.51301)



<style type="text/css">
#T_b22a6_row0_col0, #T_b22a6_row0_col1, #T_b22a6_row0_col2, #T_b22a6_row0_col3, #T_b22a6_row0_col4, #T_b22a6_row0_col5, #T_b22a6_row0_col6, #T_b22a6_row0_col7, #T_b22a6_row0_col8, #T_b22a6_row0_col9, #T_b22a6_row0_col10, #T_b22a6_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_b22a6">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b22a6_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_b22a6_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_b22a6_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_b22a6_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_b22a6_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_b22a6_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_b22a6_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_b22a6_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_b22a6_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_b22a6_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_b22a6_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_b22a6_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b22a6_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b22a6_row0_col0" class="data row0 col0" >20</td>
      <td id="T_b22a6_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_b22a6_row0_col2" class="data row0 col2" >LORE5</td>
      <td id="T_b22a6_row0_col3" class="data row0 col3" >IF hours.per.week > 35.0 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_b22a6_row0_col4" class="data row0 col4" >0.07586</td>
      <td id="T_b22a6_row0_col5" class="data row0 col5" >0.16160</td>
      <td id="T_b22a6_row0_col6" class="data row0 col6" >0.51301</td>
      <td id="T_b22a6_row0_col7" class="data row0 col7" >3</td>
      <td id="T_b22a6_row0_col8" class="data row0 col8" >0</td>
      <td id="T_b22a6_row0_col9" class="data row0 col9" >160.11351</td>
      <td id="T_b22a6_row0_col10" class="data row0 col10" >False</td>
      <td id="T_b22a6_row0_col11" class="data row0 col11" >0.00000</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_136.png)
    



### Rules for Instance 20, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.07586, Pre: 0.51301), Unique rules (diffrent features)



<style type="text/css">
#T_3c654_row0_col0, #T_3c654_row0_col1, #T_3c654_row0_col2, #T_3c654_row0_col3, #T_3c654_row0_col4, #T_3c654_row0_col5, #T_3c654_row0_col6, #T_3c654_row0_col7, #T_3c654_row0_col8, #T_3c654_row0_col9, #T_3c654_row0_col10, #T_3c654_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_3c654">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3c654_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_3c654_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_3c654_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_3c654_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_3c654_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_3c654_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_3c654_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_3c654_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_3c654_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_3c654_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_3c654_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_3c654_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3c654_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3c654_row0_col0" class="data row0 col0" >20</td>
      <td id="T_3c654_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_3c654_row0_col2" class="data row0 col2" >LORE5</td>
      <td id="T_3c654_row0_col3" class="data row0 col3" >IF hours.per.week > 35.0 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_3c654_row0_col4" class="data row0 col4" >0.07586</td>
      <td id="T_3c654_row0_col5" class="data row0 col5" >0.16160</td>
      <td id="T_3c654_row0_col6" class="data row0 col6" >0.51301</td>
      <td id="T_3c654_row0_col7" class="data row0 col7" >3</td>
      <td id="T_3c654_row0_col8" class="data row0 col8" >0</td>
      <td id="T_3c654_row0_col9" class="data row0 col9" >160.11351</td>
      <td id="T_3c654_row0_col10" class="data row0 col10" >False</td>
      <td id="T_3c654_row0_col11" class="data row0 col11" >0.00000</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_139.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_140.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_141.png)
    



### Rules for Instance 20, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.1616, Pre: 0.51301, Len: 0.51301)



<style type="text/css">
#T_7957f_row0_col0, #T_7957f_row0_col1, #T_7957f_row0_col2, #T_7957f_row0_col3, #T_7957f_row0_col4, #T_7957f_row0_col5, #T_7957f_row0_col6, #T_7957f_row0_col7, #T_7957f_row0_col8, #T_7957f_row0_col9, #T_7957f_row0_col10, #T_7957f_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_7957f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7957f_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_7957f_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_7957f_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_7957f_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_7957f_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_7957f_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_7957f_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_7957f_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_7957f_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_7957f_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_7957f_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_7957f_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7957f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_7957f_row0_col0" class="data row0 col0" >20</td>
      <td id="T_7957f_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_7957f_row0_col2" class="data row0 col2" >LORE5</td>
      <td id="T_7957f_row0_col3" class="data row0 col3" >IF hours.per.week > 35.0 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_7957f_row0_col4" class="data row0 col4" >0.07586</td>
      <td id="T_7957f_row0_col5" class="data row0 col5" >0.16160</td>
      <td id="T_7957f_row0_col6" class="data row0 col6" >0.51301</td>
      <td id="T_7957f_row0_col7" class="data row0 col7" >3</td>
      <td id="T_7957f_row0_col8" class="data row0 col8" >0</td>
      <td id="T_7957f_row0_col9" class="data row0 col9" >160.11351</td>
      <td id="T_7957f_row0_col10" class="data row0 col10" >False</td>
      <td id="T_7957f_row0_col11" class="data row0 col11" >2.48699</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 20, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.1616, Pre: 0.51301), Unique rules (diffrent features)



<style type="text/css">
#T_141a2_row0_col0, #T_141a2_row0_col1, #T_141a2_row0_col2, #T_141a2_row0_col3, #T_141a2_row0_col4, #T_141a2_row0_col5, #T_141a2_row0_col6, #T_141a2_row0_col7, #T_141a2_row0_col8, #T_141a2_row0_col9, #T_141a2_row0_col10, #T_141a2_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_141a2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_141a2_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_141a2_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_141a2_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_141a2_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_141a2_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_141a2_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_141a2_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_141a2_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_141a2_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_141a2_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_141a2_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_141a2_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_141a2_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_141a2_row0_col0" class="data row0 col0" >20</td>
      <td id="T_141a2_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_141a2_row0_col2" class="data row0 col2" >LORE5</td>
      <td id="T_141a2_row0_col3" class="data row0 col3" >IF hours.per.week > 35.0 AND occupation = Exec-managerial AND workclass = Private THEN class = >50K</td>
      <td id="T_141a2_row0_col4" class="data row0 col4" >0.07586</td>
      <td id="T_141a2_row0_col5" class="data row0 col5" >0.16160</td>
      <td id="T_141a2_row0_col6" class="data row0 col6" >0.51301</td>
      <td id="T_141a2_row0_col7" class="data row0 col7" >3</td>
      <td id="T_141a2_row0_col8" class="data row0 col8" >0</td>
      <td id="T_141a2_row0_col9" class="data row0 col9" >160.11351</td>
      <td id="T_141a2_row0_col10" class="data row0 col10" >False</td>
      <td id="T_141a2_row0_col11" class="data row0 col11" >2.48699</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_146.png)
    



## Instance 1808 (Original: >50K , Predicted: >50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>59.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Self-emp-not-inc</td>
    </tr>
    <tr>
      <th>education</th>
      <td>Bachelors</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>13</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Married-civ-spouse</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Sales</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Husband</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>15024.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>40.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 1808



<style type="text/css">
</style>
<table id="T_3681f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3681f_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_3681f_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_3681f_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_3681f_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_3681f_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_3681f_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_3681f_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_3681f_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_3681f_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_3681f_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_3681f_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3681f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3681f_row0_col0" class="data row0 col0" >1808</td>
      <td id="T_3681f_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_3681f_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_3681f_row0_col3" class="data row0 col3" >IF age > 37.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_3681f_row0_col4" class="data row0 col4" >0.00873</td>
      <td id="T_3681f_row0_col5" class="data row0 col5" >0.03279</td>
      <td id="T_3681f_row0_col6" class="data row0 col6" >0.90452</td>
      <td id="T_3681f_row0_col7" class="data row0 col7" >6</td>
      <td id="T_3681f_row0_col8" class="data row0 col8" >0</td>
      <td id="T_3681f_row0_col9" class="data row0 col9" >17.98463</td>
      <td id="T_3681f_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_3681f_row1_col0" class="data row1 col0" >1808</td>
      <td id="T_3681f_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_3681f_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_3681f_row1_col3" class="data row1 col3" >IF age > 37.0 AND capital.gain > 0.0 AND education = Bachelors AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_3681f_row1_col4" class="data row1 col4" >0.00873</td>
      <td id="T_3681f_row1_col5" class="data row1 col5" >0.03279</td>
      <td id="T_3681f_row1_col6" class="data row1 col6" >0.90452</td>
      <td id="T_3681f_row1_col7" class="data row1 col7" >6</td>
      <td id="T_3681f_row1_col8" class="data row1 col8" >0</td>
      <td id="T_3681f_row1_col9" class="data row1 col9" >17.25049</td>
      <td id="T_3681f_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_3681f_row2_col0" class="data row2 col0" >1808</td>
      <td id="T_3681f_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_3681f_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_3681f_row2_col3" class="data row2 col3" >IF age > 28.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_3681f_row2_col4" class="data row2 col4" >0.01150</td>
      <td id="T_3681f_row2_col5" class="data row2 col5" >0.04318</td>
      <td id="T_3681f_row2_col6" class="data row2 col6" >0.90458</td>
      <td id="T_3681f_row2_col7" class="data row2 col7" >6</td>
      <td id="T_3681f_row2_col8" class="data row2 col8" >0</td>
      <td id="T_3681f_row2_col9" class="data row2 col9" >18.25926</td>
      <td id="T_3681f_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_3681f_row3_col0" class="data row3 col0" >1808</td>
      <td id="T_3681f_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_3681f_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_3681f_row3_col3" class="data row3 col3" >IF age > 28.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_3681f_row3_col4" class="data row3 col4" >0.01150</td>
      <td id="T_3681f_row3_col5" class="data row3 col5" >0.04318</td>
      <td id="T_3681f_row3_col6" class="data row3 col6" >0.90458</td>
      <td id="T_3681f_row3_col7" class="data row3 col7" >7</td>
      <td id="T_3681f_row3_col8" class="data row3 col8" >0</td>
      <td id="T_3681f_row3_col9" class="data row3 col9" >17.01136</td>
      <td id="T_3681f_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_3681f_row4_col0" class="data row4 col0" >1808</td>
      <td id="T_3681f_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_3681f_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_3681f_row4_col3" class="data row4 col3" >IF age > 28.0 AND capital.gain > 0.0 AND capital.loss <= 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_3681f_row4_col4" class="data row4 col4" >0.01150</td>
      <td id="T_3681f_row4_col5" class="data row4 col5" >0.04318</td>
      <td id="T_3681f_row4_col6" class="data row4 col6" >0.90458</td>
      <td id="T_3681f_row4_col7" class="data row4 col7" >7</td>
      <td id="T_3681f_row4_col8" class="data row4 col8" >1</td>
      <td id="T_3681f_row4_col9" class="data row4 col9" >38.42346</td>
      <td id="T_3681f_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_3681f_row5_col0" class="data row5 col0" >1808</td>
      <td id="T_3681f_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_3681f_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_3681f_row5_col3" class="data row5 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_3681f_row5_col4" class="data row5 col4" >0.04923</td>
      <td id="T_3681f_row5_col5" class="data row5 col5" >0.19366</td>
      <td id="T_3681f_row5_col6" class="data row5 col6" >0.94742</td>
      <td id="T_3681f_row5_col7" class="data row5 col7" >1</td>
      <td id="T_3681f_row5_col8" class="data row5 col8" >0</td>
      <td id="T_3681f_row5_col9" class="data row5 col9" >197.42534</td>
      <td id="T_3681f_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_3681f_row6_col0" class="data row6 col0" >1808</td>
      <td id="T_3681f_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_3681f_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_3681f_row6_col3" class="data row6 col3" >IF capital.gain > 5212.2262 THEN class = >50K</td>
      <td id="T_3681f_row6_col4" class="data row6 col4" >0.04602</td>
      <td id="T_3681f_row6_col5" class="data row6 col5" >0.18054</td>
      <td id="T_3681f_row6_col6" class="data row6 col6" >0.94471</td>
      <td id="T_3681f_row6_col7" class="data row6 col7" >1</td>
      <td id="T_3681f_row6_col8" class="data row6 col8" >1</td>
      <td id="T_3681f_row6_col9" class="data row6 col9" >294.42594</td>
      <td id="T_3681f_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_3681f_row7_col0" class="data row7 col0" >1808</td>
      <td id="T_3681f_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_3681f_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_3681f_row7_col3" class="data row7 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_3681f_row7_col4" class="data row7 col4" >0.02694</td>
      <td id="T_3681f_row7_col5" class="data row7 col5" >0.10931</td>
      <td id="T_3681f_row7_col6" class="data row7 col6" >0.97720</td>
      <td id="T_3681f_row7_col7" class="data row7 col7" >1</td>
      <td id="T_3681f_row7_col8" class="data row7 col8" >0</td>
      <td id="T_3681f_row7_col9" class="data row7 col9" >176.13512</td>
      <td id="T_3681f_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_3681f_row8_col0" class="data row8 col0" >1808</td>
      <td id="T_3681f_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_3681f_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_3681f_row8_col3" class="data row8 col3" >IF capital.gain > 4386.0 THEN class = >50K</td>
      <td id="T_3681f_row8_col4" class="data row8 col4" >0.05476</td>
      <td id="T_3681f_row8_col5" class="data row8 col5" >0.19767</td>
      <td id="T_3681f_row8_col6" class="data row8 col6" >0.86939</td>
      <td id="T_3681f_row8_col7" class="data row8 col7" >1</td>
      <td id="T_3681f_row8_col8" class="data row8 col8" >1</td>
      <td id="T_3681f_row8_col9" class="data row8 col9" >365.37206</td>
      <td id="T_3681f_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_3681f_row9_col0" class="data row9 col0" >1808</td>
      <td id="T_3681f_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_3681f_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_3681f_row9_col3" class="data row9 col3" >IF age > 38.0 AND capital.gain > 4416.0 THEN class = >50K</td>
      <td id="T_3681f_row9_col4" class="data row9 col4" >0.03681</td>
      <td id="T_3681f_row9_col5" class="data row9 col5" >0.13773</td>
      <td id="T_3681f_row9_col6" class="data row9 col6" >0.90107</td>
      <td id="T_3681f_row9_col7" class="data row9 col7" >2</td>
      <td id="T_3681f_row9_col8" class="data row9 col8" >0</td>
      <td id="T_3681f_row9_col9" class="data row9 col9" >244.33531</td>
      <td id="T_3681f_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_3681f_row10_col0" class="data row10 col0" >1808</td>
      <td id="T_3681f_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_3681f_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_3681f_row10_col3" class="data row10 col3" >IF native.country != Honduras AND relationship = Husband THEN class = >50K</td>
      <td id="T_3681f_row10_col4" class="data row10 col4" >0.40479</td>
      <td id="T_3681f_row10_col5" class="data row10 col5" >0.75752</td>
      <td id="T_3681f_row10_col6" class="data row10 col6" >0.45068</td>
      <td id="T_3681f_row10_col7" class="data row10 col7" >2</td>
      <td id="T_3681f_row10_col8" class="data row10 col8" >0</td>
      <td id="T_3681f_row10_col9" class="data row10 col9" >43.34698</td>
      <td id="T_3681f_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_3681f_row11_col0" class="data row11 col0" >1808</td>
      <td id="T_3681f_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_3681f_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_3681f_row11_col3" class="data row11 col3" >IF capital.gain > 14964.5977 AND relationship != Own-child THEN class = >50K</td>
      <td id="T_3681f_row11_col4" class="data row11 col4" >0.01909</td>
      <td id="T_3681f_row11_col5" class="data row11 col5" >0.07834</td>
      <td id="T_3681f_row11_col6" class="data row11 col6" >0.98851</td>
      <td id="T_3681f_row11_col7" class="data row11 col7" >2</td>
      <td id="T_3681f_row11_col8" class="data row11 col8" >0</td>
      <td id="T_3681f_row11_col9" class="data row11 col9" >42.62766</td>
      <td id="T_3681f_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_3681f_row12_col0" class="data row12 col0" >1808</td>
      <td id="T_3681f_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_3681f_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_3681f_row12_col3" class="data row12 col3" >IF capital.gain > 5356.9038 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_3681f_row12_col4" class="data row12 col4" >0.03471</td>
      <td id="T_3681f_row12_col5" class="data row12 col5" >0.14192</td>
      <td id="T_3681f_row12_col6" class="data row12 col6" >0.98483</td>
      <td id="T_3681f_row12_col7" class="data row12 col7" >2</td>
      <td id="T_3681f_row12_col8" class="data row12 col8" >0</td>
      <td id="T_3681f_row12_col9" class="data row12 col9" >41.51357</td>
      <td id="T_3681f_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_3681f_row13_col0" class="data row13 col0" >1808</td>
      <td id="T_3681f_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_3681f_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_3681f_row13_col3" class="data row13 col3" >IF age > 42.5082 AND capital.gain > 572.1054 AND occupation != Other-service THEN class = >50K</td>
      <td id="T_3681f_row13_col4" class="data row13 col4" >0.04072</td>
      <td id="T_3681f_row13_col5" class="data row13 col5" >0.11933</td>
      <td id="T_3681f_row13_col6" class="data row13 col6" >0.70582</td>
      <td id="T_3681f_row13_col7" class="data row13 col7" >3</td>
      <td id="T_3681f_row13_col8" class="data row13 col8" >0</td>
      <td id="T_3681f_row13_col9" class="data row13 col9" >43.46254</td>
      <td id="T_3681f_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_3681f_row14_col0" class="data row14 col0" >1808</td>
      <td id="T_3681f_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_3681f_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_3681f_row14_col3" class="data row14 col3" >IF native.country != Portugal AND relationship != Own-child THEN class = >50K</td>
      <td id="T_3681f_row14_col4" class="data row14 col4" >0.84337</td>
      <td id="T_3681f_row14_col5" class="data row14 col5" >0.99217</td>
      <td id="T_3681f_row14_col6" class="data row14 col6" >0.28332</td>
      <td id="T_3681f_row14_col7" class="data row14 col7" >2</td>
      <td id="T_3681f_row14_col8" class="data row14 col8" >0</td>
      <td id="T_3681f_row14_col9" class="data row14 col9" >42.49464</td>
      <td id="T_3681f_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_3681f_row15_col0" class="data row15 col0" >1808</td>
      <td id="T_3681f_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_3681f_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_3681f_row15_col3" class="data row15 col3" >IF capital.gain > 7688.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_3681f_row15_col4" class="data row15 col4" >0.01584</td>
      <td id="T_3681f_row15_col5" class="data row15 col5" >0.06486</td>
      <td id="T_3681f_row15_col6" class="data row15 col6" >0.98615</td>
      <td id="T_3681f_row15_col7" class="data row15 col7" >2</td>
      <td id="T_3681f_row15_col8" class="data row15 col8" >0</td>
      <td id="T_3681f_row15_col9" class="data row15 col9" >13.07816</td>
      <td id="T_3681f_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_3681f_row16_col0" class="data row16 col0" >1808</td>
      <td id="T_3681f_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_3681f_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_3681f_row16_col3" class="data row16 col3" >IF age > 33.8902 AND capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_3681f_row16_col4" class="data row16 col4" >0.02303</td>
      <td id="T_3681f_row16_col5" class="data row16 col5" >0.09401</td>
      <td id="T_3681f_row16_col6" class="data row16 col6" >0.98286</td>
      <td id="T_3681f_row16_col7" class="data row16 col7" >2</td>
      <td id="T_3681f_row16_col8" class="data row16 col8" >1</td>
      <td id="T_3681f_row16_col9" class="data row16 col9" >17.98777</td>
      <td id="T_3681f_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_3681f_row17_col0" class="data row17 col0" >1808</td>
      <td id="T_3681f_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_3681f_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_3681f_row17_col3" class="data row17 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_3681f_row17_col4" class="data row17 col4" >0.02694</td>
      <td id="T_3681f_row17_col5" class="data row17 col5" >0.10931</td>
      <td id="T_3681f_row17_col6" class="data row17 col6" >0.97720</td>
      <td id="T_3681f_row17_col7" class="data row17 col7" >1</td>
      <td id="T_3681f_row17_col8" class="data row17 col8" >0</td>
      <td id="T_3681f_row17_col9" class="data row17 col9" >10.93505</td>
      <td id="T_3681f_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_3681f_row18_col0" class="data row18 col0" >1808</td>
      <td id="T_3681f_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_3681f_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_3681f_row18_col3" class="data row18 col3" >IF age > 24.452 AND capital.gain > 10520.0 THEN class = >50K</td>
      <td id="T_3681f_row18_col4" class="data row18 col4" >0.02242</td>
      <td id="T_3681f_row18_col5" class="data row18 col5" >0.09146</td>
      <td id="T_3681f_row18_col6" class="data row18 col6" >0.98239</td>
      <td id="T_3681f_row18_col7" class="data row18 col7" >2</td>
      <td id="T_3681f_row18_col8" class="data row18 col8" >0</td>
      <td id="T_3681f_row18_col9" class="data row18 col9" >9.72879</td>
      <td id="T_3681f_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_3681f_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_3681f_row19_col0" class="data row19 col0" >1808</td>
      <td id="T_3681f_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_3681f_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_3681f_row19_col3" class="data row19 col3" >IF age > 28.4951 AND capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_3681f_row19_col4" class="data row19 col4" >0.02558</td>
      <td id="T_3681f_row19_col5" class="data row19 col5" >0.10457</td>
      <td id="T_3681f_row19_col6" class="data row19 col6" >0.98456</td>
      <td id="T_3681f_row19_col7" class="data row19 col7" >2</td>
      <td id="T_3681f_row19_col8" class="data row19 col8" >1</td>
      <td id="T_3681f_row19_col9" class="data row19 col9" >21.77034</td>
      <td id="T_3681f_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 1808, Correct Prediction



<style type="text/css">
</style>
<table id="T_a7957">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a7957_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_a7957_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_a7957_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_a7957_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_a7957_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_a7957_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_a7957_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_a7957_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_a7957_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_a7957_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_a7957_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a7957_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a7957_row0_col0" class="data row0 col0" >1808</td>
      <td id="T_a7957_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_a7957_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_a7957_row0_col3" class="data row0 col3" >IF age > 37.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_a7957_row0_col4" class="data row0 col4" >0.00873</td>
      <td id="T_a7957_row0_col5" class="data row0 col5" >0.03279</td>
      <td id="T_a7957_row0_col6" class="data row0 col6" >0.90452</td>
      <td id="T_a7957_row0_col7" class="data row0 col7" >6</td>
      <td id="T_a7957_row0_col8" class="data row0 col8" >0</td>
      <td id="T_a7957_row0_col9" class="data row0 col9" >17.98463</td>
      <td id="T_a7957_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a7957_row1_col0" class="data row1 col0" >1808</td>
      <td id="T_a7957_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_a7957_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_a7957_row1_col3" class="data row1 col3" >IF age > 37.0 AND capital.gain > 0.0 AND education = Bachelors AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_a7957_row1_col4" class="data row1 col4" >0.00873</td>
      <td id="T_a7957_row1_col5" class="data row1 col5" >0.03279</td>
      <td id="T_a7957_row1_col6" class="data row1 col6" >0.90452</td>
      <td id="T_a7957_row1_col7" class="data row1 col7" >6</td>
      <td id="T_a7957_row1_col8" class="data row1 col8" >0</td>
      <td id="T_a7957_row1_col9" class="data row1 col9" >17.25049</td>
      <td id="T_a7957_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a7957_row2_col0" class="data row2 col0" >1808</td>
      <td id="T_a7957_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_a7957_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_a7957_row2_col3" class="data row2 col3" >IF age > 28.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_a7957_row2_col4" class="data row2 col4" >0.01150</td>
      <td id="T_a7957_row2_col5" class="data row2 col5" >0.04318</td>
      <td id="T_a7957_row2_col6" class="data row2 col6" >0.90458</td>
      <td id="T_a7957_row2_col7" class="data row2 col7" >6</td>
      <td id="T_a7957_row2_col8" class="data row2 col8" >0</td>
      <td id="T_a7957_row2_col9" class="data row2 col9" >18.25926</td>
      <td id="T_a7957_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a7957_row3_col0" class="data row3 col0" >1808</td>
      <td id="T_a7957_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_a7957_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_a7957_row3_col3" class="data row3 col3" >IF age > 28.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_a7957_row3_col4" class="data row3 col4" >0.01150</td>
      <td id="T_a7957_row3_col5" class="data row3 col5" >0.04318</td>
      <td id="T_a7957_row3_col6" class="data row3 col6" >0.90458</td>
      <td id="T_a7957_row3_col7" class="data row3 col7" >7</td>
      <td id="T_a7957_row3_col8" class="data row3 col8" >0</td>
      <td id="T_a7957_row3_col9" class="data row3 col9" >17.01136</td>
      <td id="T_a7957_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_a7957_row4_col0" class="data row4 col0" >1808</td>
      <td id="T_a7957_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_a7957_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_a7957_row4_col3" class="data row4 col3" >IF age > 28.0 AND capital.gain > 0.0 AND capital.loss <= 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_a7957_row4_col4" class="data row4 col4" >0.01150</td>
      <td id="T_a7957_row4_col5" class="data row4 col5" >0.04318</td>
      <td id="T_a7957_row4_col6" class="data row4 col6" >0.90458</td>
      <td id="T_a7957_row4_col7" class="data row4 col7" >7</td>
      <td id="T_a7957_row4_col8" class="data row4 col8" >1</td>
      <td id="T_a7957_row4_col9" class="data row4 col9" >38.42346</td>
      <td id="T_a7957_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_a7957_row5_col0" class="data row5 col0" >1808</td>
      <td id="T_a7957_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_a7957_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_a7957_row5_col3" class="data row5 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_a7957_row5_col4" class="data row5 col4" >0.04923</td>
      <td id="T_a7957_row5_col5" class="data row5 col5" >0.19366</td>
      <td id="T_a7957_row5_col6" class="data row5 col6" >0.94742</td>
      <td id="T_a7957_row5_col7" class="data row5 col7" >1</td>
      <td id="T_a7957_row5_col8" class="data row5 col8" >0</td>
      <td id="T_a7957_row5_col9" class="data row5 col9" >197.42534</td>
      <td id="T_a7957_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_a7957_row6_col0" class="data row6 col0" >1808</td>
      <td id="T_a7957_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_a7957_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_a7957_row6_col3" class="data row6 col3" >IF capital.gain > 5212.2262 THEN class = >50K</td>
      <td id="T_a7957_row6_col4" class="data row6 col4" >0.04602</td>
      <td id="T_a7957_row6_col5" class="data row6 col5" >0.18054</td>
      <td id="T_a7957_row6_col6" class="data row6 col6" >0.94471</td>
      <td id="T_a7957_row6_col7" class="data row6 col7" >1</td>
      <td id="T_a7957_row6_col8" class="data row6 col8" >1</td>
      <td id="T_a7957_row6_col9" class="data row6 col9" >294.42594</td>
      <td id="T_a7957_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_a7957_row7_col0" class="data row7 col0" >1808</td>
      <td id="T_a7957_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_a7957_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_a7957_row7_col3" class="data row7 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_a7957_row7_col4" class="data row7 col4" >0.02694</td>
      <td id="T_a7957_row7_col5" class="data row7 col5" >0.10931</td>
      <td id="T_a7957_row7_col6" class="data row7 col6" >0.97720</td>
      <td id="T_a7957_row7_col7" class="data row7 col7" >1</td>
      <td id="T_a7957_row7_col8" class="data row7 col8" >0</td>
      <td id="T_a7957_row7_col9" class="data row7 col9" >176.13512</td>
      <td id="T_a7957_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_a7957_row8_col0" class="data row8 col0" >1808</td>
      <td id="T_a7957_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_a7957_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_a7957_row8_col3" class="data row8 col3" >IF capital.gain > 4386.0 THEN class = >50K</td>
      <td id="T_a7957_row8_col4" class="data row8 col4" >0.05476</td>
      <td id="T_a7957_row8_col5" class="data row8 col5" >0.19767</td>
      <td id="T_a7957_row8_col6" class="data row8 col6" >0.86939</td>
      <td id="T_a7957_row8_col7" class="data row8 col7" >1</td>
      <td id="T_a7957_row8_col8" class="data row8 col8" >1</td>
      <td id="T_a7957_row8_col9" class="data row8 col9" >365.37206</td>
      <td id="T_a7957_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_a7957_row9_col0" class="data row9 col0" >1808</td>
      <td id="T_a7957_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_a7957_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_a7957_row9_col3" class="data row9 col3" >IF age > 38.0 AND capital.gain > 4416.0 THEN class = >50K</td>
      <td id="T_a7957_row9_col4" class="data row9 col4" >0.03681</td>
      <td id="T_a7957_row9_col5" class="data row9 col5" >0.13773</td>
      <td id="T_a7957_row9_col6" class="data row9 col6" >0.90107</td>
      <td id="T_a7957_row9_col7" class="data row9 col7" >2</td>
      <td id="T_a7957_row9_col8" class="data row9 col8" >0</td>
      <td id="T_a7957_row9_col9" class="data row9 col9" >244.33531</td>
      <td id="T_a7957_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_a7957_row10_col0" class="data row10 col0" >1808</td>
      <td id="T_a7957_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_a7957_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_a7957_row10_col3" class="data row10 col3" >IF native.country != Honduras AND relationship = Husband THEN class = >50K</td>
      <td id="T_a7957_row10_col4" class="data row10 col4" >0.40479</td>
      <td id="T_a7957_row10_col5" class="data row10 col5" >0.75752</td>
      <td id="T_a7957_row10_col6" class="data row10 col6" >0.45068</td>
      <td id="T_a7957_row10_col7" class="data row10 col7" >2</td>
      <td id="T_a7957_row10_col8" class="data row10 col8" >0</td>
      <td id="T_a7957_row10_col9" class="data row10 col9" >43.34698</td>
      <td id="T_a7957_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_a7957_row11_col0" class="data row11 col0" >1808</td>
      <td id="T_a7957_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_a7957_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_a7957_row11_col3" class="data row11 col3" >IF capital.gain > 14964.5977 AND relationship != Own-child THEN class = >50K</td>
      <td id="T_a7957_row11_col4" class="data row11 col4" >0.01909</td>
      <td id="T_a7957_row11_col5" class="data row11 col5" >0.07834</td>
      <td id="T_a7957_row11_col6" class="data row11 col6" >0.98851</td>
      <td id="T_a7957_row11_col7" class="data row11 col7" >2</td>
      <td id="T_a7957_row11_col8" class="data row11 col8" >0</td>
      <td id="T_a7957_row11_col9" class="data row11 col9" >42.62766</td>
      <td id="T_a7957_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_a7957_row12_col0" class="data row12 col0" >1808</td>
      <td id="T_a7957_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_a7957_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_a7957_row12_col3" class="data row12 col3" >IF capital.gain > 5356.9038 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_a7957_row12_col4" class="data row12 col4" >0.03471</td>
      <td id="T_a7957_row12_col5" class="data row12 col5" >0.14192</td>
      <td id="T_a7957_row12_col6" class="data row12 col6" >0.98483</td>
      <td id="T_a7957_row12_col7" class="data row12 col7" >2</td>
      <td id="T_a7957_row12_col8" class="data row12 col8" >0</td>
      <td id="T_a7957_row12_col9" class="data row12 col9" >41.51357</td>
      <td id="T_a7957_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_a7957_row13_col0" class="data row13 col0" >1808</td>
      <td id="T_a7957_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_a7957_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_a7957_row13_col3" class="data row13 col3" >IF age > 42.5082 AND capital.gain > 572.1054 AND occupation != Other-service THEN class = >50K</td>
      <td id="T_a7957_row13_col4" class="data row13 col4" >0.04072</td>
      <td id="T_a7957_row13_col5" class="data row13 col5" >0.11933</td>
      <td id="T_a7957_row13_col6" class="data row13 col6" >0.70582</td>
      <td id="T_a7957_row13_col7" class="data row13 col7" >3</td>
      <td id="T_a7957_row13_col8" class="data row13 col8" >0</td>
      <td id="T_a7957_row13_col9" class="data row13 col9" >43.46254</td>
      <td id="T_a7957_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_a7957_row14_col0" class="data row14 col0" >1808</td>
      <td id="T_a7957_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_a7957_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_a7957_row14_col3" class="data row14 col3" >IF native.country != Portugal AND relationship != Own-child THEN class = >50K</td>
      <td id="T_a7957_row14_col4" class="data row14 col4" >0.84337</td>
      <td id="T_a7957_row14_col5" class="data row14 col5" >0.99217</td>
      <td id="T_a7957_row14_col6" class="data row14 col6" >0.28332</td>
      <td id="T_a7957_row14_col7" class="data row14 col7" >2</td>
      <td id="T_a7957_row14_col8" class="data row14 col8" >0</td>
      <td id="T_a7957_row14_col9" class="data row14 col9" >42.49464</td>
      <td id="T_a7957_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_a7957_row15_col0" class="data row15 col0" >1808</td>
      <td id="T_a7957_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_a7957_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_a7957_row15_col3" class="data row15 col3" >IF capital.gain > 7688.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_a7957_row15_col4" class="data row15 col4" >0.01584</td>
      <td id="T_a7957_row15_col5" class="data row15 col5" >0.06486</td>
      <td id="T_a7957_row15_col6" class="data row15 col6" >0.98615</td>
      <td id="T_a7957_row15_col7" class="data row15 col7" >2</td>
      <td id="T_a7957_row15_col8" class="data row15 col8" >0</td>
      <td id="T_a7957_row15_col9" class="data row15 col9" >13.07816</td>
      <td id="T_a7957_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_a7957_row16_col0" class="data row16 col0" >1808</td>
      <td id="T_a7957_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_a7957_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_a7957_row16_col3" class="data row16 col3" >IF age > 33.8902 AND capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_a7957_row16_col4" class="data row16 col4" >0.02303</td>
      <td id="T_a7957_row16_col5" class="data row16 col5" >0.09401</td>
      <td id="T_a7957_row16_col6" class="data row16 col6" >0.98286</td>
      <td id="T_a7957_row16_col7" class="data row16 col7" >2</td>
      <td id="T_a7957_row16_col8" class="data row16 col8" >1</td>
      <td id="T_a7957_row16_col9" class="data row16 col9" >17.98777</td>
      <td id="T_a7957_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_a7957_row17_col0" class="data row17 col0" >1808</td>
      <td id="T_a7957_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_a7957_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_a7957_row17_col3" class="data row17 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_a7957_row17_col4" class="data row17 col4" >0.02694</td>
      <td id="T_a7957_row17_col5" class="data row17 col5" >0.10931</td>
      <td id="T_a7957_row17_col6" class="data row17 col6" >0.97720</td>
      <td id="T_a7957_row17_col7" class="data row17 col7" >1</td>
      <td id="T_a7957_row17_col8" class="data row17 col8" >0</td>
      <td id="T_a7957_row17_col9" class="data row17 col9" >10.93505</td>
      <td id="T_a7957_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_a7957_row18_col0" class="data row18 col0" >1808</td>
      <td id="T_a7957_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_a7957_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_a7957_row18_col3" class="data row18 col3" >IF age > 24.452 AND capital.gain > 10520.0 THEN class = >50K</td>
      <td id="T_a7957_row18_col4" class="data row18 col4" >0.02242</td>
      <td id="T_a7957_row18_col5" class="data row18 col5" >0.09146</td>
      <td id="T_a7957_row18_col6" class="data row18 col6" >0.98239</td>
      <td id="T_a7957_row18_col7" class="data row18 col7" >2</td>
      <td id="T_a7957_row18_col8" class="data row18 col8" >0</td>
      <td id="T_a7957_row18_col9" class="data row18 col9" >9.72879</td>
      <td id="T_a7957_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_a7957_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_a7957_row19_col0" class="data row19 col0" >1808</td>
      <td id="T_a7957_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_a7957_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_a7957_row19_col3" class="data row19 col3" >IF age > 28.4951 AND capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_a7957_row19_col4" class="data row19 col4" >0.02558</td>
      <td id="T_a7957_row19_col5" class="data row19 col5" >0.10457</td>
      <td id="T_a7957_row19_col6" class="data row19 col6" >0.98456</td>
      <td id="T_a7957_row19_col7" class="data row19 col7" >2</td>
      <td id="T_a7957_row19_col8" class="data row19 col8" >1</td>
      <td id="T_a7957_row19_col9" class="data row19 col9" >21.77034</td>
      <td id="T_a7957_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 1808, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_b2f4c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b2f4c_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_b2f4c_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_b2f4c_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_b2f4c_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_b2f4c_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_b2f4c_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_b2f4c_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_b2f4c_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_b2f4c_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_b2f4c_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_b2f4c_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b2f4c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b2f4c_row0_col0" class="data row0 col0" >1808</td>
      <td id="T_b2f4c_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_b2f4c_row0_col2" class="data row0 col2" >ANCHOR3</td>
      <td id="T_b2f4c_row0_col3" class="data row0 col3" >IF age > 28.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b2f4c_row0_col4" class="data row0 col4" >0.01150</td>
      <td id="T_b2f4c_row0_col5" class="data row0 col5" >0.04318</td>
      <td id="T_b2f4c_row0_col6" class="data row0 col6" >0.90458</td>
      <td id="T_b2f4c_row0_col7" class="data row0 col7" >6</td>
      <td id="T_b2f4c_row0_col8" class="data row0 col8" >0</td>
      <td id="T_b2f4c_row0_col9" class="data row0 col9" >18.25926</td>
      <td id="T_b2f4c_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_b2f4c_row1_col0" class="data row1 col0" >1808</td>
      <td id="T_b2f4c_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_b2f4c_row1_col2" class="data row1 col2" >ANCHOR4</td>
      <td id="T_b2f4c_row1_col3" class="data row1 col3" >IF age > 28.0 AND capital.gain > 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband AND sex = Male THEN class = >50K</td>
      <td id="T_b2f4c_row1_col4" class="data row1 col4" >0.01150</td>
      <td id="T_b2f4c_row1_col5" class="data row1 col5" >0.04318</td>
      <td id="T_b2f4c_row1_col6" class="data row1 col6" >0.90458</td>
      <td id="T_b2f4c_row1_col7" class="data row1 col7" >7</td>
      <td id="T_b2f4c_row1_col8" class="data row1 col8" >0</td>
      <td id="T_b2f4c_row1_col9" class="data row1 col9" >17.01136</td>
      <td id="T_b2f4c_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_b2f4c_row2_col0" class="data row2 col0" >1808</td>
      <td id="T_b2f4c_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_b2f4c_row2_col2" class="data row2 col2" >ANCHOR5</td>
      <td id="T_b2f4c_row2_col3" class="data row2 col3" >IF age > 28.0 AND capital.gain > 0.0 AND capital.loss <= 0.0 AND education = Bachelors AND education.num = 13.0 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = >50K</td>
      <td id="T_b2f4c_row2_col4" class="data row2 col4" >0.01150</td>
      <td id="T_b2f4c_row2_col5" class="data row2 col5" >0.04318</td>
      <td id="T_b2f4c_row2_col6" class="data row2 col6" >0.90458</td>
      <td id="T_b2f4c_row2_col7" class="data row2 col7" >7</td>
      <td id="T_b2f4c_row2_col8" class="data row2 col8" >1</td>
      <td id="T_b2f4c_row2_col9" class="data row2 col9" >38.42346</td>
      <td id="T_b2f4c_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_b2f4c_row3_col0" class="data row3 col0" >1808</td>
      <td id="T_b2f4c_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_b2f4c_row3_col2" class="data row3 col2" >LORE1</td>
      <td id="T_b2f4c_row3_col3" class="data row3 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_b2f4c_row3_col4" class="data row3 col4" >0.04923</td>
      <td id="T_b2f4c_row3_col5" class="data row3 col5" >0.19366</td>
      <td id="T_b2f4c_row3_col6" class="data row3 col6" >0.94742</td>
      <td id="T_b2f4c_row3_col7" class="data row3 col7" >1</td>
      <td id="T_b2f4c_row3_col8" class="data row3 col8" >0</td>
      <td id="T_b2f4c_row3_col9" class="data row3 col9" >197.42534</td>
      <td id="T_b2f4c_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_b2f4c_row4_col0" class="data row4 col0" >1808</td>
      <td id="T_b2f4c_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_b2f4c_row4_col2" class="data row4 col2" >LORE2</td>
      <td id="T_b2f4c_row4_col3" class="data row4 col3" >IF capital.gain > 5212.2262 THEN class = >50K</td>
      <td id="T_b2f4c_row4_col4" class="data row4 col4" >0.04602</td>
      <td id="T_b2f4c_row4_col5" class="data row4 col5" >0.18054</td>
      <td id="T_b2f4c_row4_col6" class="data row4 col6" >0.94471</td>
      <td id="T_b2f4c_row4_col7" class="data row4 col7" >1</td>
      <td id="T_b2f4c_row4_col8" class="data row4 col8" >1</td>
      <td id="T_b2f4c_row4_col9" class="data row4 col9" >294.42594</td>
      <td id="T_b2f4c_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_b2f4c_row5_col0" class="data row5 col0" >1808</td>
      <td id="T_b2f4c_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_b2f4c_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_b2f4c_row5_col3" class="data row5 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_b2f4c_row5_col4" class="data row5 col4" >0.02694</td>
      <td id="T_b2f4c_row5_col5" class="data row5 col5" >0.10931</td>
      <td id="T_b2f4c_row5_col6" class="data row5 col6" >0.97720</td>
      <td id="T_b2f4c_row5_col7" class="data row5 col7" >1</td>
      <td id="T_b2f4c_row5_col8" class="data row5 col8" >0</td>
      <td id="T_b2f4c_row5_col9" class="data row5 col9" >176.13512</td>
      <td id="T_b2f4c_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_b2f4c_row6_col0" class="data row6 col0" >1808</td>
      <td id="T_b2f4c_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_b2f4c_row6_col2" class="data row6 col2" >LORE4</td>
      <td id="T_b2f4c_row6_col3" class="data row6 col3" >IF capital.gain > 4386.0 THEN class = >50K</td>
      <td id="T_b2f4c_row6_col4" class="data row6 col4" >0.05476</td>
      <td id="T_b2f4c_row6_col5" class="data row6 col5" >0.19767</td>
      <td id="T_b2f4c_row6_col6" class="data row6 col6" >0.86939</td>
      <td id="T_b2f4c_row6_col7" class="data row6 col7" >1</td>
      <td id="T_b2f4c_row6_col8" class="data row6 col8" >1</td>
      <td id="T_b2f4c_row6_col9" class="data row6 col9" >365.37206</td>
      <td id="T_b2f4c_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_b2f4c_row7_col0" class="data row7 col0" >1808</td>
      <td id="T_b2f4c_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_b2f4c_row7_col2" class="data row7 col2" >LORE5</td>
      <td id="T_b2f4c_row7_col3" class="data row7 col3" >IF age > 38.0 AND capital.gain > 4416.0 THEN class = >50K</td>
      <td id="T_b2f4c_row7_col4" class="data row7 col4" >0.03681</td>
      <td id="T_b2f4c_row7_col5" class="data row7 col5" >0.13773</td>
      <td id="T_b2f4c_row7_col6" class="data row7 col6" >0.90107</td>
      <td id="T_b2f4c_row7_col7" class="data row7 col7" >2</td>
      <td id="T_b2f4c_row7_col8" class="data row7 col8" >0</td>
      <td id="T_b2f4c_row7_col9" class="data row7 col9" >244.33531</td>
      <td id="T_b2f4c_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_b2f4c_row8_col0" class="data row8 col0" >1808</td>
      <td id="T_b2f4c_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_b2f4c_row8_col2" class="data row8 col2" >LORE_SA1</td>
      <td id="T_b2f4c_row8_col3" class="data row8 col3" >IF native.country != Honduras AND relationship = Husband THEN class = >50K</td>
      <td id="T_b2f4c_row8_col4" class="data row8 col4" >0.40479</td>
      <td id="T_b2f4c_row8_col5" class="data row8 col5" >0.75752</td>
      <td id="T_b2f4c_row8_col6" class="data row8 col6" >0.45068</td>
      <td id="T_b2f4c_row8_col7" class="data row8 col7" >2</td>
      <td id="T_b2f4c_row8_col8" class="data row8 col8" >0</td>
      <td id="T_b2f4c_row8_col9" class="data row8 col9" >43.34698</td>
      <td id="T_b2f4c_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_b2f4c_row9_col0" class="data row9 col0" >1808</td>
      <td id="T_b2f4c_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_b2f4c_row9_col2" class="data row9 col2" >LORE_SA2</td>
      <td id="T_b2f4c_row9_col3" class="data row9 col3" >IF capital.gain > 14964.5977 AND relationship != Own-child THEN class = >50K</td>
      <td id="T_b2f4c_row9_col4" class="data row9 col4" >0.01909</td>
      <td id="T_b2f4c_row9_col5" class="data row9 col5" >0.07834</td>
      <td id="T_b2f4c_row9_col6" class="data row9 col6" >0.98851</td>
      <td id="T_b2f4c_row9_col7" class="data row9 col7" >2</td>
      <td id="T_b2f4c_row9_col8" class="data row9 col8" >0</td>
      <td id="T_b2f4c_row9_col9" class="data row9 col9" >42.62766</td>
      <td id="T_b2f4c_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_b2f4c_row10_col0" class="data row10 col0" >1808</td>
      <td id="T_b2f4c_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_b2f4c_row10_col2" class="data row10 col2" >LORE_SA3</td>
      <td id="T_b2f4c_row10_col3" class="data row10 col3" >IF capital.gain > 5356.9038 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_b2f4c_row10_col4" class="data row10 col4" >0.03471</td>
      <td id="T_b2f4c_row10_col5" class="data row10 col5" >0.14192</td>
      <td id="T_b2f4c_row10_col6" class="data row10 col6" >0.98483</td>
      <td id="T_b2f4c_row10_col7" class="data row10 col7" >2</td>
      <td id="T_b2f4c_row10_col8" class="data row10 col8" >0</td>
      <td id="T_b2f4c_row10_col9" class="data row10 col9" >41.51357</td>
      <td id="T_b2f4c_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_b2f4c_row11_col0" class="data row11 col0" >1808</td>
      <td id="T_b2f4c_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_b2f4c_row11_col2" class="data row11 col2" >LORE_SA4</td>
      <td id="T_b2f4c_row11_col3" class="data row11 col3" >IF age > 42.5082 AND capital.gain > 572.1054 AND occupation != Other-service THEN class = >50K</td>
      <td id="T_b2f4c_row11_col4" class="data row11 col4" >0.04072</td>
      <td id="T_b2f4c_row11_col5" class="data row11 col5" >0.11933</td>
      <td id="T_b2f4c_row11_col6" class="data row11 col6" >0.70582</td>
      <td id="T_b2f4c_row11_col7" class="data row11 col7" >3</td>
      <td id="T_b2f4c_row11_col8" class="data row11 col8" >0</td>
      <td id="T_b2f4c_row11_col9" class="data row11 col9" >43.46254</td>
      <td id="T_b2f4c_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_b2f4c_row12_col0" class="data row12 col0" >1808</td>
      <td id="T_b2f4c_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_b2f4c_row12_col2" class="data row12 col2" >LORE_SA5</td>
      <td id="T_b2f4c_row12_col3" class="data row12 col3" >IF native.country != Portugal AND relationship != Own-child THEN class = >50K</td>
      <td id="T_b2f4c_row12_col4" class="data row12 col4" >0.84337</td>
      <td id="T_b2f4c_row12_col5" class="data row12 col5" >0.99217</td>
      <td id="T_b2f4c_row12_col6" class="data row12 col6" >0.28332</td>
      <td id="T_b2f4c_row12_col7" class="data row12 col7" >2</td>
      <td id="T_b2f4c_row12_col8" class="data row12 col8" >0</td>
      <td id="T_b2f4c_row12_col9" class="data row12 col9" >42.49464</td>
      <td id="T_b2f4c_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_b2f4c_row13_col0" class="data row13 col0" >1808</td>
      <td id="T_b2f4c_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_b2f4c_row13_col2" class="data row13 col2" >EXPLAN1</td>
      <td id="T_b2f4c_row13_col3" class="data row13 col3" >IF capital.gain > 7688.0 AND relationship = Husband THEN class = >50K</td>
      <td id="T_b2f4c_row13_col4" class="data row13 col4" >0.01584</td>
      <td id="T_b2f4c_row13_col5" class="data row13 col5" >0.06486</td>
      <td id="T_b2f4c_row13_col6" class="data row13 col6" >0.98615</td>
      <td id="T_b2f4c_row13_col7" class="data row13 col7" >2</td>
      <td id="T_b2f4c_row13_col8" class="data row13 col8" >0</td>
      <td id="T_b2f4c_row13_col9" class="data row13 col9" >13.07816</td>
      <td id="T_b2f4c_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_b2f4c_row14_col0" class="data row14 col0" >1808</td>
      <td id="T_b2f4c_row14_col1" class="data row14 col1" >EXPLAN</td>
      <td id="T_b2f4c_row14_col2" class="data row14 col2" >EXPLAN2</td>
      <td id="T_b2f4c_row14_col3" class="data row14 col3" >IF age > 33.8902 AND capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_b2f4c_row14_col4" class="data row14 col4" >0.02303</td>
      <td id="T_b2f4c_row14_col5" class="data row14 col5" >0.09401</td>
      <td id="T_b2f4c_row14_col6" class="data row14 col6" >0.98286</td>
      <td id="T_b2f4c_row14_col7" class="data row14 col7" >2</td>
      <td id="T_b2f4c_row14_col8" class="data row14 col8" >1</td>
      <td id="T_b2f4c_row14_col9" class="data row14 col9" >17.98777</td>
      <td id="T_b2f4c_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_b2f4c_row15_col0" class="data row15 col0" >1808</td>
      <td id="T_b2f4c_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_b2f4c_row15_col2" class="data row15 col2" >EXPLAN3</td>
      <td id="T_b2f4c_row15_col3" class="data row15 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_b2f4c_row15_col4" class="data row15 col4" >0.02694</td>
      <td id="T_b2f4c_row15_col5" class="data row15 col5" >0.10931</td>
      <td id="T_b2f4c_row15_col6" class="data row15 col6" >0.97720</td>
      <td id="T_b2f4c_row15_col7" class="data row15 col7" >1</td>
      <td id="T_b2f4c_row15_col8" class="data row15 col8" >0</td>
      <td id="T_b2f4c_row15_col9" class="data row15 col9" >10.93505</td>
      <td id="T_b2f4c_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_b2f4c_row16_col0" class="data row16 col0" >1808</td>
      <td id="T_b2f4c_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_b2f4c_row16_col2" class="data row16 col2" >EXPLAN4</td>
      <td id="T_b2f4c_row16_col3" class="data row16 col3" >IF age > 24.452 AND capital.gain > 10520.0 THEN class = >50K</td>
      <td id="T_b2f4c_row16_col4" class="data row16 col4" >0.02242</td>
      <td id="T_b2f4c_row16_col5" class="data row16 col5" >0.09146</td>
      <td id="T_b2f4c_row16_col6" class="data row16 col6" >0.98239</td>
      <td id="T_b2f4c_row16_col7" class="data row16 col7" >2</td>
      <td id="T_b2f4c_row16_col8" class="data row16 col8" >0</td>
      <td id="T_b2f4c_row16_col9" class="data row16 col9" >9.72879</td>
      <td id="T_b2f4c_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_b2f4c_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_b2f4c_row17_col0" class="data row17 col0" >1808</td>
      <td id="T_b2f4c_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_b2f4c_row17_col2" class="data row17 col2" >EXPLAN5</td>
      <td id="T_b2f4c_row17_col3" class="data row17 col3" >IF age > 28.4951 AND capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_b2f4c_row17_col4" class="data row17 col4" >0.02558</td>
      <td id="T_b2f4c_row17_col5" class="data row17 col5" >0.10457</td>
      <td id="T_b2f4c_row17_col6" class="data row17 col6" >0.98456</td>
      <td id="T_b2f4c_row17_col7" class="data row17 col7" >2</td>
      <td id="T_b2f4c_row17_col8" class="data row17 col8" >1</td>
      <td id="T_b2f4c_row17_col9" class="data row17 col9" >21.77034</td>
      <td id="T_b2f4c_row17_col10" class="data row17 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 1808, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.84337, Pre: 0.98851)



<style type="text/css">
#T_e34fb_row2_col0, #T_e34fb_row2_col1, #T_e34fb_row2_col2, #T_e34fb_row2_col3, #T_e34fb_row2_col4, #T_e34fb_row2_col5, #T_e34fb_row2_col6, #T_e34fb_row2_col7, #T_e34fb_row2_col8, #T_e34fb_row2_col9, #T_e34fb_row2_col10, #T_e34fb_row2_col11 {
  font-weight: bold;
}
</style>
<table id="T_e34fb">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e34fb_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e34fb_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e34fb_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e34fb_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e34fb_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e34fb_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e34fb_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e34fb_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e34fb_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e34fb_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e34fb_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_e34fb_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e34fb_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e34fb_row0_col0" class="data row0 col0" >1808</td>
      <td id="T_e34fb_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_e34fb_row0_col2" class="data row0 col2" >LORE1</td>
      <td id="T_e34fb_row0_col3" class="data row0 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_e34fb_row0_col4" class="data row0 col4" >0.04923</td>
      <td id="T_e34fb_row0_col5" class="data row0 col5" >0.19366</td>
      <td id="T_e34fb_row0_col6" class="data row0 col6" >0.94742</td>
      <td id="T_e34fb_row0_col7" class="data row0 col7" >1</td>
      <td id="T_e34fb_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e34fb_row0_col9" class="data row0 col9" >197.42534</td>
      <td id="T_e34fb_row0_col10" class="data row0 col10" >False</td>
      <td id="T_e34fb_row0_col11" class="data row0 col11" >0.79520</td>
    </tr>
    <tr>
      <th id="T_e34fb_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e34fb_row1_col0" class="data row1 col0" >1808</td>
      <td id="T_e34fb_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_e34fb_row1_col2" class="data row1 col2" >LORE4</td>
      <td id="T_e34fb_row1_col3" class="data row1 col3" >IF capital.gain > 4386.0 THEN class = >50K</td>
      <td id="T_e34fb_row1_col4" class="data row1 col4" >0.05476</td>
      <td id="T_e34fb_row1_col5" class="data row1 col5" >0.19767</td>
      <td id="T_e34fb_row1_col6" class="data row1 col6" >0.86939</td>
      <td id="T_e34fb_row1_col7" class="data row1 col7" >1</td>
      <td id="T_e34fb_row1_col8" class="data row1 col8" >1</td>
      <td id="T_e34fb_row1_col9" class="data row1 col9" >365.37206</td>
      <td id="T_e34fb_row1_col10" class="data row1 col10" >False</td>
      <td id="T_e34fb_row1_col11" class="data row1 col11" >0.79756</td>
    </tr>
    <tr>
      <th id="T_e34fb_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e34fb_row2_col0" class="data row2 col0" >1808</td>
      <td id="T_e34fb_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_e34fb_row2_col2" class="data row2 col2" >LORE_SA1</td>
      <td id="T_e34fb_row2_col3" class="data row2 col3" >IF native.country != Honduras AND relationship = Husband THEN class = >50K</td>
      <td id="T_e34fb_row2_col4" class="data row2 col4" >0.40479</td>
      <td id="T_e34fb_row2_col5" class="data row2 col5" >0.75752</td>
      <td id="T_e34fb_row2_col6" class="data row2 col6" >0.45068</td>
      <td id="T_e34fb_row2_col7" class="data row2 col7" >2</td>
      <td id="T_e34fb_row2_col8" class="data row2 col8" >0</td>
      <td id="T_e34fb_row2_col9" class="data row2 col9" >43.34698</td>
      <td id="T_e34fb_row2_col10" class="data row2 col10" >False</td>
      <td id="T_e34fb_row2_col11" class="data row2 col11" >0.69398</td>
    </tr>
    <tr>
      <th id="T_e34fb_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e34fb_row3_col0" class="data row3 col0" >1808</td>
      <td id="T_e34fb_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_e34fb_row3_col2" class="data row3 col2" >LORE_SA2</td>
      <td id="T_e34fb_row3_col3" class="data row3 col3" >IF capital.gain > 14964.5977 AND relationship != Own-child THEN class = >50K</td>
      <td id="T_e34fb_row3_col4" class="data row3 col4" >0.01909</td>
      <td id="T_e34fb_row3_col5" class="data row3 col5" >0.07834</td>
      <td id="T_e34fb_row3_col6" class="data row3 col6" >0.98851</td>
      <td id="T_e34fb_row3_col7" class="data row3 col7" >2</td>
      <td id="T_e34fb_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e34fb_row3_col9" class="data row3 col9" >42.62766</td>
      <td id="T_e34fb_row3_col10" class="data row3 col10" >False</td>
      <td id="T_e34fb_row3_col11" class="data row3 col11" >0.82428</td>
    </tr>
    <tr>
      <th id="T_e34fb_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e34fb_row4_col0" class="data row4 col0" >1808</td>
      <td id="T_e34fb_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_e34fb_row4_col2" class="data row4 col2" >LORE_SA3</td>
      <td id="T_e34fb_row4_col3" class="data row4 col3" >IF capital.gain > 5356.9038 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_e34fb_row4_col4" class="data row4 col4" >0.03471</td>
      <td id="T_e34fb_row4_col5" class="data row4 col5" >0.14192</td>
      <td id="T_e34fb_row4_col6" class="data row4 col6" >0.98483</td>
      <td id="T_e34fb_row4_col7" class="data row4 col7" >2</td>
      <td id="T_e34fb_row4_col8" class="data row4 col8" >0</td>
      <td id="T_e34fb_row4_col9" class="data row4 col9" >41.51357</td>
      <td id="T_e34fb_row4_col10" class="data row4 col10" >False</td>
      <td id="T_e34fb_row4_col11" class="data row4 col11" >0.80867</td>
    </tr>
    <tr>
      <th id="T_e34fb_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e34fb_row5_col0" class="data row5 col0" >1808</td>
      <td id="T_e34fb_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_e34fb_row5_col2" class="data row5 col2" >LORE_SA5</td>
      <td id="T_e34fb_row5_col3" class="data row5 col3" >IF native.country != Portugal AND relationship != Own-child THEN class = >50K</td>
      <td id="T_e34fb_row5_col4" class="data row5 col4" >0.84337</td>
      <td id="T_e34fb_row5_col5" class="data row5 col5" >0.99217</td>
      <td id="T_e34fb_row5_col6" class="data row5 col6" >0.28332</td>
      <td id="T_e34fb_row5_col7" class="data row5 col7" >2</td>
      <td id="T_e34fb_row5_col8" class="data row5 col8" >0</td>
      <td id="T_e34fb_row5_col9" class="data row5 col9" >42.49464</td>
      <td id="T_e34fb_row5_col10" class="data row5 col10" >False</td>
      <td id="T_e34fb_row5_col11" class="data row5 col11" >0.70519</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_157.png)
    



### Rules for Instance 1808, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.84337, Pre: 0.98851), Unique rules (diffrent features)



<style type="text/css">
#T_c1754_row1_col0, #T_c1754_row1_col1, #T_c1754_row1_col2, #T_c1754_row1_col3, #T_c1754_row1_col4, #T_c1754_row1_col5, #T_c1754_row1_col6, #T_c1754_row1_col7, #T_c1754_row1_col8, #T_c1754_row1_col9, #T_c1754_row1_col10, #T_c1754_row1_col11 {
  font-weight: bold;
}
</style>
<table id="T_c1754">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c1754_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_c1754_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_c1754_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_c1754_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_c1754_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_c1754_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_c1754_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_c1754_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_c1754_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_c1754_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_c1754_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_c1754_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c1754_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_c1754_row0_col0" class="data row0 col0" >1808</td>
      <td id="T_c1754_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_c1754_row0_col2" class="data row0 col2" >LORE1</td>
      <td id="T_c1754_row0_col3" class="data row0 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_c1754_row0_col4" class="data row0 col4" >0.04923</td>
      <td id="T_c1754_row0_col5" class="data row0 col5" >0.19366</td>
      <td id="T_c1754_row0_col6" class="data row0 col6" >0.94742</td>
      <td id="T_c1754_row0_col7" class="data row0 col7" >1</td>
      <td id="T_c1754_row0_col8" class="data row0 col8" >0</td>
      <td id="T_c1754_row0_col9" class="data row0 col9" >197.42534</td>
      <td id="T_c1754_row0_col10" class="data row0 col10" >False</td>
      <td id="T_c1754_row0_col11" class="data row0 col11" >0.79520</td>
    </tr>
    <tr>
      <th id="T_c1754_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_c1754_row1_col0" class="data row1 col0" >1808</td>
      <td id="T_c1754_row1_col1" class="data row1 col1" >LORE_SA</td>
      <td id="T_c1754_row1_col2" class="data row1 col2" >LORE_SA1</td>
      <td id="T_c1754_row1_col3" class="data row1 col3" >IF native.country != Honduras AND relationship = Husband THEN class = >50K</td>
      <td id="T_c1754_row1_col4" class="data row1 col4" >0.40479</td>
      <td id="T_c1754_row1_col5" class="data row1 col5" >0.75752</td>
      <td id="T_c1754_row1_col6" class="data row1 col6" >0.45068</td>
      <td id="T_c1754_row1_col7" class="data row1 col7" >2</td>
      <td id="T_c1754_row1_col8" class="data row1 col8" >0</td>
      <td id="T_c1754_row1_col9" class="data row1 col9" >43.34698</td>
      <td id="T_c1754_row1_col10" class="data row1 col10" >False</td>
      <td id="T_c1754_row1_col11" class="data row1 col11" >0.69398</td>
    </tr>
    <tr>
      <th id="T_c1754_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_c1754_row2_col0" class="data row2 col0" >1808</td>
      <td id="T_c1754_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_c1754_row2_col2" class="data row2 col2" >LORE_SA2</td>
      <td id="T_c1754_row2_col3" class="data row2 col3" >IF capital.gain > 14964.5977 AND relationship != Own-child THEN class = >50K</td>
      <td id="T_c1754_row2_col4" class="data row2 col4" >0.01909</td>
      <td id="T_c1754_row2_col5" class="data row2 col5" >0.07834</td>
      <td id="T_c1754_row2_col6" class="data row2 col6" >0.98851</td>
      <td id="T_c1754_row2_col7" class="data row2 col7" >2</td>
      <td id="T_c1754_row2_col8" class="data row2 col8" >0</td>
      <td id="T_c1754_row2_col9" class="data row2 col9" >42.62766</td>
      <td id="T_c1754_row2_col10" class="data row2 col10" >False</td>
      <td id="T_c1754_row2_col11" class="data row2 col11" >0.82428</td>
    </tr>
    <tr>
      <th id="T_c1754_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_c1754_row3_col0" class="data row3 col0" >1808</td>
      <td id="T_c1754_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_c1754_row3_col2" class="data row3 col2" >LORE_SA3</td>
      <td id="T_c1754_row3_col3" class="data row3 col3" >IF capital.gain > 5356.9038 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_c1754_row3_col4" class="data row3 col4" >0.03471</td>
      <td id="T_c1754_row3_col5" class="data row3 col5" >0.14192</td>
      <td id="T_c1754_row3_col6" class="data row3 col6" >0.98483</td>
      <td id="T_c1754_row3_col7" class="data row3 col7" >2</td>
      <td id="T_c1754_row3_col8" class="data row3 col8" >0</td>
      <td id="T_c1754_row3_col9" class="data row3 col9" >41.51357</td>
      <td id="T_c1754_row3_col10" class="data row3 col10" >False</td>
      <td id="T_c1754_row3_col11" class="data row3 col11" >0.80867</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_160.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_161.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_162.png)
    



### Rules for Instance 1808, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99217, Pre: 0.98851, Len: 0.28332)



<style type="text/css">
#T_e7148_row0_col0, #T_e7148_row0_col1, #T_e7148_row0_col2, #T_e7148_row0_col3, #T_e7148_row0_col4, #T_e7148_row0_col5, #T_e7148_row0_col6, #T_e7148_row0_col7, #T_e7148_row0_col8, #T_e7148_row0_col9, #T_e7148_row0_col10, #T_e7148_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_e7148">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e7148_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e7148_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e7148_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e7148_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e7148_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e7148_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e7148_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e7148_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e7148_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e7148_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e7148_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_e7148_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e7148_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e7148_row0_col0" class="data row0 col0" >1808</td>
      <td id="T_e7148_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_e7148_row0_col2" class="data row0 col2" >LORE1</td>
      <td id="T_e7148_row0_col3" class="data row0 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_e7148_row0_col4" class="data row0 col4" >0.04923</td>
      <td id="T_e7148_row0_col5" class="data row0 col5" >0.19366</td>
      <td id="T_e7148_row0_col6" class="data row0 col6" >0.94742</td>
      <td id="T_e7148_row0_col7" class="data row0 col7" >1</td>
      <td id="T_e7148_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e7148_row0_col9" class="data row0 col9" >197.42534</td>
      <td id="T_e7148_row0_col10" class="data row0 col10" >False</td>
      <td id="T_e7148_row0_col11" class="data row0 col11" >1.07375</td>
    </tr>
    <tr>
      <th id="T_e7148_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e7148_row1_col0" class="data row1 col0" >1808</td>
      <td id="T_e7148_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_e7148_row1_col2" class="data row1 col2" >LORE3</td>
      <td id="T_e7148_row1_col3" class="data row1 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_e7148_row1_col4" class="data row1 col4" >0.02694</td>
      <td id="T_e7148_row1_col5" class="data row1 col5" >0.10931</td>
      <td id="T_e7148_row1_col6" class="data row1 col6" >0.97720</td>
      <td id="T_e7148_row1_col7" class="data row1 col7" >1</td>
      <td id="T_e7148_row1_col8" class="data row1 col8" >0</td>
      <td id="T_e7148_row1_col9" class="data row1 col9" >176.13512</td>
      <td id="T_e7148_row1_col10" class="data row1 col10" >False</td>
      <td id="T_e7148_row1_col11" class="data row1 col11" >1.13719</td>
    </tr>
    <tr>
      <th id="T_e7148_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e7148_row2_col0" class="data row2 col0" >1808</td>
      <td id="T_e7148_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_e7148_row2_col2" class="data row2 col2" >LORE4</td>
      <td id="T_e7148_row2_col3" class="data row2 col3" >IF capital.gain > 4386.0 THEN class = >50K</td>
      <td id="T_e7148_row2_col4" class="data row2 col4" >0.05476</td>
      <td id="T_e7148_row2_col5" class="data row2 col5" >0.19767</td>
      <td id="T_e7148_row2_col6" class="data row2 col6" >0.86939</td>
      <td id="T_e7148_row2_col7" class="data row2 col7" >1</td>
      <td id="T_e7148_row2_col8" class="data row2 col8" >1</td>
      <td id="T_e7148_row2_col9" class="data row2 col9" >365.37206</td>
      <td id="T_e7148_row2_col10" class="data row2 col10" >False</td>
      <td id="T_e7148_row2_col11" class="data row2 col11" >1.07659</td>
    </tr>
    <tr>
      <th id="T_e7148_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e7148_row3_col0" class="data row3 col0" >1808</td>
      <td id="T_e7148_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_e7148_row3_col2" class="data row3 col2" >LORE_SA1</td>
      <td id="T_e7148_row3_col3" class="data row3 col3" >IF native.country != Honduras AND relationship = Husband THEN class = >50K</td>
      <td id="T_e7148_row3_col4" class="data row3 col4" >0.40479</td>
      <td id="T_e7148_row3_col5" class="data row3 col5" >0.75752</td>
      <td id="T_e7148_row3_col6" class="data row3 col6" >0.45068</td>
      <td id="T_e7148_row3_col7" class="data row3 col7" >2</td>
      <td id="T_e7148_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e7148_row3_col9" class="data row3 col9" >43.34698</td>
      <td id="T_e7148_row3_col10" class="data row3 col10" >False</td>
      <td id="T_e7148_row3_col11" class="data row3 col11" >1.81420</td>
    </tr>
    <tr>
      <th id="T_e7148_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e7148_row4_col0" class="data row4 col0" >1808</td>
      <td id="T_e7148_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_e7148_row4_col2" class="data row4 col2" >LORE_SA2</td>
      <td id="T_e7148_row4_col3" class="data row4 col3" >IF capital.gain > 14964.5977 AND relationship != Own-child THEN class = >50K</td>
      <td id="T_e7148_row4_col4" class="data row4 col4" >0.01909</td>
      <td id="T_e7148_row4_col5" class="data row4 col5" >0.07834</td>
      <td id="T_e7148_row4_col6" class="data row4 col6" >0.98851</td>
      <td id="T_e7148_row4_col7" class="data row4 col7" >2</td>
      <td id="T_e7148_row4_col8" class="data row4 col8" >0</td>
      <td id="T_e7148_row4_col9" class="data row4 col9" >42.62766</td>
      <td id="T_e7148_row4_col10" class="data row4 col10" >False</td>
      <td id="T_e7148_row4_col11" class="data row4 col11" >1.94476</td>
    </tr>
    <tr>
      <th id="T_e7148_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e7148_row5_col0" class="data row5 col0" >1808</td>
      <td id="T_e7148_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_e7148_row5_col2" class="data row5 col2" >LORE_SA3</td>
      <td id="T_e7148_row5_col3" class="data row5 col3" >IF capital.gain > 5356.9038 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_e7148_row5_col4" class="data row5 col4" >0.03471</td>
      <td id="T_e7148_row5_col5" class="data row5 col5" >0.14192</td>
      <td id="T_e7148_row5_col6" class="data row5 col6" >0.98483</td>
      <td id="T_e7148_row5_col7" class="data row5 col7" >2</td>
      <td id="T_e7148_row5_col8" class="data row5 col8" >0</td>
      <td id="T_e7148_row5_col9" class="data row5 col9" >41.51357</td>
      <td id="T_e7148_row5_col10" class="data row5 col10" >False</td>
      <td id="T_e7148_row5_col11" class="data row5 col11" >1.91571</td>
    </tr>
    <tr>
      <th id="T_e7148_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_e7148_row6_col0" class="data row6 col0" >1808</td>
      <td id="T_e7148_row6_col1" class="data row6 col1" >LORE_SA</td>
      <td id="T_e7148_row6_col2" class="data row6 col2" >LORE_SA5</td>
      <td id="T_e7148_row6_col3" class="data row6 col3" >IF native.country != Portugal AND relationship != Own-child THEN class = >50K</td>
      <td id="T_e7148_row6_col4" class="data row6 col4" >0.84337</td>
      <td id="T_e7148_row6_col5" class="data row6 col5" >0.99217</td>
      <td id="T_e7148_row6_col6" class="data row6 col6" >0.28332</td>
      <td id="T_e7148_row6_col7" class="data row6 col7" >2</td>
      <td id="T_e7148_row6_col8" class="data row6 col8" >0</td>
      <td id="T_e7148_row6_col9" class="data row6 col9" >42.49464</td>
      <td id="T_e7148_row6_col10" class="data row6 col10" >False</td>
      <td id="T_e7148_row6_col11" class="data row6 col11" >1.85588</td>
    </tr>
    <tr>
      <th id="T_e7148_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_e7148_row7_col0" class="data row7 col0" >1808</td>
      <td id="T_e7148_row7_col1" class="data row7 col1" >EXPLAN</td>
      <td id="T_e7148_row7_col2" class="data row7 col2" >EXPLAN3</td>
      <td id="T_e7148_row7_col3" class="data row7 col3" >IF capital.gain > 7688.0 THEN class = >50K</td>
      <td id="T_e7148_row7_col4" class="data row7 col4" >0.02694</td>
      <td id="T_e7148_row7_col5" class="data row7 col5" >0.10931</td>
      <td id="T_e7148_row7_col6" class="data row7 col6" >0.97720</td>
      <td id="T_e7148_row7_col7" class="data row7 col7" >1</td>
      <td id="T_e7148_row7_col8" class="data row7 col8" >0</td>
      <td id="T_e7148_row7_col9" class="data row7 col9" >10.93505</td>
      <td id="T_e7148_row7_col10" class="data row7 col10" >False</td>
      <td id="T_e7148_row7_col11" class="data row7 col11" >1.13719</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 1808, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99217, Pre: 0.98851), Unique rules (diffrent features)



<style type="text/css">
#T_37333_row0_col0, #T_37333_row0_col1, #T_37333_row0_col2, #T_37333_row0_col3, #T_37333_row0_col4, #T_37333_row0_col5, #T_37333_row0_col6, #T_37333_row0_col7, #T_37333_row0_col8, #T_37333_row0_col9, #T_37333_row0_col10, #T_37333_row0_col11 {
  font-weight: bold;
}
</style>
<table id="T_37333">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_37333_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_37333_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_37333_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_37333_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_37333_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_37333_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_37333_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_37333_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_37333_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_37333_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_37333_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_37333_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_37333_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_37333_row0_col0" class="data row0 col0" >1808</td>
      <td id="T_37333_row0_col1" class="data row0 col1" >LORE</td>
      <td id="T_37333_row0_col2" class="data row0 col2" >LORE1</td>
      <td id="T_37333_row0_col3" class="data row0 col3" >IF capital.gain > 5013.0 THEN class = >50K</td>
      <td id="T_37333_row0_col4" class="data row0 col4" >0.04923</td>
      <td id="T_37333_row0_col5" class="data row0 col5" >0.19366</td>
      <td id="T_37333_row0_col6" class="data row0 col6" >0.94742</td>
      <td id="T_37333_row0_col7" class="data row0 col7" >1</td>
      <td id="T_37333_row0_col8" class="data row0 col8" >0</td>
      <td id="T_37333_row0_col9" class="data row0 col9" >197.42534</td>
      <td id="T_37333_row0_col10" class="data row0 col10" >False</td>
      <td id="T_37333_row0_col11" class="data row0 col11" >1.07375</td>
    </tr>
    <tr>
      <th id="T_37333_level0_row1" class="row_heading level0 row1" >3</th>
      <td id="T_37333_row1_col0" class="data row1 col0" >1808</td>
      <td id="T_37333_row1_col1" class="data row1 col1" >LORE_SA</td>
      <td id="T_37333_row1_col2" class="data row1 col2" >LORE_SA1</td>
      <td id="T_37333_row1_col3" class="data row1 col3" >IF native.country != Honduras AND relationship = Husband THEN class = >50K</td>
      <td id="T_37333_row1_col4" class="data row1 col4" >0.40479</td>
      <td id="T_37333_row1_col5" class="data row1 col5" >0.75752</td>
      <td id="T_37333_row1_col6" class="data row1 col6" >0.45068</td>
      <td id="T_37333_row1_col7" class="data row1 col7" >2</td>
      <td id="T_37333_row1_col8" class="data row1 col8" >0</td>
      <td id="T_37333_row1_col9" class="data row1 col9" >43.34698</td>
      <td id="T_37333_row1_col10" class="data row1 col10" >False</td>
      <td id="T_37333_row1_col11" class="data row1 col11" >1.81420</td>
    </tr>
    <tr>
      <th id="T_37333_level0_row2" class="row_heading level0 row2" >4</th>
      <td id="T_37333_row2_col0" class="data row2 col0" >1808</td>
      <td id="T_37333_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_37333_row2_col2" class="data row2 col2" >LORE_SA2</td>
      <td id="T_37333_row2_col3" class="data row2 col3" >IF capital.gain > 14964.5977 AND relationship != Own-child THEN class = >50K</td>
      <td id="T_37333_row2_col4" class="data row2 col4" >0.01909</td>
      <td id="T_37333_row2_col5" class="data row2 col5" >0.07834</td>
      <td id="T_37333_row2_col6" class="data row2 col6" >0.98851</td>
      <td id="T_37333_row2_col7" class="data row2 col7" >2</td>
      <td id="T_37333_row2_col8" class="data row2 col8" >0</td>
      <td id="T_37333_row2_col9" class="data row2 col9" >42.62766</td>
      <td id="T_37333_row2_col10" class="data row2 col10" >False</td>
      <td id="T_37333_row2_col11" class="data row2 col11" >1.94476</td>
    </tr>
    <tr>
      <th id="T_37333_level0_row3" class="row_heading level0 row3" >5</th>
      <td id="T_37333_row3_col0" class="data row3 col0" >1808</td>
      <td id="T_37333_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_37333_row3_col2" class="data row3 col2" >LORE_SA3</td>
      <td id="T_37333_row3_col3" class="data row3 col3" >IF capital.gain > 5356.9038 AND marital.status = Married-civ-spouse THEN class = >50K</td>
      <td id="T_37333_row3_col4" class="data row3 col4" >0.03471</td>
      <td id="T_37333_row3_col5" class="data row3 col5" >0.14192</td>
      <td id="T_37333_row3_col6" class="data row3 col6" >0.98483</td>
      <td id="T_37333_row3_col7" class="data row3 col7" >2</td>
      <td id="T_37333_row3_col8" class="data row3 col8" >0</td>
      <td id="T_37333_row3_col9" class="data row3 col9" >41.51357</td>
      <td id="T_37333_row3_col10" class="data row3 col10" >False</td>
      <td id="T_37333_row3_col11" class="data row3 col11" >1.91571</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_167.png)
    



## Instance 12191 (Original: <=50K , Predicted: <=50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>22.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>HS-grad</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>9</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Never-married</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Sales</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Not-in-family</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>30.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 12191



<style type="text/css">
</style>
<table id="T_2d2b1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2d2b1_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_2d2b1_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_2d2b1_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_2d2b1_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_2d2b1_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_2d2b1_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_2d2b1_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_2d2b1_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_2d2b1_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_2d2b1_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_2d2b1_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2d2b1_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_2d2b1_row0_col0" class="data row0 col0" >12191</td>
      <td id="T_2d2b1_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_2d2b1_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_2d2b1_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row0_col4" class="data row0 col4" >0.63088</td>
      <td id="T_2d2b1_row0_col5" class="data row0 col5" >0.71733</td>
      <td id="T_2d2b1_row0_col6" class="data row0 col6" >0.86320</td>
      <td id="T_2d2b1_row0_col7" class="data row0 col7" >3</td>
      <td id="T_2d2b1_row0_col8" class="data row0 col8" >0</td>
      <td id="T_2d2b1_row0_col9" class="data row0 col9" >1.95820</td>
      <td id="T_2d2b1_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_2d2b1_row1_col0" class="data row1 col0" >12191</td>
      <td id="T_2d2b1_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_2d2b1_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_2d2b1_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row1_col4" class="data row1 col4" >0.69287</td>
      <td id="T_2d2b1_row1_col5" class="data row1 col5" >0.77489</td>
      <td id="T_2d2b1_row1_col6" class="data row1 col6" >0.84904</td>
      <td id="T_2d2b1_row1_col7" class="data row1 col7" >3</td>
      <td id="T_2d2b1_row1_col8" class="data row1 col8" >0</td>
      <td id="T_2d2b1_row1_col9" class="data row1 col9" >2.31952</td>
      <td id="T_2d2b1_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_2d2b1_row2_col0" class="data row2 col0" >12191</td>
      <td id="T_2d2b1_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_2d2b1_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_2d2b1_row2_col3" class="data row2 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row2_col4" class="data row2 col4" >0.48710</td>
      <td id="T_2d2b1_row2_col5" class="data row2 col5" >0.56643</td>
      <td id="T_2d2b1_row2_col6" class="data row2 col6" >0.88281</td>
      <td id="T_2d2b1_row2_col7" class="data row2 col7" >2</td>
      <td id="T_2d2b1_row2_col8" class="data row2 col8" >2</td>
      <td id="T_2d2b1_row2_col9" class="data row2 col9" >6.01046</td>
      <td id="T_2d2b1_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_2d2b1_row3_col0" class="data row3 col0" >12191</td>
      <td id="T_2d2b1_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_2d2b1_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_2d2b1_row3_col3" class="data row3 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row3_col4" class="data row3 col4" >0.66269</td>
      <td id="T_2d2b1_row3_col5" class="data row3 col5" >0.73091</td>
      <td id="T_2d2b1_row3_col6" class="data row3 col6" >0.83733</td>
      <td id="T_2d2b1_row3_col7" class="data row3 col7" >3</td>
      <td id="T_2d2b1_row3_col8" class="data row3 col8" >0</td>
      <td id="T_2d2b1_row3_col9" class="data row3 col9" >2.33662</td>
      <td id="T_2d2b1_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_2d2b1_row4_col0" class="data row4 col0" >12191</td>
      <td id="T_2d2b1_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_2d2b1_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_2d2b1_row4_col3" class="data row4 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row4_col4" class="data row4 col4" >0.69595</td>
      <td id="T_2d2b1_row4_col5" class="data row4 col5" >0.75357</td>
      <td id="T_2d2b1_row4_col6" class="data row4 col6" >0.82203</td>
      <td id="T_2d2b1_row4_col7" class="data row4 col7" >2</td>
      <td id="T_2d2b1_row4_col8" class="data row4 col8" >0</td>
      <td id="T_2d2b1_row4_col9" class="data row4 col9" >2.09261</td>
      <td id="T_2d2b1_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_2d2b1_row5_col0" class="data row5 col0" >12191</td>
      <td id="T_2d2b1_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_2d2b1_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_2d2b1_row5_col3" class="data row5 col3" >IF capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row5_col4" class="data row5 col4" >0.91598</td>
      <td id="T_2d2b1_row5_col5" class="data row5 col5" >0.95810</td>
      <td id="T_2d2b1_row5_col6" class="data row5 col6" >0.79408</td>
      <td id="T_2d2b1_row5_col7" class="data row5 col7" >1</td>
      <td id="T_2d2b1_row5_col8" class="data row5 col8" >0</td>
      <td id="T_2d2b1_row5_col9" class="data row5 col9" >190.90707</td>
      <td id="T_2d2b1_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_2d2b1_row6_col0" class="data row6 col0" >12191</td>
      <td id="T_2d2b1_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_2d2b1_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_2d2b1_row6_col3" class="data row6 col3" >IF capital.gain <= 1407.808 THEN class = <=50K</td>
      <td id="T_2d2b1_row6_col4" class="data row6 col4" >0.91896</td>
      <td id="T_2d2b1_row6_col5" class="data row6 col5" >0.96203</td>
      <td id="T_2d2b1_row6_col6" class="data row6 col6" >0.79475</td>
      <td id="T_2d2b1_row6_col7" class="data row6 col7" >1</td>
      <td id="T_2d2b1_row6_col8" class="data row6 col8" >0</td>
      <td id="T_2d2b1_row6_col9" class="data row6 col9" >196.93938</td>
      <td id="T_2d2b1_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_2d2b1_row7_col0" class="data row7 col0" >12191</td>
      <td id="T_2d2b1_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_2d2b1_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_2d2b1_row7_col3" class="data row7 col3" >IF capital.gain <= 8066.0229 THEN class = <=50K</td>
      <td id="T_2d2b1_row7_col4" class="data row7 col4" >0.97315</td>
      <td id="T_2d2b1_row7_col5" class="data row7 col5" >0.99925</td>
      <td id="T_2d2b1_row7_col6" class="data row7 col6" >0.77953</td>
      <td id="T_2d2b1_row7_col7" class="data row7 col7" >1</td>
      <td id="T_2d2b1_row7_col8" class="data row7 col8" >0</td>
      <td id="T_2d2b1_row7_col9" class="data row7 col9" >288.81475</td>
      <td id="T_2d2b1_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_2d2b1_row8_col0" class="data row8 col0" >12191</td>
      <td id="T_2d2b1_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_2d2b1_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_2d2b1_row8_col3" class="data row8 col3" >IF capital.gain <= 594.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row8_col4" class="data row8 col4" >0.91725</td>
      <td id="T_2d2b1_row8_col5" class="data row8 col5" >0.95978</td>
      <td id="T_2d2b1_row8_col6" class="data row8 col6" >0.79437</td>
      <td id="T_2d2b1_row8_col7" class="data row8 col7" >1</td>
      <td id="T_2d2b1_row8_col8" class="data row8 col8" >0</td>
      <td id="T_2d2b1_row8_col9" class="data row8 col9" >316.98152</td>
      <td id="T_2d2b1_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_2d2b1_row9_col0" class="data row9 col0" >12191</td>
      <td id="T_2d2b1_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_2d2b1_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_2d2b1_row9_col3" class="data row9 col3" >IF capital.gain <= 6229.6605 THEN class = <=50K</td>
      <td id="T_2d2b1_row9_col4" class="data row9 col4" >0.95450</td>
      <td id="T_2d2b1_row9_col5" class="data row9 col5" >0.99717</td>
      <td id="T_2d2b1_row9_col6" class="data row9 col6" >0.79311</td>
      <td id="T_2d2b1_row9_col7" class="data row9 col7" >1</td>
      <td id="T_2d2b1_row9_col8" class="data row9 col8" >0</td>
      <td id="T_2d2b1_row9_col9" class="data row9 col9" >329.93644</td>
      <td id="T_2d2b1_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_2d2b1_row10_col0" class="data row10 col0" >12191</td>
      <td id="T_2d2b1_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_2d2b1_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_2d2b1_row10_col3" class="data row10 col3" >IF capital.gain <= 370.1838 THEN class = <=50K</td>
      <td id="T_2d2b1_row10_col4" class="data row10 col4" >0.91620</td>
      <td id="T_2d2b1_row10_col5" class="data row10 col5" >0.95839</td>
      <td id="T_2d2b1_row10_col6" class="data row10 col6" >0.79413</td>
      <td id="T_2d2b1_row10_col7" class="data row10 col7" >1</td>
      <td id="T_2d2b1_row10_col8" class="data row10 col8" >0</td>
      <td id="T_2d2b1_row10_col9" class="data row10 col9" >59.51019</td>
      <td id="T_2d2b1_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_2d2b1_row11_col0" class="data row11 col0" >12191</td>
      <td id="T_2d2b1_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_2d2b1_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_2d2b1_row11_col3" class="data row11 col3" >IF capital.gain <= 9458.4502 THEN class = <=50K</td>
      <td id="T_2d2b1_row11_col4" class="data row11 col4" >0.97543</td>
      <td id="T_2d2b1_row11_col5" class="data row11 col5" >0.99925</td>
      <td id="T_2d2b1_row11_col6" class="data row11 col6" >0.77771</td>
      <td id="T_2d2b1_row11_col7" class="data row11 col7" >1</td>
      <td id="T_2d2b1_row11_col8" class="data row11 col8" >0</td>
      <td id="T_2d2b1_row11_col9" class="data row11 col9" >35.10025</td>
      <td id="T_2d2b1_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_2d2b1_row12_col0" class="data row12 col0" >12191</td>
      <td id="T_2d2b1_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_2d2b1_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_2d2b1_row12_col3" class="data row12 col3" >IF capital.gain <= 7040.3074 THEN class = <=50K</td>
      <td id="T_2d2b1_row12_col4" class="data row12 col4" >0.95621</td>
      <td id="T_2d2b1_row12_col5" class="data row12 col5" >0.99890</td>
      <td id="T_2d2b1_row12_col6" class="data row12 col6" >0.79306</td>
      <td id="T_2d2b1_row12_col7" class="data row12 col7" >1</td>
      <td id="T_2d2b1_row12_col8" class="data row12 col8" >0</td>
      <td id="T_2d2b1_row12_col9" class="data row12 col9" >37.75637</td>
      <td id="T_2d2b1_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_2d2b1_row13_col0" class="data row13 col0" >12191</td>
      <td id="T_2d2b1_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_2d2b1_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_2d2b1_row13_col3" class="data row13 col3" >IF capital.gain <= 270.5606 AND workclass != Self-emp-not-inc THEN class = <=50K</td>
      <td id="T_2d2b1_row13_col4" class="data row13 col4" >0.84631</td>
      <td id="T_2d2b1_row13_col5" class="data row13 col5" >0.88863</td>
      <td id="T_2d2b1_row13_col6" class="data row13 col6" >0.79714</td>
      <td id="T_2d2b1_row13_col7" class="data row13 col7" >2</td>
      <td id="T_2d2b1_row13_col8" class="data row13 col8" >0</td>
      <td id="T_2d2b1_row13_col9" class="data row13 col9" >38.45775</td>
      <td id="T_2d2b1_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_2d2b1_row14_col0" class="data row14 col0" >12191</td>
      <td id="T_2d2b1_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_2d2b1_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_2d2b1_row14_col3" class="data row14 col3" >IF capital.gain <= 8606.7402 THEN class = <=50K</td>
      <td id="T_2d2b1_row14_col4" class="data row14 col4" >0.97315</td>
      <td id="T_2d2b1_row14_col5" class="data row14 col5" >0.99925</td>
      <td id="T_2d2b1_row14_col6" class="data row14 col6" >0.77953</td>
      <td id="T_2d2b1_row14_col7" class="data row14 col7" >1</td>
      <td id="T_2d2b1_row14_col8" class="data row14 col8" >0</td>
      <td id="T_2d2b1_row14_col9" class="data row14 col9" >40.04660</td>
      <td id="T_2d2b1_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_2d2b1_row15_col0" class="data row15 col0" >12191</td>
      <td id="T_2d2b1_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_2d2b1_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_2d2b1_row15_col3" class="data row15 col3" >IF hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row15_col4" class="data row15 col4" >0.21231</td>
      <td id="T_2d2b1_row15_col5" class="data row15 col5" >0.25574</td>
      <td id="T_2d2b1_row15_col6" class="data row15 col6" >0.91445</td>
      <td id="T_2d2b1_row15_col7" class="data row15 col7" >1</td>
      <td id="T_2d2b1_row15_col8" class="data row15 col8" >0</td>
      <td id="T_2d2b1_row15_col9" class="data row15 col9" >3.96192</td>
      <td id="T_2d2b1_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_2d2b1_row16_col0" class="data row16 col0" >12191</td>
      <td id="T_2d2b1_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_2d2b1_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_2d2b1_row16_col3" class="data row16 col3" >IF age <= 31.0 AND capital.gain <= 0.0 AND native.country = United-States THEN class = <=50K</td>
      <td id="T_2d2b1_row16_col4" class="data row16 col4" >0.30638</td>
      <td id="T_2d2b1_row16_col5" class="data row16 col5" >0.37728</td>
      <td id="T_2d2b1_row16_col6" class="data row16 col6" >0.93484</td>
      <td id="T_2d2b1_row16_col7" class="data row16 col7" >3</td>
      <td id="T_2d2b1_row16_col8" class="data row16 col8" >0</td>
      <td id="T_2d2b1_row16_col9" class="data row16 col9" >4.10710</td>
      <td id="T_2d2b1_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_2d2b1_row17_col0" class="data row17 col0" >12191</td>
      <td id="T_2d2b1_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_2d2b1_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_2d2b1_row17_col3" class="data row17 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row17_col4" class="data row17 col4" >0.16322</td>
      <td id="T_2d2b1_row17_col5" class="data row17 col5" >0.20251</td>
      <td id="T_2d2b1_row17_col6" class="data row17 col6" >0.94194</td>
      <td id="T_2d2b1_row17_col7" class="data row17 col7" >2</td>
      <td id="T_2d2b1_row17_col8" class="data row17 col8" >0</td>
      <td id="T_2d2b1_row17_col9" class="data row17 col9" >3.88605</td>
      <td id="T_2d2b1_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_2d2b1_row18_col0" class="data row18 col0" >12191</td>
      <td id="T_2d2b1_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_2d2b1_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_2d2b1_row18_col3" class="data row18 col3" >IF age <= 29.043 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row18_col4" class="data row18 col4" >0.28725</td>
      <td id="T_2d2b1_row18_col5" class="data row18 col5" >0.36173</td>
      <td id="T_2d2b1_row18_col6" class="data row18 col6" >0.95601</td>
      <td id="T_2d2b1_row18_col7" class="data row18 col7" >2</td>
      <td id="T_2d2b1_row18_col8" class="data row18 col8" >0</td>
      <td id="T_2d2b1_row18_col9" class="data row18 col9" >4.05909</td>
      <td id="T_2d2b1_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_2d2b1_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_2d2b1_row19_col0" class="data row19 col0" >12191</td>
      <td id="T_2d2b1_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_2d2b1_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_2d2b1_row19_col3" class="data row19 col3" >IF age <= 32.5613 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_2d2b1_row19_col4" class="data row19 col4" >0.36135</td>
      <td id="T_2d2b1_row19_col5" class="data row19 col5" >0.44091</td>
      <td id="T_2d2b1_row19_col6" class="data row19 col6" >0.92630</td>
      <td id="T_2d2b1_row19_col7" class="data row19 col7" >2</td>
      <td id="T_2d2b1_row19_col8" class="data row19 col8" >0</td>
      <td id="T_2d2b1_row19_col9" class="data row19 col9" >3.99501</td>
      <td id="T_2d2b1_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12191, Correct Prediction



<style type="text/css">
</style>
<table id="T_96d38">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_96d38_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_96d38_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_96d38_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_96d38_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_96d38_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_96d38_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_96d38_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_96d38_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_96d38_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_96d38_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_96d38_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_96d38_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_96d38_row0_col0" class="data row0 col0" >12191</td>
      <td id="T_96d38_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_96d38_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_96d38_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_96d38_row0_col4" class="data row0 col4" >0.63088</td>
      <td id="T_96d38_row0_col5" class="data row0 col5" >0.71733</td>
      <td id="T_96d38_row0_col6" class="data row0 col6" >0.86320</td>
      <td id="T_96d38_row0_col7" class="data row0 col7" >3</td>
      <td id="T_96d38_row0_col8" class="data row0 col8" >0</td>
      <td id="T_96d38_row0_col9" class="data row0 col9" >1.95820</td>
      <td id="T_96d38_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_96d38_row1_col0" class="data row1 col0" >12191</td>
      <td id="T_96d38_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_96d38_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_96d38_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_96d38_row1_col4" class="data row1 col4" >0.69287</td>
      <td id="T_96d38_row1_col5" class="data row1 col5" >0.77489</td>
      <td id="T_96d38_row1_col6" class="data row1 col6" >0.84904</td>
      <td id="T_96d38_row1_col7" class="data row1 col7" >3</td>
      <td id="T_96d38_row1_col8" class="data row1 col8" >0</td>
      <td id="T_96d38_row1_col9" class="data row1 col9" >2.31952</td>
      <td id="T_96d38_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_96d38_row2_col0" class="data row2 col0" >12191</td>
      <td id="T_96d38_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_96d38_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_96d38_row2_col3" class="data row2 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_96d38_row2_col4" class="data row2 col4" >0.48710</td>
      <td id="T_96d38_row2_col5" class="data row2 col5" >0.56643</td>
      <td id="T_96d38_row2_col6" class="data row2 col6" >0.88281</td>
      <td id="T_96d38_row2_col7" class="data row2 col7" >2</td>
      <td id="T_96d38_row2_col8" class="data row2 col8" >2</td>
      <td id="T_96d38_row2_col9" class="data row2 col9" >6.01046</td>
      <td id="T_96d38_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_96d38_row3_col0" class="data row3 col0" >12191</td>
      <td id="T_96d38_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_96d38_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_96d38_row3_col3" class="data row3 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_96d38_row3_col4" class="data row3 col4" >0.66269</td>
      <td id="T_96d38_row3_col5" class="data row3 col5" >0.73091</td>
      <td id="T_96d38_row3_col6" class="data row3 col6" >0.83733</td>
      <td id="T_96d38_row3_col7" class="data row3 col7" >3</td>
      <td id="T_96d38_row3_col8" class="data row3 col8" >0</td>
      <td id="T_96d38_row3_col9" class="data row3 col9" >2.33662</td>
      <td id="T_96d38_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_96d38_row4_col0" class="data row4 col0" >12191</td>
      <td id="T_96d38_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_96d38_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_96d38_row4_col3" class="data row4 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_96d38_row4_col4" class="data row4 col4" >0.69595</td>
      <td id="T_96d38_row4_col5" class="data row4 col5" >0.75357</td>
      <td id="T_96d38_row4_col6" class="data row4 col6" >0.82203</td>
      <td id="T_96d38_row4_col7" class="data row4 col7" >2</td>
      <td id="T_96d38_row4_col8" class="data row4 col8" >0</td>
      <td id="T_96d38_row4_col9" class="data row4 col9" >2.09261</td>
      <td id="T_96d38_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_96d38_row5_col0" class="data row5 col0" >12191</td>
      <td id="T_96d38_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_96d38_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_96d38_row5_col3" class="data row5 col3" >IF capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_96d38_row5_col4" class="data row5 col4" >0.91598</td>
      <td id="T_96d38_row5_col5" class="data row5 col5" >0.95810</td>
      <td id="T_96d38_row5_col6" class="data row5 col6" >0.79408</td>
      <td id="T_96d38_row5_col7" class="data row5 col7" >1</td>
      <td id="T_96d38_row5_col8" class="data row5 col8" >0</td>
      <td id="T_96d38_row5_col9" class="data row5 col9" >190.90707</td>
      <td id="T_96d38_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_96d38_row6_col0" class="data row6 col0" >12191</td>
      <td id="T_96d38_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_96d38_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_96d38_row6_col3" class="data row6 col3" >IF capital.gain <= 1407.808 THEN class = <=50K</td>
      <td id="T_96d38_row6_col4" class="data row6 col4" >0.91896</td>
      <td id="T_96d38_row6_col5" class="data row6 col5" >0.96203</td>
      <td id="T_96d38_row6_col6" class="data row6 col6" >0.79475</td>
      <td id="T_96d38_row6_col7" class="data row6 col7" >1</td>
      <td id="T_96d38_row6_col8" class="data row6 col8" >0</td>
      <td id="T_96d38_row6_col9" class="data row6 col9" >196.93938</td>
      <td id="T_96d38_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_96d38_row7_col0" class="data row7 col0" >12191</td>
      <td id="T_96d38_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_96d38_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_96d38_row7_col3" class="data row7 col3" >IF capital.gain <= 8066.0229 THEN class = <=50K</td>
      <td id="T_96d38_row7_col4" class="data row7 col4" >0.97315</td>
      <td id="T_96d38_row7_col5" class="data row7 col5" >0.99925</td>
      <td id="T_96d38_row7_col6" class="data row7 col6" >0.77953</td>
      <td id="T_96d38_row7_col7" class="data row7 col7" >1</td>
      <td id="T_96d38_row7_col8" class="data row7 col8" >0</td>
      <td id="T_96d38_row7_col9" class="data row7 col9" >288.81475</td>
      <td id="T_96d38_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_96d38_row8_col0" class="data row8 col0" >12191</td>
      <td id="T_96d38_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_96d38_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_96d38_row8_col3" class="data row8 col3" >IF capital.gain <= 594.0 THEN class = <=50K</td>
      <td id="T_96d38_row8_col4" class="data row8 col4" >0.91725</td>
      <td id="T_96d38_row8_col5" class="data row8 col5" >0.95978</td>
      <td id="T_96d38_row8_col6" class="data row8 col6" >0.79437</td>
      <td id="T_96d38_row8_col7" class="data row8 col7" >1</td>
      <td id="T_96d38_row8_col8" class="data row8 col8" >0</td>
      <td id="T_96d38_row8_col9" class="data row8 col9" >316.98152</td>
      <td id="T_96d38_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_96d38_row9_col0" class="data row9 col0" >12191</td>
      <td id="T_96d38_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_96d38_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_96d38_row9_col3" class="data row9 col3" >IF capital.gain <= 6229.6605 THEN class = <=50K</td>
      <td id="T_96d38_row9_col4" class="data row9 col4" >0.95450</td>
      <td id="T_96d38_row9_col5" class="data row9 col5" >0.99717</td>
      <td id="T_96d38_row9_col6" class="data row9 col6" >0.79311</td>
      <td id="T_96d38_row9_col7" class="data row9 col7" >1</td>
      <td id="T_96d38_row9_col8" class="data row9 col8" >0</td>
      <td id="T_96d38_row9_col9" class="data row9 col9" >329.93644</td>
      <td id="T_96d38_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_96d38_row10_col0" class="data row10 col0" >12191</td>
      <td id="T_96d38_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_96d38_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_96d38_row10_col3" class="data row10 col3" >IF capital.gain <= 370.1838 THEN class = <=50K</td>
      <td id="T_96d38_row10_col4" class="data row10 col4" >0.91620</td>
      <td id="T_96d38_row10_col5" class="data row10 col5" >0.95839</td>
      <td id="T_96d38_row10_col6" class="data row10 col6" >0.79413</td>
      <td id="T_96d38_row10_col7" class="data row10 col7" >1</td>
      <td id="T_96d38_row10_col8" class="data row10 col8" >0</td>
      <td id="T_96d38_row10_col9" class="data row10 col9" >59.51019</td>
      <td id="T_96d38_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_96d38_row11_col0" class="data row11 col0" >12191</td>
      <td id="T_96d38_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_96d38_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_96d38_row11_col3" class="data row11 col3" >IF capital.gain <= 9458.4502 THEN class = <=50K</td>
      <td id="T_96d38_row11_col4" class="data row11 col4" >0.97543</td>
      <td id="T_96d38_row11_col5" class="data row11 col5" >0.99925</td>
      <td id="T_96d38_row11_col6" class="data row11 col6" >0.77771</td>
      <td id="T_96d38_row11_col7" class="data row11 col7" >1</td>
      <td id="T_96d38_row11_col8" class="data row11 col8" >0</td>
      <td id="T_96d38_row11_col9" class="data row11 col9" >35.10025</td>
      <td id="T_96d38_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_96d38_row12_col0" class="data row12 col0" >12191</td>
      <td id="T_96d38_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_96d38_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_96d38_row12_col3" class="data row12 col3" >IF capital.gain <= 7040.3074 THEN class = <=50K</td>
      <td id="T_96d38_row12_col4" class="data row12 col4" >0.95621</td>
      <td id="T_96d38_row12_col5" class="data row12 col5" >0.99890</td>
      <td id="T_96d38_row12_col6" class="data row12 col6" >0.79306</td>
      <td id="T_96d38_row12_col7" class="data row12 col7" >1</td>
      <td id="T_96d38_row12_col8" class="data row12 col8" >0</td>
      <td id="T_96d38_row12_col9" class="data row12 col9" >37.75637</td>
      <td id="T_96d38_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_96d38_row13_col0" class="data row13 col0" >12191</td>
      <td id="T_96d38_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_96d38_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_96d38_row13_col3" class="data row13 col3" >IF capital.gain <= 270.5606 AND workclass != Self-emp-not-inc THEN class = <=50K</td>
      <td id="T_96d38_row13_col4" class="data row13 col4" >0.84631</td>
      <td id="T_96d38_row13_col5" class="data row13 col5" >0.88863</td>
      <td id="T_96d38_row13_col6" class="data row13 col6" >0.79714</td>
      <td id="T_96d38_row13_col7" class="data row13 col7" >2</td>
      <td id="T_96d38_row13_col8" class="data row13 col8" >0</td>
      <td id="T_96d38_row13_col9" class="data row13 col9" >38.45775</td>
      <td id="T_96d38_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_96d38_row14_col0" class="data row14 col0" >12191</td>
      <td id="T_96d38_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_96d38_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_96d38_row14_col3" class="data row14 col3" >IF capital.gain <= 8606.7402 THEN class = <=50K</td>
      <td id="T_96d38_row14_col4" class="data row14 col4" >0.97315</td>
      <td id="T_96d38_row14_col5" class="data row14 col5" >0.99925</td>
      <td id="T_96d38_row14_col6" class="data row14 col6" >0.77953</td>
      <td id="T_96d38_row14_col7" class="data row14 col7" >1</td>
      <td id="T_96d38_row14_col8" class="data row14 col8" >0</td>
      <td id="T_96d38_row14_col9" class="data row14 col9" >40.04660</td>
      <td id="T_96d38_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_96d38_row15_col0" class="data row15 col0" >12191</td>
      <td id="T_96d38_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_96d38_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_96d38_row15_col3" class="data row15 col3" >IF hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_96d38_row15_col4" class="data row15 col4" >0.21231</td>
      <td id="T_96d38_row15_col5" class="data row15 col5" >0.25574</td>
      <td id="T_96d38_row15_col6" class="data row15 col6" >0.91445</td>
      <td id="T_96d38_row15_col7" class="data row15 col7" >1</td>
      <td id="T_96d38_row15_col8" class="data row15 col8" >0</td>
      <td id="T_96d38_row15_col9" class="data row15 col9" >3.96192</td>
      <td id="T_96d38_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_96d38_row16_col0" class="data row16 col0" >12191</td>
      <td id="T_96d38_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_96d38_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_96d38_row16_col3" class="data row16 col3" >IF age <= 31.0 AND capital.gain <= 0.0 AND native.country = United-States THEN class = <=50K</td>
      <td id="T_96d38_row16_col4" class="data row16 col4" >0.30638</td>
      <td id="T_96d38_row16_col5" class="data row16 col5" >0.37728</td>
      <td id="T_96d38_row16_col6" class="data row16 col6" >0.93484</td>
      <td id="T_96d38_row16_col7" class="data row16 col7" >3</td>
      <td id="T_96d38_row16_col8" class="data row16 col8" >0</td>
      <td id="T_96d38_row16_col9" class="data row16 col9" >4.10710</td>
      <td id="T_96d38_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_96d38_row17_col0" class="data row17 col0" >12191</td>
      <td id="T_96d38_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_96d38_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_96d38_row17_col3" class="data row17 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_96d38_row17_col4" class="data row17 col4" >0.16322</td>
      <td id="T_96d38_row17_col5" class="data row17 col5" >0.20251</td>
      <td id="T_96d38_row17_col6" class="data row17 col6" >0.94194</td>
      <td id="T_96d38_row17_col7" class="data row17 col7" >2</td>
      <td id="T_96d38_row17_col8" class="data row17 col8" >0</td>
      <td id="T_96d38_row17_col9" class="data row17 col9" >3.88605</td>
      <td id="T_96d38_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_96d38_row18_col0" class="data row18 col0" >12191</td>
      <td id="T_96d38_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_96d38_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_96d38_row18_col3" class="data row18 col3" >IF age <= 29.043 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_96d38_row18_col4" class="data row18 col4" >0.28725</td>
      <td id="T_96d38_row18_col5" class="data row18 col5" >0.36173</td>
      <td id="T_96d38_row18_col6" class="data row18 col6" >0.95601</td>
      <td id="T_96d38_row18_col7" class="data row18 col7" >2</td>
      <td id="T_96d38_row18_col8" class="data row18 col8" >0</td>
      <td id="T_96d38_row18_col9" class="data row18 col9" >4.05909</td>
      <td id="T_96d38_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_96d38_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_96d38_row19_col0" class="data row19 col0" >12191</td>
      <td id="T_96d38_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_96d38_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_96d38_row19_col3" class="data row19 col3" >IF age <= 32.5613 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_96d38_row19_col4" class="data row19 col4" >0.36135</td>
      <td id="T_96d38_row19_col5" class="data row19 col5" >0.44091</td>
      <td id="T_96d38_row19_col6" class="data row19 col6" >0.92630</td>
      <td id="T_96d38_row19_col7" class="data row19 col7" >2</td>
      <td id="T_96d38_row19_col8" class="data row19 col8" >0</td>
      <td id="T_96d38_row19_col9" class="data row19 col9" >3.99501</td>
      <td id="T_96d38_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12191, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_25f89">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_25f89_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_25f89_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_25f89_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_25f89_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_25f89_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_25f89_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_25f89_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_25f89_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_25f89_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_25f89_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_25f89_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_25f89_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_25f89_row0_col0" class="data row0 col0" >12191</td>
      <td id="T_25f89_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_25f89_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_25f89_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_25f89_row0_col4" class="data row0 col4" >0.63088</td>
      <td id="T_25f89_row0_col5" class="data row0 col5" >0.71733</td>
      <td id="T_25f89_row0_col6" class="data row0 col6" >0.86320</td>
      <td id="T_25f89_row0_col7" class="data row0 col7" >3</td>
      <td id="T_25f89_row0_col8" class="data row0 col8" >0</td>
      <td id="T_25f89_row0_col9" class="data row0 col9" >1.95820</td>
      <td id="T_25f89_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_25f89_row1_col0" class="data row1 col0" >12191</td>
      <td id="T_25f89_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_25f89_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_25f89_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_25f89_row1_col4" class="data row1 col4" >0.69287</td>
      <td id="T_25f89_row1_col5" class="data row1 col5" >0.77489</td>
      <td id="T_25f89_row1_col6" class="data row1 col6" >0.84904</td>
      <td id="T_25f89_row1_col7" class="data row1 col7" >3</td>
      <td id="T_25f89_row1_col8" class="data row1 col8" >0</td>
      <td id="T_25f89_row1_col9" class="data row1 col9" >2.31952</td>
      <td id="T_25f89_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_25f89_row2_col0" class="data row2 col0" >12191</td>
      <td id="T_25f89_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_25f89_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_25f89_row2_col3" class="data row2 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_25f89_row2_col4" class="data row2 col4" >0.48710</td>
      <td id="T_25f89_row2_col5" class="data row2 col5" >0.56643</td>
      <td id="T_25f89_row2_col6" class="data row2 col6" >0.88281</td>
      <td id="T_25f89_row2_col7" class="data row2 col7" >2</td>
      <td id="T_25f89_row2_col8" class="data row2 col8" >2</td>
      <td id="T_25f89_row2_col9" class="data row2 col9" >6.01046</td>
      <td id="T_25f89_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_25f89_row3_col0" class="data row3 col0" >12191</td>
      <td id="T_25f89_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_25f89_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_25f89_row3_col3" class="data row3 col3" >IF age <= 47.0 AND capital.gain <= 0.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_25f89_row3_col4" class="data row3 col4" >0.66269</td>
      <td id="T_25f89_row3_col5" class="data row3 col5" >0.73091</td>
      <td id="T_25f89_row3_col6" class="data row3 col6" >0.83733</td>
      <td id="T_25f89_row3_col7" class="data row3 col7" >3</td>
      <td id="T_25f89_row3_col8" class="data row3 col8" >0</td>
      <td id="T_25f89_row3_col9" class="data row3 col9" >2.33662</td>
      <td id="T_25f89_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_25f89_row4_col0" class="data row4 col0" >12191</td>
      <td id="T_25f89_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_25f89_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_25f89_row4_col3" class="data row4 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_25f89_row4_col4" class="data row4 col4" >0.69595</td>
      <td id="T_25f89_row4_col5" class="data row4 col5" >0.75357</td>
      <td id="T_25f89_row4_col6" class="data row4 col6" >0.82203</td>
      <td id="T_25f89_row4_col7" class="data row4 col7" >2</td>
      <td id="T_25f89_row4_col8" class="data row4 col8" >0</td>
      <td id="T_25f89_row4_col9" class="data row4 col9" >2.09261</td>
      <td id="T_25f89_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_25f89_row5_col0" class="data row5 col0" >12191</td>
      <td id="T_25f89_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_25f89_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_25f89_row5_col3" class="data row5 col3" >IF capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_25f89_row5_col4" class="data row5 col4" >0.91598</td>
      <td id="T_25f89_row5_col5" class="data row5 col5" >0.95810</td>
      <td id="T_25f89_row5_col6" class="data row5 col6" >0.79408</td>
      <td id="T_25f89_row5_col7" class="data row5 col7" >1</td>
      <td id="T_25f89_row5_col8" class="data row5 col8" >0</td>
      <td id="T_25f89_row5_col9" class="data row5 col9" >190.90707</td>
      <td id="T_25f89_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_25f89_row6_col0" class="data row6 col0" >12191</td>
      <td id="T_25f89_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_25f89_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_25f89_row6_col3" class="data row6 col3" >IF capital.gain <= 1407.808 THEN class = <=50K</td>
      <td id="T_25f89_row6_col4" class="data row6 col4" >0.91896</td>
      <td id="T_25f89_row6_col5" class="data row6 col5" >0.96203</td>
      <td id="T_25f89_row6_col6" class="data row6 col6" >0.79475</td>
      <td id="T_25f89_row6_col7" class="data row6 col7" >1</td>
      <td id="T_25f89_row6_col8" class="data row6 col8" >0</td>
      <td id="T_25f89_row6_col9" class="data row6 col9" >196.93938</td>
      <td id="T_25f89_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_25f89_row7_col0" class="data row7 col0" >12191</td>
      <td id="T_25f89_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_25f89_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_25f89_row7_col3" class="data row7 col3" >IF capital.gain <= 8066.0229 THEN class = <=50K</td>
      <td id="T_25f89_row7_col4" class="data row7 col4" >0.97315</td>
      <td id="T_25f89_row7_col5" class="data row7 col5" >0.99925</td>
      <td id="T_25f89_row7_col6" class="data row7 col6" >0.77953</td>
      <td id="T_25f89_row7_col7" class="data row7 col7" >1</td>
      <td id="T_25f89_row7_col8" class="data row7 col8" >0</td>
      <td id="T_25f89_row7_col9" class="data row7 col9" >288.81475</td>
      <td id="T_25f89_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_25f89_row8_col0" class="data row8 col0" >12191</td>
      <td id="T_25f89_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_25f89_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_25f89_row8_col3" class="data row8 col3" >IF capital.gain <= 594.0 THEN class = <=50K</td>
      <td id="T_25f89_row8_col4" class="data row8 col4" >0.91725</td>
      <td id="T_25f89_row8_col5" class="data row8 col5" >0.95978</td>
      <td id="T_25f89_row8_col6" class="data row8 col6" >0.79437</td>
      <td id="T_25f89_row8_col7" class="data row8 col7" >1</td>
      <td id="T_25f89_row8_col8" class="data row8 col8" >0</td>
      <td id="T_25f89_row8_col9" class="data row8 col9" >316.98152</td>
      <td id="T_25f89_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_25f89_row9_col0" class="data row9 col0" >12191</td>
      <td id="T_25f89_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_25f89_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_25f89_row9_col3" class="data row9 col3" >IF capital.gain <= 6229.6605 THEN class = <=50K</td>
      <td id="T_25f89_row9_col4" class="data row9 col4" >0.95450</td>
      <td id="T_25f89_row9_col5" class="data row9 col5" >0.99717</td>
      <td id="T_25f89_row9_col6" class="data row9 col6" >0.79311</td>
      <td id="T_25f89_row9_col7" class="data row9 col7" >1</td>
      <td id="T_25f89_row9_col8" class="data row9 col8" >0</td>
      <td id="T_25f89_row9_col9" class="data row9 col9" >329.93644</td>
      <td id="T_25f89_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_25f89_row10_col0" class="data row10 col0" >12191</td>
      <td id="T_25f89_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_25f89_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_25f89_row10_col3" class="data row10 col3" >IF capital.gain <= 370.1838 THEN class = <=50K</td>
      <td id="T_25f89_row10_col4" class="data row10 col4" >0.91620</td>
      <td id="T_25f89_row10_col5" class="data row10 col5" >0.95839</td>
      <td id="T_25f89_row10_col6" class="data row10 col6" >0.79413</td>
      <td id="T_25f89_row10_col7" class="data row10 col7" >1</td>
      <td id="T_25f89_row10_col8" class="data row10 col8" >0</td>
      <td id="T_25f89_row10_col9" class="data row10 col9" >59.51019</td>
      <td id="T_25f89_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_25f89_row11_col0" class="data row11 col0" >12191</td>
      <td id="T_25f89_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_25f89_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_25f89_row11_col3" class="data row11 col3" >IF capital.gain <= 9458.4502 THEN class = <=50K</td>
      <td id="T_25f89_row11_col4" class="data row11 col4" >0.97543</td>
      <td id="T_25f89_row11_col5" class="data row11 col5" >0.99925</td>
      <td id="T_25f89_row11_col6" class="data row11 col6" >0.77771</td>
      <td id="T_25f89_row11_col7" class="data row11 col7" >1</td>
      <td id="T_25f89_row11_col8" class="data row11 col8" >0</td>
      <td id="T_25f89_row11_col9" class="data row11 col9" >35.10025</td>
      <td id="T_25f89_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_25f89_row12_col0" class="data row12 col0" >12191</td>
      <td id="T_25f89_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_25f89_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_25f89_row12_col3" class="data row12 col3" >IF capital.gain <= 7040.3074 THEN class = <=50K</td>
      <td id="T_25f89_row12_col4" class="data row12 col4" >0.95621</td>
      <td id="T_25f89_row12_col5" class="data row12 col5" >0.99890</td>
      <td id="T_25f89_row12_col6" class="data row12 col6" >0.79306</td>
      <td id="T_25f89_row12_col7" class="data row12 col7" >1</td>
      <td id="T_25f89_row12_col8" class="data row12 col8" >0</td>
      <td id="T_25f89_row12_col9" class="data row12 col9" >37.75637</td>
      <td id="T_25f89_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_25f89_row13_col0" class="data row13 col0" >12191</td>
      <td id="T_25f89_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_25f89_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_25f89_row13_col3" class="data row13 col3" >IF capital.gain <= 270.5606 AND workclass != Self-emp-not-inc THEN class = <=50K</td>
      <td id="T_25f89_row13_col4" class="data row13 col4" >0.84631</td>
      <td id="T_25f89_row13_col5" class="data row13 col5" >0.88863</td>
      <td id="T_25f89_row13_col6" class="data row13 col6" >0.79714</td>
      <td id="T_25f89_row13_col7" class="data row13 col7" >2</td>
      <td id="T_25f89_row13_col8" class="data row13 col8" >0</td>
      <td id="T_25f89_row13_col9" class="data row13 col9" >38.45775</td>
      <td id="T_25f89_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_25f89_row14_col0" class="data row14 col0" >12191</td>
      <td id="T_25f89_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_25f89_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_25f89_row14_col3" class="data row14 col3" >IF capital.gain <= 8606.7402 THEN class = <=50K</td>
      <td id="T_25f89_row14_col4" class="data row14 col4" >0.97315</td>
      <td id="T_25f89_row14_col5" class="data row14 col5" >0.99925</td>
      <td id="T_25f89_row14_col6" class="data row14 col6" >0.77953</td>
      <td id="T_25f89_row14_col7" class="data row14 col7" >1</td>
      <td id="T_25f89_row14_col8" class="data row14 col8" >0</td>
      <td id="T_25f89_row14_col9" class="data row14 col9" >40.04660</td>
      <td id="T_25f89_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_25f89_row15_col0" class="data row15 col0" >12191</td>
      <td id="T_25f89_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_25f89_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_25f89_row15_col3" class="data row15 col3" >IF hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_25f89_row15_col4" class="data row15 col4" >0.21231</td>
      <td id="T_25f89_row15_col5" class="data row15 col5" >0.25574</td>
      <td id="T_25f89_row15_col6" class="data row15 col6" >0.91445</td>
      <td id="T_25f89_row15_col7" class="data row15 col7" >1</td>
      <td id="T_25f89_row15_col8" class="data row15 col8" >0</td>
      <td id="T_25f89_row15_col9" class="data row15 col9" >3.96192</td>
      <td id="T_25f89_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_25f89_row16_col0" class="data row16 col0" >12191</td>
      <td id="T_25f89_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_25f89_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_25f89_row16_col3" class="data row16 col3" >IF age <= 31.0 AND capital.gain <= 0.0 AND native.country = United-States THEN class = <=50K</td>
      <td id="T_25f89_row16_col4" class="data row16 col4" >0.30638</td>
      <td id="T_25f89_row16_col5" class="data row16 col5" >0.37728</td>
      <td id="T_25f89_row16_col6" class="data row16 col6" >0.93484</td>
      <td id="T_25f89_row16_col7" class="data row16 col7" >3</td>
      <td id="T_25f89_row16_col8" class="data row16 col8" >0</td>
      <td id="T_25f89_row16_col9" class="data row16 col9" >4.10710</td>
      <td id="T_25f89_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_25f89_row17_col0" class="data row17 col0" >12191</td>
      <td id="T_25f89_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_25f89_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_25f89_row17_col3" class="data row17 col3" >IF capital.gain <= 0.0 AND hours.per.week <= 33.0 THEN class = <=50K</td>
      <td id="T_25f89_row17_col4" class="data row17 col4" >0.16322</td>
      <td id="T_25f89_row17_col5" class="data row17 col5" >0.20251</td>
      <td id="T_25f89_row17_col6" class="data row17 col6" >0.94194</td>
      <td id="T_25f89_row17_col7" class="data row17 col7" >2</td>
      <td id="T_25f89_row17_col8" class="data row17 col8" >0</td>
      <td id="T_25f89_row17_col9" class="data row17 col9" >3.88605</td>
      <td id="T_25f89_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_25f89_row18_col0" class="data row18 col0" >12191</td>
      <td id="T_25f89_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_25f89_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_25f89_row18_col3" class="data row18 col3" >IF age <= 29.043 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_25f89_row18_col4" class="data row18 col4" >0.28725</td>
      <td id="T_25f89_row18_col5" class="data row18 col5" >0.36173</td>
      <td id="T_25f89_row18_col6" class="data row18 col6" >0.95601</td>
      <td id="T_25f89_row18_col7" class="data row18 col7" >2</td>
      <td id="T_25f89_row18_col8" class="data row18 col8" >0</td>
      <td id="T_25f89_row18_col9" class="data row18 col9" >4.05909</td>
      <td id="T_25f89_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_25f89_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_25f89_row19_col0" class="data row19 col0" >12191</td>
      <td id="T_25f89_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_25f89_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_25f89_row19_col3" class="data row19 col3" >IF age <= 32.5613 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_25f89_row19_col4" class="data row19 col4" >0.36135</td>
      <td id="T_25f89_row19_col5" class="data row19 col5" >0.44091</td>
      <td id="T_25f89_row19_col6" class="data row19 col6" >0.92630</td>
      <td id="T_25f89_row19_col7" class="data row19 col7" >2</td>
      <td id="T_25f89_row19_col8" class="data row19 col8" >0</td>
      <td id="T_25f89_row19_col9" class="data row19 col9" >3.99501</td>
      <td id="T_25f89_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12191, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.97543, Pre: 0.95601)



<style type="text/css">
#T_53a18_row8_col0, #T_53a18_row8_col1, #T_53a18_row8_col2, #T_53a18_row8_col3, #T_53a18_row8_col4, #T_53a18_row8_col5, #T_53a18_row8_col6, #T_53a18_row8_col7, #T_53a18_row8_col8, #T_53a18_row8_col9, #T_53a18_row8_col10, #T_53a18_row8_col11 {
  font-weight: bold;
}
</style>
<table id="T_53a18">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_53a18_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_53a18_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_53a18_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_53a18_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_53a18_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_53a18_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_53a18_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_53a18_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_53a18_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_53a18_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_53a18_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_53a18_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_53a18_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_53a18_row0_col0" class="data row0 col0" >12191</td>
      <td id="T_53a18_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_53a18_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_53a18_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_53a18_row0_col4" class="data row0 col4" >0.63088</td>
      <td id="T_53a18_row0_col5" class="data row0 col5" >0.71733</td>
      <td id="T_53a18_row0_col6" class="data row0 col6" >0.86320</td>
      <td id="T_53a18_row0_col7" class="data row0 col7" >3</td>
      <td id="T_53a18_row0_col8" class="data row0 col8" >0</td>
      <td id="T_53a18_row0_col9" class="data row0 col9" >1.95820</td>
      <td id="T_53a18_row0_col10" class="data row0 col10" >False</td>
      <td id="T_53a18_row0_col11" class="data row0 col11" >0.35683</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_53a18_row1_col0" class="data row1 col0" >12191</td>
      <td id="T_53a18_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_53a18_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_53a18_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_53a18_row1_col4" class="data row1 col4" >0.69287</td>
      <td id="T_53a18_row1_col5" class="data row1 col5" >0.77489</td>
      <td id="T_53a18_row1_col6" class="data row1 col6" >0.84904</td>
      <td id="T_53a18_row1_col7" class="data row1 col7" >3</td>
      <td id="T_53a18_row1_col8" class="data row1 col8" >0</td>
      <td id="T_53a18_row1_col9" class="data row1 col9" >2.31952</td>
      <td id="T_53a18_row1_col10" class="data row1 col10" >False</td>
      <td id="T_53a18_row1_col11" class="data row1 col11" >0.30213</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_53a18_row2_col0" class="data row2 col0" >12191</td>
      <td id="T_53a18_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_53a18_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_53a18_row2_col3" class="data row2 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_53a18_row2_col4" class="data row2 col4" >0.48710</td>
      <td id="T_53a18_row2_col5" class="data row2 col5" >0.56643</td>
      <td id="T_53a18_row2_col6" class="data row2 col6" >0.88281</td>
      <td id="T_53a18_row2_col7" class="data row2 col7" >2</td>
      <td id="T_53a18_row2_col8" class="data row2 col8" >2</td>
      <td id="T_53a18_row2_col9" class="data row2 col9" >6.01046</td>
      <td id="T_53a18_row2_col10" class="data row2 col10" >False</td>
      <td id="T_53a18_row2_col11" class="data row2 col11" >0.49379</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_53a18_row3_col0" class="data row3 col0" >12191</td>
      <td id="T_53a18_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_53a18_row3_col2" class="data row3 col2" >ANCHOR5</td>
      <td id="T_53a18_row3_col3" class="data row3 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_53a18_row3_col4" class="data row3 col4" >0.69595</td>
      <td id="T_53a18_row3_col5" class="data row3 col5" >0.75357</td>
      <td id="T_53a18_row3_col6" class="data row3 col6" >0.82203</td>
      <td id="T_53a18_row3_col7" class="data row3 col7" >2</td>
      <td id="T_53a18_row3_col8" class="data row3 col8" >0</td>
      <td id="T_53a18_row3_col9" class="data row3 col9" >2.09261</td>
      <td id="T_53a18_row3_col10" class="data row3 col10" >False</td>
      <td id="T_53a18_row3_col11" class="data row3 col11" >0.30994</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_53a18_row4_col0" class="data row4 col0" >12191</td>
      <td id="T_53a18_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_53a18_row4_col2" class="data row4 col2" >LORE2</td>
      <td id="T_53a18_row4_col3" class="data row4 col3" >IF capital.gain <= 1407.808 THEN class = <=50K</td>
      <td id="T_53a18_row4_col4" class="data row4 col4" >0.91896</td>
      <td id="T_53a18_row4_col5" class="data row4 col5" >0.96203</td>
      <td id="T_53a18_row4_col6" class="data row4 col6" >0.79475</td>
      <td id="T_53a18_row4_col7" class="data row4 col7" >1</td>
      <td id="T_53a18_row4_col8" class="data row4 col8" >0</td>
      <td id="T_53a18_row4_col9" class="data row4 col9" >196.93938</td>
      <td id="T_53a18_row4_col10" class="data row4 col10" >False</td>
      <td id="T_53a18_row4_col11" class="data row4 col11" >0.17086</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_53a18_row5_col0" class="data row5 col0" >12191</td>
      <td id="T_53a18_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_53a18_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_53a18_row5_col3" class="data row5 col3" >IF capital.gain <= 8066.0229 THEN class = <=50K</td>
      <td id="T_53a18_row5_col4" class="data row5 col4" >0.97315</td>
      <td id="T_53a18_row5_col5" class="data row5 col5" >0.99925</td>
      <td id="T_53a18_row5_col6" class="data row5 col6" >0.77953</td>
      <td id="T_53a18_row5_col7" class="data row5 col7" >1</td>
      <td id="T_53a18_row5_col8" class="data row5 col8" >0</td>
      <td id="T_53a18_row5_col9" class="data row5 col9" >288.81475</td>
      <td id="T_53a18_row5_col10" class="data row5 col10" >False</td>
      <td id="T_53a18_row5_col11" class="data row5 col11" >0.17649</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_53a18_row6_col0" class="data row6 col0" >12191</td>
      <td id="T_53a18_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_53a18_row6_col2" class="data row6 col2" >LORE5</td>
      <td id="T_53a18_row6_col3" class="data row6 col3" >IF capital.gain <= 6229.6605 THEN class = <=50K</td>
      <td id="T_53a18_row6_col4" class="data row6 col4" >0.95450</td>
      <td id="T_53a18_row6_col5" class="data row6 col5" >0.99717</td>
      <td id="T_53a18_row6_col6" class="data row6 col6" >0.79311</td>
      <td id="T_53a18_row6_col7" class="data row6 col7" >1</td>
      <td id="T_53a18_row6_col8" class="data row6 col8" >0</td>
      <td id="T_53a18_row6_col9" class="data row6 col9" >329.93644</td>
      <td id="T_53a18_row6_col10" class="data row6 col10" >False</td>
      <td id="T_53a18_row6_col11" class="data row6 col11" >0.16424</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_53a18_row7_col0" class="data row7 col0" >12191</td>
      <td id="T_53a18_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_53a18_row7_col2" class="data row7 col2" >LORE_SA2</td>
      <td id="T_53a18_row7_col3" class="data row7 col3" >IF capital.gain <= 9458.4502 THEN class = <=50K</td>
      <td id="T_53a18_row7_col4" class="data row7 col4" >0.97543</td>
      <td id="T_53a18_row7_col5" class="data row7 col5" >0.99925</td>
      <td id="T_53a18_row7_col6" class="data row7 col6" >0.77771</td>
      <td id="T_53a18_row7_col7" class="data row7 col7" >1</td>
      <td id="T_53a18_row7_col8" class="data row7 col8" >0</td>
      <td id="T_53a18_row7_col9" class="data row7 col9" >35.10025</td>
      <td id="T_53a18_row7_col10" class="data row7 col10" >False</td>
      <td id="T_53a18_row7_col11" class="data row7 col11" >0.17830</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_53a18_row8_col0" class="data row8 col0" >12191</td>
      <td id="T_53a18_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_53a18_row8_col2" class="data row8 col2" >LORE_SA3</td>
      <td id="T_53a18_row8_col3" class="data row8 col3" >IF capital.gain <= 7040.3074 THEN class = <=50K</td>
      <td id="T_53a18_row8_col4" class="data row8 col4" >0.95621</td>
      <td id="T_53a18_row8_col5" class="data row8 col5" >0.99890</td>
      <td id="T_53a18_row8_col6" class="data row8 col6" >0.79306</td>
      <td id="T_53a18_row8_col7" class="data row8 col7" >1</td>
      <td id="T_53a18_row8_col8" class="data row8 col8" >0</td>
      <td id="T_53a18_row8_col9" class="data row8 col9" >37.75637</td>
      <td id="T_53a18_row8_col10" class="data row8 col10" >False</td>
      <td id="T_53a18_row8_col11" class="data row8 col11" >0.16408</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_53a18_row9_col0" class="data row9 col0" >12191</td>
      <td id="T_53a18_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_53a18_row9_col2" class="data row9 col2" >LORE_SA4</td>
      <td id="T_53a18_row9_col3" class="data row9 col3" >IF capital.gain <= 270.5606 AND workclass != Self-emp-not-inc THEN class = <=50K</td>
      <td id="T_53a18_row9_col4" class="data row9 col4" >0.84631</td>
      <td id="T_53a18_row9_col5" class="data row9 col5" >0.88863</td>
      <td id="T_53a18_row9_col6" class="data row9 col6" >0.79714</td>
      <td id="T_53a18_row9_col7" class="data row9 col7" >2</td>
      <td id="T_53a18_row9_col8" class="data row9 col8" >0</td>
      <td id="T_53a18_row9_col9" class="data row9 col9" >38.45775</td>
      <td id="T_53a18_row9_col10" class="data row9 col10" >False</td>
      <td id="T_53a18_row9_col11" class="data row9 col11" >0.20472</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_53a18_row10_col0" class="data row10 col0" >12191</td>
      <td id="T_53a18_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_53a18_row10_col2" class="data row10 col2" >LORE_SA5</td>
      <td id="T_53a18_row10_col3" class="data row10 col3" >IF capital.gain <= 8606.7402 THEN class = <=50K</td>
      <td id="T_53a18_row10_col4" class="data row10 col4" >0.97315</td>
      <td id="T_53a18_row10_col5" class="data row10 col5" >0.99925</td>
      <td id="T_53a18_row10_col6" class="data row10 col6" >0.77953</td>
      <td id="T_53a18_row10_col7" class="data row10 col7" >1</td>
      <td id="T_53a18_row10_col8" class="data row10 col8" >0</td>
      <td id="T_53a18_row10_col9" class="data row10 col9" >40.04660</td>
      <td id="T_53a18_row10_col10" class="data row10 col10" >False</td>
      <td id="T_53a18_row10_col11" class="data row10 col11" >0.17649</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_53a18_row11_col0" class="data row11 col0" >12191</td>
      <td id="T_53a18_row11_col1" class="data row11 col1" >EXPLAN</td>
      <td id="T_53a18_row11_col2" class="data row11 col2" >EXPLAN2</td>
      <td id="T_53a18_row11_col3" class="data row11 col3" >IF age <= 31.0 AND capital.gain <= 0.0 AND native.country = United-States THEN class = <=50K</td>
      <td id="T_53a18_row11_col4" class="data row11 col4" >0.30638</td>
      <td id="T_53a18_row11_col5" class="data row11 col5" >0.37728</td>
      <td id="T_53a18_row11_col6" class="data row11 col6" >0.93484</td>
      <td id="T_53a18_row11_col7" class="data row11 col7" >3</td>
      <td id="T_53a18_row11_col8" class="data row11 col8" >0</td>
      <td id="T_53a18_row11_col9" class="data row11 col9" >4.10710</td>
      <td id="T_53a18_row11_col10" class="data row11 col10" >False</td>
      <td id="T_53a18_row11_col11" class="data row11 col11" >0.66938</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_53a18_row12_col0" class="data row12 col0" >12191</td>
      <td id="T_53a18_row12_col1" class="data row12 col1" >EXPLAN</td>
      <td id="T_53a18_row12_col2" class="data row12 col2" >EXPLAN4</td>
      <td id="T_53a18_row12_col3" class="data row12 col3" >IF age <= 29.043 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_53a18_row12_col4" class="data row12 col4" >0.28725</td>
      <td id="T_53a18_row12_col5" class="data row12 col5" >0.36173</td>
      <td id="T_53a18_row12_col6" class="data row12 col6" >0.95601</td>
      <td id="T_53a18_row12_col7" class="data row12 col7" >2</td>
      <td id="T_53a18_row12_col8" class="data row12 col8" >0</td>
      <td id="T_53a18_row12_col9" class="data row12 col9" >4.05909</td>
      <td id="T_53a18_row12_col10" class="data row12 col10" >False</td>
      <td id="T_53a18_row12_col11" class="data row12 col11" >0.68818</td>
    </tr>
    <tr>
      <th id="T_53a18_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_53a18_row13_col0" class="data row13 col0" >12191</td>
      <td id="T_53a18_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_53a18_row13_col2" class="data row13 col2" >EXPLAN5</td>
      <td id="T_53a18_row13_col3" class="data row13 col3" >IF age <= 32.5613 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_53a18_row13_col4" class="data row13 col4" >0.36135</td>
      <td id="T_53a18_row13_col5" class="data row13 col5" >0.44091</td>
      <td id="T_53a18_row13_col6" class="data row13 col6" >0.92630</td>
      <td id="T_53a18_row13_col7" class="data row13 col7" >2</td>
      <td id="T_53a18_row13_col8" class="data row13 col8" >0</td>
      <td id="T_53a18_row13_col9" class="data row13 col9" >3.99501</td>
      <td id="T_53a18_row13_col10" class="data row13 col10" >False</td>
      <td id="T_53a18_row13_col11" class="data row13 col11" >0.61480</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_178.png)
    



### Rules for Instance 12191, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.97543, Pre: 0.95601), Unique rules (diffrent features)



<style type="text/css">
#T_d0435_row2_col0, #T_d0435_row2_col1, #T_d0435_row2_col2, #T_d0435_row2_col3, #T_d0435_row2_col4, #T_d0435_row2_col5, #T_d0435_row2_col6, #T_d0435_row2_col7, #T_d0435_row2_col8, #T_d0435_row2_col9, #T_d0435_row2_col10, #T_d0435_row2_col11 {
  font-weight: bold;
}
</style>
<table id="T_d0435">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d0435_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_d0435_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_d0435_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_d0435_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_d0435_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_d0435_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_d0435_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_d0435_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_d0435_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_d0435_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_d0435_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_d0435_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d0435_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_d0435_row0_col0" class="data row0 col0" >12191</td>
      <td id="T_d0435_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_d0435_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_d0435_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_d0435_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_d0435_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_d0435_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_d0435_row0_col7" class="data row0 col7" >3</td>
      <td id="T_d0435_row0_col8" class="data row0 col8" >0</td>
      <td id="T_d0435_row0_col9" class="data row0 col9" >2.31952</td>
      <td id="T_d0435_row0_col10" class="data row0 col10" >False</td>
      <td id="T_d0435_row0_col11" class="data row0 col11" >0.30213</td>
    </tr>
    <tr>
      <th id="T_d0435_level0_row1" class="row_heading level0 row1" >3</th>
      <td id="T_d0435_row1_col0" class="data row1 col0" >12191</td>
      <td id="T_d0435_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_d0435_row1_col2" class="data row1 col2" >ANCHOR5</td>
      <td id="T_d0435_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_d0435_row1_col4" class="data row1 col4" >0.69595</td>
      <td id="T_d0435_row1_col5" class="data row1 col5" >0.75357</td>
      <td id="T_d0435_row1_col6" class="data row1 col6" >0.82203</td>
      <td id="T_d0435_row1_col7" class="data row1 col7" >2</td>
      <td id="T_d0435_row1_col8" class="data row1 col8" >0</td>
      <td id="T_d0435_row1_col9" class="data row1 col9" >2.09261</td>
      <td id="T_d0435_row1_col10" class="data row1 col10" >False</td>
      <td id="T_d0435_row1_col11" class="data row1 col11" >0.30994</td>
    </tr>
    <tr>
      <th id="T_d0435_level0_row2" class="row_heading level0 row2" >8</th>
      <td id="T_d0435_row2_col0" class="data row2 col0" >12191</td>
      <td id="T_d0435_row2_col1" class="data row2 col1" >LORE_SA</td>
      <td id="T_d0435_row2_col2" class="data row2 col2" >LORE_SA3</td>
      <td id="T_d0435_row2_col3" class="data row2 col3" >IF capital.gain <= 7040.3074 THEN class = <=50K</td>
      <td id="T_d0435_row2_col4" class="data row2 col4" >0.95621</td>
      <td id="T_d0435_row2_col5" class="data row2 col5" >0.99890</td>
      <td id="T_d0435_row2_col6" class="data row2 col6" >0.79306</td>
      <td id="T_d0435_row2_col7" class="data row2 col7" >1</td>
      <td id="T_d0435_row2_col8" class="data row2 col8" >0</td>
      <td id="T_d0435_row2_col9" class="data row2 col9" >37.75637</td>
      <td id="T_d0435_row2_col10" class="data row2 col10" >False</td>
      <td id="T_d0435_row2_col11" class="data row2 col11" >0.16408</td>
    </tr>
    <tr>
      <th id="T_d0435_level0_row3" class="row_heading level0 row3" >9</th>
      <td id="T_d0435_row3_col0" class="data row3 col0" >12191</td>
      <td id="T_d0435_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_d0435_row3_col2" class="data row3 col2" >LORE_SA4</td>
      <td id="T_d0435_row3_col3" class="data row3 col3" >IF capital.gain <= 270.5606 AND workclass != Self-emp-not-inc THEN class = <=50K</td>
      <td id="T_d0435_row3_col4" class="data row3 col4" >0.84631</td>
      <td id="T_d0435_row3_col5" class="data row3 col5" >0.88863</td>
      <td id="T_d0435_row3_col6" class="data row3 col6" >0.79714</td>
      <td id="T_d0435_row3_col7" class="data row3 col7" >2</td>
      <td id="T_d0435_row3_col8" class="data row3 col8" >0</td>
      <td id="T_d0435_row3_col9" class="data row3 col9" >38.45775</td>
      <td id="T_d0435_row3_col10" class="data row3 col10" >False</td>
      <td id="T_d0435_row3_col11" class="data row3 col11" >0.20472</td>
    </tr>
    <tr>
      <th id="T_d0435_level0_row4" class="row_heading level0 row4" >11</th>
      <td id="T_d0435_row4_col0" class="data row4 col0" >12191</td>
      <td id="T_d0435_row4_col1" class="data row4 col1" >EXPLAN</td>
      <td id="T_d0435_row4_col2" class="data row4 col2" >EXPLAN2</td>
      <td id="T_d0435_row4_col3" class="data row4 col3" >IF age <= 31.0 AND capital.gain <= 0.0 AND native.country = United-States THEN class = <=50K</td>
      <td id="T_d0435_row4_col4" class="data row4 col4" >0.30638</td>
      <td id="T_d0435_row4_col5" class="data row4 col5" >0.37728</td>
      <td id="T_d0435_row4_col6" class="data row4 col6" >0.93484</td>
      <td id="T_d0435_row4_col7" class="data row4 col7" >3</td>
      <td id="T_d0435_row4_col8" class="data row4 col8" >0</td>
      <td id="T_d0435_row4_col9" class="data row4 col9" >4.10710</td>
      <td id="T_d0435_row4_col10" class="data row4 col10" >False</td>
      <td id="T_d0435_row4_col11" class="data row4 col11" >0.66938</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_181.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_182.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_183.png)
    



### Rules for Instance 12191, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99925, Pre: 0.95601, Len: 0.77953)



<style type="text/css">
#T_b1b49_row6_col0, #T_b1b49_row6_col1, #T_b1b49_row6_col2, #T_b1b49_row6_col3, #T_b1b49_row6_col4, #T_b1b49_row6_col5, #T_b1b49_row6_col6, #T_b1b49_row6_col7, #T_b1b49_row6_col8, #T_b1b49_row6_col9, #T_b1b49_row6_col10, #T_b1b49_row6_col11 {
  font-weight: bold;
}
</style>
<table id="T_b1b49">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b1b49_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_b1b49_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_b1b49_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_b1b49_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_b1b49_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_b1b49_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_b1b49_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_b1b49_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_b1b49_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_b1b49_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_b1b49_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_b1b49_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b1b49_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b1b49_row0_col0" class="data row0 col0" >12191</td>
      <td id="T_b1b49_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_b1b49_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_b1b49_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_b1b49_row0_col4" class="data row0 col4" >0.63088</td>
      <td id="T_b1b49_row0_col5" class="data row0 col5" >0.71733</td>
      <td id="T_b1b49_row0_col6" class="data row0 col6" >0.86320</td>
      <td id="T_b1b49_row0_col7" class="data row0 col7" >3</td>
      <td id="T_b1b49_row0_col8" class="data row0 col8" >0</td>
      <td id="T_b1b49_row0_col9" class="data row0 col9" >1.95820</td>
      <td id="T_b1b49_row0_col10" class="data row0 col10" >False</td>
      <td id="T_b1b49_row0_col11" class="data row0 col11" >2.24022</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_b1b49_row1_col0" class="data row1 col0" >12191</td>
      <td id="T_b1b49_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_b1b49_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_b1b49_row1_col3" class="data row1 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_b1b49_row1_col4" class="data row1 col4" >0.69287</td>
      <td id="T_b1b49_row1_col5" class="data row1 col5" >0.77489</td>
      <td id="T_b1b49_row1_col6" class="data row1 col6" >0.84904</td>
      <td id="T_b1b49_row1_col7" class="data row1 col7" >3</td>
      <td id="T_b1b49_row1_col8" class="data row1 col8" >0</td>
      <td id="T_b1b49_row1_col9" class="data row1 col9" >2.31952</td>
      <td id="T_b1b49_row1_col10" class="data row1 col10" >False</td>
      <td id="T_b1b49_row1_col11" class="data row1 col11" >2.23434</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_b1b49_row2_col0" class="data row2 col0" >12191</td>
      <td id="T_b1b49_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_b1b49_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_b1b49_row2_col3" class="data row2 col3" >IF age <= 37.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_b1b49_row2_col4" class="data row2 col4" >0.48710</td>
      <td id="T_b1b49_row2_col5" class="data row2 col5" >0.56643</td>
      <td id="T_b1b49_row2_col6" class="data row2 col6" >0.88281</td>
      <td id="T_b1b49_row2_col7" class="data row2 col7" >2</td>
      <td id="T_b1b49_row2_col8" class="data row2 col8" >2</td>
      <td id="T_b1b49_row2_col9" class="data row2 col9" >6.01046</td>
      <td id="T_b1b49_row2_col10" class="data row2 col10" >False</td>
      <td id="T_b1b49_row2_col11" class="data row2 col11" >1.29701</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_b1b49_row3_col0" class="data row3 col0" >12191</td>
      <td id="T_b1b49_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_b1b49_row3_col2" class="data row3 col2" >ANCHOR5</td>
      <td id="T_b1b49_row3_col3" class="data row3 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_b1b49_row3_col4" class="data row3 col4" >0.69595</td>
      <td id="T_b1b49_row3_col5" class="data row3 col5" >0.75357</td>
      <td id="T_b1b49_row3_col6" class="data row3 col6" >0.82203</td>
      <td id="T_b1b49_row3_col7" class="data row3 col7" >2</td>
      <td id="T_b1b49_row3_col8" class="data row3 col8" >0</td>
      <td id="T_b1b49_row3_col9" class="data row3 col9" >2.09261</td>
      <td id="T_b1b49_row3_col10" class="data row3 col10" >False</td>
      <td id="T_b1b49_row3_col11" class="data row3 col11" >1.25214</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_b1b49_row4_col0" class="data row4 col0" >12191</td>
      <td id="T_b1b49_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_b1b49_row4_col2" class="data row4 col2" >LORE2</td>
      <td id="T_b1b49_row4_col3" class="data row4 col3" >IF capital.gain <= 1407.808 THEN class = <=50K</td>
      <td id="T_b1b49_row4_col4" class="data row4 col4" >0.91896</td>
      <td id="T_b1b49_row4_col5" class="data row4 col5" >0.96203</td>
      <td id="T_b1b49_row4_col6" class="data row4 col6" >0.79475</td>
      <td id="T_b1b49_row4_col7" class="data row4 col7" >1</td>
      <td id="T_b1b49_row4_col8" class="data row4 col8" >0</td>
      <td id="T_b1b49_row4_col9" class="data row4 col9" >196.93938</td>
      <td id="T_b1b49_row4_col10" class="data row4 col10" >False</td>
      <td id="T_b1b49_row4_col11" class="data row4 col11" >0.27568</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_b1b49_row5_col0" class="data row5 col0" >12191</td>
      <td id="T_b1b49_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_b1b49_row5_col2" class="data row5 col2" >LORE3</td>
      <td id="T_b1b49_row5_col3" class="data row5 col3" >IF capital.gain <= 8066.0229 THEN class = <=50K</td>
      <td id="T_b1b49_row5_col4" class="data row5 col4" >0.97315</td>
      <td id="T_b1b49_row5_col5" class="data row5 col5" >0.99925</td>
      <td id="T_b1b49_row5_col6" class="data row5 col6" >0.77953</td>
      <td id="T_b1b49_row5_col7" class="data row5 col7" >1</td>
      <td id="T_b1b49_row5_col8" class="data row5 col8" >0</td>
      <td id="T_b1b49_row5_col9" class="data row5 col9" >288.81475</td>
      <td id="T_b1b49_row5_col10" class="data row5 col10" >False</td>
      <td id="T_b1b49_row5_col11" class="data row5 col11" >0.28240</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_b1b49_row6_col0" class="data row6 col0" >12191</td>
      <td id="T_b1b49_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_b1b49_row6_col2" class="data row6 col2" >LORE5</td>
      <td id="T_b1b49_row6_col3" class="data row6 col3" >IF capital.gain <= 6229.6605 THEN class = <=50K</td>
      <td id="T_b1b49_row6_col4" class="data row6 col4" >0.95450</td>
      <td id="T_b1b49_row6_col5" class="data row6 col5" >0.99717</td>
      <td id="T_b1b49_row6_col6" class="data row6 col6" >0.79311</td>
      <td id="T_b1b49_row6_col7" class="data row6 col7" >1</td>
      <td id="T_b1b49_row6_col8" class="data row6 col8" >0</td>
      <td id="T_b1b49_row6_col9" class="data row6 col9" >329.93644</td>
      <td id="T_b1b49_row6_col10" class="data row6 col10" >False</td>
      <td id="T_b1b49_row6_col11" class="data row6 col11" >0.27413</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_b1b49_row7_col0" class="data row7 col0" >12191</td>
      <td id="T_b1b49_row7_col1" class="data row7 col1" >LORE_SA</td>
      <td id="T_b1b49_row7_col2" class="data row7 col2" >LORE_SA3</td>
      <td id="T_b1b49_row7_col3" class="data row7 col3" >IF capital.gain <= 7040.3074 THEN class = <=50K</td>
      <td id="T_b1b49_row7_col4" class="data row7 col4" >0.95621</td>
      <td id="T_b1b49_row7_col5" class="data row7 col5" >0.99890</td>
      <td id="T_b1b49_row7_col6" class="data row7 col6" >0.79306</td>
      <td id="T_b1b49_row7_col7" class="data row7 col7" >1</td>
      <td id="T_b1b49_row7_col8" class="data row7 col8" >0</td>
      <td id="T_b1b49_row7_col9" class="data row7 col9" >37.75637</td>
      <td id="T_b1b49_row7_col10" class="data row7 col10" >False</td>
      <td id="T_b1b49_row7_col11" class="data row7 col11" >0.27415</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_b1b49_row8_col0" class="data row8 col0" >12191</td>
      <td id="T_b1b49_row8_col1" class="data row8 col1" >LORE_SA</td>
      <td id="T_b1b49_row8_col2" class="data row8 col2" >LORE_SA4</td>
      <td id="T_b1b49_row8_col3" class="data row8 col3" >IF capital.gain <= 270.5606 AND workclass != Self-emp-not-inc THEN class = <=50K</td>
      <td id="T_b1b49_row8_col4" class="data row8 col4" >0.84631</td>
      <td id="T_b1b49_row8_col5" class="data row8 col5" >0.88863</td>
      <td id="T_b1b49_row8_col6" class="data row8 col6" >0.79714</td>
      <td id="T_b1b49_row8_col7" class="data row8 col7" >2</td>
      <td id="T_b1b49_row8_col8" class="data row8 col8" >0</td>
      <td id="T_b1b49_row8_col9" class="data row8 col9" >38.45775</td>
      <td id="T_b1b49_row8_col10" class="data row8 col10" >False</td>
      <td id="T_b1b49_row8_col11" class="data row8 col11" >1.23573</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_b1b49_row9_col0" class="data row9 col0" >12191</td>
      <td id="T_b1b49_row9_col1" class="data row9 col1" >LORE_SA</td>
      <td id="T_b1b49_row9_col2" class="data row9 col2" >LORE_SA5</td>
      <td id="T_b1b49_row9_col3" class="data row9 col3" >IF capital.gain <= 8606.7402 THEN class = <=50K</td>
      <td id="T_b1b49_row9_col4" class="data row9 col4" >0.97315</td>
      <td id="T_b1b49_row9_col5" class="data row9 col5" >0.99925</td>
      <td id="T_b1b49_row9_col6" class="data row9 col6" >0.77953</td>
      <td id="T_b1b49_row9_col7" class="data row9 col7" >1</td>
      <td id="T_b1b49_row9_col8" class="data row9 col8" >0</td>
      <td id="T_b1b49_row9_col9" class="data row9 col9" >40.04660</td>
      <td id="T_b1b49_row9_col10" class="data row9 col10" >False</td>
      <td id="T_b1b49_row9_col11" class="data row9 col11" >0.28240</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_b1b49_row10_col0" class="data row10 col0" >12191</td>
      <td id="T_b1b49_row10_col1" class="data row10 col1" >EXPLAN</td>
      <td id="T_b1b49_row10_col2" class="data row10 col2" >EXPLAN1</td>
      <td id="T_b1b49_row10_col3" class="data row10 col3" >IF hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_b1b49_row10_col4" class="data row10 col4" >0.21231</td>
      <td id="T_b1b49_row10_col5" class="data row10 col5" >0.25574</td>
      <td id="T_b1b49_row10_col6" class="data row10 col6" >0.91445</td>
      <td id="T_b1b49_row10_col7" class="data row10 col7" >1</td>
      <td id="T_b1b49_row10_col8" class="data row10 col8" >0</td>
      <td id="T_b1b49_row10_col9" class="data row10 col9" >3.96192</td>
      <td id="T_b1b49_row10_col10" class="data row10 col10" >False</td>
      <td id="T_b1b49_row10_col11" class="data row10 col11" >0.77662</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_b1b49_row11_col0" class="data row11 col0" >12191</td>
      <td id="T_b1b49_row11_col1" class="data row11 col1" >EXPLAN</td>
      <td id="T_b1b49_row11_col2" class="data row11 col2" >EXPLAN2</td>
      <td id="T_b1b49_row11_col3" class="data row11 col3" >IF age <= 31.0 AND capital.gain <= 0.0 AND native.country = United-States THEN class = <=50K</td>
      <td id="T_b1b49_row11_col4" class="data row11 col4" >0.30638</td>
      <td id="T_b1b49_row11_col5" class="data row11 col5" >0.37728</td>
      <td id="T_b1b49_row11_col6" class="data row11 col6" >0.93484</td>
      <td id="T_b1b49_row11_col7" class="data row11 col7" >3</td>
      <td id="T_b1b49_row11_col8" class="data row11 col8" >0</td>
      <td id="T_b1b49_row11_col9" class="data row11 col9" >4.10710</td>
      <td id="T_b1b49_row11_col10" class="data row11 col10" >False</td>
      <td id="T_b1b49_row11_col11" class="data row11 col11" >2.30603</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_b1b49_row12_col0" class="data row12 col0" >12191</td>
      <td id="T_b1b49_row12_col1" class="data row12 col1" >EXPLAN</td>
      <td id="T_b1b49_row12_col2" class="data row12 col2" >EXPLAN4</td>
      <td id="T_b1b49_row12_col3" class="data row12 col3" >IF age <= 29.043 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_b1b49_row12_col4" class="data row12 col4" >0.28725</td>
      <td id="T_b1b49_row12_col5" class="data row12 col5" >0.36173</td>
      <td id="T_b1b49_row12_col6" class="data row12 col6" >0.95601</td>
      <td id="T_b1b49_row12_col7" class="data row12 col7" >2</td>
      <td id="T_b1b49_row12_col8" class="data row12 col8" >0</td>
      <td id="T_b1b49_row12_col9" class="data row12 col9" >4.05909</td>
      <td id="T_b1b49_row12_col10" class="data row12 col10" >False</td>
      <td id="T_b1b49_row12_col11" class="data row12 col11" >1.37695</td>
    </tr>
    <tr>
      <th id="T_b1b49_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_b1b49_row13_col0" class="data row13 col0" >12191</td>
      <td id="T_b1b49_row13_col1" class="data row13 col1" >EXPLAN</td>
      <td id="T_b1b49_row13_col2" class="data row13 col2" >EXPLAN5</td>
      <td id="T_b1b49_row13_col3" class="data row13 col3" >IF age <= 32.5613 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_b1b49_row13_col4" class="data row13 col4" >0.36135</td>
      <td id="T_b1b49_row13_col5" class="data row13 col5" >0.44091</td>
      <td id="T_b1b49_row13_col6" class="data row13 col6" >0.92630</td>
      <td id="T_b1b49_row13_col7" class="data row13 col7" >2</td>
      <td id="T_b1b49_row13_col8" class="data row13 col8" >0</td>
      <td id="T_b1b49_row13_col9" class="data row13 col9" >3.99501</td>
      <td id="T_b1b49_row13_col10" class="data row13 col10" >False</td>
      <td id="T_b1b49_row13_col11" class="data row13 col11" >1.34245</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 12191, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99925, Pre: 0.95601), Unique rules (diffrent features)



<style type="text/css">
#T_0cd65_row2_col0, #T_0cd65_row2_col1, #T_0cd65_row2_col2, #T_0cd65_row2_col3, #T_0cd65_row2_col4, #T_0cd65_row2_col5, #T_0cd65_row2_col6, #T_0cd65_row2_col7, #T_0cd65_row2_col8, #T_0cd65_row2_col9, #T_0cd65_row2_col10, #T_0cd65_row2_col11 {
  font-weight: bold;
}
</style>
<table id="T_0cd65">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0cd65_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_0cd65_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_0cd65_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_0cd65_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_0cd65_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_0cd65_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_0cd65_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_0cd65_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_0cd65_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_0cd65_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_0cd65_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_0cd65_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0cd65_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_0cd65_row0_col0" class="data row0 col0" >12191</td>
      <td id="T_0cd65_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_0cd65_row0_col2" class="data row0 col2" >ANCHOR2</td>
      <td id="T_0cd65_row0_col3" class="data row0 col3" >IF capital.gain <= 0.0 AND capital.loss <= 0.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_0cd65_row0_col4" class="data row0 col4" >0.69287</td>
      <td id="T_0cd65_row0_col5" class="data row0 col5" >0.77489</td>
      <td id="T_0cd65_row0_col6" class="data row0 col6" >0.84904</td>
      <td id="T_0cd65_row0_col7" class="data row0 col7" >3</td>
      <td id="T_0cd65_row0_col8" class="data row0 col8" >0</td>
      <td id="T_0cd65_row0_col9" class="data row0 col9" >2.31952</td>
      <td id="T_0cd65_row0_col10" class="data row0 col10" >False</td>
      <td id="T_0cd65_row0_col11" class="data row0 col11" >2.23434</td>
    </tr>
    <tr>
      <th id="T_0cd65_level0_row1" class="row_heading level0 row1" >3</th>
      <td id="T_0cd65_row1_col0" class="data row1 col0" >12191</td>
      <td id="T_0cd65_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_0cd65_row1_col2" class="data row1 col2" >ANCHOR5</td>
      <td id="T_0cd65_row1_col3" class="data row1 col3" >IF age <= 47.0 AND capital.gain <= 0.0 THEN class = <=50K</td>
      <td id="T_0cd65_row1_col4" class="data row1 col4" >0.69595</td>
      <td id="T_0cd65_row1_col5" class="data row1 col5" >0.75357</td>
      <td id="T_0cd65_row1_col6" class="data row1 col6" >0.82203</td>
      <td id="T_0cd65_row1_col7" class="data row1 col7" >2</td>
      <td id="T_0cd65_row1_col8" class="data row1 col8" >0</td>
      <td id="T_0cd65_row1_col9" class="data row1 col9" >2.09261</td>
      <td id="T_0cd65_row1_col10" class="data row1 col10" >False</td>
      <td id="T_0cd65_row1_col11" class="data row1 col11" >1.25214</td>
    </tr>
    <tr>
      <th id="T_0cd65_level0_row2" class="row_heading level0 row2" >6</th>
      <td id="T_0cd65_row2_col0" class="data row2 col0" >12191</td>
      <td id="T_0cd65_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_0cd65_row2_col2" class="data row2 col2" >LORE5</td>
      <td id="T_0cd65_row2_col3" class="data row2 col3" >IF capital.gain <= 6229.6605 THEN class = <=50K</td>
      <td id="T_0cd65_row2_col4" class="data row2 col4" >0.95450</td>
      <td id="T_0cd65_row2_col5" class="data row2 col5" >0.99717</td>
      <td id="T_0cd65_row2_col6" class="data row2 col6" >0.79311</td>
      <td id="T_0cd65_row2_col7" class="data row2 col7" >1</td>
      <td id="T_0cd65_row2_col8" class="data row2 col8" >0</td>
      <td id="T_0cd65_row2_col9" class="data row2 col9" >329.93644</td>
      <td id="T_0cd65_row2_col10" class="data row2 col10" >False</td>
      <td id="T_0cd65_row2_col11" class="data row2 col11" >0.27413</td>
    </tr>
    <tr>
      <th id="T_0cd65_level0_row3" class="row_heading level0 row3" >8</th>
      <td id="T_0cd65_row3_col0" class="data row3 col0" >12191</td>
      <td id="T_0cd65_row3_col1" class="data row3 col1" >LORE_SA</td>
      <td id="T_0cd65_row3_col2" class="data row3 col2" >LORE_SA4</td>
      <td id="T_0cd65_row3_col3" class="data row3 col3" >IF capital.gain <= 270.5606 AND workclass != Self-emp-not-inc THEN class = <=50K</td>
      <td id="T_0cd65_row3_col4" class="data row3 col4" >0.84631</td>
      <td id="T_0cd65_row3_col5" class="data row3 col5" >0.88863</td>
      <td id="T_0cd65_row3_col6" class="data row3 col6" >0.79714</td>
      <td id="T_0cd65_row3_col7" class="data row3 col7" >2</td>
      <td id="T_0cd65_row3_col8" class="data row3 col8" >0</td>
      <td id="T_0cd65_row3_col9" class="data row3 col9" >38.45775</td>
      <td id="T_0cd65_row3_col10" class="data row3 col10" >False</td>
      <td id="T_0cd65_row3_col11" class="data row3 col11" >1.23573</td>
    </tr>
    <tr>
      <th id="T_0cd65_level0_row4" class="row_heading level0 row4" >10</th>
      <td id="T_0cd65_row4_col0" class="data row4 col0" >12191</td>
      <td id="T_0cd65_row4_col1" class="data row4 col1" >EXPLAN</td>
      <td id="T_0cd65_row4_col2" class="data row4 col2" >EXPLAN1</td>
      <td id="T_0cd65_row4_col3" class="data row4 col3" >IF hours.per.week <= 35.0 THEN class = <=50K</td>
      <td id="T_0cd65_row4_col4" class="data row4 col4" >0.21231</td>
      <td id="T_0cd65_row4_col5" class="data row4 col5" >0.25574</td>
      <td id="T_0cd65_row4_col6" class="data row4 col6" >0.91445</td>
      <td id="T_0cd65_row4_col7" class="data row4 col7" >1</td>
      <td id="T_0cd65_row4_col8" class="data row4 col8" >0</td>
      <td id="T_0cd65_row4_col9" class="data row4 col9" >3.96192</td>
      <td id="T_0cd65_row4_col10" class="data row4 col10" >False</td>
      <td id="T_0cd65_row4_col11" class="data row4 col11" >0.77662</td>
    </tr>
    <tr>
      <th id="T_0cd65_level0_row5" class="row_heading level0 row5" >11</th>
      <td id="T_0cd65_row5_col0" class="data row5 col0" >12191</td>
      <td id="T_0cd65_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_0cd65_row5_col2" class="data row5 col2" >EXPLAN2</td>
      <td id="T_0cd65_row5_col3" class="data row5 col3" >IF age <= 31.0 AND capital.gain <= 0.0 AND native.country = United-States THEN class = <=50K</td>
      <td id="T_0cd65_row5_col4" class="data row5 col4" >0.30638</td>
      <td id="T_0cd65_row5_col5" class="data row5 col5" >0.37728</td>
      <td id="T_0cd65_row5_col6" class="data row5 col6" >0.93484</td>
      <td id="T_0cd65_row5_col7" class="data row5 col7" >3</td>
      <td id="T_0cd65_row5_col8" class="data row5 col8" >0</td>
      <td id="T_0cd65_row5_col9" class="data row5 col9" >4.10710</td>
      <td id="T_0cd65_row5_col10" class="data row5 col10" >False</td>
      <td id="T_0cd65_row5_col11" class="data row5 col11" >2.30603</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_188.png)
    



## Instance 3374 (Original: <=50K , Predicted: <=50K)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>59.0</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>Private</td>
    </tr>
    <tr>
      <th>education</th>
      <td>Some-college</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>10</td>
    </tr>
    <tr>
      <th>marital.status</th>
      <td>Married-civ-spouse</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>Sales</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>Husband</td>
    </tr>
    <tr>
      <th>race</th>
      <td>White</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>4064.0</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>40.0</td>
    </tr>
    <tr>
      <th>native.country</th>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



### Rules for Instance 3374



<style type="text/css">
</style>
<table id="T_60481">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_60481_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_60481_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_60481_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_60481_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_60481_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_60481_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_60481_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_60481_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_60481_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_60481_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_60481_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_60481_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_60481_row0_col0" class="data row0 col0" >3374</td>
      <td id="T_60481_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_60481_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_60481_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 40.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_60481_row0_col4" class="data row0 col4" >0.53032</td>
      <td id="T_60481_row0_col5" class="data row0 col5" >0.60117</td>
      <td id="T_60481_row0_col6" class="data row0 col6" >0.86059</td>
      <td id="T_60481_row0_col7" class="data row0 col7" >3</td>
      <td id="T_60481_row0_col8" class="data row0 col8" >0</td>
      <td id="T_60481_row0_col9" class="data row0 col9" >6.03093</td>
      <td id="T_60481_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_60481_row1_col0" class="data row1 col0" >3374</td>
      <td id="T_60481_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_60481_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_60481_row1_col3" class="data row1 col3" >IF education = Some-college AND education.num = 10.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_60481_row1_col4" class="data row1 col4" >0.18063</td>
      <td id="T_60481_row1_col5" class="data row1 col5" >0.20164</td>
      <td id="T_60481_row1_col6" class="data row1 col6" >0.84746</td>
      <td id="T_60481_row1_col7" class="data row1 col7" >3</td>
      <td id="T_60481_row1_col8" class="data row1 col8" >0</td>
      <td id="T_60481_row1_col9" class="data row1 col9" >7.79334</td>
      <td id="T_60481_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_60481_row2_col0" class="data row2 col0" >3374</td>
      <td id="T_60481_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_60481_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_60481_row2_col3" class="data row2 col3" >IF education = Some-college AND education.num = 10.0 AND race = White THEN class = <=50K</td>
      <td id="T_60481_row2_col4" class="data row2 col4" >0.19042</td>
      <td id="T_60481_row2_col5" class="data row2 col5" >0.20008</td>
      <td id="T_60481_row2_col6" class="data row2 col6" >0.79770</td>
      <td id="T_60481_row2_col7" class="data row2 col7" >3</td>
      <td id="T_60481_row2_col8" class="data row2 col8" >0</td>
      <td id="T_60481_row2_col9" class="data row2 col9" >4.62174</td>
      <td id="T_60481_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_60481_row3_col0" class="data row3 col0" >3374</td>
      <td id="T_60481_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_60481_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_60481_row3_col3" class="data row3 col3" >IF education = Some-college AND education.num = 10.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_60481_row3_col4" class="data row3 col4" >0.17133</td>
      <td id="T_60481_row3_col5" class="data row3 col5" >0.18667</td>
      <td id="T_60481_row3_col6" class="data row3 col6" >0.82714</td>
      <td id="T_60481_row3_col7" class="data row3 col7" >3</td>
      <td id="T_60481_row3_col8" class="data row3 col8" >0</td>
      <td id="T_60481_row3_col9" class="data row3 col9" >7.23871</td>
      <td id="T_60481_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_60481_row4_col0" class="data row4 col0" >3374</td>
      <td id="T_60481_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_60481_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_60481_row4_col3" class="data row4 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_60481_row4_col4" class="data row4 col4" >0.58600</td>
      <td id="T_60481_row4_col5" class="data row4 col5" >0.65064</td>
      <td id="T_60481_row4_col6" class="data row4 col6" >0.84292</td>
      <td id="T_60481_row4_col7" class="data row4 col7" >3</td>
      <td id="T_60481_row4_col8" class="data row4 col8" >0</td>
      <td id="T_60481_row4_col9" class="data row4 col9" >7.78208</td>
      <td id="T_60481_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_60481_row5_col0" class="data row5 col0" >3374</td>
      <td id="T_60481_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_60481_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_60481_row5_col3" class="data row5 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_60481_row5_col4" class="data row5 col4" >0.94226</td>
      <td id="T_60481_row5_col5" class="data row5 col5" >0.98942</td>
      <td id="T_60481_row5_col6" class="data row5 col6" >0.79717</td>
      <td id="T_60481_row5_col7" class="data row5 col7" >1</td>
      <td id="T_60481_row5_col8" class="data row5 col8" >0</td>
      <td id="T_60481_row5_col9" class="data row5 col9" >270.15156</td>
      <td id="T_60481_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_60481_row6_col0" class="data row6 col0" >3374</td>
      <td id="T_60481_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_60481_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_60481_row6_col3" class="data row6 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_60481_row6_col4" class="data row6 col4" >0.89523</td>
      <td id="T_60481_row6_col5" class="data row6 col5" >0.95891</td>
      <td id="T_60481_row6_col6" class="data row6 col6" >0.81317</td>
      <td id="T_60481_row6_col7" class="data row6 col7" >2</td>
      <td id="T_60481_row6_col8" class="data row6 col8" >0</td>
      <td id="T_60481_row6_col9" class="data row6 col9" >110.73017</td>
      <td id="T_60481_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_60481_row7_col0" class="data row7 col0" >3374</td>
      <td id="T_60481_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_60481_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_60481_row7_col3" class="data row7 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 1628.0 AND education = Some-college AND occupation = Sales THEN class = <=50K</td>
      <td id="T_60481_row7_col4" class="data row7 col4" >0.02887</td>
      <td id="T_60481_row7_col5" class="data row7 col5" >0.03138</td>
      <td id="T_60481_row7_col6" class="data row7 col6" >0.82523</td>
      <td id="T_60481_row7_col7" class="data row7 col7" >4</td>
      <td id="T_60481_row7_col8" class="data row7 col8" >0</td>
      <td id="T_60481_row7_col9" class="data row7 col9" >226.12810</td>
      <td id="T_60481_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_60481_row8_col0" class="data row8 col0" >3374</td>
      <td id="T_60481_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_60481_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_60481_row8_col3" class="data row8 col3" >IF capital.gain <= 4064.0 AND education = Some-college AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_60481_row8_col4" class="data row8 col4" >0.16093</td>
      <td id="T_60481_row8_col5" class="data row8 col5" >0.18621</td>
      <td id="T_60481_row8_col6" class="data row8 col6" >0.87841</td>
      <td id="T_60481_row8_col7" class="data row8 col7" >3</td>
      <td id="T_60481_row8_col8" class="data row8 col8" >0</td>
      <td id="T_60481_row8_col9" class="data row8 col9" >357.85710</td>
      <td id="T_60481_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_60481_row9_col0" class="data row9 col0" >3374</td>
      <td id="T_60481_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_60481_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_60481_row9_col3" class="data row9 col3" >IF age <= 59.0 AND capital.gain <= 4064.0 AND capital.loss <= 0.0 AND hours.per.week > 37.4221 AND hours.per.week <= 42.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_60481_row9_col4" class="data row9 col4" >0.32836</td>
      <td id="T_60481_row9_col5" class="data row9 col5" >0.36843</td>
      <td id="T_60481_row9_col6" class="data row9 col6" >0.85182</td>
      <td id="T_60481_row9_col7" class="data row9 col7" >6</td>
      <td id="T_60481_row9_col8" class="data row9 col8" >0</td>
      <td id="T_60481_row9_col9" class="data row9 col9" >203.32216</td>
      <td id="T_60481_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_60481_row10_col0" class="data row10 col0" >3374</td>
      <td id="T_60481_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_60481_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_60481_row10_col3" class="data row10 col3" >IF capital.gain <= 7513.6748 AND capital.loss <= 1748.2952 AND education != Masters AND education.num != 13.0 THEN class = <=50K</td>
      <td id="T_60481_row10_col4" class="data row10 col4" >0.74245</td>
      <td id="T_60481_row10_col5" class="data row10 col5" >0.83009</td>
      <td id="T_60481_row10_col6" class="data row10 col6" >0.84878</td>
      <td id="T_60481_row10_col7" class="data row10 col7" >4</td>
      <td id="T_60481_row10_col8" class="data row10 col8" >0</td>
      <td id="T_60481_row10_col9" class="data row10 col9" >35.40669</td>
      <td id="T_60481_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_60481_row11_col0" class="data row11 col0" >3374</td>
      <td id="T_60481_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_60481_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_60481_row11_col3" class="data row11 col3" >IF age <= 62.9463 AND capital.gain <= 5132.76 AND capital.loss <= 1772.5395 AND education != Masters AND occupation = Sales AND workclass != Self-emp-inc THEN class = <=50K</td>
      <td id="T_60481_row11_col4" class="data row11 col4" >0.08736</td>
      <td id="T_60481_row11_col5" class="data row11 col5" >0.09415</td>
      <td id="T_60481_row11_col6" class="data row11 col6" >0.81818</td>
      <td id="T_60481_row11_col7" class="data row11 col7" >6</td>
      <td id="T_60481_row11_col8" class="data row11 col8" >0</td>
      <td id="T_60481_row11_col9" class="data row11 col9" >41.83923</td>
      <td id="T_60481_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_60481_row12_col0" class="data row12 col0" >3374</td>
      <td id="T_60481_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_60481_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_60481_row12_col3" class="data row12 col3" >IF capital.gain <= 10347.709 AND capital.loss <= 1466.2245 AND education = Some-college AND occupation != Exec-managerial AND occupation != Tech-support AND workclass != Federal-gov THEN class = <=50K</td>
      <td id="T_60481_row12_col4" class="data row12 col4" >0.17405</td>
      <td id="T_60481_row12_col5" class="data row12 col5" >0.19598</td>
      <td id="T_60481_row12_col6" class="data row12 col6" >0.85480</td>
      <td id="T_60481_row12_col7" class="data row12 col7" >6</td>
      <td id="T_60481_row12_col8" class="data row12 col8" >0</td>
      <td id="T_60481_row12_col9" class="data row12 col9" >36.25605</td>
      <td id="T_60481_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_60481_row13_col0" class="data row13 col0" >3374</td>
      <td id="T_60481_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_60481_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_60481_row13_col3" class="data row13 col3" >IF capital.gain <= 5055.0842 AND capital.loss <= 1917.7106 AND education = Some-college AND workclass != Local-gov THEN class = <=50K</td>
      <td id="T_60481_row13_col4" class="data row13 col4" >0.20183</td>
      <td id="T_60481_row13_col5" class="data row13 col5" >0.22279</td>
      <td id="T_60481_row13_col6" class="data row13 col6" >0.83804</td>
      <td id="T_60481_row13_col7" class="data row13 col7" >4</td>
      <td id="T_60481_row13_col8" class="data row13 col8" >0</td>
      <td id="T_60481_row13_col9" class="data row13 col9" >39.17354</td>
      <td id="T_60481_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_60481_row14_col0" class="data row14 col0" >3374</td>
      <td id="T_60481_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_60481_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_60481_row14_col3" class="data row14 col3" >IF capital.gain <= 10441.4766 AND capital.loss <= 2929.3966 AND occupation = Sales THEN class = <=50K</td>
      <td id="T_60481_row14_col4" class="data row14 col4" >0.10789</td>
      <td id="T_60481_row14_col5" class="data row14 col5" >0.10732</td>
      <td id="T_60481_row14_col6" class="data row14 col6" >0.75519</td>
      <td id="T_60481_row14_col7" class="data row14 col7" >3</td>
      <td id="T_60481_row14_col8" class="data row14 col8" >0</td>
      <td id="T_60481_row14_col9" class="data row14 col9" >47.81815</td>
      <td id="T_60481_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_60481_row15_col0" class="data row15 col0" >3374</td>
      <td id="T_60481_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_60481_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_60481_row15_col3" class="data row15 col3" >IF age > 46.0 AND capital.gain <= 5079.369 AND capital.loss <= 1855.7073 AND education = Some-college AND hours.per.week <= 40.0002 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = <=50K</td>
      <td id="T_60481_row15_col4" class="data row15 col4" >0.01518</td>
      <td id="T_60481_row15_col5" class="data row15 col5" >0.01167</td>
      <td id="T_60481_row15_col6" class="data row15 col6" >0.58382</td>
      <td id="T_60481_row15_col7" class="data row15 col7" >7</td>
      <td id="T_60481_row15_col8" class="data row15 col8" >0</td>
      <td id="T_60481_row15_col9" class="data row15 col9" >9.05432</td>
      <td id="T_60481_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_60481_row16_col0" class="data row16 col0" >3374</td>
      <td id="T_60481_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_60481_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_60481_row16_col3" class="data row16 col3" >IF capital.gain > 3325.0 AND capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_60481_row16_col4" class="data row16 col4" >0.01430</td>
      <td id="T_60481_row16_col5" class="data row16 col5" >0.01480</td>
      <td id="T_60481_row16_col6" class="data row16 col6" >0.78528</td>
      <td id="T_60481_row16_col7" class="data row16 col7" >2</td>
      <td id="T_60481_row16_col8" class="data row16 col8" >0</td>
      <td id="T_60481_row16_col9" class="data row16 col9" >1.88651</td>
      <td id="T_60481_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_60481_row17_col0" class="data row17 col0" >3374</td>
      <td id="T_60481_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_60481_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_60481_row17_col3" class="data row17 col3" >IF age > 33.0877 AND capital.gain <= 5013.0 AND education = Some-college AND education.num = 10.0 AND marital.status = Married-civ-spouse AND occupation = Sales AND race = White AND relationship = Husband THEN class = <=50K</td>
      <td id="T_60481_row17_col4" class="data row17 col4" >0.00759</td>
      <td id="T_60481_row17_col5" class="data row17 col5" >0.00503</td>
      <td id="T_60481_row17_col6" class="data row17 col6" >0.50289</td>
      <td id="T_60481_row17_col7" class="data row17 col7" >8</td>
      <td id="T_60481_row17_col8" class="data row17 col8" >0</td>
      <td id="T_60481_row17_col9" class="data row17 col9" >14.37766</td>
      <td id="T_60481_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_60481_row18_col0" class="data row18 col0" >3374</td>
      <td id="T_60481_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_60481_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_60481_row18_col3" class="data row18 col3" >IF capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_60481_row18_col4" class="data row18 col4" >0.95082</td>
      <td id="T_60481_row18_col5" class="data row18 col5" >0.99665</td>
      <td id="T_60481_row18_col6" class="data row18 col6" >0.79576</td>
      <td id="T_60481_row18_col7" class="data row18 col7" >1</td>
      <td id="T_60481_row18_col8" class="data row18 col8" >0</td>
      <td id="T_60481_row18_col9" class="data row18 col9" >1.82929</td>
      <td id="T_60481_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_60481_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_60481_row19_col0" class="data row19 col0" >3374</td>
      <td id="T_60481_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_60481_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_60481_row19_col3" class="data row19 col3" >IF capital.gain > 3908.0 AND capital.gain <= 6418.0 AND capital.loss <= 896.9302 AND marital.status = Married-civ-spouse AND occupation = Sales AND relationship = Husband THEN class = <=50K</td>
      <td id="T_60481_row19_col4" class="data row19 col4" >0.00132</td>
      <td id="T_60481_row19_col5" class="data row19 col5" >0.00087</td>
      <td id="T_60481_row19_col6" class="data row19 col6" >0.50000</td>
      <td id="T_60481_row19_col7" class="data row19 col7" >6</td>
      <td id="T_60481_row19_col8" class="data row19 col8" >0</td>
      <td id="T_60481_row19_col9" class="data row19 col9" >13.75265</td>
      <td id="T_60481_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 3374, Correct Prediction



<style type="text/css">
</style>
<table id="T_6e666">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6e666_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_6e666_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_6e666_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_6e666_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_6e666_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_6e666_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_6e666_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_6e666_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_6e666_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_6e666_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_6e666_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6e666_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_6e666_row0_col0" class="data row0 col0" >3374</td>
      <td id="T_6e666_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_6e666_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_6e666_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 40.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_6e666_row0_col4" class="data row0 col4" >0.53032</td>
      <td id="T_6e666_row0_col5" class="data row0 col5" >0.60117</td>
      <td id="T_6e666_row0_col6" class="data row0 col6" >0.86059</td>
      <td id="T_6e666_row0_col7" class="data row0 col7" >3</td>
      <td id="T_6e666_row0_col8" class="data row0 col8" >0</td>
      <td id="T_6e666_row0_col9" class="data row0 col9" >6.03093</td>
      <td id="T_6e666_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_6e666_row1_col0" class="data row1 col0" >3374</td>
      <td id="T_6e666_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_6e666_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_6e666_row1_col3" class="data row1 col3" >IF education = Some-college AND education.num = 10.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_6e666_row1_col4" class="data row1 col4" >0.18063</td>
      <td id="T_6e666_row1_col5" class="data row1 col5" >0.20164</td>
      <td id="T_6e666_row1_col6" class="data row1 col6" >0.84746</td>
      <td id="T_6e666_row1_col7" class="data row1 col7" >3</td>
      <td id="T_6e666_row1_col8" class="data row1 col8" >0</td>
      <td id="T_6e666_row1_col9" class="data row1 col9" >7.79334</td>
      <td id="T_6e666_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_6e666_row2_col0" class="data row2 col0" >3374</td>
      <td id="T_6e666_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_6e666_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_6e666_row2_col3" class="data row2 col3" >IF education = Some-college AND education.num = 10.0 AND race = White THEN class = <=50K</td>
      <td id="T_6e666_row2_col4" class="data row2 col4" >0.19042</td>
      <td id="T_6e666_row2_col5" class="data row2 col5" >0.20008</td>
      <td id="T_6e666_row2_col6" class="data row2 col6" >0.79770</td>
      <td id="T_6e666_row2_col7" class="data row2 col7" >3</td>
      <td id="T_6e666_row2_col8" class="data row2 col8" >0</td>
      <td id="T_6e666_row2_col9" class="data row2 col9" >4.62174</td>
      <td id="T_6e666_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_6e666_row3_col0" class="data row3 col0" >3374</td>
      <td id="T_6e666_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_6e666_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_6e666_row3_col3" class="data row3 col3" >IF education = Some-college AND education.num = 10.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_6e666_row3_col4" class="data row3 col4" >0.17133</td>
      <td id="T_6e666_row3_col5" class="data row3 col5" >0.18667</td>
      <td id="T_6e666_row3_col6" class="data row3 col6" >0.82714</td>
      <td id="T_6e666_row3_col7" class="data row3 col7" >3</td>
      <td id="T_6e666_row3_col8" class="data row3 col8" >0</td>
      <td id="T_6e666_row3_col9" class="data row3 col9" >7.23871</td>
      <td id="T_6e666_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_6e666_row4_col0" class="data row4 col0" >3374</td>
      <td id="T_6e666_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_6e666_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_6e666_row4_col3" class="data row4 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_6e666_row4_col4" class="data row4 col4" >0.58600</td>
      <td id="T_6e666_row4_col5" class="data row4 col5" >0.65064</td>
      <td id="T_6e666_row4_col6" class="data row4 col6" >0.84292</td>
      <td id="T_6e666_row4_col7" class="data row4 col7" >3</td>
      <td id="T_6e666_row4_col8" class="data row4 col8" >0</td>
      <td id="T_6e666_row4_col9" class="data row4 col9" >7.78208</td>
      <td id="T_6e666_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_6e666_row5_col0" class="data row5 col0" >3374</td>
      <td id="T_6e666_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_6e666_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_6e666_row5_col3" class="data row5 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_6e666_row5_col4" class="data row5 col4" >0.94226</td>
      <td id="T_6e666_row5_col5" class="data row5 col5" >0.98942</td>
      <td id="T_6e666_row5_col6" class="data row5 col6" >0.79717</td>
      <td id="T_6e666_row5_col7" class="data row5 col7" >1</td>
      <td id="T_6e666_row5_col8" class="data row5 col8" >0</td>
      <td id="T_6e666_row5_col9" class="data row5 col9" >270.15156</td>
      <td id="T_6e666_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_6e666_row6_col0" class="data row6 col0" >3374</td>
      <td id="T_6e666_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_6e666_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_6e666_row6_col3" class="data row6 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_6e666_row6_col4" class="data row6 col4" >0.89523</td>
      <td id="T_6e666_row6_col5" class="data row6 col5" >0.95891</td>
      <td id="T_6e666_row6_col6" class="data row6 col6" >0.81317</td>
      <td id="T_6e666_row6_col7" class="data row6 col7" >2</td>
      <td id="T_6e666_row6_col8" class="data row6 col8" >0</td>
      <td id="T_6e666_row6_col9" class="data row6 col9" >110.73017</td>
      <td id="T_6e666_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_6e666_row7_col0" class="data row7 col0" >3374</td>
      <td id="T_6e666_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_6e666_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_6e666_row7_col3" class="data row7 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 1628.0 AND education = Some-college AND occupation = Sales THEN class = <=50K</td>
      <td id="T_6e666_row7_col4" class="data row7 col4" >0.02887</td>
      <td id="T_6e666_row7_col5" class="data row7 col5" >0.03138</td>
      <td id="T_6e666_row7_col6" class="data row7 col6" >0.82523</td>
      <td id="T_6e666_row7_col7" class="data row7 col7" >4</td>
      <td id="T_6e666_row7_col8" class="data row7 col8" >0</td>
      <td id="T_6e666_row7_col9" class="data row7 col9" >226.12810</td>
      <td id="T_6e666_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_6e666_row8_col0" class="data row8 col0" >3374</td>
      <td id="T_6e666_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_6e666_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_6e666_row8_col3" class="data row8 col3" >IF capital.gain <= 4064.0 AND education = Some-college AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_6e666_row8_col4" class="data row8 col4" >0.16093</td>
      <td id="T_6e666_row8_col5" class="data row8 col5" >0.18621</td>
      <td id="T_6e666_row8_col6" class="data row8 col6" >0.87841</td>
      <td id="T_6e666_row8_col7" class="data row8 col7" >3</td>
      <td id="T_6e666_row8_col8" class="data row8 col8" >0</td>
      <td id="T_6e666_row8_col9" class="data row8 col9" >357.85710</td>
      <td id="T_6e666_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_6e666_row9_col0" class="data row9 col0" >3374</td>
      <td id="T_6e666_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_6e666_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_6e666_row9_col3" class="data row9 col3" >IF age <= 59.0 AND capital.gain <= 4064.0 AND capital.loss <= 0.0 AND hours.per.week > 37.4221 AND hours.per.week <= 42.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_6e666_row9_col4" class="data row9 col4" >0.32836</td>
      <td id="T_6e666_row9_col5" class="data row9 col5" >0.36843</td>
      <td id="T_6e666_row9_col6" class="data row9 col6" >0.85182</td>
      <td id="T_6e666_row9_col7" class="data row9 col7" >6</td>
      <td id="T_6e666_row9_col8" class="data row9 col8" >0</td>
      <td id="T_6e666_row9_col9" class="data row9 col9" >203.32216</td>
      <td id="T_6e666_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_6e666_row10_col0" class="data row10 col0" >3374</td>
      <td id="T_6e666_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_6e666_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_6e666_row10_col3" class="data row10 col3" >IF capital.gain <= 7513.6748 AND capital.loss <= 1748.2952 AND education != Masters AND education.num != 13.0 THEN class = <=50K</td>
      <td id="T_6e666_row10_col4" class="data row10 col4" >0.74245</td>
      <td id="T_6e666_row10_col5" class="data row10 col5" >0.83009</td>
      <td id="T_6e666_row10_col6" class="data row10 col6" >0.84878</td>
      <td id="T_6e666_row10_col7" class="data row10 col7" >4</td>
      <td id="T_6e666_row10_col8" class="data row10 col8" >0</td>
      <td id="T_6e666_row10_col9" class="data row10 col9" >35.40669</td>
      <td id="T_6e666_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_6e666_row11_col0" class="data row11 col0" >3374</td>
      <td id="T_6e666_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_6e666_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_6e666_row11_col3" class="data row11 col3" >IF age <= 62.9463 AND capital.gain <= 5132.76 AND capital.loss <= 1772.5395 AND education != Masters AND occupation = Sales AND workclass != Self-emp-inc THEN class = <=50K</td>
      <td id="T_6e666_row11_col4" class="data row11 col4" >0.08736</td>
      <td id="T_6e666_row11_col5" class="data row11 col5" >0.09415</td>
      <td id="T_6e666_row11_col6" class="data row11 col6" >0.81818</td>
      <td id="T_6e666_row11_col7" class="data row11 col7" >6</td>
      <td id="T_6e666_row11_col8" class="data row11 col8" >0</td>
      <td id="T_6e666_row11_col9" class="data row11 col9" >41.83923</td>
      <td id="T_6e666_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_6e666_row12_col0" class="data row12 col0" >3374</td>
      <td id="T_6e666_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_6e666_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_6e666_row12_col3" class="data row12 col3" >IF capital.gain <= 10347.709 AND capital.loss <= 1466.2245 AND education = Some-college AND occupation != Exec-managerial AND occupation != Tech-support AND workclass != Federal-gov THEN class = <=50K</td>
      <td id="T_6e666_row12_col4" class="data row12 col4" >0.17405</td>
      <td id="T_6e666_row12_col5" class="data row12 col5" >0.19598</td>
      <td id="T_6e666_row12_col6" class="data row12 col6" >0.85480</td>
      <td id="T_6e666_row12_col7" class="data row12 col7" >6</td>
      <td id="T_6e666_row12_col8" class="data row12 col8" >0</td>
      <td id="T_6e666_row12_col9" class="data row12 col9" >36.25605</td>
      <td id="T_6e666_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_6e666_row13_col0" class="data row13 col0" >3374</td>
      <td id="T_6e666_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_6e666_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_6e666_row13_col3" class="data row13 col3" >IF capital.gain <= 5055.0842 AND capital.loss <= 1917.7106 AND education = Some-college AND workclass != Local-gov THEN class = <=50K</td>
      <td id="T_6e666_row13_col4" class="data row13 col4" >0.20183</td>
      <td id="T_6e666_row13_col5" class="data row13 col5" >0.22279</td>
      <td id="T_6e666_row13_col6" class="data row13 col6" >0.83804</td>
      <td id="T_6e666_row13_col7" class="data row13 col7" >4</td>
      <td id="T_6e666_row13_col8" class="data row13 col8" >0</td>
      <td id="T_6e666_row13_col9" class="data row13 col9" >39.17354</td>
      <td id="T_6e666_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_6e666_row14_col0" class="data row14 col0" >3374</td>
      <td id="T_6e666_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_6e666_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_6e666_row14_col3" class="data row14 col3" >IF capital.gain <= 10441.4766 AND capital.loss <= 2929.3966 AND occupation = Sales THEN class = <=50K</td>
      <td id="T_6e666_row14_col4" class="data row14 col4" >0.10789</td>
      <td id="T_6e666_row14_col5" class="data row14 col5" >0.10732</td>
      <td id="T_6e666_row14_col6" class="data row14 col6" >0.75519</td>
      <td id="T_6e666_row14_col7" class="data row14 col7" >3</td>
      <td id="T_6e666_row14_col8" class="data row14 col8" >0</td>
      <td id="T_6e666_row14_col9" class="data row14 col9" >47.81815</td>
      <td id="T_6e666_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_6e666_row15_col0" class="data row15 col0" >3374</td>
      <td id="T_6e666_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_6e666_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_6e666_row15_col3" class="data row15 col3" >IF age > 46.0 AND capital.gain <= 5079.369 AND capital.loss <= 1855.7073 AND education = Some-college AND hours.per.week <= 40.0002 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = <=50K</td>
      <td id="T_6e666_row15_col4" class="data row15 col4" >0.01518</td>
      <td id="T_6e666_row15_col5" class="data row15 col5" >0.01167</td>
      <td id="T_6e666_row15_col6" class="data row15 col6" >0.58382</td>
      <td id="T_6e666_row15_col7" class="data row15 col7" >7</td>
      <td id="T_6e666_row15_col8" class="data row15 col8" >0</td>
      <td id="T_6e666_row15_col9" class="data row15 col9" >9.05432</td>
      <td id="T_6e666_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_6e666_row16_col0" class="data row16 col0" >3374</td>
      <td id="T_6e666_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_6e666_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_6e666_row16_col3" class="data row16 col3" >IF capital.gain > 3325.0 AND capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_6e666_row16_col4" class="data row16 col4" >0.01430</td>
      <td id="T_6e666_row16_col5" class="data row16 col5" >0.01480</td>
      <td id="T_6e666_row16_col6" class="data row16 col6" >0.78528</td>
      <td id="T_6e666_row16_col7" class="data row16 col7" >2</td>
      <td id="T_6e666_row16_col8" class="data row16 col8" >0</td>
      <td id="T_6e666_row16_col9" class="data row16 col9" >1.88651</td>
      <td id="T_6e666_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_6e666_row17_col0" class="data row17 col0" >3374</td>
      <td id="T_6e666_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_6e666_row17_col2" class="data row17 col2" >EXPLAN3</td>
      <td id="T_6e666_row17_col3" class="data row17 col3" >IF age > 33.0877 AND capital.gain <= 5013.0 AND education = Some-college AND education.num = 10.0 AND marital.status = Married-civ-spouse AND occupation = Sales AND race = White AND relationship = Husband THEN class = <=50K</td>
      <td id="T_6e666_row17_col4" class="data row17 col4" >0.00759</td>
      <td id="T_6e666_row17_col5" class="data row17 col5" >0.00503</td>
      <td id="T_6e666_row17_col6" class="data row17 col6" >0.50289</td>
      <td id="T_6e666_row17_col7" class="data row17 col7" >8</td>
      <td id="T_6e666_row17_col8" class="data row17 col8" >0</td>
      <td id="T_6e666_row17_col9" class="data row17 col9" >14.37766</td>
      <td id="T_6e666_row17_col10" class="data row17 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_6e666_row18_col0" class="data row18 col0" >3374</td>
      <td id="T_6e666_row18_col1" class="data row18 col1" >EXPLAN</td>
      <td id="T_6e666_row18_col2" class="data row18 col2" >EXPLAN4</td>
      <td id="T_6e666_row18_col3" class="data row18 col3" >IF capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_6e666_row18_col4" class="data row18 col4" >0.95082</td>
      <td id="T_6e666_row18_col5" class="data row18 col5" >0.99665</td>
      <td id="T_6e666_row18_col6" class="data row18 col6" >0.79576</td>
      <td id="T_6e666_row18_col7" class="data row18 col7" >1</td>
      <td id="T_6e666_row18_col8" class="data row18 col8" >0</td>
      <td id="T_6e666_row18_col9" class="data row18 col9" >1.82929</td>
      <td id="T_6e666_row18_col10" class="data row18 col10" >False</td>
    </tr>
    <tr>
      <th id="T_6e666_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_6e666_row19_col0" class="data row19 col0" >3374</td>
      <td id="T_6e666_row19_col1" class="data row19 col1" >EXPLAN</td>
      <td id="T_6e666_row19_col2" class="data row19 col2" >EXPLAN5</td>
      <td id="T_6e666_row19_col3" class="data row19 col3" >IF capital.gain > 3908.0 AND capital.gain <= 6418.0 AND capital.loss <= 896.9302 AND marital.status = Married-civ-spouse AND occupation = Sales AND relationship = Husband THEN class = <=50K</td>
      <td id="T_6e666_row19_col4" class="data row19 col4" >0.00132</td>
      <td id="T_6e666_row19_col5" class="data row19 col5" >0.00087</td>
      <td id="T_6e666_row19_col6" class="data row19 col6" >0.50000</td>
      <td id="T_6e666_row19_col7" class="data row19 col7" >6</td>
      <td id="T_6e666_row19_col8" class="data row19 col8" >0</td>
      <td id="T_6e666_row19_col9" class="data row19 col9" >13.75265</td>
      <td id="T_6e666_row19_col10" class="data row19 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 3374, Min_treshold (Cov 0.01, Cov_class 0.01, Pre 0.01)



<style type="text/css">
</style>
<table id="T_e12be">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e12be_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e12be_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e12be_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e12be_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e12be_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e12be_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e12be_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e12be_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e12be_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e12be_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e12be_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e12be_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e12be_row0_col0" class="data row0 col0" >3374</td>
      <td id="T_e12be_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_e12be_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_e12be_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 40.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e12be_row0_col4" class="data row0 col4" >0.53032</td>
      <td id="T_e12be_row0_col5" class="data row0 col5" >0.60117</td>
      <td id="T_e12be_row0_col6" class="data row0 col6" >0.86059</td>
      <td id="T_e12be_row0_col7" class="data row0 col7" >3</td>
      <td id="T_e12be_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e12be_row0_col9" class="data row0 col9" >6.03093</td>
      <td id="T_e12be_row0_col10" class="data row0 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e12be_row1_col0" class="data row1 col0" >3374</td>
      <td id="T_e12be_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_e12be_row1_col2" class="data row1 col2" >ANCHOR2</td>
      <td id="T_e12be_row1_col3" class="data row1 col3" >IF education = Some-college AND education.num = 10.0 AND hours.per.week <= 45.0 THEN class = <=50K</td>
      <td id="T_e12be_row1_col4" class="data row1 col4" >0.18063</td>
      <td id="T_e12be_row1_col5" class="data row1 col5" >0.20164</td>
      <td id="T_e12be_row1_col6" class="data row1 col6" >0.84746</td>
      <td id="T_e12be_row1_col7" class="data row1 col7" >3</td>
      <td id="T_e12be_row1_col8" class="data row1 col8" >0</td>
      <td id="T_e12be_row1_col9" class="data row1 col9" >7.79334</td>
      <td id="T_e12be_row1_col10" class="data row1 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e12be_row2_col0" class="data row2 col0" >3374</td>
      <td id="T_e12be_row2_col1" class="data row2 col1" >ANCHOR</td>
      <td id="T_e12be_row2_col2" class="data row2 col2" >ANCHOR3</td>
      <td id="T_e12be_row2_col3" class="data row2 col3" >IF education = Some-college AND education.num = 10.0 AND race = White THEN class = <=50K</td>
      <td id="T_e12be_row2_col4" class="data row2 col4" >0.19042</td>
      <td id="T_e12be_row2_col5" class="data row2 col5" >0.20008</td>
      <td id="T_e12be_row2_col6" class="data row2 col6" >0.79770</td>
      <td id="T_e12be_row2_col7" class="data row2 col7" >3</td>
      <td id="T_e12be_row2_col8" class="data row2 col8" >0</td>
      <td id="T_e12be_row2_col9" class="data row2 col9" >4.62174</td>
      <td id="T_e12be_row2_col10" class="data row2 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e12be_row3_col0" class="data row3 col0" >3374</td>
      <td id="T_e12be_row3_col1" class="data row3 col1" >ANCHOR</td>
      <td id="T_e12be_row3_col2" class="data row3 col2" >ANCHOR4</td>
      <td id="T_e12be_row3_col3" class="data row3 col3" >IF education = Some-college AND education.num = 10.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e12be_row3_col4" class="data row3 col4" >0.17133</td>
      <td id="T_e12be_row3_col5" class="data row3 col5" >0.18667</td>
      <td id="T_e12be_row3_col6" class="data row3 col6" >0.82714</td>
      <td id="T_e12be_row3_col7" class="data row3 col7" >3</td>
      <td id="T_e12be_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e12be_row3_col9" class="data row3 col9" >7.23871</td>
      <td id="T_e12be_row3_col10" class="data row3 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e12be_row4_col0" class="data row4 col0" >3374</td>
      <td id="T_e12be_row4_col1" class="data row4 col1" >ANCHOR</td>
      <td id="T_e12be_row4_col2" class="data row4 col2" >ANCHOR5</td>
      <td id="T_e12be_row4_col3" class="data row4 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e12be_row4_col4" class="data row4 col4" >0.58600</td>
      <td id="T_e12be_row4_col5" class="data row4 col5" >0.65064</td>
      <td id="T_e12be_row4_col6" class="data row4 col6" >0.84292</td>
      <td id="T_e12be_row4_col7" class="data row4 col7" >3</td>
      <td id="T_e12be_row4_col8" class="data row4 col8" >0</td>
      <td id="T_e12be_row4_col9" class="data row4 col9" >7.78208</td>
      <td id="T_e12be_row4_col10" class="data row4 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e12be_row5_col0" class="data row5 col0" >3374</td>
      <td id="T_e12be_row5_col1" class="data row5 col1" >LORE</td>
      <td id="T_e12be_row5_col2" class="data row5 col2" >LORE1</td>
      <td id="T_e12be_row5_col3" class="data row5 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_e12be_row5_col4" class="data row5 col4" >0.94226</td>
      <td id="T_e12be_row5_col5" class="data row5 col5" >0.98942</td>
      <td id="T_e12be_row5_col6" class="data row5 col6" >0.79717</td>
      <td id="T_e12be_row5_col7" class="data row5 col7" >1</td>
      <td id="T_e12be_row5_col8" class="data row5 col8" >0</td>
      <td id="T_e12be_row5_col9" class="data row5 col9" >270.15156</td>
      <td id="T_e12be_row5_col10" class="data row5 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_e12be_row6_col0" class="data row6 col0" >3374</td>
      <td id="T_e12be_row6_col1" class="data row6 col1" >LORE</td>
      <td id="T_e12be_row6_col2" class="data row6 col2" >LORE2</td>
      <td id="T_e12be_row6_col3" class="data row6 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_e12be_row6_col4" class="data row6 col4" >0.89523</td>
      <td id="T_e12be_row6_col5" class="data row6 col5" >0.95891</td>
      <td id="T_e12be_row6_col6" class="data row6 col6" >0.81317</td>
      <td id="T_e12be_row6_col7" class="data row6 col7" >2</td>
      <td id="T_e12be_row6_col8" class="data row6 col8" >0</td>
      <td id="T_e12be_row6_col9" class="data row6 col9" >110.73017</td>
      <td id="T_e12be_row6_col10" class="data row6 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_e12be_row7_col0" class="data row7 col0" >3374</td>
      <td id="T_e12be_row7_col1" class="data row7 col1" >LORE</td>
      <td id="T_e12be_row7_col2" class="data row7 col2" >LORE3</td>
      <td id="T_e12be_row7_col3" class="data row7 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 1628.0 AND education = Some-college AND occupation = Sales THEN class = <=50K</td>
      <td id="T_e12be_row7_col4" class="data row7 col4" >0.02887</td>
      <td id="T_e12be_row7_col5" class="data row7 col5" >0.03138</td>
      <td id="T_e12be_row7_col6" class="data row7 col6" >0.82523</td>
      <td id="T_e12be_row7_col7" class="data row7 col7" >4</td>
      <td id="T_e12be_row7_col8" class="data row7 col8" >0</td>
      <td id="T_e12be_row7_col9" class="data row7 col9" >226.12810</td>
      <td id="T_e12be_row7_col10" class="data row7 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_e12be_row8_col0" class="data row8 col0" >3374</td>
      <td id="T_e12be_row8_col1" class="data row8 col1" >LORE</td>
      <td id="T_e12be_row8_col2" class="data row8 col2" >LORE4</td>
      <td id="T_e12be_row8_col3" class="data row8 col3" >IF capital.gain <= 4064.0 AND education = Some-college AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_e12be_row8_col4" class="data row8 col4" >0.16093</td>
      <td id="T_e12be_row8_col5" class="data row8 col5" >0.18621</td>
      <td id="T_e12be_row8_col6" class="data row8 col6" >0.87841</td>
      <td id="T_e12be_row8_col7" class="data row8 col7" >3</td>
      <td id="T_e12be_row8_col8" class="data row8 col8" >0</td>
      <td id="T_e12be_row8_col9" class="data row8 col9" >357.85710</td>
      <td id="T_e12be_row8_col10" class="data row8 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_e12be_row9_col0" class="data row9 col0" >3374</td>
      <td id="T_e12be_row9_col1" class="data row9 col1" >LORE</td>
      <td id="T_e12be_row9_col2" class="data row9 col2" >LORE5</td>
      <td id="T_e12be_row9_col3" class="data row9 col3" >IF age <= 59.0 AND capital.gain <= 4064.0 AND capital.loss <= 0.0 AND hours.per.week > 37.4221 AND hours.per.week <= 42.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e12be_row9_col4" class="data row9 col4" >0.32836</td>
      <td id="T_e12be_row9_col5" class="data row9 col5" >0.36843</td>
      <td id="T_e12be_row9_col6" class="data row9 col6" >0.85182</td>
      <td id="T_e12be_row9_col7" class="data row9 col7" >6</td>
      <td id="T_e12be_row9_col8" class="data row9 col8" >0</td>
      <td id="T_e12be_row9_col9" class="data row9 col9" >203.32216</td>
      <td id="T_e12be_row9_col10" class="data row9 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_e12be_row10_col0" class="data row10 col0" >3374</td>
      <td id="T_e12be_row10_col1" class="data row10 col1" >LORE_SA</td>
      <td id="T_e12be_row10_col2" class="data row10 col2" >LORE_SA1</td>
      <td id="T_e12be_row10_col3" class="data row10 col3" >IF capital.gain <= 7513.6748 AND capital.loss <= 1748.2952 AND education != Masters AND education.num != 13.0 THEN class = <=50K</td>
      <td id="T_e12be_row10_col4" class="data row10 col4" >0.74245</td>
      <td id="T_e12be_row10_col5" class="data row10 col5" >0.83009</td>
      <td id="T_e12be_row10_col6" class="data row10 col6" >0.84878</td>
      <td id="T_e12be_row10_col7" class="data row10 col7" >4</td>
      <td id="T_e12be_row10_col8" class="data row10 col8" >0</td>
      <td id="T_e12be_row10_col9" class="data row10 col9" >35.40669</td>
      <td id="T_e12be_row10_col10" class="data row10 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_e12be_row11_col0" class="data row11 col0" >3374</td>
      <td id="T_e12be_row11_col1" class="data row11 col1" >LORE_SA</td>
      <td id="T_e12be_row11_col2" class="data row11 col2" >LORE_SA2</td>
      <td id="T_e12be_row11_col3" class="data row11 col3" >IF age <= 62.9463 AND capital.gain <= 5132.76 AND capital.loss <= 1772.5395 AND education != Masters AND occupation = Sales AND workclass != Self-emp-inc THEN class = <=50K</td>
      <td id="T_e12be_row11_col4" class="data row11 col4" >0.08736</td>
      <td id="T_e12be_row11_col5" class="data row11 col5" >0.09415</td>
      <td id="T_e12be_row11_col6" class="data row11 col6" >0.81818</td>
      <td id="T_e12be_row11_col7" class="data row11 col7" >6</td>
      <td id="T_e12be_row11_col8" class="data row11 col8" >0</td>
      <td id="T_e12be_row11_col9" class="data row11 col9" >41.83923</td>
      <td id="T_e12be_row11_col10" class="data row11 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_e12be_row12_col0" class="data row12 col0" >3374</td>
      <td id="T_e12be_row12_col1" class="data row12 col1" >LORE_SA</td>
      <td id="T_e12be_row12_col2" class="data row12 col2" >LORE_SA3</td>
      <td id="T_e12be_row12_col3" class="data row12 col3" >IF capital.gain <= 10347.709 AND capital.loss <= 1466.2245 AND education = Some-college AND occupation != Exec-managerial AND occupation != Tech-support AND workclass != Federal-gov THEN class = <=50K</td>
      <td id="T_e12be_row12_col4" class="data row12 col4" >0.17405</td>
      <td id="T_e12be_row12_col5" class="data row12 col5" >0.19598</td>
      <td id="T_e12be_row12_col6" class="data row12 col6" >0.85480</td>
      <td id="T_e12be_row12_col7" class="data row12 col7" >6</td>
      <td id="T_e12be_row12_col8" class="data row12 col8" >0</td>
      <td id="T_e12be_row12_col9" class="data row12 col9" >36.25605</td>
      <td id="T_e12be_row12_col10" class="data row12 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_e12be_row13_col0" class="data row13 col0" >3374</td>
      <td id="T_e12be_row13_col1" class="data row13 col1" >LORE_SA</td>
      <td id="T_e12be_row13_col2" class="data row13 col2" >LORE_SA4</td>
      <td id="T_e12be_row13_col3" class="data row13 col3" >IF capital.gain <= 5055.0842 AND capital.loss <= 1917.7106 AND education = Some-college AND workclass != Local-gov THEN class = <=50K</td>
      <td id="T_e12be_row13_col4" class="data row13 col4" >0.20183</td>
      <td id="T_e12be_row13_col5" class="data row13 col5" >0.22279</td>
      <td id="T_e12be_row13_col6" class="data row13 col6" >0.83804</td>
      <td id="T_e12be_row13_col7" class="data row13 col7" >4</td>
      <td id="T_e12be_row13_col8" class="data row13 col8" >0</td>
      <td id="T_e12be_row13_col9" class="data row13 col9" >39.17354</td>
      <td id="T_e12be_row13_col10" class="data row13 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_e12be_row14_col0" class="data row14 col0" >3374</td>
      <td id="T_e12be_row14_col1" class="data row14 col1" >LORE_SA</td>
      <td id="T_e12be_row14_col2" class="data row14 col2" >LORE_SA5</td>
      <td id="T_e12be_row14_col3" class="data row14 col3" >IF capital.gain <= 10441.4766 AND capital.loss <= 2929.3966 AND occupation = Sales THEN class = <=50K</td>
      <td id="T_e12be_row14_col4" class="data row14 col4" >0.10789</td>
      <td id="T_e12be_row14_col5" class="data row14 col5" >0.10732</td>
      <td id="T_e12be_row14_col6" class="data row14 col6" >0.75519</td>
      <td id="T_e12be_row14_col7" class="data row14 col7" >3</td>
      <td id="T_e12be_row14_col8" class="data row14 col8" >0</td>
      <td id="T_e12be_row14_col9" class="data row14 col9" >47.81815</td>
      <td id="T_e12be_row14_col10" class="data row14 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_e12be_row15_col0" class="data row15 col0" >3374</td>
      <td id="T_e12be_row15_col1" class="data row15 col1" >EXPLAN</td>
      <td id="T_e12be_row15_col2" class="data row15 col2" >EXPLAN1</td>
      <td id="T_e12be_row15_col3" class="data row15 col3" >IF age > 46.0 AND capital.gain <= 5079.369 AND capital.loss <= 1855.7073 AND education = Some-college AND hours.per.week <= 40.0002 AND marital.status = Married-civ-spouse AND relationship = Husband THEN class = <=50K</td>
      <td id="T_e12be_row15_col4" class="data row15 col4" >0.01518</td>
      <td id="T_e12be_row15_col5" class="data row15 col5" >0.01167</td>
      <td id="T_e12be_row15_col6" class="data row15 col6" >0.58382</td>
      <td id="T_e12be_row15_col7" class="data row15 col7" >7</td>
      <td id="T_e12be_row15_col8" class="data row15 col8" >0</td>
      <td id="T_e12be_row15_col9" class="data row15 col9" >9.05432</td>
      <td id="T_e12be_row15_col10" class="data row15 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_e12be_row16_col0" class="data row16 col0" >3374</td>
      <td id="T_e12be_row16_col1" class="data row16 col1" >EXPLAN</td>
      <td id="T_e12be_row16_col2" class="data row16 col2" >EXPLAN2</td>
      <td id="T_e12be_row16_col3" class="data row16 col3" >IF capital.gain > 3325.0 AND capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_e12be_row16_col4" class="data row16 col4" >0.01430</td>
      <td id="T_e12be_row16_col5" class="data row16 col5" >0.01480</td>
      <td id="T_e12be_row16_col6" class="data row16 col6" >0.78528</td>
      <td id="T_e12be_row16_col7" class="data row16 col7" >2</td>
      <td id="T_e12be_row16_col8" class="data row16 col8" >0</td>
      <td id="T_e12be_row16_col9" class="data row16 col9" >1.88651</td>
      <td id="T_e12be_row16_col10" class="data row16 col10" >False</td>
    </tr>
    <tr>
      <th id="T_e12be_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_e12be_row17_col0" class="data row17 col0" >3374</td>
      <td id="T_e12be_row17_col1" class="data row17 col1" >EXPLAN</td>
      <td id="T_e12be_row17_col2" class="data row17 col2" >EXPLAN4</td>
      <td id="T_e12be_row17_col3" class="data row17 col3" >IF capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_e12be_row17_col4" class="data row17 col4" >0.95082</td>
      <td id="T_e12be_row17_col5" class="data row17 col5" >0.99665</td>
      <td id="T_e12be_row17_col6" class="data row17 col6" >0.79576</td>
      <td id="T_e12be_row17_col7" class="data row17 col7" >1</td>
      <td id="T_e12be_row17_col8" class="data row17 col8" >0</td>
      <td id="T_e12be_row17_col9" class="data row17 col9" >1.82929</td>
      <td id="T_e12be_row17_col10" class="data row17 col10" >False</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 3374, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.95082, Pre: 0.87841)



<style type="text/css">
#T_45212_row1_col0, #T_45212_row1_col1, #T_45212_row1_col2, #T_45212_row1_col3, #T_45212_row1_col4, #T_45212_row1_col5, #T_45212_row1_col6, #T_45212_row1_col7, #T_45212_row1_col8, #T_45212_row1_col9, #T_45212_row1_col10, #T_45212_row1_col11 {
  font-weight: bold;
}
</style>
<table id="T_45212">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_45212_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_45212_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_45212_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_45212_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_45212_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_45212_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_45212_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_45212_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_45212_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_45212_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_45212_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_45212_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_45212_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_45212_row0_col0" class="data row0 col0" >3374</td>
      <td id="T_45212_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_45212_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_45212_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 40.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_45212_row0_col4" class="data row0 col4" >0.53032</td>
      <td id="T_45212_row0_col5" class="data row0 col5" >0.60117</td>
      <td id="T_45212_row0_col6" class="data row0 col6" >0.86059</td>
      <td id="T_45212_row0_col7" class="data row0 col7" >3</td>
      <td id="T_45212_row0_col8" class="data row0 col8" >0</td>
      <td id="T_45212_row0_col9" class="data row0 col9" >6.03093</td>
      <td id="T_45212_row0_col10" class="data row0 col10" >False</td>
      <td id="T_45212_row0_col11" class="data row0 col11" >0.42088</td>
    </tr>
    <tr>
      <th id="T_45212_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_45212_row1_col0" class="data row1 col0" >3374</td>
      <td id="T_45212_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_45212_row1_col2" class="data row1 col2" >LORE1</td>
      <td id="T_45212_row1_col3" class="data row1 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_45212_row1_col4" class="data row1 col4" >0.94226</td>
      <td id="T_45212_row1_col5" class="data row1 col5" >0.98942</td>
      <td id="T_45212_row1_col6" class="data row1 col6" >0.79717</td>
      <td id="T_45212_row1_col7" class="data row1 col7" >1</td>
      <td id="T_45212_row1_col8" class="data row1 col8" >0</td>
      <td id="T_45212_row1_col9" class="data row1 col9" >270.15156</td>
      <td id="T_45212_row1_col10" class="data row1 col10" >False</td>
      <td id="T_45212_row1_col11" class="data row1 col11" >0.08169</td>
    </tr>
    <tr>
      <th id="T_45212_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_45212_row2_col0" class="data row2 col0" >3374</td>
      <td id="T_45212_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_45212_row2_col2" class="data row2 col2" >LORE2</td>
      <td id="T_45212_row2_col3" class="data row2 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_45212_row2_col4" class="data row2 col4" >0.89523</td>
      <td id="T_45212_row2_col5" class="data row2 col5" >0.95891</td>
      <td id="T_45212_row2_col6" class="data row2 col6" >0.81317</td>
      <td id="T_45212_row2_col7" class="data row2 col7" >2</td>
      <td id="T_45212_row2_col8" class="data row2 col8" >0</td>
      <td id="T_45212_row2_col9" class="data row2 col9" >110.73017</td>
      <td id="T_45212_row2_col10" class="data row2 col10" >False</td>
      <td id="T_45212_row2_col11" class="data row2 col11" >0.08571</td>
    </tr>
    <tr>
      <th id="T_45212_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_45212_row3_col0" class="data row3 col0" >3374</td>
      <td id="T_45212_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_45212_row3_col2" class="data row3 col2" >LORE4</td>
      <td id="T_45212_row3_col3" class="data row3 col3" >IF capital.gain <= 4064.0 AND education = Some-college AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_45212_row3_col4" class="data row3 col4" >0.16093</td>
      <td id="T_45212_row3_col5" class="data row3 col5" >0.18621</td>
      <td id="T_45212_row3_col6" class="data row3 col6" >0.87841</td>
      <td id="T_45212_row3_col7" class="data row3 col7" >3</td>
      <td id="T_45212_row3_col8" class="data row3 col8" >0</td>
      <td id="T_45212_row3_col9" class="data row3 col9" >357.85710</td>
      <td id="T_45212_row3_col10" class="data row3 col10" >False</td>
      <td id="T_45212_row3_col11" class="data row3 col11" >0.78989</td>
    </tr>
    <tr>
      <th id="T_45212_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_45212_row4_col0" class="data row4 col0" >3374</td>
      <td id="T_45212_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_45212_row4_col2" class="data row4 col2" >LORE_SA1</td>
      <td id="T_45212_row4_col3" class="data row4 col3" >IF capital.gain <= 7513.6748 AND capital.loss <= 1748.2952 AND education != Masters AND education.num != 13.0 THEN class = <=50K</td>
      <td id="T_45212_row4_col4" class="data row4 col4" >0.74245</td>
      <td id="T_45212_row4_col5" class="data row4 col5" >0.83009</td>
      <td id="T_45212_row4_col6" class="data row4 col6" >0.84878</td>
      <td id="T_45212_row4_col7" class="data row4 col7" >4</td>
      <td id="T_45212_row4_col8" class="data row4 col8" >0</td>
      <td id="T_45212_row4_col9" class="data row4 col9" >35.40669</td>
      <td id="T_45212_row4_col10" class="data row4 col10" >False</td>
      <td id="T_45212_row4_col11" class="data row4 col11" >0.21047</td>
    </tr>
    <tr>
      <th id="T_45212_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_45212_row5_col0" class="data row5 col0" >3374</td>
      <td id="T_45212_row5_col1" class="data row5 col1" >EXPLAN</td>
      <td id="T_45212_row5_col2" class="data row5 col2" >EXPLAN4</td>
      <td id="T_45212_row5_col3" class="data row5 col3" >IF capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_45212_row5_col4" class="data row5 col4" >0.95082</td>
      <td id="T_45212_row5_col5" class="data row5 col5" >0.99665</td>
      <td id="T_45212_row5_col6" class="data row5 col6" >0.79576</td>
      <td id="T_45212_row5_col7" class="data row5 col7" >1</td>
      <td id="T_45212_row5_col8" class="data row5 col8" >0</td>
      <td id="T_45212_row5_col9" class="data row5 col9" >1.82929</td>
      <td id="T_45212_row5_col10" class="data row5 col10" >False</td>
      <td id="T_45212_row5_col11" class="data row5 col11" >0.08265</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_199.png)
    



### Rules for Instance 3374, Non-dominated (Cov↑, Pre↑), Ideal (Cov: 0.95082, Pre: 0.87841), Unique rules (diffrent features)



<style type="text/css">
#T_04627_row1_col0, #T_04627_row1_col1, #T_04627_row1_col2, #T_04627_row1_col3, #T_04627_row1_col4, #T_04627_row1_col5, #T_04627_row1_col6, #T_04627_row1_col7, #T_04627_row1_col8, #T_04627_row1_col9, #T_04627_row1_col10, #T_04627_row1_col11 {
  font-weight: bold;
}
</style>
<table id="T_04627">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_04627_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_04627_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_04627_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_04627_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_04627_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_04627_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_04627_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_04627_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_04627_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_04627_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_04627_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_04627_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_04627_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_04627_row0_col0" class="data row0 col0" >3374</td>
      <td id="T_04627_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_04627_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_04627_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 40.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_04627_row0_col4" class="data row0 col4" >0.53032</td>
      <td id="T_04627_row0_col5" class="data row0 col5" >0.60117</td>
      <td id="T_04627_row0_col6" class="data row0 col6" >0.86059</td>
      <td id="T_04627_row0_col7" class="data row0 col7" >3</td>
      <td id="T_04627_row0_col8" class="data row0 col8" >0</td>
      <td id="T_04627_row0_col9" class="data row0 col9" >6.03093</td>
      <td id="T_04627_row0_col10" class="data row0 col10" >False</td>
      <td id="T_04627_row0_col11" class="data row0 col11" >0.42088</td>
    </tr>
    <tr>
      <th id="T_04627_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_04627_row1_col0" class="data row1 col0" >3374</td>
      <td id="T_04627_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_04627_row1_col2" class="data row1 col2" >LORE1</td>
      <td id="T_04627_row1_col3" class="data row1 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_04627_row1_col4" class="data row1 col4" >0.94226</td>
      <td id="T_04627_row1_col5" class="data row1 col5" >0.98942</td>
      <td id="T_04627_row1_col6" class="data row1 col6" >0.79717</td>
      <td id="T_04627_row1_col7" class="data row1 col7" >1</td>
      <td id="T_04627_row1_col8" class="data row1 col8" >0</td>
      <td id="T_04627_row1_col9" class="data row1 col9" >270.15156</td>
      <td id="T_04627_row1_col10" class="data row1 col10" >False</td>
      <td id="T_04627_row1_col11" class="data row1 col11" >0.08169</td>
    </tr>
    <tr>
      <th id="T_04627_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_04627_row2_col0" class="data row2 col0" >3374</td>
      <td id="T_04627_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_04627_row2_col2" class="data row2 col2" >LORE2</td>
      <td id="T_04627_row2_col3" class="data row2 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_04627_row2_col4" class="data row2 col4" >0.89523</td>
      <td id="T_04627_row2_col5" class="data row2 col5" >0.95891</td>
      <td id="T_04627_row2_col6" class="data row2 col6" >0.81317</td>
      <td id="T_04627_row2_col7" class="data row2 col7" >2</td>
      <td id="T_04627_row2_col8" class="data row2 col8" >0</td>
      <td id="T_04627_row2_col9" class="data row2 col9" >110.73017</td>
      <td id="T_04627_row2_col10" class="data row2 col10" >False</td>
      <td id="T_04627_row2_col11" class="data row2 col11" >0.08571</td>
    </tr>
    <tr>
      <th id="T_04627_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_04627_row3_col0" class="data row3 col0" >3374</td>
      <td id="T_04627_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_04627_row3_col2" class="data row3 col2" >LORE4</td>
      <td id="T_04627_row3_col3" class="data row3 col3" >IF capital.gain <= 4064.0 AND education = Some-college AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_04627_row3_col4" class="data row3 col4" >0.16093</td>
      <td id="T_04627_row3_col5" class="data row3 col5" >0.18621</td>
      <td id="T_04627_row3_col6" class="data row3 col6" >0.87841</td>
      <td id="T_04627_row3_col7" class="data row3 col7" >3</td>
      <td id="T_04627_row3_col8" class="data row3 col8" >0</td>
      <td id="T_04627_row3_col9" class="data row3 col9" >357.85710</td>
      <td id="T_04627_row3_col10" class="data row3 col10" >False</td>
      <td id="T_04627_row3_col11" class="data row3 col11" >0.78989</td>
    </tr>
    <tr>
      <th id="T_04627_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_04627_row4_col0" class="data row4 col0" >3374</td>
      <td id="T_04627_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_04627_row4_col2" class="data row4 col2" >LORE_SA1</td>
      <td id="T_04627_row4_col3" class="data row4 col3" >IF capital.gain <= 7513.6748 AND capital.loss <= 1748.2952 AND education != Masters AND education.num != 13.0 THEN class = <=50K</td>
      <td id="T_04627_row4_col4" class="data row4 col4" >0.74245</td>
      <td id="T_04627_row4_col5" class="data row4 col5" >0.83009</td>
      <td id="T_04627_row4_col6" class="data row4 col6" >0.84878</td>
      <td id="T_04627_row4_col7" class="data row4 col7" >4</td>
      <td id="T_04627_row4_col8" class="data row4 col8" >0</td>
      <td id="T_04627_row4_col9" class="data row4 col9" >35.40669</td>
      <td id="T_04627_row4_col10" class="data row4 col10" >False</td>
      <td id="T_04627_row4_col11" class="data row4 col11" >0.21047</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_202.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_203.png)
    



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_204.png)
    



### Rules for Instance 3374, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99665, Pre: 0.87841, Len: 0.79576)



<style type="text/css">
#T_c06c8_row2_col0, #T_c06c8_row2_col1, #T_c06c8_row2_col2, #T_c06c8_row2_col3, #T_c06c8_row2_col4, #T_c06c8_row2_col5, #T_c06c8_row2_col6, #T_c06c8_row2_col7, #T_c06c8_row2_col8, #T_c06c8_row2_col9, #T_c06c8_row2_col10, #T_c06c8_row2_col11 {
  font-weight: bold;
}
</style>
<table id="T_c06c8">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c06c8_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_c06c8_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_c06c8_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_c06c8_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_c06c8_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_c06c8_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_c06c8_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_c06c8_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_c06c8_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_c06c8_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_c06c8_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_c06c8_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c06c8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_c06c8_row0_col0" class="data row0 col0" >3374</td>
      <td id="T_c06c8_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_c06c8_row0_col2" class="data row0 col2" >ANCHOR1</td>
      <td id="T_c06c8_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 40.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_c06c8_row0_col4" class="data row0 col4" >0.53032</td>
      <td id="T_c06c8_row0_col5" class="data row0 col5" >0.60117</td>
      <td id="T_c06c8_row0_col6" class="data row0 col6" >0.86059</td>
      <td id="T_c06c8_row0_col7" class="data row0 col7" >3</td>
      <td id="T_c06c8_row0_col8" class="data row0 col8" >0</td>
      <td id="T_c06c8_row0_col9" class="data row0 col9" >6.03093</td>
      <td id="T_c06c8_row0_col10" class="data row0 col10" >False</td>
      <td id="T_c06c8_row0_col11" class="data row0 col11" >2.23951</td>
    </tr>
    <tr>
      <th id="T_c06c8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_c06c8_row1_col0" class="data row1 col0" >3374</td>
      <td id="T_c06c8_row1_col1" class="data row1 col1" >ANCHOR</td>
      <td id="T_c06c8_row1_col2" class="data row1 col2" >ANCHOR5</td>
      <td id="T_c06c8_row1_col3" class="data row1 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_c06c8_row1_col4" class="data row1 col4" >0.58600</td>
      <td id="T_c06c8_row1_col5" class="data row1 col5" >0.65064</td>
      <td id="T_c06c8_row1_col6" class="data row1 col6" >0.84292</td>
      <td id="T_c06c8_row1_col7" class="data row1 col7" >3</td>
      <td id="T_c06c8_row1_col8" class="data row1 col8" >0</td>
      <td id="T_c06c8_row1_col9" class="data row1 col9" >7.78208</td>
      <td id="T_c06c8_row1_col10" class="data row1 col10" >False</td>
      <td id="T_c06c8_row1_col11" class="data row1 col11" >2.23151</td>
    </tr>
    <tr>
      <th id="T_c06c8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_c06c8_row2_col0" class="data row2 col0" >3374</td>
      <td id="T_c06c8_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_c06c8_row2_col2" class="data row2 col2" >LORE1</td>
      <td id="T_c06c8_row2_col3" class="data row2 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_c06c8_row2_col4" class="data row2 col4" >0.94226</td>
      <td id="T_c06c8_row2_col5" class="data row2 col5" >0.98942</td>
      <td id="T_c06c8_row2_col6" class="data row2 col6" >0.79717</td>
      <td id="T_c06c8_row2_col7" class="data row2 col7" >1</td>
      <td id="T_c06c8_row2_col8" class="data row2 col8" >0</td>
      <td id="T_c06c8_row2_col9" class="data row2 col9" >270.15156</td>
      <td id="T_c06c8_row2_col10" class="data row2 col10" >False</td>
      <td id="T_c06c8_row2_col11" class="data row2 col11" >0.21992</td>
    </tr>
    <tr>
      <th id="T_c06c8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_c06c8_row3_col0" class="data row3 col0" >3374</td>
      <td id="T_c06c8_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_c06c8_row3_col2" class="data row3 col2" >LORE2</td>
      <td id="T_c06c8_row3_col3" class="data row3 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_c06c8_row3_col4" class="data row3 col4" >0.89523</td>
      <td id="T_c06c8_row3_col5" class="data row3 col5" >0.95891</td>
      <td id="T_c06c8_row3_col6" class="data row3 col6" >0.81317</td>
      <td id="T_c06c8_row3_col7" class="data row3 col7" >2</td>
      <td id="T_c06c8_row3_col8" class="data row3 col8" >0</td>
      <td id="T_c06c8_row3_col9" class="data row3 col9" >110.73017</td>
      <td id="T_c06c8_row3_col10" class="data row3 col10" >False</td>
      <td id="T_c06c8_row3_col11" class="data row3 col11" >1.20660</td>
    </tr>
    <tr>
      <th id="T_c06c8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_c06c8_row4_col0" class="data row4 col0" >3374</td>
      <td id="T_c06c8_row4_col1" class="data row4 col1" >LORE</td>
      <td id="T_c06c8_row4_col2" class="data row4 col2" >LORE4</td>
      <td id="T_c06c8_row4_col3" class="data row4 col3" >IF capital.gain <= 4064.0 AND education = Some-college AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_c06c8_row4_col4" class="data row4 col4" >0.16093</td>
      <td id="T_c06c8_row4_col5" class="data row4 col5" >0.18621</td>
      <td id="T_c06c8_row4_col6" class="data row4 col6" >0.87841</td>
      <td id="T_c06c8_row4_col7" class="data row4 col7" >3</td>
      <td id="T_c06c8_row4_col8" class="data row4 col8" >0</td>
      <td id="T_c06c8_row4_col9" class="data row4 col9" >357.85710</td>
      <td id="T_c06c8_row4_col10" class="data row4 col10" >False</td>
      <td id="T_c06c8_row4_col11" class="data row4 col11" >2.34851</td>
    </tr>
    <tr>
      <th id="T_c06c8_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_c06c8_row5_col0" class="data row5 col0" >3374</td>
      <td id="T_c06c8_row5_col1" class="data row5 col1" >LORE_SA</td>
      <td id="T_c06c8_row5_col2" class="data row5 col2" >LORE_SA1</td>
      <td id="T_c06c8_row5_col3" class="data row5 col3" >IF capital.gain <= 7513.6748 AND capital.loss <= 1748.2952 AND education != Masters AND education.num != 13.0 THEN class = <=50K</td>
      <td id="T_c06c8_row5_col4" class="data row5 col4" >0.74245</td>
      <td id="T_c06c8_row5_col5" class="data row5 col5" >0.83009</td>
      <td id="T_c06c8_row5_col6" class="data row5 col6" >0.84878</td>
      <td id="T_c06c8_row5_col7" class="data row5 col7" >4</td>
      <td id="T_c06c8_row5_col8" class="data row5 col8" >0</td>
      <td id="T_c06c8_row5_col9" class="data row5 col9" >35.40669</td>
      <td id="T_c06c8_row5_col10" class="data row5 col10" >False</td>
      <td id="T_c06c8_row5_col11" class="data row5 col11" >3.20870</td>
    </tr>
    <tr>
      <th id="T_c06c8_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_c06c8_row6_col0" class="data row6 col0" >3374</td>
      <td id="T_c06c8_row6_col1" class="data row6 col1" >EXPLAN</td>
      <td id="T_c06c8_row6_col2" class="data row6 col2" >EXPLAN4</td>
      <td id="T_c06c8_row6_col3" class="data row6 col3" >IF capital.gain <= 5060.0 THEN class = <=50K</td>
      <td id="T_c06c8_row6_col4" class="data row6 col4" >0.95082</td>
      <td id="T_c06c8_row6_col5" class="data row6 col5" >0.99665</td>
      <td id="T_c06c8_row6_col6" class="data row6 col6" >0.79576</td>
      <td id="T_c06c8_row6_col7" class="data row6 col7" >1</td>
      <td id="T_c06c8_row6_col8" class="data row6 col8" >0</td>
      <td id="T_c06c8_row6_col9" class="data row6 col9" >1.82929</td>
      <td id="T_c06c8_row6_col10" class="data row6 col10" >False</td>
      <td id="T_c06c8_row6_col11" class="data row6 col11" >0.22033</td>
    </tr>
  </tbody>
</table>




### Rules for Instance 3374, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov: 0.99665, Pre: 0.87841), Unique rules (diffrent features)



<style type="text/css">
#T_e11de_row1_col0, #T_e11de_row1_col1, #T_e11de_row1_col2, #T_e11de_row1_col3, #T_e11de_row1_col4, #T_e11de_row1_col5, #T_e11de_row1_col6, #T_e11de_row1_col7, #T_e11de_row1_col8, #T_e11de_row1_col9, #T_e11de_row1_col10, #T_e11de_row1_col11 {
  font-weight: bold;
}
</style>
<table id="T_e11de">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e11de_level0_col0" class="col_heading level0 col0" >Instance_Name</th>
      <th id="T_e11de_level0_col1" class="col_heading level0 col1" >Explainer</th>
      <th id="T_e11de_level0_col2" class="col_heading level0 col2" >Rule_ID</th>
      <th id="T_e11de_level0_col3" class="col_heading level0 col3" >Rule</th>
      <th id="T_e11de_level0_col4" class="col_heading level0 col4" >Cov</th>
      <th id="T_e11de_level0_col5" class="col_heading level0 col5" >Cov_class</th>
      <th id="T_e11de_level0_col6" class="col_heading level0 col6" >Pre</th>
      <th id="T_e11de_level0_col7" class="col_heading level0 col7" >Len</th>
      <th id="T_e11de_level0_col8" class="col_heading level0 col8" >Reject</th>
      <th id="T_e11de_level0_col9" class="col_heading level0 col9" >Elapsed_time</th>
      <th id="T_e11de_level0_col10" class="col_heading level0 col10" >Iter_Limit</th>
      <th id="T_e11de_level0_col11" class="col_heading level0 col11" >Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e11de_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_e11de_row0_col0" class="data row0 col0" >3374</td>
      <td id="T_e11de_row0_col1" class="data row0 col1" >ANCHOR</td>
      <td id="T_e11de_row0_col2" class="data row0 col2" >ANCHOR5</td>
      <td id="T_e11de_row0_col3" class="data row0 col3" >IF capital.loss <= 0.0 AND hours.per.week <= 45.0 AND workclass = Private THEN class = <=50K</td>
      <td id="T_e11de_row0_col4" class="data row0 col4" >0.58600</td>
      <td id="T_e11de_row0_col5" class="data row0 col5" >0.65064</td>
      <td id="T_e11de_row0_col6" class="data row0 col6" >0.84292</td>
      <td id="T_e11de_row0_col7" class="data row0 col7" >3</td>
      <td id="T_e11de_row0_col8" class="data row0 col8" >0</td>
      <td id="T_e11de_row0_col9" class="data row0 col9" >7.78208</td>
      <td id="T_e11de_row0_col10" class="data row0 col10" >False</td>
      <td id="T_e11de_row0_col11" class="data row0 col11" >2.23151</td>
    </tr>
    <tr>
      <th id="T_e11de_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_e11de_row1_col0" class="data row1 col0" >3374</td>
      <td id="T_e11de_row1_col1" class="data row1 col1" >LORE</td>
      <td id="T_e11de_row1_col2" class="data row1 col2" >LORE1</td>
      <td id="T_e11de_row1_col3" class="data row1 col3" >IF capital.gain <= 4064.0 THEN class = <=50K</td>
      <td id="T_e11de_row1_col4" class="data row1 col4" >0.94226</td>
      <td id="T_e11de_row1_col5" class="data row1 col5" >0.98942</td>
      <td id="T_e11de_row1_col6" class="data row1 col6" >0.79717</td>
      <td id="T_e11de_row1_col7" class="data row1 col7" >1</td>
      <td id="T_e11de_row1_col8" class="data row1 col8" >0</td>
      <td id="T_e11de_row1_col9" class="data row1 col9" >270.15156</td>
      <td id="T_e11de_row1_col10" class="data row1 col10" >False</td>
      <td id="T_e11de_row1_col11" class="data row1 col11" >0.21992</td>
    </tr>
    <tr>
      <th id="T_e11de_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_e11de_row2_col0" class="data row2 col0" >3374</td>
      <td id="T_e11de_row2_col1" class="data row2 col1" >LORE</td>
      <td id="T_e11de_row2_col2" class="data row2 col2" >LORE2</td>
      <td id="T_e11de_row2_col3" class="data row2 col3" >IF capital.gain <= 4064.0 AND capital.loss <= 0.0 THEN class = <=50K</td>
      <td id="T_e11de_row2_col4" class="data row2 col4" >0.89523</td>
      <td id="T_e11de_row2_col5" class="data row2 col5" >0.95891</td>
      <td id="T_e11de_row2_col6" class="data row2 col6" >0.81317</td>
      <td id="T_e11de_row2_col7" class="data row2 col7" >2</td>
      <td id="T_e11de_row2_col8" class="data row2 col8" >0</td>
      <td id="T_e11de_row2_col9" class="data row2 col9" >110.73017</td>
      <td id="T_e11de_row2_col10" class="data row2 col10" >False</td>
      <td id="T_e11de_row2_col11" class="data row2 col11" >1.20660</td>
    </tr>
    <tr>
      <th id="T_e11de_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_e11de_row3_col0" class="data row3 col0" >3374</td>
      <td id="T_e11de_row3_col1" class="data row3 col1" >LORE</td>
      <td id="T_e11de_row3_col2" class="data row3 col2" >LORE4</td>
      <td id="T_e11de_row3_col3" class="data row3 col3" >IF capital.gain <= 4064.0 AND education = Some-college AND hours.per.week <= 40.0 THEN class = <=50K</td>
      <td id="T_e11de_row3_col4" class="data row3 col4" >0.16093</td>
      <td id="T_e11de_row3_col5" class="data row3 col5" >0.18621</td>
      <td id="T_e11de_row3_col6" class="data row3 col6" >0.87841</td>
      <td id="T_e11de_row3_col7" class="data row3 col7" >3</td>
      <td id="T_e11de_row3_col8" class="data row3 col8" >0</td>
      <td id="T_e11de_row3_col9" class="data row3 col9" >357.85710</td>
      <td id="T_e11de_row3_col10" class="data row3 col10" >False</td>
      <td id="T_e11de_row3_col11" class="data row3 col11" >2.34851</td>
    </tr>
    <tr>
      <th id="T_e11de_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_e11de_row4_col0" class="data row4 col0" >3374</td>
      <td id="T_e11de_row4_col1" class="data row4 col1" >LORE_SA</td>
      <td id="T_e11de_row4_col2" class="data row4 col2" >LORE_SA1</td>
      <td id="T_e11de_row4_col3" class="data row4 col3" >IF capital.gain <= 7513.6748 AND capital.loss <= 1748.2952 AND education != Masters AND education.num != 13.0 THEN class = <=50K</td>
      <td id="T_e11de_row4_col4" class="data row4 col4" >0.74245</td>
      <td id="T_e11de_row4_col5" class="data row4 col5" >0.83009</td>
      <td id="T_e11de_row4_col6" class="data row4 col6" >0.84878</td>
      <td id="T_e11de_row4_col7" class="data row4 col7" >4</td>
      <td id="T_e11de_row4_col8" class="data row4 col8" >0</td>
      <td id="T_e11de_row4_col9" class="data row4 col9" >35.40669</td>
      <td id="T_e11de_row4_col10" class="data row4 col10" >False</td>
      <td id="T_e11de_row4_col11" class="data row4 col11" >3.20870</td>
    </tr>
  </tbody>
</table>




    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_209.png)
    



### Average Number of Filtered Rules at Each Step



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg Filtered (Correct Prediction)</th>
      <th>Avg Filtered (Threshold Filter)</th>
      <th>Avg Filtered (Non-dominated 1)</th>
      <th>Avg Filtered (Non-dominated 2)</th>
      <th>Avg Filtered (Unique Non-dominated 1)</th>
      <th>Avg Filtered (Unique Non-dominated 2)</th>
    </tr>
    <tr>
      <th>Explainer</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ANCHOR</th>
      <td>0.0</td>
      <td>1.2</td>
      <td>1.777778</td>
      <td>1.777778</td>
      <td>2.444444</td>
      <td>2.555556</td>
    </tr>
    <tr>
      <th>LORE</th>
      <td>0.3</td>
      <td>0.2</td>
      <td>1.900000</td>
      <td>1.700000</td>
      <td>2.900000</td>
      <td>2.700000</td>
    </tr>
    <tr>
      <th>LORE_SA</th>
      <td>0.0</td>
      <td>0.4</td>
      <td>2.200000</td>
      <td>2.300000</td>
      <td>2.800000</td>
      <td>2.900000</td>
    </tr>
    <tr>
      <th>EXPLAN</th>
      <td>0.0</td>
      <td>0.9</td>
      <td>2.333333</td>
      <td>2.111111</td>
      <td>3.333333</td>
      <td>3.222222</td>
    </tr>
    <tr>
      <th>Overall Average</th>
      <td>0.3</td>
      <td>2.7</td>
      <td>7.800000</td>
      <td>7.500000</td>
      <td>3.100000</td>
      <td>3.300000</td>
    </tr>
  </tbody>
</table>
</div>



### Average Metrics for Non-Dominated Unique Rules (Cov↑, Pre↑)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pre</th>
      <th>Cov</th>
      <th>Cov_class</th>
      <th>Len</th>
      <th>Reject</th>
      <th>Elapsed_time</th>
      <th>Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ANCHOR</th>
      <td>0.853389</td>
      <td>0.497422</td>
      <td>0.560760</td>
      <td>2.957143</td>
      <td>0.057143</td>
      <td>2.783175</td>
      <td>0.380511</td>
    </tr>
    <tr>
      <th>EXPLAN</th>
      <td>0.920405</td>
      <td>0.140992</td>
      <td>0.201230</td>
      <td>3.333333</td>
      <td>0.000000</td>
      <td>4.088502</td>
      <td>0.625475</td>
    </tr>
    <tr>
      <th>LORE</th>
      <td>0.758908</td>
      <td>0.472701</td>
      <td>0.587999</td>
      <td>1.944444</td>
      <td>0.000000</td>
      <td>123.733865</td>
      <td>0.351265</td>
    </tr>
    <tr>
      <th>LORE_SA</th>
      <td>0.785804</td>
      <td>0.483738</td>
      <td>0.606699</td>
      <td>2.687500</td>
      <td>0.062500</td>
      <td>28.014247</td>
      <td>0.417264</td>
    </tr>
    <tr>
      <th>Global_Average</th>
      <td>0.820425</td>
      <td>0.415071</td>
      <td>0.509276</td>
      <td>2.656667</td>
      <td>0.030000</td>
      <td>46.057733</td>
      <td>0.430531</td>
    </tr>
  </tbody>
</table>
</div>



### Average Metrics for Non-Dominated Unique Rules (Cov_class↑, Pre↑, Len↓)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pre</th>
      <th>Cov</th>
      <th>Cov_class</th>
      <th>Len</th>
      <th>Reject</th>
      <th>Elapsed_time</th>
      <th>Distance_idp_eucl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ANCHOR</th>
      <td>0.851030</td>
      <td>0.507575</td>
      <td>0.570308</td>
      <td>2.964286</td>
      <td>0.071429</td>
      <td>3.052796</td>
      <td>2.279082</td>
    </tr>
    <tr>
      <th>EXPLAN</th>
      <td>0.918706</td>
      <td>0.133153</td>
      <td>0.191102</td>
      <td>3.166667</td>
      <td>0.000000</td>
      <td>4.076404</td>
      <td>2.679050</td>
    </tr>
    <tr>
      <th>LORE</th>
      <td>0.760163</td>
      <td>0.525350</td>
      <td>0.635496</td>
      <td>1.800000</td>
      <td>0.000000</td>
      <td>144.335948</td>
      <td>1.260211</td>
    </tr>
    <tr>
      <th>LORE_SA</th>
      <td>0.775746</td>
      <td>0.505834</td>
      <td>0.634839</td>
      <td>2.760417</td>
      <td>0.083333</td>
      <td>28.584294</td>
      <td>2.231624</td>
    </tr>
    <tr>
      <th>Global_Average</th>
      <td>0.815388</td>
      <td>0.440391</td>
      <td>0.534595</td>
      <td>2.575269</td>
      <td>0.037634</td>
      <td>55.414897</td>
      <td>2.015580</td>
    </tr>
  </tbody>
</table>
</div>



## Overall Heatmap – Non-dominated Rules (Cov↑, Pre↑)



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_217.png)
    



## Overall Heatmap – Non-dominated Rules (Cov_class↑, Pre↑, Len↓)



    
![png](Results_Analysis_Adult_RF_25_06_files/Results_Analysis_Adult_RF_25_06_0_219.png)
    



```python

```
