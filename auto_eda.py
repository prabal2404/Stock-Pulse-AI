import pandas as pd

def smart_eda(df, label_col_name='Label'):
    print("\n>>>>>>>>>>>>>>>>>Running Smart_EDA interpretation.......\n")

    # Shape and null values check
    print(f"Total Rows : {df.shape[0]} | Columns : {df.shape[1]} ")
    nulls = df.isnull().mean()*100 # every column's % null values present in the specific column
    columns_to_drop = nulls[nulls>30].index.tolist()
    if columns_to_drop:
        print(f'Drop Columns Due to high null(NaN) values (>30%) : {columns_to_drop}')

    else:
        print("No columns with high null values (NaN)")

    # Label distribution 
    if label_col_name in df.columns:
        counts = df[label_col_name].value_counts(normalize=True) # 'noramlize = True' will give proportion of each among all instead of just counts
        print(f"\n Label Distirbution : {counts}")

        if counts.max() > 0.70:
            print(" Label imbalance detected !!!!!!!!!!!!!")
        else:
            print('Labels looks balanced')

    else:
        print('No labels found !!!!!!!!')

    # Correlation check
    print("\n Checking correlation between features.......")
    corr = df.select_dtypes(include='number').corr()
    checked = set()
    high_corr_pairs = []

    for i in corr.columns:
        for j in corr.columns:
            if i != j:
                pair = tuple(sorted((i, j)))
                if pair not in checked and abs(corr.loc[i, j]) >= 0.9:
                    high_corr_pairs.append((i, j, corr.loc[i, j]))
                    checked.add(pair)
                    
    if high_corr_pairs:
        print("Highly correlated feature pairs (>= 0.9):")
        for pair in high_corr_pairs:
            print(f"{pair[0]} | {pair[1]} = {pair[2]:.2f}")
        print("Suggest dropping one of each correlated pair.")
    else:
        print("No highly correlated features found in data.")

    # check skewness if there
    print("\n Checking Skewness........")
    num_cols = df.select_dtypes(include='number').columns
    skewness = df[num_cols].skew() # will return pandas series: index-->num_col and values-->skewness value >0<
    skewed = skewness[abs(skewness) > 1] # return filteres 'Series' for skewness having more than >1 or >-1
    if not skewed.empty:
        print(f"Skewed columns detected (|skew| > 1):")
        print(skewed)
        print("Suggest: Use StandardScaler.")
    else:
        print("Features are normally distributed.")

    print("\n ********************** Summary Recommendations **************************")
    if columns_to_drop:
        print(f"- Drop columns: {columns_to_drop}\n")
    if counts.max() > 0.7:
        print("-\n Handle label imbalance using SMOTE or `class_weight='balanced'`")
    if high_corr_pairs:
        print("- Drop one feature from each correlated pair")
    if not skewed.empty:
        print("- Apply Standard scaling")

    print("\n********************************** Smart EDA Completed *****************************\n")