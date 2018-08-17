# Labasi CNN Work Flow

The purpose of this project was to prepare code for training a CNN for recognizing glyphs from the  labasi database as their respective signs. Due to time constraints, thorough training was not possible,  but a workflow with snippets of code is provided below. Of particular note for those interested is the section on writing and reading tfrecord files (formatting one's own image files for use in tensorflow) and preparing one-hots for the image files, of which I include snippets below. To view the code in full please see labasi_cnn_WORKING.ipynb for the final code with comments provided.

## File Preparation

Because there are cases of signs with very few instances of glyphs, a threshold of glyphs-per-sign should be used to prepare the files. In this case, 50 was used. Each sign group is assigned an integer within the range of total groups that qualify over the threshold. This integer is later used as an id and turned into a one-hot for training.

For visualisation of statistics regarding the labasi data, please see labasi_visualization_recovered.ipynb. From the data gathered, it appears that it would only make sense to try to classify the glyphs based on sign or period if one were to do artificial amplification of the data. 

## Thresholding and Id Prep

Example of preparing the files to include integer ids that can later be converted to one-hots, and thresholding glyphs-per-sign.

df1 = pd.read_csv('/Volumes/IMVDrive/cfdb-django/glyphs-aligned-w-std_sign-images.csv', usecols=['sign', 'glyph'])
df_group = df1.groupby(by=['sign'])
df_group = sorted(df_group, key=lambda x: len(x[1])) #https://stackoverflow.com/questions/22291395/sorting-the-grouped-data-as-per-group-size-in-pandas

train_list = pd.DataFrame(data=None, columns=['sign', 'glyph','onehot'])
val_list = pd.DataFrame(data=None, columns=['sign', 'glyph','onehot'])
test_list = pd.DataFrame(data=None, columns=['sign', 'glyph','onehot'])

group_count = 0
print("Thresholding number of glyphs per sign at 50...")
first = True
for name,group in df_group:
    if len(group) >= 50:
        group_count = group_count+1
        
        # id column of identical integers to identify each instance of the group
        col = []
        for g in range(len(group)):
            col.append(group_count)
        col_df = pd.DataFrame(col, columns=['onehot'])
        
        # assign sections of the column to train, validate, and test groups
        train_col = col[0:int(0.9*len(group))]
        val_col = col[int(0.9*len(group)):int(0.95*len(group))]
        test_col = col[int(0.95*len(group)):]
        
        # assign sections of the orginal group to train, validate, and test groups
        train_group = group[0:int(0.9*len(group))] 
        val_group = group[int(0.9*len(group)):int(0.95*len(group))]
        test_group = group[int(0.95*len(group)):]
        
        # join the onehot column to the original group
        train_group = train_group.assign(onehot=train_col)
        val_group = val_group.assign(onehot=val_col)
        test_group = test_group.assign(onehot=test_col)
        
        # for each successive group append the data to the growing list of train, validate, and test groups
        train_list = pd.concat([train_list, train_group], join_axes=[train_list.columns], ignore_index=True)
        val_list = pd.concat([val_list, val_group], join_axes=[val_list.columns], ignore_index=True)
        test_list = pd.concat([test_list, test_group], join_axes=[test_list.columns], ignore_index=True)
        
print("Thresholding finished.")

## 

# --------------------------------------------------------------------------------------

# labasi-lab
a repo to explore data from the labasi-project

## Task 1: Get to know the data

Gather basic statics about the data like:

* How many Glyphs | Sings | Tablets
* How many Glyphs per Sign | Tablet (total numbers, mean, min, max)
* Glyphs per Period | Scribe | ... (total numbers, mean, min, max)

Visualize your results with e.g. bar charts


## Task 2: Define what you'd like to train

Based on the outcome of Task 1 define potential use cases for training specific classifier/models
e.g. Train a model to classify
* Glyphs by Period
* Glyphs by Tablet
* Glyphs by Sign


## Task 3: Feature definition

It is most likely necessary to somehow numberize the images of the traning data. This will involve some image (pre)processing, e.g. removing color, scale, rotate, cut, ....
