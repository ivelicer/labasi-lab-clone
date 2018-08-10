# Progress update Fri 10 Aug 2018
1) Have figured out how to use git operations from laptop, but the cfdb_django folder from which I work is stored in my usb because my laptop cannot store all the data. I have been trying to find a way to be able to push from the usb, but this is still in progress...

2) Writing images and labels as a tfRecordFile appears to function. Reading from the tfRecordFile results in empty tensors, however. I have focused on trying to fix this today, but without success. I will stop by on Monday to discuss.

3) Layers for a CNN are prepared, but need to wait on the reading issue to be resolved.

#--------------

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
