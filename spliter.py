import splitfolders

# Split with a rario.
# To only split into training and validation set, sat atuple to ratio 'ratio' i.e, (0.8, .2).
# Splitfolder.ratio(dir_path, outputpath="output", seed 1337, ratio=(.8, .1, .1), group_prefix=None)

splitfolders.ratio("input",  output="output", seed = 1337, ratio=(.8, .1, .1), group_prefix=None)
