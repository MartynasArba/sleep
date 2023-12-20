**Code for sleep period extraction**
Might not work correctly, can also be messy - NOT A FINAL VERSION
In general, takes data, converts to spectral features, calculates a UMAP, clusters the UMAP by agglomerative clustering, sorts clusters into sleep states.
For efficiency, agglomerative clustering is used to train a random forest classifier instead.
