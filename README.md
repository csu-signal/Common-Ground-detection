# Common-Ground-detection

Paper: Common Ground Tracking in Multimodal Dialogue

Authors: Ibrahim Khebour, Kenneth Lai, Mariah Bradford, Yifan Zhu, Richard Brutti, Christopher Tam, Jingxuan Tu, Benjamin Ibarra, Nathaniel Blanchard, Nikhil Krishnaswamy and James Pustejovsky

Published at LREC-COLING 2024

## Repo Content

CG_3_pipeline_integration merges all parts of the model together (move classifier, propositional extractor, and updates the Banks). This is primarily what you need to run the code, other files are only needed if you want to explore different paths with this project.

Move_Classifier_final.ipynb Creates the move classifier as presented in the paper. This also includes code used for grid search for hyperparameter optimization, and more detailed performance results

Featurization.ipynb constructs the feature vectors for GAMR and actions

Mapping to oracle.ipynb maps the gamr, action and cps annotation to oracle segments.

utt_encoding.py generates all possible propositons and their embeddings.

## Data Used

Data is located at https://drive.google.com/open?id=1B4pzoS7E4iXu8R2OKZiJOySmM8IKHWCp&usp=drive_fs (Need permission to access)

The Google Drive link also includes saved checkpoint of the 10 final models.
