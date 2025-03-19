# Rule-based Product Matching
 # Product Matching Model

## Overview
This project implements a **Product Matching Model** that identifies similar products between a master product list and a dataset using a combination of text preprocessing, TF-IDF vectorization, cosine similarity, fuzzy matching, price normalization and weighted confidence score.

## Features
- **Arabic Text Normalization**: Handles variations in Arabic text by removing tashkeel, normalizing specific characters, and cleaning redundant words.
- **Text Preprocessing**: Removes unwanted words and applies regex-based cleaning.
- **TF-IDF Vectorization**: Converts product names into numerical features to compute similarity scores.
- **Cosine Similarity**: Measures text-based similarity between product names.
- **Fuzzy Matching**: Uses the Levenshtein distance to compare string similarities.
- **Price Similarity**: Ensures that price variations are considered in the matching process.
- **Optimized Weights**: Optimized scoring weights according to performance evaluation.
- **Weighted Scoring System**: Combines cosine similarity, fuzzy matching, and price similarity to determine the best product match.

## Dependencies
This model requires the following Python libraries:
```bash
pip install numpy pandas scikit-learn fuzzywuzzy python-Levenshtein openpyxl
```

## Example usage
```bash
python matcher.py <file.xlsx> <MasterSheet> <DatasetSheet>
```


## Notes
- Ensure that both product names in English and Arabic exist in the dataset for better matching.
- The `words_to_remove` list can be updated to refine the text cleaning process.
- Ensure the data has a 'seller_price_name' column 
- The evaluation and Visualization provided in the notebook are based on the ground truths (SKUs) provided in the dataset sheet



