# isupply-task
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
pip install numpy pandas scikit-learn fuzzywuzzy openpyxl
```

## Usage
1. Place the dataset in the working directory.
2. Run the script to match products between the master file and the dataset.
3. The output file `Matched Dataset.xlsx` will be generated with matched SKUs and confidence scores.

### Example Usage:
```python
import pandas as pd

masterfile = pd.read_excel("Product Matching Dataset.xlsx", sheet_name="Master File")
dataset = pd.read_excel("Product Matching Dataset.xlsx", sheet_name="Dataset")

matched_results = match_products(masterfile, dataset)
matched_results.to_excel("Matched Dataset.xlsx", index=False)
print("✅ Matching Completed and Saved!")
```

## File Structure
```
- Product Matching Model
  |-- product_matching.py (Main script)
  |-- Product Matching Dataset.xlsx (Input data)
  |-- Matched Dataset.xlsx (Output file)
  |-- README.md (Project documentation)
```

## Notes
- Ensure that both product names in English and Arabic exist in the dataset for better matching.
- The `words_to_remove` list can be updated to refine the text cleaning process.
- Ensure the data has a 'seller_price_name' column or modify the column name in the pipeline



