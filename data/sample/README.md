# H&M Sample Dataset

Sample dataset created from H&M Fashion Recommendations dataset for development and testing.

## Contents

- **articles_sample.parquet** - 1,000 products with images
- **customers_sample.parquet** - 4,946 customers who purchased these products
- **transactions_sample.parquet** - 5,000 purchase transactions
- **images/** - 100 product images

## Statistics

| File | Rows | Columns |
|------|------|---------|
| articles_sample.parquet | 1,000 | 25 |
| customers_sample.parquet | 4,946 | 7 |
| transactions_sample.parquet | 5,000 | 4 |

## Schema

### Articles
Schema({'article_id': Int64, 'product_code': Int64, 'prod_name': String, 'product_type_no': Int64, 'product_type_name': String, 'product_group_name': String, 'graphical_appearance_no': Int64, 'graphical_appearance_name': String, 'colour_group_code': Int64, 'colour_group_name': String, 'perceived_colour_value_id': Int64, 'perceived_colour_value_name': String, 'perceived_colour_master_id': Int64, 'perceived_colour_master_name': String, 'department_no': Int64, 'department_name': String, 'index_code': String, 'index_name': String, 'index_group_no': Int64, 'index_group_name': String, 'section_no': Int64, 'section_name': String, 'garment_group_no': Int64, 'garment_group_name': String, 'detail_desc': String})

### Customers
Schema({'customer_id': String, 'FN': Float64, 'Active': Float64, 'club_member_status': String, 'fashion_news_frequency': String, 'age': Int64, 'postal_code': String})

### Transactions
Schema({'t_dat': String, 'customer_id': String, 'article_id': Int64, 'price': Float64})

## Usage

```python
import polars as pl

# Load sample data
articles = pl.read_parquet("data/sample/articles_sample.parquet")
customers = pl.read_parquet("data/sample/customers_sample.parquet")
transactions = pl.read_parquet("data/sample/transactions_sample.parquet")
```

## Source

Full dataset: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

Sample created: 2026-01-28 (via create_sample_data.py)
Sampling seed: 42 (reproducible)
