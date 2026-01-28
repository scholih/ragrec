-- RagRec Database Schema
-- PostgreSQL schema for H&M fashion recommendation data

-- Products/Articles table
CREATE TABLE IF NOT EXISTS products (
    article_id BIGINT PRIMARY KEY,
    product_code BIGINT NOT NULL,
    prod_name TEXT NOT NULL,
    product_type_no INTEGER,
    product_type_name TEXT,
    product_group_name TEXT,
    graphical_appearance_no INTEGER,
    graphical_appearance_name TEXT,
    colour_group_code INTEGER,
    colour_group_name TEXT,
    perceived_colour_value_id INTEGER,
    perceived_colour_value_name TEXT,
    perceived_colour_master_id INTEGER,
    perceived_colour_master_name TEXT,
    department_no INTEGER,
    department_name TEXT,
    index_code TEXT,
    index_name TEXT,
    index_group_no INTEGER,
    index_group_name TEXT,
    section_no INTEGER,
    section_name TEXT,
    garment_group_no INTEGER,
    garment_group_name TEXT,
    detail_desc TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on product_code for lookups
CREATE INDEX IF NOT EXISTS idx_products_product_code ON products(product_code);
CREATE INDEX IF NOT EXISTS idx_products_product_type ON products(product_type_no);
CREATE INDEX IF NOT EXISTS idx_products_department ON products(department_no);

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    fn REAL,  -- FN value (unknown meaning in dataset)
    active REAL,  -- Activity score
    club_member_status TEXT,
    fashion_news_frequency TEXT,
    age INTEGER,
    age_bracket TEXT,  -- Derived: 'under_20', '20-29', '30-39', '40-49', '50-59', '60+'
    postal_code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on customer attributes for segmentation
CREATE INDEX IF NOT EXISTS idx_customers_age_bracket ON customers(age_bracket);
CREATE INDEX IF NOT EXISTS idx_customers_club_status ON customers(club_member_status);
CREATE INDEX IF NOT EXISTS idx_customers_postal ON customers(postal_code);

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_date DATE NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    article_id BIGINT NOT NULL REFERENCES products(article_id),
    price REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_article ON transactions(article_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_customer_date ON transactions(customer_id, transaction_date DESC);

-- Product embeddings table (will be populated by Phase 1.1)
CREATE TABLE IF NOT EXISTS product_embeddings (
    article_id BIGINT PRIMARY KEY REFERENCES products(article_id),
    embedding vector(768),  -- SigLIP embedding dimension
    model_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector similarity index (will be used for visual search)
CREATE INDEX IF NOT EXISTS idx_product_embeddings_vector
    ON product_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Stats view for monitoring
CREATE OR REPLACE VIEW data_stats AS
SELECT
    (SELECT COUNT(*) FROM products) as products_count,
    (SELECT COUNT(*) FROM customers) as customers_count,
    (SELECT COUNT(*) FROM transactions) as transactions_count,
    (SELECT COUNT(*) FROM product_embeddings) as embeddings_count,
    (SELECT MIN(transaction_date) FROM transactions) as earliest_transaction,
    (SELECT MAX(transaction_date) FROM transactions) as latest_transaction;
