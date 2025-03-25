"""
Spark utilities for processing MIMIC dataset.

This module provides functions for efficient processing of the MIMIC dataset
using Apache Spark, which is essential for handling the scale of the data.
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, expr, when, udf
from pyspark.sql.types import DoubleType, ArrayType, StringType
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_spark(app_name="MIMIC-Processing", memory="16g", cores=4):
    """Initialize a Spark session with appropriate configuration.
    
    Args:
        app_name: Name of the Spark application
        memory: Memory allocation for driver
        cores: Number of cores to use
        
    Returns:
        Initialized SparkSession
    """
    logger.info(f"Initializing Spark with {memory} memory and {cores} cores")
    
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.executor.cores", str(cores))
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate())
    
    return spark


def load_mimic_data(data_path, table_name, spark=None):
    """Load MIMIC data into a Spark DataFrame.
    
    Args:
        data_path: Path to MIMIC data
        table_name: Name of the table to load (e.g., 'chartevents', 'patients')
        spark: SparkSession (created if None)
        
    Returns:
        Spark DataFrame with loaded data
    """
    if spark is None:
        spark = initialize_spark()
    
    # Determine file format
    file_path = os.path.join(data_path, f"{table_name}")
    
    if os.path.exists(f"{file_path}.csv"):
        logger.info(f"Loading {table_name} from CSV")
        df = spark.read.csv(f"{file_path}.csv", header=True, inferSchema=True)
    elif os.path.exists(f"{file_path}.parquet") or os.path.isdir(f"{file_path}.parquet"):
        logger.info(f"Loading {table_name} from Parquet")
        df = spark.read.parquet(f"{file_path}.parquet")
    else:
        raise FileNotFoundError(f"Could not find {table_name} in {data_path}")
    
    # Optimize with caching for frequently used tables
    if table_name in ['chartevents', 'patients', 'admissions']:
        df = df.cache()
    
    logger.info(f"Loaded {table_name}: {df.count()} rows with {len(df.columns)} columns")
    return df


def extract_features(spark, data_path, output_path, features=None, window_size=24, step_size=1):
    """Extract time-series features from MIMIC data.
    
    Args:
        spark: SparkSession
        data_path: Path to MIMIC data
        output_path: Path to save processed data
        features: List of specific features to extract (None for auto-select)
        window_size: Window size in hours for temporal aggregation
        step_size: Step size in hours for sliding windows
        
    Returns:
        Path to processed data
    """
    logger.info("Starting feature extraction pipeline")
    
    # Load relevant tables
    chartevents = load_mimic_data(data_path, "chartevents", spark)
    d_items = load_mimic_data(data_path, "d_items", spark)
    patients = load_mimic_data(data_path, "patients", spark)
    admissions = load_mimic_data(data_path, "admissions", spark)
    
    # Register tables for SQL queries
    chartevents.createOrReplaceTempView("chartevents")
    d_items.createOrReplaceTempView("d_items")
    patients.createOrReplaceTempView("patients")
    admissions.createOrReplaceTempView("admissions")
    
    # Get most common chartevents if features not specified
    if features is None:
        logger.info("Automatically selecting most common chartevents")
        feature_count = spark.sql("""
            SELECT itemid, COUNT(*) as count
            FROM chartevents
            WHERE valuenum IS NOT NULL
            GROUP BY itemid
            ORDER BY count DESC
            LIMIT 50
        """)
        
        features = [row.itemid for row in feature_count.collect()]
        logger.info(f"Selected {len(features)} features: {features[:5]}...")
    
    # Get patient demographics
    logger.info("Extracting patient demographics")
    demographics = spark.sql("""
        SELECT 
            p.subject_id,
            p.gender,
            DATEDIFF(a.admittime, p.dob) / 365.25 as age,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            a.deathtime,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.insurance,
            a.language,
            a.marital_status,
            a.ethnicity,
            CASE WHEN a.deathtime IS NOT NULL THEN 1 ELSE 0 END as mortality,
            CASE WHEN a.hospital_expire_flag = 1 THEN 1 ELSE 0 END as hospital_mortality
        FROM 
            patients p
        JOIN 
            admissions a ON p.subject_id = a.subject_id
    """)
    
    # Create feature pivot query
    pivot_columns = ", ".join([f"MAX(CASE WHEN itemid = {f} THEN valuenum ELSE NULL END) as feature_{f}" 
                              for f in features])
    
    logger.info("Extracting time-series features with temporal windows")
    feature_query = f"""
        SELECT 
            c.subject_id,
            c.hadm_id,
            CAST(c.charttime as timestamp) as charttime,
            {pivot_columns}
        FROM 
            chartevents c
        WHERE 
            c.valuenum IS NOT NULL
        GROUP BY 
            c.subject_id, c.hadm_id, c.charttime
        ORDER BY
            c.subject_id, c.hadm_id, c.charttime
    """
    
    time_series_features = spark.sql(feature_query)
    
    # Create windows with proper timestamps
    logger.info(f"Creating sliding windows: size={window_size}h, step={step_size}h")
    time_series_features.createOrReplaceTempView("time_series_features")
    
    # Create window aggregation
    window_features = spark.sql(f"""
        SELECT
            t1.subject_id,
            t1.hadm_id,
            t1.charttime as window_end,
            {', '.join([f'AVG(t2.feature_{f}) as feature_{f}' for f in features])}
        FROM
            time_series_features t1
        JOIN
            time_series_features t2
        ON
            t1.subject_id = t2.subject_id AND
            t1.hadm_id = t2.hadm_id AND
            t2.charttime >= t1.charttime - INTERVAL {window_size} HOURS AND
            t2.charttime <= t1.charttime
        GROUP BY
            t1.subject_id, t1.hadm_id, t1.charttime
        HAVING
            COUNT(*) > 0
    """)
    
    # Join with demographics to add outcomes
    logger.info("Joining features with patient demographics and outcomes")
    final_dataset = window_features.join(
        demographics,
        (window_features.subject_id == demographics.subject_id) & 
        (window_features.hadm_id == demographics.hadm_id),
        "inner"
    ).select(
        window_features["*"],
        demographics.age,
        demographics.gender,
        demographics.admission_type,
        demographics.insurance,
        demographics.ethnicity,
        demographics.mortality,
        demographics.hospital_mortality
    )
    
    # Write output
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "processed_mimic")
    
    logger.info(f"Writing processed data to {output_file}")
    final_dataset.write.parquet(output_file)
    
    # Also save a small sample as CSV for inspection
    sample = final_dataset.limit(1000)
    sample.toPandas().to_csv(f"{output_file}_sample.csv", index=False)
    
    logger.info("Feature extraction complete")
    return output_file


def create_train_validate_test_split(spark, data_path, output_path, val_ratio=0.15, test_ratio=0.15):
    """Split processed data into train, validation, and test sets.
    
    Args:
        spark: SparkSession
        data_path: Path to processed data
        output_path: Path to save split data
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Dictionary with paths to train, val, and test data
    """
    logger.info("Creating train/validation/test splits")
    
    # Load processed data
    data = spark.read.parquet(data_path)
    
    # Get unique patients
    unique_patients = data.select("subject_id").distinct()
    patient_count = unique_patients.count()
    
    logger.info(f"Found {patient_count} unique patients")
    
    # Create random splits by patient ID to prevent leakage
    train_patients, val_test_patients = unique_patients.randomSplit([1.0 - val_ratio - test_ratio, val_ratio + test_ratio])
    val_patients, test_patients = val_test_patients.randomSplit([val_ratio / (val_ratio + test_ratio), test_ratio / (val_ratio + test_ratio)])
    
    # Save patient IDs for reference
    train_patient_ids = [row.subject_id for row in train_patients.collect()]
    val_patient_ids = [row.subject_id for row in val_patients.collect()]
    test_patient_ids = [row.subject_id for row in test_patients.collect()]
    
    logger.info(f"Split counts - Train: {len(train_patient_ids)}, Val: {len(val_patient_ids)}, Test: {len(test_patient_ids)}")
    
    # Create datasets based on patient IDs
    train_data = data.filter(col("subject_id").isin(train_patient_ids))
    val_data = data.filter(col("subject_id").isin(val_patient_ids))
    test_data = data.filter(col("subject_id").isin(test_patient_ids))
    
    # Save splits
    os.makedirs(output_path, exist_ok=True)
    train_path = os.path.join(output_path, "train")
    val_path = os.path.join(output_path, "val")
    test_path = os.path.join(output_path, "test")
    
    logger.info(f"Saving splits to {output_path}")
    train_data.write.parquet(train_path)
    val_data.write.parquet(val_path)
    test_data.write.parquet(test_path)
    
    # Return paths
    return {
        "train": train_path,
        "val": val_path,
        "test": test_path
    }


def main(data_path, output_path, features=None):
    """Main function to process MIMIC data with Spark.
    
    Args:
        data_path: Path to raw MIMIC data
        output_path: Path to save processed data
        features: List of specific features to extract
        
    Returns:
        Dictionary with paths to train, val, and test data
    """
    spark = initialize_spark()
    
    try:
        # Extract features
        processed_data_path = extract_features(spark, data_path, output_path, features)
        
        # Create data splits
        split_paths = create_train_validate_test_split(spark, processed_data_path, output_path)
        
        return split_paths
    finally:
        # Clean up
        spark.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process MIMIC data with Spark")
    parser.add_argument("--data_path", required=True, help="Path to raw MIMIC data")
    parser.add_argument("--output_path", required=True, help="Path to save processed data")
    parser.add_argument("--features", nargs="+", type=int, help="Specific features to extract")
    
    args = parser.parse_args()
    
    main(args.data_path, args.output_path, args.features)
