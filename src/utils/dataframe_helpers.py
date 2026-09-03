import pandas as pd

def optimize_dataframe_memory(df):
    """
    Optimizes a Pandas DataFrame's memory usage by downcasting numerical types
    and converting object columns to 'category' where appropriate.
    
    Args:
        df (pd.DataFrame): The DataFrame to optimize.
    
    Returns:
        pd.DataFrame: The memory-optimized DataFrame.
    """
    initial_memory = df.memory_usage(deep=True).sum() / (1024**3)
    print(f"  Initial DataFrame memory: {initial_memory:.4f} GB", flush = True)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            # Heuristic for category conversion
            if num_unique_values / num_total_values < 0.5: 
                df[col] = df[col].astype('category')
        elif 'float' in str(col_type):
            df.loc[:, col] = pd.to_numeric(df[col], downcast='float')
        elif 'int' in str(col_type):
            df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')

    final_memory = df.memory_usage(deep=True).sum() / (1024**3)
    print(
        f"  Optimized DataFrame memory: {final_memory:.4f} GB" 
        f" (Reduced by {(initial_memory - final_memory) / initial_memory * 100:.4f}%)", 
        flush = True
    )
    return df


def get_df_size(df):
    """
    Helper function to get and print the size of a DataFrame in GB.
    """
    size_gb = df.memory_usage(deep=True).sum() / (1024**3)
    return size_gb
