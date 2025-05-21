if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import duckdb

@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    str_query = """
        SELECT CAST(PULocationID  AS STRING) as PULocationID,
                CAST(DOLocationID  as STRING) as DOLocationID
        FROM df
                WHERE EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime))/60.0 
        BETWEEN 1.0 AND 60.0        
"""

## One Hot Encode 
    df_one_hot = duckdb.sql(str_query).df()
    df_one_hot.shape
    list_df_feb = df_one_hot.to_dict(orient='records')
    vec = DictVectorizer() 
    mat_feb = vec.fit_transform(list_df_feb)


    ## Prepare train data 
    str_query = """ 
        SELECT 
            EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime))/60.0 
                AS trip_duration_minutes
        FROM  df
        WHERE EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime))/60.0 
                BETWEEN 1.0 AND 60.0  
    """

    y = duckdb.sql(str_query).df().to_numpy()
    X = mat_feb 

    ## Train an predict model 
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    rmse = root_mean_squared_error(y, y_pred)

    print(model.intercept_)

    return {
    "model": model,
    "vectorizer": vec
}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
