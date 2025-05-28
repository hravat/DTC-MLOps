import sklearn
import argparse
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# In[5]:

parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")
parser.add_argument("--arg_month", type=int, help="Month 2 digits",required=True)
parser.add_argument("--arg_year", type=int, help="Year in 4 digits",required=True)

args = parser.parse_args()

print(f'######### File running for year {args.arg_year:04d} and month {args.arg_month:02d} ############')



with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[8]:

file_name=f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.arg_year:04d}-{args.arg_month:02d}.parquet'
df = read_data(file_name)


print('############### Data Read Complete ################')

# In[9]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


print(f"#########  The average of predictions is :- {float(y_pred.mean())}  #########")


df['ride_id'] = f'{args.arg_year:04d}/{args.arg_month:02d}_' + df.index.astype('str')



df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'prediction': y_pred
})



output_file = f'../data/yellow_tripdata_{args.arg_year:04d}-{args.arg_month:02d}-results.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)



print(f'######### Resules written to {output_file} #########')





