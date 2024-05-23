# Remote data storage 
S3: `s3://real-estate-realtor-buddy/`

Necessary commands to set up AWS credentials in local. 
```
export AWS_ACCESS_KEY_ID='your_access_key_id'
export AWS_SECRET_ACCESS_KEY='your_secret_access_key'
```

## Zillow data scrape service 
Script name: `zillow_data_scrape_service.py`

We use the script to download raw listing data and save those as json in the data directory. You may have to set up your AWS access credentials to download necessary files. 

