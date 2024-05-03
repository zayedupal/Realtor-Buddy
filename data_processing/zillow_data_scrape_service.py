import datetime
import os
import time

import pandas as pd
import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def get_property_details_by_zpid(zpid):
	time.sleep(0.25)
	url = "https://zillow-com4.p.rapidapi.com/properties/details"

	querystring = {"zpid": zpid}

	headers = {
		"X-RapidAPI-Key": os.environ['RAPIDAPI_KEY'],
		"X-RapidAPI-Host": "zillow-com4.p.rapidapi.com"
	}

	response = requests.get(url, headers=headers, params=querystring)

	data = None
	if response.status_code == 200:
		try:
			data = response.json()  # Parse JSON response (handle potential errors)
		except json.JSONDecodeError:
			print(f"Error: Could not parse JSON response for zpid in get_property_details_by_zpid: {zpid}")
	else:
		print(f"Error: API call failed in get_property_details_by_zpid with response: {response}\n")

	return data

def get_listing_by_location(location, resultsPerPage, page=1):
	# time.sleep(0.5)
	url = "https://zillow-com4.p.rapidapi.com/properties/search"
	querystring = {
		"location": location,
		"resultsPerPage": resultsPerPage,
		"page": page,
		"status": "forSale",
		"sort": "recentlyChanged"
	}

	headers = {
		"X-RapidAPI-Key": os.environ['RAPIDAPI_KEY'],
		"X-RapidAPI-Host": "zillow-com4.p.rapidapi.com"
	}
	response = requests.get(url, headers=headers, params=querystring)

	data = None
	if response.status_code == 200:
		try:
			data = response.json()
			# for each zpid get the property details
			for prop in tqdm(data['data']):
				prop_details = get_property_details_by_zpid(prop['zpid'])
				prop['details'] = prop_details['data']
		except json.JSONDecodeError:
			print(f"Error: Could not parse JSON response in get_listing_by_location for iteration: {page}")
	else:
		print(f"Error: API call failed in get_listing_by_location with response: {response}\n")

	return data

def get_listings_by_location_batch(location="New York, NY", total_listings=5000,
								   output_filename="New York_NY.json"):

	data = get_listing_by_location(location, total_listings)

	# If you want the result as a JSON string
	# merged_json = json.dumps(merged_list)
	merged_json = data

	# Save merged data to JSON file
	with open(output_filename, 'w') as outfile:
		json.dump(merged_json, outfile)
		# outfile.write(merged_json)

def get_data_by_city(city, count):
	city_formatted = city.replace(",", "_")
	get_listings_by_location_batch(location=city, total_listings=count,
								   output_filename=f"data/{city_formatted}_{count}_{datetime.date.today().strftime('%m-%d-%Y')}.json")

if __name__ == '__main__':
	top_cities_df = pd.read_csv("data/others/us-cities-table-for-georgia - top_cities.csv")
	for city in tqdm(top_cities_df['top cities'].tolist()):
		get_data_by_city(city, count=1000)

