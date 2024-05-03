PRIMARY_FIELD = ['zpid']
DESCRIPTION_FIELDS = [
    'bathrooms', 'bedrooms', 'livingarea', 'yearbuilt', 'propertytype',
    'location_latitude', 'location_longitude',
    'address_zipcode', 'address_city', 'address_state',
    'price_value', 'price_pricepersquarefoot',
    'details_description', 'details_zestimate', 'details_timeonzillow', 'details_resofacts_parkingfeatures',
    'details_resofacts_garageparkingcapacity', 'details_resofacts_hasattachedgarage',
    'details_resofacts_laundryfeatures',
    'details_resofacts_appliances', 'details_resofacts_exteriorfeatures', 'details_resofacts_patioandporchfeatures',
    'details_resofacts_hoafee', 'details_resofacts_communityfeatures', 'details_homeinsights'
]

DESCRIPTION_DICT = {
    'bathrooms' : 'Number of Bathrooms = '
    , 'bedrooms' : 'Number of bedrooms = '
    , 'livingarea' : 'living area '
    , 'yearbuilt' : 'The property was built at '
    , 'propertytype' : 'property type '
    # ,'location_latitude'
    # , 'location_longitude'
    # ,'address_zipcode'
    # , 'address_city'
    # , 'address_state'
    # ,'price_value'
    # , 'price_pricepersquarefoot'
    # ,'details_description'
    # , 'details_zestimate'
    # , 'details_timeonzillow'
    , 'details_resofacts_parkingfeatures' : 'These are the parking features '
    ,'details_resofacts_garageparkingcapacity' : 'The garage parking capacity '
    , 'details_resofacts_hasattachedgarage' : 'has attached garage? '
    ,'details_resofacts_laundryfeatures' : 'The laundry features '
    ,'details_resofacts_appliances' : 'appliances '
    , 'details_resofacts_exteriorfeatures' : 'exterior features '
    , 'details_resofacts_patioandporchfeatures': 'patio and porch features '
    ,'details_resofacts_hoafee': 'hoa fee '
    , 'details_resofacts_communityfeatures': 'community features '
    , 'details_homeinsights': 'home insights '
}

IMAGE_FIELDS = ['details_photourlshighres']
METADATA_FIELDS = [
    'bathrooms', 'bedrooms', 'livingarea', 'yearbuilt', 'propertytype',
    'location_latitude', 'location_longitude',
    'address_streetaddress', 'address_zipcode', 'address_city', 'address_state',
    'price_value', 'price_pricepersquarefoot'
]
PRECISION_FIELDS = [
    'bathrooms', 'bedrooms', 'propertytype',
    'address_city', 'address_state', 'price_value'
]
TABLE_NAME = 'property'
TEXT_DATA_QUERY = f"""SELECT {", ".join(PRIMARY_FIELD+DESCRIPTION_FIELDS)} FROM {TABLE_NAME}"""
PRECISION_DATA_QUERY = f"""SELECT {", ".join(PRIMARY_FIELD+PRECISION_FIELDS)} FROM {TABLE_NAME}"""
IMG_DATA_QUERY = f"""SELECT {", ".join(PRIMARY_FIELD+IMAGE_FIELDS+METADATA_FIELDS)} FROM {TABLE_NAME}"""

