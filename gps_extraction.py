import os
import requests
from PIL import Image, ExifTags
from fractions import Fraction  # Import for conversion

# Load Google Maps API Key (Store securely)
GOOGLE_MAPS_API_KEY = "Your API Key"

def convert_to_degrees(value):
    """Convert EXIF GPS coordinates to decimal degrees."""
    if isinstance(value, tuple):
        return float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600
    return float(value)

def extract_gps(image_path):
    """Extract GPS coordinates from image metadata (EXIF)."""
    image = Image.open(image_path)
    exif_data = image._getexif()

    if not exif_data:
        return {"gps_latitude": None, "gps_longitude": None, "location": "Unknown"}

    gps_info = {}
    for tag, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for gps_tag, gps_value in value.items():
                gps_name = ExifTags.GPSTAGS.get(gps_tag, gps_tag)
                gps_info[gps_name] = gps_value

    if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        lon = convert_to_degrees(gps_info["GPSLongitude"])
        lat_ref = gps_info["GPSLatitudeRef"]
        lon_ref = gps_info["GPSLongitudeRef"]

        lat_final = lat * (-1 if lat_ref == "S" else 1)
        lon_final = lon * (-1 if lon_ref == "W" else 1)

        # Fetch human-readable address
        location = get_location_from_gps(lat_final, lon_final)

        return {"gps_latitude": float(lat_final), "gps_longitude": float(lon_final), "location": location}
    
    return {"gps_latitude": None, "gps_longitude": None, "location": "Unknown"}

def get_location_from_gps(lat, lon):
    """Fetches location name from GPS coordinates using Google Maps API."""
    if GOOGLE_MAPS_API_KEY:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0]["formatted_address"]
    return "Unknown Location"
