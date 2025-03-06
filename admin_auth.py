import bcrypt
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/road_damage_db"
client = MongoClient(MONGO_URI)
db = client["road_damage_db"]
admin_collection = db["admins"]

# Initialize admin (Run this once to create an admin user)
def create_admin():
    email = "admin4@example.com"
    password = "admin1233"
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    if admin_collection.find_one({"email": email}):
        print("Admin already exists.")
    else:
        admin_collection.insert_one({"email": email, "password": hashed_pw})
        print("Admin created successfully.")

# Verify admin credentials
def verify_admin(email, password):
    admin = admin_collection.find_one({"email": email})
    if admin and bcrypt.checkpw(password.encode("utf-8"), admin["password"]):
        return True
    return False

# Run this script directly to create an admin user
if __name__ == "__main__":
    create_admin()

