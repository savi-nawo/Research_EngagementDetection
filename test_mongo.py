from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["mall_analytics"]
col = db["engagement_logs"]

col.insert_one({"test": "hello"})
print("Inserted successfully")
