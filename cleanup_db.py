from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.mall_analytics  # replace with your DB name

KEEP = 30  # number of rows to keep in each collection

# Loop over all collections in the database
for col_name in db.list_collection_names():
    col = db[col_name]
    
    # Get the last KEEP documents
    latest = col.find().sort("_id", -1).limit(KEEP)
    latest_ids = [doc["_id"] for doc in latest]
    
    # Delete all other documents
    deleted = col.delete_many({"_id": {"$nin": latest_ids}})
    
    print(f"Collection '{col_name}': Old rows deleted = {deleted.deleted_count}")
