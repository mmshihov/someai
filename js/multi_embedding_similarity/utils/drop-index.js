const { MongoClient, ServerApiVersion } = require('mongodb');

const config = require("../config.json");

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(config.mongo.connectionString);

const DB_NAME = "SAMC"
const COLLECTION_NAME = "BETA_AAI"
const VECTOR_INDEX_NAME = "BETA_AAI_VECTOR"


async function run() {
    try {
        await client.connect();

        let collection = client.db(DB_NAME).collection(COLLECTION_NAME);

        const result = await collection.dropSearchIndex(VECTOR_INDEX_NAME);

        console.log(result);
    } finally {
        await client.close();
    }
}
run().catch(console.dir);
