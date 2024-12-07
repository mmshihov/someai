const { MongoClient, ServerApiVersion } = require('mongodb');

const config = require("../config.json");

const client = new MongoClient(config.mongo.connectionString);

async function run() {
    try {

        await client.connect();
        let collection = await client.db(config.mongo.db).createCollection(config.mongo.collection);
        //let collection = client.db(config.mongo.db).collection(config.mongo.collection);

        const index = {
                 name: config.mongo.vectorIndex,
                 type: "vectorSearch",
                 definition: {
                    "fields": [
                      {
                       "type": "vector",
                       "numDimensions": config.mongo.vectorDimension,
                       "path":          config.mongo.vectorFieldName,
                       "similarity":    config.mongo.similarity
                      },
                      {
                        "type": "filter",
                        "path": config.mongo.tenantIdFieldName
                      }                   
                    ]
                 }
             }

        const result = await collection.createSearchIndex(index);
        console.log(result);
        console.log("DONE");
    } finally {
        await client.close();
    }
}
run().catch(console.dir);
