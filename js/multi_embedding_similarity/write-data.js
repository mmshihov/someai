'use strict'

//const debug = require("debug")("sam_container:aimsSet:debug");
const info = require("debug")("patchout:write:info");

const { MongoClient } = require('mongodb');
const config = require("./config.json");
const fnames = require("./data/fnames.json");

const TENANT_NAME = config.mongo.tenantId;


async function saveItem(vector, vectorIndex, alias, tenantId) {
    info("aimsSet(%o, %s, %s)", vector, alias, tenantId);

    const client = new MongoClient(config.mongo.connectionString);

    const database = client.db(config.mongo.db);
    const collection = database.collection(config.mongo.collection);
    
    let doc = { 
      alias,     
      vectorIndex, 
      [config.mongo.tenantIdFieldName]: tenantId,
      [config.mongo.vectorFieldName]: vector
    };

    const insertResult = await collection.insertOne(doc);

    let result = {
      aimId: insertResult.insertedId.toHexString()
    };

    info("aimsSet(...) returns %o", result);
    return result;
}

async function main() {
  info("Writing is started...");

  for (let f of fnames) {
    if (f.type === "original") {

      let vectorIndex = 0;
      const embeddings = require(`./data/embeddings/${f.name}.json`);
      for (let e of embeddings.audio.embeddings) {

        console.log("fname:", embeddings.audio.name, "embedding[", vectorIndex, "]")
        await saveItem(e, vectorIndex, embeddings.audio.name, TENANT_NAME);

        vectorIndex++;
      }

      info("Timeout...");
      await new Promise(r => setTimeout(r, 2000)); // just timeout for mongo
    }
  }

//  const vectorObj = require(`./data/${model}.json`);
//  for (let item of vectorObj.vectors) {
//    //await saveItem(item.vector, item.alias, `${MODEL}_tenant`);
//  }
}

main().then((res, rej) => {console.log(res)}).catch(console.dir);
