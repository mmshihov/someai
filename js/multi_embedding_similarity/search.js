const info = require("debug")("patchout:search:info");

const { MongoClient } = require('mongodb');

const config = require("./config.json")
const fnames = require("./data/fnames.json");

const TENANT_NAME = config.mongo.tenantId;
const LIMIT=10;

async function search(vector, limit, tenant) {
    const client = new MongoClient(config.mongo.connectionString);

    const database = client.db(config.mongo.db);
    const collection = database.collection(config.mongo.collection);

    const agg = [
      {
        '$vectorSearch': {
          'index': config.mongo.vectorIndex,
          'path': config.mongo.vectorFieldName,
          'queryVector': vector,
          'numCandidates': 150,
          'limit': limit,
          'filter': { [config.mongo.tenantIdFieldName]: tenant } // additional filtering by tenancy
        }
      }, {
        '$project': {
          '_id': 0,
          'aimId': '$_id',
          'alias': 1,
          [config.mongo.tenantIdFieldName]: 1,
          [config.mongo.vectorFieldName]: 1,
          'distance': {
            '$meta': 'vectorSearchScore'
          }
        }
      }
    ];      

    const aggregateResult = collection.aggregate(agg);
    
    let result = [];

    await aggregateResult.forEach((item) => {
      result.push({
        distance: item.distance,
        alias:    item.alias,
        // tenant:   item[config.mongo.tenantIdFieldName]
      })      
    });

    return result;
}

function compareResults(ra, rb) {
  if (ra.distance > rb.distance) {
    return -1;
  } else if (ra.distance < rb.distance) {
    return 1;
  }
  return 0;
}

// finds max distance
function filterResult(searchResults) {
  let filtered = [];
  filtered.push(searchResults[0]);
  for (let i=1; i<searchResults.length; i++) {

    let isAppendNeeded = true;
    for (let j=0; j<filtered.length; j++) {

      if (filtered[j].alias == searchResults[i].alias) {
        filtered[j].distance = Math.max(filtered[j].distance, searchResults[i].distance);
        isAppendNeeded = false;
        break;
      }
    }

    if (isAppendNeeded) {
      filtered.push(searchResults[i]);
    }
  }

  filtered.sort(compareResults);
  return filtered;
}

function combineResults(prev, curr) {
  let comby = [];
  for (let i=0; i<prev.length; i++) {

    let isAppendNeed = true;
    for (let j=0; j<curr.length; j++) {

      if (prev[i].alias == curr[j].alias) {
        let newSearch = {
          distance: prev[i].distance + curr[j].distance,
          alias: prev[i].alias,
        };
        comby.push(newSearch)
        isAppendNeed = false;
        break;
      }

    }

    if (isAppendNeed) {
      comby.push(prev[i]);
    }
  }

  for (let j=0; j<curr.length; j++) {

    let isAppendNeed = true;
    for (let i=0; i<prev.length; i++) {

      if (prev[i].alias == curr[j].alias) {
        isAppendNeed = false;
        break;
      }
    }

    if (isAppendNeed) {
      comby.push(curr[j]);
    }
  }

  comby.sort(compareResults);
  return comby;
}

function guessResultsForAudio(results, originalAudioName) {
  let prevFiltered = filterResult(results[0]);

  for (let i = 1; i < results.length; i++) {
    let filtered = filterResult(results[i]);
    prevFiltered = combineResults(prevFiltered, filtered);
  }

  for (let i=0; i < prevFiltered.length; i++) {
    prevFiltered[i].distance = prevFiltered[i].distance / results.length;
  }
  return {isGuess: prevFiltered[0].alias == originalAudioName, variants: prevFiltered}; //guess: TODO: bool
}

async function main() {
  info("Started...");

  let guessCount = 0;
  let failCount = 0;

  for (let f of fnames) {
    if (f.type === "diffusion") {
      const embeddings = require(`./data/embeddings/${f.name}.json`);

      let searchResults = []
      for (let e of embeddings.audio.embeddings) {
        let r = await search(e, LIMIT, TENANT_NAME);
        searchResults.push(r);
        // info("Song(%s): %o", embeddings.audio.name, r);
      }

      let guess = guessResultsForAudio(searchResults, f.original);
      info("song: %s -> %o", f.name, guess);      
      if (guess.isGuess) {
        guessCount++;
      } else {
        failCount++;
      }
    }
  }

  info("Total guess:fail = %i:%i", guessCount, failCount);
}

main().then((res, rej) => {console.log(res)}).catch(console.dir);


